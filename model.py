import math
from typing import Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.optimization import Adafactor


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        context_size: int,
        bidirectional: bool = True,
        attn_pdrop: float = 0.1,
        relative_attn_n_buckets: int = 32,
        relative_attn_max_distance: int = 128,
    ) -> None:
        super().__init__()
        assert d_model % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        # output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        # self.resid_dropout = nn.Dropout(resid_pdrop)
        if not bidirectional:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(context_size, context_size)).view(
                    1, 1, context_size, context_size
                ),
            )
        self.bidirectional = bidirectional
        self.n_head = n_head
        self.d_model = d_model
        self.relative_attn_n_buckets = relative_attn_n_buckets
        self.relative_attn_max_distance = relative_attn_max_distance

        self.relative_attn_bias = nn.Embedding(
            self.relative_attn_n_buckets, self.n_head
        )

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, n_buckets=32, max_distance=128
    ):
        """
        code ref: https://github.com/huggingface/transformers/blob/v4.21.2/src/transformers/models/t5/modeling_t5.py#L374
        """
        relative_buckets = 0
        if bidirectional:
            n_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * n_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position)
            )
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = n_buckets // 2
        is_small = relative_position < max_exact

        # reserve a bucket for tokens > max_distance.
        # NOTE: the original Mesh Tensorflow and Huggingface implementation don't do this,
        # and it results in max_distance not working as described in the paper.
        max_exact -= 1
        n_buckets -= 1

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (n_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, n_buckets),
        )

        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias

        code ref: https://github.com/huggingface/transformers/blob/v4.21.2/src/transformers/models/t5/modeling_t5.py#L421
        """
        if device is None:
            device = self.relative_attn_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[
            :, None
        ]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[
            None, :
        ]
        relative_position = (
            memory_position - context_position
        )  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=self.bidirectional,
            n_buckets=self.relative_attn_n_buckets,
            max_distance=self.relative_attn_max_distance,
        )
        values = self.relative_attn_bias(
            relative_position_bucket
        )  # shape (query_length, key_length, num_heads)
        values = einops.rearrange(
            values, "Q K (B nh) -> B nh Q K", B=1
        )  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        queries: Tensor,
        keys: Optional[Tensor] = None,
        values: Optional[Tensor] = None,
    ) -> Tensor:
        # batch size, sequence length, embedding dimensionality (d_model)
        # do self attention if only one input
        if keys is None:
            keys = queries
        if values is None:
            values = queries

        qT = queries.size(1)
        kT = keys.size(1)

        q = self.q_proj(queries)
        k = self.k_proj(keys)
        v = self.v_proj(values)

        q, k, v = map(
            lambda t: einops.rearrange(
                t,
                "B T (nh hs) -> B nh T hs",
                nh=self.n_head,
                hs=self.d_model // self.n_head,
            ),
            (q, k, v),
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        position_bias = self.compute_bias(qT, kT)
        att += position_bias
        if not self.bidirectional:
            att = att.masked_fill(self.bias[:, :, :qT, :kT] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        o = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        o = einops.rearrange(
            o, "B nh T hs -> B T (nh hs)"
        )  # re-assemble all head outputs side by side

        # output projection
        o = self.out_proj(o)
        return o


class RMSLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        """Root Mean Square Layer Normalization

        code ref: https://github.com/huggingface/transformers/blob/b487096b02307cd6e0f132b676cdcc7255fe8e74/src/transformers/models/t5/modeling_t5.py#L239
        """
        super().__init__()
        self.w = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)  # B, T, 1
        x = x * torch.rsqrt(variance + 1e-6)  # B, T, d_model
        return self.w * x


class SwiGLUMLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        mlpf_pdrop: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.w12 = nn.Linear(d_model, 2 * 4 * d_model, bias=bias)
        self.dropout = nn.Dropout(mlpf_pdrop)
        self.w3 = nn.Linear(4 * d_model, d_model, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        x = F.silu(x1) * x2
        x = self.dropout(x)
        return self.w3(x)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        context_size: int,
        mlpf_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        relative_attn_n_buckets: int = 32,
        relative_attn_max_distance: int = 128,
    ) -> None:
        super().__init__()

        self.ln_1 = RMSLayerNorm(d_model)
        self.attn = MultiheadAttention(
            d_model,
            n_head,
            context_size,
            bidirectional=True,
            attn_pdrop=attn_pdrop,
            relative_attn_n_buckets=relative_attn_n_buckets,
            relative_attn_max_distance=relative_attn_max_distance,
        )
        self.dropout_1 = nn.Dropout(resid_pdrop)
        self.ln_2 = RMSLayerNorm(d_model)
        self.mlpf = SwiGLUMLP(d_model, mlpf_pdrop, bias=False)
        self.dropout_2 = nn.Dropout(resid_pdrop)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.dropout_1(self.attn(self.ln_1(x)))
        x = x + self.dropout_2(self.mlpf(self.ln_2(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        context_size: int,
        mlpf_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        relative_attn_n_buckets: int = 32,
        relative_attn_max_distance: int = 128,
    ) -> None:
        super().__init__()
        self.ln_1 = RMSLayerNorm(d_model)
        self.self_attn = MultiheadAttention(
            d_model,
            n_head,
            context_size,
            bidirectional=False,
            attn_pdrop=attn_pdrop,
            relative_attn_n_buckets=relative_attn_n_buckets,
            relative_attn_max_distance=relative_attn_max_distance,
        )
        self.dropout_1 = nn.Dropout(resid_pdrop)
        self.ln_2 = RMSLayerNorm(d_model)
        self.cross_attn = MultiheadAttention(
            d_model,
            n_head,
            context_size,
            bidirectional=True,
            attn_pdrop=attn_pdrop,
            relative_attn_n_buckets=relative_attn_n_buckets,
            relative_attn_max_distance=relative_attn_max_distance,
        )
        self.dropout_2 = nn.Dropout(resid_pdrop)
        self.ln_3 = RMSLayerNorm(d_model)
        self.mlpf = SwiGLUMLP(d_model, mlpf_pdrop, bias=False)
        self.dropout_3 = nn.Dropout(resid_pdrop)

    def forward(self, x: Tensor, hidden_states: Tensor) -> Tensor:
        x = x + self.dropout_1(self.self_attn(self.ln_1(x)))
        x = x + self.dropout_2(
            self.cross_attn(self.ln_2(x), hidden_states, hidden_states)
        )
        x = x + self.dropout_3(self.mlpf(self.ln_3(x)))
        return x


class T5(nn.Module):
    def __init__(
        self,
        n_encoder_layer: int,
        n_decoder_layer: int,
        n_head: int,
        d_model: int,
        vocab_size: int,
        encoder_context_size: int,
        decoder_context_size: int,
        stack_pdrop: float = 0.1,
        mlpf_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        relative_attn_n_buckets: int = 32,
        relative_attn_max_distance: int = 128,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.encoder_context_size = encoder_context_size
        self.decoder_context_size = decoder_context_size
        self.shared_embedding = nn.Embedding(vocab_size, d_model)

        self.encoder = nn.ModuleDict(
            {
                "wte": self.shared_embedding,
                "drop_in": nn.Dropout(stack_pdrop),
                "h": nn.ModuleList(
                    [
                        EncoderBlock(
                            d_model,
                            n_head,
                            encoder_context_size,
                            mlpf_pdrop,
                            attn_pdrop,
                            resid_pdrop,
                            relative_attn_n_buckets,
                            relative_attn_max_distance,
                        )
                        for _ in range(n_encoder_layer)
                    ]
                ),
                "drop_out": nn.Dropout(stack_pdrop),
            }
        )

        self.decoder = nn.ModuleDict(
            {
                "wte": self.shared_embedding,
                "drop_in": nn.Dropout(stack_pdrop),
                "h": nn.ModuleList(
                    [
                        DecoderBlock(
                            d_model,
                            n_head,
                            decoder_context_size,
                            mlpf_pdrop,
                            attn_pdrop,
                            resid_pdrop,
                            relative_attn_n_buckets,
                            relative_attn_max_distance,
                        )
                        for _ in range(n_decoder_layer)
                    ]
                ),
                "drop_out": nn.Dropout(stack_pdrop),
            }
        )

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        src_idx: Tensor,
        dst_idx: Tensor,
        targets: Optional[Tensor] = None,
    ) -> Tensor:
        # st, dt = src_idx.size(1), dst_idx.size(1)
        # assert (
        #     st <= self.context_size and dt <= self.context_size
        # ), f"Cannot forward sequence of length {st} or {dt}, context size is only {self.context_size}"

        # forward through encoder stack
        src_emb = self.encoder.wte(src_idx)
        x = self.encoder.drop_in(src_emb)
        for block in self.encoder.h:
            x = block(x)
        hidden_states = self.encoder.drop_out(x)

        # forward through decoder stack
        dst_emb = self.decoder.wte(dst_idx)
        x = self.decoder.drop_in(dst_emb)
        for block in self.decoder.h:
            x = block(x, hidden_states)
        x = self.decoder.drop_out(x)

        # unembedding
        logits = self.lm_head(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def configure_optimizers(self):
        return Adafactor(self.parameters())

    @torch.no_grad()
    def generate(
        self, encoder_idx, decoder_idx, max_new_tokens, temperature=1.0, top_k=None
    ):
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            encoder_idx_cond = (
                encoder_idx
                if encoder_idx.size(1) <= self.encoder_context_size
                else encoder_idx[:, -self.encoder_context_size :]
            )
            decoder_idx_cond = (
                decoder_idx
                if decoder_idx.size(1) <= self.decoder_context_size
                else decoder_idx[:, -self.decoder_context_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(encoder_idx_cond, decoder_idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            decoder_idx = torch.cat((decoder_idx, idx_next), dim=1)

        return decoder_idx
