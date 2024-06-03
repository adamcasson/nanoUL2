from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW


class RotaryEmbedding(nn.Module):
    """rotary embedding impl. from huggingface modelling_llama.py
    """
    def __init__(
        self,
        d_model: int,
        context_size: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.d_model = d_model
        self.context_size = context_size
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.d_model, 2, dtype=torch.int64).float() / self.d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = context_size

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.LongTensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        rope_theta: float,
        context_size: int,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        head_dim = d_model // n_head
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.rotary_emb = RotaryEmbedding(
            head_dim,
            context_size=context_size,
            base=rope_theta,
        )

    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.n_head, C // self.n_head)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        # TODO: replace with custom PrefixLM kernel
        x = (
            F.scaled_dot_product_attention(q, k, v, is_causal=True)
            .transpose(1, 2)
            .reshape(B, N, C)
        )

        x = self.proj(x)
        return x


class FeedForward(nn.Module):
    """SwiGLU MLP with no biases
    """
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
    ) -> None:
        super().__init__()
        assert d_hidden % 2 == 0
        self.fc12 = nn.Linear(d_model, d_hidden * 2, bias=False)
        self.act = nn.SiLU()
        self.fc3 = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc12(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * self.act(x2)
        x = self.fc3(x)
        return x


class RMSNorm(nn.Module):
    """RMS norm impl. from mistral model.py
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        n_head: int,
        rope_theta: float,
        context_size: int,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = Attention(
             d_model,
             n_head=n_head,
             rope_theta=rope_theta,
             context_size=context_size,
        )

        self.norm2 = RMSNorm(d_model)
        self.ffn = FeedForward(
            d_model=d_model,
            d_hidden=d_hidden,
        )

    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), position_ids)
        x = x + self.ffn(self.norm2(x))
        return x


@dataclass
class TransformerConfig:
    context_size: int = 1024
    vocab_size: int = 50432 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 256 for efficiency and space for sentinel tokens
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    d_hidden: int = 2048
    rope_theta: float = 10000.0


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)

        self.blocks = nn.ModuleList(
            [
                Block(
                    d_model=config.d_model,
                    d_hidden=config.d_hidden,
                    n_head=config.n_head,
                    rope_theta=config.rope_theta,
                    context_size=config.context_size,
                )
                for _ in range(config.n_layer)
            ]
        )

        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embeddings.weight.numel()
        return n_params

    def forward(
            self,
            x: torch.LongTensor,
            position_ids: torch.LongTensor,
            targets: Optional[torch.LongTensor] = None,
        ) -> torch.Tensor:
        x = self.embeddings(x)

        for blk in self.blocks:
            x = blk(x, position_ids)

        x = self.norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: tuple, device_type: str) -> torch.optim.Optimizer:
        return AdamW(self.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)

    def estimate_mfu(self, fwdbwd_per_iter: float, dt: float) -> float:
        """ estimate model flops utilization (MFU) in units of 4090 bfloat16 per FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.d_model//cfg.n_head, cfg.context_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of 4090 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt)  # per second
        flops_promised = 165e12  # 4090 GPU bloat16 peak flops is 165 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad
    def generate(self, idx: torch.LongTensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None) -> torch.LongTensor:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        position_ids = torch.arange(0, self.config.context_size).repeat(idx.size(0), 1).to(idx.device)
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.context_size else idx[:, -self.config.context_size:]
            # crop position ids to match current length of idx_cond
            position_ids_cond = position_ids[:, :idx_cond.size(1)]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond, position_ids_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
