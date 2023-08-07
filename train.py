import math
import os
import random
import time
from functools import partial

import h5py
import numpy as np
import torch
from transformers import AutoTokenizer

from model import T5

# I/O
out_dir = "out"
eval_interval = 10
log_interval = 1
eval_iters = 1
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
# wandb logging
wandb_log = True  # disabled by default
wandb_project = "tinystories"
wandb_run_name = "ul2"  # 'run' + str(time.time())
# data
gradient_accumulation_steps = 5 * 8
batch_size = 16
# model
encoder_block_size = 256
decoder_block_size = 128
# adafactor optimizer
max_iters = 100  # total number of training iterations
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
# exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained("ul2-tinystories-tokenizer")


def mask_spans(tokens, mu, r, tokenizer, prefix_lm=False, prepend=None):
    # masked_tokens = tokens[:]

    encoder_inputs = [tokenizer.vocab[prepend]] if prepend is not None else []
    targets = []

    # Original T5 code reused tokens at the end of vocab for sentinels
    # https://github.com/google-research/text-to-text-transfer-transformer/blob/258fd30687e6c60d18b7204d009dc5c753142987/t5/data/preprocessors.py#L3106C6-L3106C6
    sentinel_id = tokenizer.vocab_size - 1

    if prefix_lm:
        # n = 1
        mu = max(1, int(len(tokens) * r))
        start = max(
            0, len(tokens) - random.randint(1, int(2 * mu))
        )  # max to handle start < 0
        encoder_inputs += tokens[:start] + [sentinel_id]
        targets += tokens[start:]
        # for i in range(start, len(tokens)):
        #     masked_tokens[i] = 0

    else:
        # n = ceil(len(tokens) / mu)
        prev_span_unmasked = False
        start = 0
        end = 0
        while start < len(tokens):
            # uniform random span length
            length = random.randint(1, int(2 * mu))
            end = min(start + length, len(tokens))

            # randomly decide if span should be masked
            if np.random.binomial(1, p=r):
                encoder_inputs.append(sentinel_id)
                targets += tokens[start:end]
                # for i in range(start, end):
                #     masked_tokens[i] = 0
                prev_span_unmasked = False
                sentinel_id -= 1
            else:
                encoder_inputs += tokens[start:end]
                # if previous span was also unmasked we don't need to keep adding the sentinel token
                if not prev_span_unmasked:
                    targets.append(sentinel_id)
                    prev_span_unmasked = True
            start = end

    targets.append(tokenizer.eos_token_id)
    decoder_inputs = (
        [tokenizer.vocab[prepend]] + targets[:-1]
        if prepend is not None
        else [tokenizer.eos_token_id] + targets[:-1]
    )

    return encoder_inputs, decoder_inputs, targets


# Create mixture-of-denoisers
r_denoisers = [
    partial(mask_spans, mu=3, r=0.15, tokenizer=tokenizer, prepend="[R]"),
    partial(mask_spans, mu=8, r=0.15, tokenizer=tokenizer, prepend="[R]"),
]

s_denoisers = [
    partial(
        mask_spans, mu=None, r=0.25, tokenizer=tokenizer, prefix_lm=True, prepend="[S]"
    ),
]

x_denoisers = [
    partial(mask_spans, mu=3, r=0.5, tokenizer=tokenizer, prepend="[X]"),
    partial(mask_spans, mu=8, r=0.5, tokenizer=tokenizer, prepend="[X]"),
    partial(mask_spans, mu=32, r=0.15, tokenizer=tokenizer, prepend="[X]"),
    partial(mask_spans, mu=32, r=0.5, tokenizer=tokenizer, prepend="[X]"),
]

denoisers = r_denoisers + s_denoisers + x_denoisers


# Open file
train_file = h5py.File("data/tinystories/train.h5", "r")
train_data = train_file["train"]
val_file = h5py.File("data/tinystories/validation.h5", "r")
val_data = val_file["validation"]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = random.randint(0, len(data) - batch_size)
    seq = data[ix : ix + batch_size]
    rand_denoiser = random.choice(denoisers)

    encoder_inputs = torch.zeros(
        batch_size,
        encoder_block_size,
        dtype=torch.int64,
        pin_memory=True,
    )
    decoder_inputs = torch.zeros(
        batch_size,
        decoder_block_size,
        dtype=torch.int64,
        pin_memory=True,
    )
    targets = (
        torch.zeros(
            batch_size,
            decoder_block_size,
            dtype=torch.int64,
            pin_memory=True,
        )
        - 1  # empty targets get -1 so they're ignored in the loss
    )

    e_lengths = []
    d_lengths = []

    for i, s in enumerate(seq):
        # print(ix, i)
        e, d, t = rand_denoiser(s.tolist())

        e_lengths.append(min(encoder_block_size, len(e)))
        d_lengths.append(min(decoder_block_size, len(d)))

        encoder_inputs[i, : len(e)] = torch.tensor(e[:encoder_block_size])
        decoder_inputs[i, : len(d)] = torch.tensor(d[:decoder_block_size])
        targets[i, : len(t)] = torch.tensor(t[:decoder_block_size])

    return (
        encoder_inputs.cuda(),
        decoder_inputs.cuda(),
        targets.cuda(),
        e_lengths,
        d_lengths,
    )


iter_num = 0
best_val_loss = 1e9

model_args = dict(
    n_encoder_layer=2,
    n_decoder_layer=2,
    n_head=8,
    d_model=512,
    vocab_size=tokenizer.vocab_size,
    encoder_context_size=encoder_block_size,
    decoder_context_size=decoder_block_size,
    relative_attn_n_buckets=16,
    relative_attn_max_distance=64,
)
model = T5(**model_args)
model.cuda()

optimizer = model.configure_optimizers()


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            EX, DX, Y, _, _ = get_batch(split)
            logits, loss = model(EX, DX, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# logging
if wandb_log:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

EX, DX, Y, _, _ = get_batch("train")
t0 = time.time()
local_iter_num = 0

while True:
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    # "lr": lr,
                    # "mfu": running_mfu*100, # convert to percentage
                }
            )
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

    for micro_step in range(gradient_accumulation_steps):
        logits, loss = model(EX, DX, Y)
        loss = (
            loss / gradient_accumulation_steps
        )  # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        EX, DX, Y, _, _ = get_batch("train")

        loss.backward()

    # clip the gradient
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        # if local_iter_num >= 5: # let the training loop settle a bit

        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
