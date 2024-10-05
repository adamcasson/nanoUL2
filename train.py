"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
from functools import partial

import numpy as np
import tiktoken
import torch
from torch.nn.attention.flex_attention import create_block_mask, or_masks
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from denoising import (
    denoise,
    noise_span_to_unique_sentinel,
    nonnoise_span_to_unique_sentinel,
    random_prefix_noise_mask,
    random_spans_helper,
    random_spans_noise_mask,
)
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# UL2 hyperparams
causal_only = False # if True, falls back to regular fully causal langauge modeling
mean_noise_span_lengths = [3.0, 8.0, 3.0, 8.0, 64.0, 64.0, None] # mean length of corrupted spans, use None for S-denoisers
noise_densities = [0.15, 0.15, 0.5, 0.5, 0.15, 0.5, 0.25] # rate of corruption
optional_task_prefixes = ["[NLU]", "[NLU]", "[NLG]", "[NLG]", "[NLG]", "[NLG]", "[S2S]"] # mode token prepended to input
rates = [0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.22] # probability of picking each denoiser, the paper says ~20% for S-denoisers
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

assert len(mean_noise_span_lengths) == len(noise_densities) == len(optional_task_prefixes) == len(rates)
assert set(optional_task_prefixes).issubset({"[NLU]", "[S2S]", "[NLG]", None})
noise_hparams = [(mu, r, pre) for mu, r, pre in zip(mean_noise_span_lengths, noise_densities, optional_task_prefixes)]

# we need the tokenizer for tokenizing any mode prefix tokens
gpt2_base_enc = tiktoken.get_encoding("gpt2")

# map mode tokens start at the end of GPT-2 vocab
mode_tokens = {
        "[NLU]": gpt2_base_enc.max_token_value + 1,
        "[S2S]": gpt2_base_enc.max_token_value + 2,
        "[NLG]": gpt2_base_enc.max_token_value + 3,
}

# map sentinel tokens starting at the end of the vocab, i.e.
# {
#     "<|sentinel_token_0|>": 50431,
#     "<|sentinel_token_1|>": 50430,
#     etc.
# }
sentinel_tokens = {f"<|sentinel_token_{i}|>": vidx for i, vidx in enumerate(range(50432-1, gpt2_base_enc.max_token_value + 3, -1))}

# extend gpt2 tokenizer with the sentinel tokens
enc = tiktoken.Encoding(
    name="gpt2_ul2",
    pat_str=gpt2_base_enc._pat_str,
    mergeable_ranks=gpt2_base_enc._mergeable_ranks,
    special_tokens={
        **gpt2_base_enc._special_tokens,
        **mode_tokens,
        **sentinel_tokens,
    }
)

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
   
    if split == 'train' and not causal_only:
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        # set y array to -1 to ignore xentropy loss at that position
        x = torch.zeros(batch_size, block_size, dtype=torch.int64, pin_memory=True)
        y = torch.zeros(batch_size, block_size, dtype=torch.int64, pin_memory=True) - 1
        position_ids = torch.arange(0, block_size, dtype=torch.int64, pin_memory=True).repeat(batch_size, 1)
        # position_ids = torch.zeros(batch_size, block_size, dtype=torch.int64, pin_memory=True)
        prefix_lens = []

        for i in range(batch_size):
            # select a random set of denoiser params according the probability in `rates`
            noise_span_length, noise_density, task_prefix = noise_hparams[np.random.choice(len(noise_hparams), p=rates)]
            task_prefix_tokens = [] if task_prefix is None else torch.tensor(enc.encode(task_prefix, allowed_special="all"), dtype=torch.int64)
            
            # if the denoiser isn't an S-denoiser (PrefixLM)
            if noise_span_length is not None:
                # helper that tells us to denoise a sequence of size N with the chosen hparams
                # to ensure it fits as closely as possible to our context size to minimize padding
                random_chunk_size, _ = random_spans_helper(
                    block_size - len(task_prefix_tokens) + 1,
                    noise_density,
                    noise_span_length,
                    extra_tokens_per_span_inputs=1,
                    extra_tokens_per_span_targets=1,
                    decoder_only=True,
                )
            else:
                # for S-denoisers
                random_chunk_size = block_size - len(task_prefix_tokens) + 1
            
            # select a random chunk of tokens from the dataset
            ix = torch.randint(len(data) - random_chunk_size, (1,))
            random_chunk = torch.from_numpy((data[ix:ix+random_chunk_size]).astype(np.int64))

            noise_mask_fn = partial(
                random_spans_noise_mask,
                noise_density=noise_density,
                mean_noise_span_length=noise_span_length
            ) if noise_span_length is not None else partial(random_prefix_noise_mask, noise_density=noise_density)
           
            # corrupt the sequence and make the inputs/targets (will be concat together for decoder-only model)
            noise_mask = noise_mask_fn(len(random_chunk))
            inputs, targets = denoise(
                random_chunk,
                noise_mask,
                inputs_fn=partial(noise_span_to_unique_sentinel, vocab_size=enc.n_vocab),
                targets_fn=partial(nonnoise_span_to_unique_sentinel, vocab_size=enc.n_vocab),
            )

            # for PrefixLM task, strip the sentinel tokens for decoder-only setting
            if noise_span_length is None:
                inputs = inputs[:-1]
                targets = targets[1:]
           
            input_start, input_end = len(task_prefix_tokens), len(task_prefix_tokens) + len(inputs)

            if task_prefix is not None:
                x[i, :len(task_prefix_tokens)] = task_prefix_tokens

            x[i, input_start:input_end] = inputs
            x[i, input_end:input_end+len(targets)-1] = targets[:-1]
            y[i, input_end:input_end+len(targets)-1] = targets[1:]

            prefix_lens.append(input_end)
    else:
        # for val (or causal_only=True) we can do full CasualLM objective
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix]).pin_memory()
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix]).pin_memory()
        prefix_lens = None
    
    if device_type == 'cuda':
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    prefix_lens = torch.tensor(prefix_lens, dtype=torch.int32).to(device) if prefix_lens is not None else None
    
    def _get_attn_mask_fn(prefix_lens=None):
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        def prefix_mask(b, h, q_idx, kv_idx):
            return kv_idx < prefix_lens[b]

        if prefix_lens is None:
            return causal_mask
        else:
            return or_masks(prefix_mask, causal_mask)
    
    B, S = x.shape
    attn_mask = create_block_mask(_get_attn_mask_fn(prefix_lens), B, None, S, S, device=x.device)
    
    return x, y, attn_mask
# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50432 rounded up for efficiency and special tokens)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50432
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, attn_mask = get_batch(split)
            with ctx:
                logits, loss = model(X, attn_mask, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y, attn_mask = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, attn_mask, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y, attn_mask = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
