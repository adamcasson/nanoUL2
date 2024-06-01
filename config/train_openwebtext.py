# config for training Transformer (124M)
# $ python train.py config/train_openwebtext.py

out_dir = 'out'

wandb_log = True
wandb_project = 'openwebtext'
wandb_run_name='ul2-decoder-124M'

# these make the total batch size be ~0.5M
# 16 batch size * 1024 context size * 30 gradaccum = 491,520
batch_size = 16
context_size = 1024
gradient_accumulation_steps = 30

# this makes total number of tokens be ~2.5B, which is "chinchilla optimal" for default model size (124M params * 20 tokens/param)
max_iters = 5000
lr_decay_iters = 5000
warmup_iters = 500

# eval stuff
eval_interval = 100
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

