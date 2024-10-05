# config for training Transformer (124M)
# $ python train.py config/train_fineweb_openmoe.py

# out_dir = 'out-fineweb-openmoe-prefix-lm-kernel'
out_dir = 'out-fineweb-openmoe-flex-prefix'

wandb_log = True
wandb_project = 'fineweb-og-nanogpt'
# wandb_run_name='ul2-decoder-openmoe-prefix-lm-kernel-124M'
wandb_run_name = 'ul2-decoder-openmoe-flex-prefix-124M'

compile = True

dataset = 'fineweb'
# these make the total batch size be ~0.5M
# 16 batch size * 1024 context size * 30 gradaccum = 491,520
batch_size = 16
block_size = 1024
gradient_accumulation_steps = 30

# this makes total number of tokens be ~2.5B, which is "chinchilla optimal" for default model size (124M * 20)
learning_rate = 6e-4
max_iters = 5000
lr_decay_iters = 5000
warmup_iters = 500

# eval stuff
eval_interval = 50
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# UL2 hyperparams
mean_noise_span_lengths = [3.0, 8.0, 3.0, 8.0, 64.0, None] # use None for S-denoisers
noise_densities = [0.15, 0.15, 0.5, 0.5, 0.5, 0.5]
optional_task_prefixes = ["[NLU]", "[NLU]", "[NLG]", "[NLG]", "[NLG]", None]
# optional_task_prefixes = [None] * len(noise_densities)
rates = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]
