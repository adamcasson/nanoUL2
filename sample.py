"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext

import torch
from transformers import AutoTokenizer

from model import T5

# -----------------------------------------------------------------------------
init_from = (
    "resume"  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
)
out_dir = "out"  # ignored if init_from is not 'resume'
encoder_prompt = (
    "Once upon a time,"  # Can also specify a file, use as: "FILE:prompt.txt"
)
decoder_start = "[S]"  # or "[R]" or "[X]"
num_samples = 10  # number of samples to draw
max_new_tokens = 128  # number of tokens generated in each sample
temperature = (
    0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
)
top_k = (
    200  # retain only the top_k most likely tokens, clamp others to have 0 probability
)
seed = 1337
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
# dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
# exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast

# init from a model saved in a specific directory
ckpt_path = os.path.join(out_dir, "ckpt.pt")
checkpoint = torch.load(ckpt_path, map_location=device)
model = T5(**checkpoint["model_args"])
state_dict = checkpoint["model"]
model.load_state_dict(state_dict)

model.eval()
model.to(device)

tokenizer = AutoTokenizer.from_pretrained("ul2-tinystories-tokenizer")
encode = lambda s: tokenizer.encode(s)
decode = lambda l: tokenizer.decode(l)

# encode the beginning of the prompt
if encoder_prompt.startswith("FILE:"):
    with open(encoder_prompt[5:], "r", encoding="utf-8") as f:
        encoder_prompt = f.read()
encoder_ids = encode(encoder_prompt)
decoder_ids = encode(decoder_start)
ex = torch.tensor(encoder_ids, dtype=torch.long, device=device)[None, ...]
dx = torch.tensor(decoder_ids, dtype=torch.long, device=device)[None, ...]

# run generation
with torch.no_grad():
    for k in range(num_samples):
        y = model.generate(ex, dx, max_new_tokens, temperature=temperature, top_k=top_k)
        print(decode(y[0].tolist()))
        print("---------------")
