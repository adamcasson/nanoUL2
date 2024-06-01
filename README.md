# nanoUL2

A port of [nanoGPT](https://github.com/karpathy/nanoGPT) to enable simple training of decoder-only transformers with the [UL2 pretraining paradigm](https://arxiv.org/abs/2205.05131).

![ul2-full](assets/ul2-full.png)

UL2 generalizes common pretraining objectives from a perspective of denoising and unifies them into a single pretraining objective by using a mixture-of-denoisers. R-denoising (regular) is the objective used in the T5 paper. S-denoising (sequential) is prefix language modeling when the end of a sequence is corrupted or regular causal language modeling when the entire sequence is corrupted. X-denoising (extreme) has a high corruption rate either from many spans or long spans of corrupted text.

![ul2-mod](assets/ul2-mod.png)

## install

```
pip install torch numpy datasets tiktoken wandb tqdm
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3

## quickstart

If you are familiar with [nanoGPT](https://github.com/karpathy/nanoGPT) then the instructions and code should look mostly the same.

To get started we can train a model on the [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) dataset. First, we need to download and preprocess the data by tokenizing, encoding into ints, and saving it:

```
$ python data/openwebtext/prepare.py
```

The encoded data will be saved in `train.bin` and `val.bin` in `data/openwebtext/`. 

Training on a single GPU can be kicked off by running:

```
$ python train.py config/train_openwebtext.py
```

By default, the model is the similar to GPT-2 124M in nanoGPT with a context length of 1024, but modernized by using SwiGLU, rotary positional embeddings, RMSNorm. The GPT-2 tokenizer from `tiktoken` is still used.

## todos
- support for encodcer-decoder models like original UL2 paper
- prefix LM flash attention kernel
- better initialization
- more modern datasets (FineWeb subset?)
- support for mixture of objectives scheduling

## troubleshooting
If you run into issues with FlashAttention, try using pytorch 2.1. Newer versions gave me issues when using bf16 on my 4090 and I haven't taken the time to resolve that yet.
