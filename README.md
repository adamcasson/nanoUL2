# nanoUL2

A port of [nanoGPT](https://github.com/karpathy/nanoGPT) to enable simple training of [T5-style models](https://arxiv.org/abs/1910.10683) with the [UL2 pretraining paradigm](https://arxiv.org/abs/2205.05131).

![ul2-full](assets/ul2-full.png)

UL2 generalizes common pretraining objectives from a perspective of denoising and unifies them into a single pretraining objective by using a mixture-of-denoisers. R-denoising (regular) is the objective used in the T5 paper. S-denoising (sequential) can be similar to prefix language modeling when the end of a sequence is corrupted or regular autoregressive language modeling when the entire sequence is corrupted. X-denoising (extreme) has a high corruption rate either from many spans or long spans of corrupted text.

![ul2-mod](assets/ul2-mod.png)

## install

```
pip install torch numpy transformers datasets h5py wandb tqdm
```

Dependencies:

- [pytorch](https://pytorch.org)
- [numpy](https://numpy.org/install/)
-  `transformers` for tokenizers and optimizers
-  `datasets` for huggingface datasets (if you want to download + preprocess TinyStories)
-  `h5py` for efficient data loading
-  `wandb` for optional logging
-  `tqdm` for progress bars

## quickstart

If you are familiar with [nanoGPT](https://github.com/karpathy/nanoGPT) then the instructions and code should look mostly the same.

To get started we can train a model on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset. First, we need to download and preprocess the data by tokenizing, encoding into ints, and saving it:

```
$ python data/tinystories/prepare.py
```

The encoded data will be saved in `train.h5` and `validation.h5` in `data/tinystories/`. 

`train.py` should look almost identical as well minus a few features at the moment (mainly DDP and AMP). Training on a single GPU can be kicked off by running:

```
$ python train.py
```

## todos
- features
  - match other settings to paper
  - resume from checkpoint
  - cleaner model configuration
- optimizations
  - mixed precision
  - sequence packing
  - data loading
  - ddp
