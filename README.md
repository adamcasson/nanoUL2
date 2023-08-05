# nanoUL2

A port of [nanoGPT](https://github.com/karpathy/nanoGPT) to enable simple training of [T5-style models](https://arxiv.org/abs/1910.10683) with the [UL2 pretraining paradigm](https://arxiv.org/abs/2205.05131).

![ul2-full](https://github.com/adamcasson/nanoUL2/assets/6784558/97b2a365-cd30-474e-9149-cd2677a656fc)

UL2 generalizes common pretraining objectives from a perspective of denoising and unifies them into a single pretraining objective by using a mixture-of-denoisers. R-denoising (regular) is the objective used in the T5 paper. S-denoising (sequential) can be similar to prefix language modeling when the end of a sequence is corrupted or regular autoregressive language modeling when the entire sequence is corrupted. X-denoising (extreme) has a high corruption rate either from many spans or long spans of corrupted text.

![ul2-mod](https://github.com/adamcasson/nanoUL2/assets/6784558/3526de61-566b-484b-9c8c-fdcdce78be3d)

# todos
- features
  - adafactor
  - match other settings to paper
  - resume from checkpoint
  - decoding for inference
  - cleaner model configuration
- optimizations
  - flash attention
  - sequence packing
  - data loading
