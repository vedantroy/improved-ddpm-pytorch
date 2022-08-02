# Improved Denoised Diffusion Probabilistic Models
This is my implementation of [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) for learning purposes.

## Samples
<p float="left">
  <img src="./generated_faces/0000.png" width="64" />
  <img src="./generated_faces/0001.png" width="64" />
  <img src="./generated_faces/0002.png" width="64" />
  <img src="./generated_faces/0003.png" width="64" />
  <img src="./generated_faces/0004.png" width="64" />
  <img src="./generated_faces/0005.png" width="64" />
  <img src="./generated_faces/0006.png" width="64" />
  <img src="./generated_faces/0007.png" width="64" />
  <img src="./generated_faces/0008.png" width="64" />
  <img src="./generated_faces/0009.png" width="64" />
</p>

The results of training on CelebHQ for 64500 batches where each batch had 8 samples. Took ~ 4 hrs on a V100. Used the fixed_variance.yaml config.

## Features
Implemented:
- [x] $L_\text{simple}$ objective
- [x] Cosine schedule
- [x] Training + Generating

TODO:
- [ ] $L_\text{hybrid}$ objective / learned variance
    - There's a prototype, but I'm running into some issues
- [ ] Faster sampling 
    - Requires implementing $L_\text{hybrid}$

Unplanned:
- [ ] $L_\text{vlb}$ objective with loss-aware sampler

## Repository Guide
This repository is *super* messy. It has a lot of scratch files that represent my attempts at figuring things out / debugging escapades. Nevertheless, here's the overview:
- `openai-ddm/` contains a forked version of [improved-diffusion](https://github.com/openai/improved-diffusion). I spent a while stripping out features (fp16, checkpointing) to get to the bare-minimum training loop.
- All training is done using [composer](https://github.com/mosaicml/composer) which supports fp16, checkpointing, automatic wandb logging, etc.
- `tests/` has some tests that verify my UNet / diffusion is identical to OpenAI's unet/diffusion
    - I haven't written tests for sampling yet
- [run_unet.py](./run_unet.py) prints out the architecture of the UNet in a friendly form & verifies the UNet works
- [make_torchdata.py](./data_scripts/make_torchdata.py) contains a script to generate a dataset from a folder of image files. It outputs parquet files.

Anyways, a ton of stuff here is not well-documented. E.g, you can't use "basic.yaml". It doesn't work since I haven't implemented learned variances yet. So, proceed with caution ...

## Why?
This repository is for my personal use. But, there are a few nice things you might get from it:
- The diffusion code is written simply w/o support for a ton of different options in a single class so it's easy to understand
- Same goes for the UNet
- (TODO), I will port some comments from my ml-experiments repository to the UNet, so you can see why certain things are done
- Usage of MosaicML's trainer means you don't get bogged down by FP16/checkpointing/logging & can focus on the important stuff

## Credit
- The "losses.py" and "nn.py" file inside of the "diffusion" folder are copy-pasted from the OpenAI codebase. I haven't had time to re-implement them yet. 
- The `for_timesteps` function is heavily based off of a function in lucidrain's imagen-pytorch repository.