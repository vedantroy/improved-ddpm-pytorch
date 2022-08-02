"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import torch as th
import torch.distributed as dist
import torchvision

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    add_dict_to_argparser,
)
import yahp as hp
from params import SpacedGaussianDiffusionHParams, DataHParams
from iddpm.iddpm import UNetParams, StrippedIDDPM


@dataclass
class CombinedHParams(hp.Hparams):
    # OpenAI's diffusion
    spaced_diffusion: SpacedGaussianDiffusionHParams = hp.required("diffusion")
    # My UNet
    unet: UNetParams = hp.required("unet")
    data: DataHParams = hp.required("data")

    def initialize_object(self):
        return self.unet.initialize_object(), self.spaced_diffusion.initialize_object()


def main():
    args = create_argparser().parse_args()
    config = CombinedHParams.create("./config_combined.yaml", None, cli_args=False)
    model, diffusion = config.initialize_object()
    iddpm = StrippedIDDPM(model)
    state = th.load(args.model_path)["state"]["model"]
    iddpm.load_state_dict(state)
    model = iddpm.model

    dist_util.setup_dist()
    OUT_DIR = "./images_combined"
    out_dir_path = Path(OUT_DIR)
    logger.configure(OUT_DIR)

    logger.log("creating model and diffusion...")
    model.to(dist_util.dev())
    model.eval()

    logger.log(
        f"sampling {args.num_samples} images with {args.batch_size} samples per batch"
    )
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        sample_fn = diffusion.p_sample_loop
        sample = sample_fn(
            model,
            (args.batch_size, 3, config.data.image_size, config.data.image_size),
            clip_denoised=args.clip_denoised,
            progress=True,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        for idx, cur_sample in enumerate(sample):
            path = out_dir_path / f"{idx:04d}.png"
            torchvision.io.write_png(cur_sample.cpu(), str(path))

        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
