from typing import List
from types import SimpleNamespace
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch as th
from composer import ComposerModel
import yahp as hp
import torchmetrics

from unet.unet import UNet
from diffusion.diffusion import (
    FixedSmallVarianceGaussianDiffusion,
    GaussianDiffusion,
    LearnedVarianceGaussianDiffusion,
    cosine_betas,
)


@dataclass
class UNetParams(hp.Hparams):
    in_channels: int = hp.required("# input channels")
    out_channels: int = hp.required("# output channels")
    # (C in [0] under Appendix A "Hyperparameters")
    model_channels: int = hp.required("# model channels")
    channel_mult: List[int] = hp.required("the channel multipliers")
    layer_attn: List[bool] = hp.required(
        "whether to use attention between ResNet blocks"
    )
    res_blocks: int = hp.required("# ResNet blocks")
    attention_heads: int = hp.required("# attention heads")

    def initialize_object(self):
        return UNet(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            model_channels=self.model_channels,
            channel_mult=self.channel_mult,
            layer_attn=self.layer_attn,
            num_res_blocks=self.res_blocks,
            num_heads=self.attention_heads,
        )


@dataclass
class DiffusionParams(hp.Hparams):
    steps: int = hp.required("# diffusion steps")
    schedule: str = hp.required("diffusion schedule")
    learn_sigma: bool = hp.required("whether to learn sigma")

    def initialize_object(self):
        assert self.schedule == "cosine", "Only cosine schedule is supported"
        betas = cosine_betas(self.steps)
        return (
            LearnedVarianceGaussianDiffusion(betas)
            if self.learn_sigma
            else FixedSmallVarianceGaussianDiffusion(betas)
        )


@dataclass
class IDDPMConfig(hp.Hparams):
    unet: UNetParams = hp.required("the UNet model")
    diffusion: DiffusionParams = hp.required("Gaussian diffusion parameters")

    def initialize_object(self):
        unet, diffusion = (
            self.unet.initialize_object(),
            self.diffusion.initialize_object(),
        )
        return IDDPM(unet, diffusion)


class IDDPM(ComposerModel):
    def __init__(self, unet: UNet, diffusion: GaussianDiffusion):
        super().__init__()
        self.model = unet
        self.diffusion = diffusion

        self.val_mse_loss = torchmetrics.MeanMetric(nan_strategy="error")
        self.val_vb_loss = torchmetrics.MeanMetric(nan_strategy="error")

    def forward(self, batch):
        assert len(batch.shape) == 4
        # print(batch[-1, -1, -1, -1])
        N, *_ = batch.shape
        # normalize images to [-1, 1]
        x_0 = ((batch / 255) * 2) - 1

        # Only support uniform sampling
        t = th.randint(self.diffusion.n_timesteps, (N,)).to(device=batch.device)

        noise = th.randn_like(x_0)
        x_t = self.diffusion.q_sample(x_0, t, noise)
        model_out = self.model(x_t, t)
        x_0_pred = self.diffusion.predict_x0_from_eps(x_t=x_t, t=t, eps=model_out)
        d = dict(
            noise=noise,
            model_out=model_out,
            # for logging & LearnedVarianceGaussianDiffusion
            x_t=x_t,
            t=t,
            # for logging
            x_0_pred=x_0_pred,
            # for LearnedVarianceGaussianDiffusion
            x_0=x_0,
        )
        # TODO: Use NamedTuple
        return SimpleNamespace(**d)

    def metrics(self, train: bool = False):
        assert not train, f"Metrics are not available for training"
        return torchmetrics.MetricCollection(
            {
                "mse": self.val_mse_loss,
                "vb": self.val_vb_loss,
            }
        )

    def validate(self, batch):
        out = self.forward(batch)
        losses = self.diffusion.losses_validation(
            model_output=out.model_out,
            noise=out.noise,
            x_0=out.x_0,
            x_t=out.x_t,
            t=out.t,
        )
        print(f"Losses: {losses}")
        return th.mean(losses.mse)
        # return th.mean(losses.mse), th.mean(losses.vb)
        # return losses.mse, losses.vb
        # return losses.mse, losses.vb

    def loss(self, out, micro_batch):
        losses = None
        if isinstance(self.diffusion, FixedSmallVarianceGaussianDiffusion):
            losses = self.diffusion.losses_training(
                model_output=out.model_out, noise=out.noise
            )
        elif isinstance(self.diffusion, LearnedVarianceGaussianDiffusion):
            losses = self.diffusion.losses_training(
                model_output=out.model_out,
                noise=out.noise,
                x_0=out.x_0,
                x_t=out.x_t,
                t=out.t,
            )
        else:
            raise Exception("Unsupported diffusion")

        losses_mean = {k: th.mean(v) for k, v in losses.items()}
        # Composer takes the average of the collated losses before summing
        # whereas the OpenAI trainer sums before averaging
        # The 2 should be the same? b/c
        # avg(1, 2, 3, 4) + avg(5, 6, 7, 8) = avg(1 + 5, 2 + 6, 3 + 7, 4 + 8)?
        return list(losses_mean.values())
