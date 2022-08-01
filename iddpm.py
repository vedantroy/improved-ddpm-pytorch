from typing import List
from types import SimpleNamespace
from dataclasses import dataclass

import torch as th
from composer import ComposerModel
import yahp as hp
from unet.unet import UNet
from diffusion.diffusion import FixedVarianceGaussianDiffusion, cosine_betas


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

    def initialize_object(self):
        assert self.schedule == "cosine", "Only cosine schedule is supported"
        betas = cosine_betas(self.steps)
        return FixedVarianceGaussianDiffusion(betas)


@dataclass
class TrainerConfig(hp.Hparams):
    unet: UNetParams = hp.required("the UNet model")
    diffusion: DiffusionParams = hp.required("Gaussian diffusion parameters")

    def initialize_object(self):
        return self.unet.initialize_object(), self.diffusion.initialize_object()


class IDDPM(ComposerModel):
    def __init__(self, unet: UNet, diffusion: FixedVarianceGaussianDiffusion):
        super().__init__()
        self.model = unet
        self.diffusion = diffusion

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
            # for logging
            x_t=x_t,
            t=t,
            x_0_pred=x_0_pred,
        )
        # TODO: Use NamedTuple
        return SimpleNamespace(**d)

    def loss(self, out, micro_batch):
        mse_loss = self.diffusion.training_losses_with_model_output(
            model_output=out.model_out,
            noise=out.noise,
        )
        return th.mean(mse_loss)
