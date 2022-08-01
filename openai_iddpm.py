from dataclasses import dataclass
import yahp as hp
import torch as th
from types import SimpleNamespace
from composer import ComposerModel
from tests.openai_code.gaussian_diffusion import (
    GaussianDiffusion,
    ModelMeanType,
    ModelVarType,
    LossType,
)
from tests.openai_code.unet import UNetModel
from diffusion.diffusion import cosine_betas

from typing import List

@dataclass
class DiffusionParams(hp.Hparams):
    steps: int = hp.required("# diffusion steps")
    schedule: str = hp.required("diffusion schedule")

    def initialize_object(self):
        assert self.schedule == "cosine", "Only cosine schedule is supported"
        # Using my beta values instead of OpenAI's
        betas = cosine_betas(self.steps)
        return GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps=True
        )

    
@dataclass
class UNetParams(hp.Hparams):
    in_channels: int = hp.required("# input channels")
    out_channels: int = hp.required("# output channels")
    model_channels: int = hp.required("# model channels")
    channel_mult: List[int] = hp.required("the channel multipliers")
    res_blocks: int = hp.required("# ResNet blocks")
    attention_heads: int = hp.required("# attention heads")
    attention_resolutions: List[int] = hp.required("where to apply attention")

    def initialize_object(self):
        return UNetModel(
            in_channels=self.in_channels,
            model_channels=self.model_channels,
            out_channels=self.out_channels,
            num_res_blocks=self.res_blocks,
            attention_resolutions=self.attention_resolutions,
            dropout=0,
            channel_mult=self.channel_mult,
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=False,
            num_heads=self.attention_heads,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
        )


@dataclass
class TrainerConfig(hp.Hparams):
    unet: UNetParams = hp.required("the UNet model")
    diffusion: DiffusionParams = hp.required("Gaussian diffusion parameters")

    def initialize_object(self):
        return self.unet.initialize_object(), self.diffusion.initialize_object()

class OpenAIIDDPM(ComposerModel):
    def __init__(self, unet, diffusion: GaussianDiffusion):
        super().__init__()
        self.model = unet
        self.diffusion = diffusion

    def forward(self, batch):
        assert len(batch.shape) == 4
        # print(batch[-1, -1, -1, -1])
        N, *_ = batch.shape
        # normalize images to [-1, 1]
        batch = ((batch / 255) * 2) - 1

        # Only support uniform sampling
        t = th.randint(self.diffusion.num_timesteps, (N,)).to(device=batch.device)

        x_0 = batch
        d = dict(x_0=x_0, t=t)
        return SimpleNamespace(**d)

    def loss(self, out, micro_batch):
        losses = self.diffusion.training_losses(model=self.model, x_start=out.x_0, t=out.t)
        assert (losses["loss"] == losses["mse"]).all()
        return th.mean(losses["loss"])

def manual_train(dl, diffusion, unet):
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from composer.optim import DecoupledAdamW
    from torch.optim.adamw import AdamW

    unet = unet.cuda()
    optimizer = AdamW(unet.parameters(), lr=1e-4, betas=(0.9, 0.95))
    # scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=1_000)

    for batch in dl:
        optimizer.zero_grad()
        batch = batch.cuda()

        assert len(batch.shape) == 4
        N, *_ = batch.shape
        print(N)
        # normalize images to [-1, 1]
        batch = ((batch / 255) * 2) - 1
        x_0 = batch

        # Only support uniform sampling
        t = th.randint(diffusion.num_timesteps, (N,)).to(device=x_0.device)
        losses = diffusion.training_losses(model=unet, x_start=x_0, t=t)
        assert (losses["loss"] == losses["mse"]).all()
        mean = th.mean(losses["loss"])
        print(mean)
        mean.backward()
        optimizer.step()
        # scheduler.step()