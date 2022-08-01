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
from my_iddpm import UNetParams
from diffusion.diffusion import cosine_betas

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
