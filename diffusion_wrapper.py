from operator import mod
from tests.openai_code.gaussian_diffusion import (
    GaussianDiffusion,
    ModelMeanType,
    ModelVarType,
    LossType,
)


class WrappedOpenAIGaussianDiffusion:
    def __init__(self, betas):
        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
        )
        self.n_timesteps = self.diffusion.num_timesteps

    def q_sample(self, x_0, t, noise):
        return self.diffusion.q_sample(x_0, t, noise)

    def training_losses(self, model_out, x_0, x_t, t, noise):
        fake_model = lambda *args, r=model_out: model_out
        losses = self.diffusion.training_losses(
            model=fake_model, x_start=x_0, t=t, noise=noise
        )
        assert (losses["loss"] == losses["mse"]).all()
        return losses["loss"], None
