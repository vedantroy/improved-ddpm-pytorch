from abc import ABC, abstractmethod

import torch as th

from .diffusion import GaussianDiffusion, ModelValues, for_timesteps


def xor(a, b):
    return bool(a) + bool(b) == 1


class Sampler(ABC):
    def __init__(self, diffusion: GaussianDiffusion):
        self.diffusion = diffusion

    @abstractmethod
    def sample(self, *, model_out: ModelValues, x_t, t, idx, **kwargs):
        pass

    def sample_loop_progressive(
        self, *, model, noise, shape, threshold, device, **kwargs
    ):
        assert xor(
            noise, shape
        ), f"Either noise or shape must be specified, but not both or neither"

        img = N = None
        img = noise if noise else th.randn(shape, device=device)
        N = img.shape[0]

        for idx, _t in enumerate(reversed(range(self.diffusion.n_timesteps))):
            t = th.tensor([_t] * N, device=device)
            with th.no_grad():
                model_out = self.diffusion.p_mean_variance(
                    model=model, x_t=img, t=t, threshold=threshold
                )
                img = self.sample(
                    model_out=model_out,
                    x_t=img,
                    t=t,
                    idx=idx,
                    **kwargs,
                )
                yield img


class DDPMSampler(Sampler):
    def sample(self, *, model_out, x_t, t, idx):
        N = t.shape[0]
        noise = th.randn_like(x_t)
        # no noise when t == 0
        nonzero_mask = (t != 0).float().view(N, *([1] * (len(x_t.shape) - 1)))
        sample = (
            model_out.mean + nonzero_mask * th.exp(0.5 * model_out.log_var) * noise
        )
        return sample


class DDIMSampler(Sampler):
    def __init__(self, diffusion: GaussianDiffusion, eta: float):
        super().__init__(diffusion)
        self.eta = eta

    def sample(self, *, model_out, x_t, t, idx):
        alpha_bar = for_timesteps(self.diffusion.alphas_cumprod, t, x_t)
        alpha_bar_prev = for_timesteps(self.diffusion.alphas_cumprod_prev, t, x_t)

        # I think we don't use beta directly here b/c it's a float64
        # and truncating would be more imprecise ??
        beta = alpha_bar / alpha_bar_prev
        sigma = (
            self.eta * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * th.sqrt(1 - beta)
        )

        # (12 in [1])
        mean_pred = (
            model_out.mean * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma**2) * model_out.model_eps
        )

        N = t.shape[0]
        nonzero_mask = (t != 0).float().view(N, *([1] * (len(x_t.shape) - 1)))
        return mean_pred + nonzero_mask * sigma * th.randn_like(mean_pred)
