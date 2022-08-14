from abc import ABC, abstractmethod

import torch as th

from .diffusion import GaussianDiffusion


def xor(a, b):
    return bool(a) + bool(b) == 1


class Sampler(ABC):
    def __init__(self, diffusion: GaussianDiffusion):
        self.diffusion = diffusion

    @abstractmethod
    def sample(self, *, p_mean_var, x_t, t, idx, **kwargs):
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
                p_mean_var = self.diffusion.p_mean_variance(
                    model=model, x_t=img, t=t, threshold=threshold
                )
                img = self.sample(
                    p_mean_var=p_mean_var,
                    x_t=img,
                    t=t,
                    idx=idx,
                    **kwargs,
                )
                yield img

class DDPMSampler(Sampler):
    def sample(self, *, p_mean_var, x_t, t, idx):
        N = t.shape[0]
        noise = th.randn_like(x_t)
        # no noise when t == 0
        nonzero_mask = (t != 0).float().view(N, *([1] * (len(x_t.shape) - 1)))
        sample = p_mean_var.mean + nonzero_mask * th.exp(0.5 * p_mean_var.log_var) * noise
        return sample

class DDIMSampler(Sampler):
    def sample(self, *, p_mean_var, x_t, t, idx, eta):
        pass