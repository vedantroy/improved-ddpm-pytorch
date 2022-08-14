import repackage

repackage.up()

import torch as th
from torch import testing

from diffusion.diffusion import (
    FixedSmallVarianceGaussianDiffusion,
    cosine_betas,
    for_timesteps,
)

def test_equivalence_simple():
    T = 1000
    betas = cosine_betas(T)
    gd = FixedSmallVarianceGaussianDiffusion(betas, f64_debug=True)

    dtype = th.float64
    x_t = th.randn((1, 3, 64, 64), dtype=dtype)
    N, C, H, W = x_t.shape

    eps = th.randn((N, C, H, W), dtype=dtype)
    assert x_t.shape == eps.shape

    for timestep in range(0, T):
        timestep_tensor = th.Tensor([timestep]).to(dtype=th.int64)
        t = lambda x: for_timesteps(x, timestep_tensor, x_t)

        sqrt_alphas = th.sqrt(t(gd.alphas))
        term1 = (
            x_t
            * (sqrt_alphas * (1 - t(gd.alphas_cumprod_prev)))
            / (1 - t(gd.alphas_cumprod))
        )
        term2 = (x_t - th.sqrt(1 - t(gd.alphas_cumprod)) * eps) * (
            (th.sqrt(t(gd.alphas_cumprod_prev)) * t(gd.betas))
            / ((1 - t(gd.alphas_cumprod)) * t(gd.sqrt_alphas_cumprod))
        )

        simplified = (1 / sqrt_alphas) * (
            x_t - (t(gd.betas) / th.sqrt(1 - t(gd.alphas_cumprod))) * eps
        )

        testing.assert_close(term1 + term2, simplified)

    print("test_equivalence_simple passed")


def test_equivalence():
    T = 1000
    betas = cosine_betas(T)
    gd = FixedSmallVarianceGaussianDiffusion(betas, f64_debug=True)

    dtype = th.float64
    x_0 = th.randn((1, 3, 64, 64), dtype=dtype)
    N, C, H, W = x_0.shape

    eps = th.randn((N, C, H, W), dtype=dtype)
    assert x_0.shape == eps.shape

    for timestep in range(0, T):
        t_tensor = th.Tensor([timestep]).to(dtype=th.int64)  # 8 is random choice
        assert t_tensor.shape[0] == N
        x_t = gd.q_sample(x_0=x_0, t=t_tensor, noise=eps)
        mean1 = gd.q_posterior_mean(x_0=x_0, x_t=x_t, t=t_tensor)
        t = lambda x: for_timesteps(x, t_tensor, x_t)


        sqrt_alphas = th.sqrt(t(gd.alphas))
        simplified = (1 / sqrt_alphas) * (
            x_t - (t(gd.betas) / th.sqrt(1 - t(gd.alphas_cumprod))) * eps
        )

        testing.assert_close(mean1, simplified)

    print("test_equivalence passed")


test_equivalence_simple()
test_equivalence()