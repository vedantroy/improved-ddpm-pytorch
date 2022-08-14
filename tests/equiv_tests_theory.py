import repackage

repackage.up()

import torch as th
from torch import testing

from diffusion.diffusion import (
    FixedSmallVarianceGaussianDiffusion,
    cosine_betas,
    for_timesteps,
)


def assert_dtype(*x, dtype=th.float32):
    for idx, y in enumerate(x):
        assert y.dtype == dtype, f"tensor {idx}: {y.dtype} instead of {dtype}"


def test_equivalence():
    T = 1000
    betas = cosine_betas(T)
    gd = FixedSmallVarianceGaussianDiffusion(betas, use_f32=False)

    dtype = th.float64
    x_0 = th.randn((1, 3, 64, 64), dtype=dtype)
    N, C, H, W = x_0.shape

    eps = th.randn((N, C, H, W), dtype=dtype)
    assert x_0.shape == eps.shape
    assert_dtype(eps, x_0, dtype=dtype)

    t = th.Tensor([999]).to(dtype=th.int64)  # 8 is random choice
    assert t.shape[0] == N
    x_t = gd.q_sample(x_0=x_0, t=t, noise=eps)
    assert_dtype(x_t, dtype=dtype)

    mean1 = gd.q_posterior_mean(x_0=x_0, x_t=x_t, t=t)
    assert_dtype(mean1, dtype=dtype)
    assert_dtype(
        mean1, gd.sqrt_alphas_cumprod, gd.sqrt_one_minus_alphas_cumprod, dtype=dtype
    )

    mean2 = (1 / for_timesteps(gd.sqrt_alphas_cumprod, t, eps)) * (
        x_t
        - (
            (for_timesteps(gd.betas, t, eps))
            / (for_timesteps(gd.sqrt_one_minus_alphas_cumprod, t, eps))
        )
        * eps
    )
    print(mean2.type())

    testing.assert_close(mean1, mean2)


# This test is failing & idk why
# test_equivalence()


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


test_equivalence_simple()
