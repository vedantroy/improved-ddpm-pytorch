# Allows me to import things from parent directories
# TL;DR -- Python is confusing & this is a hack
# https://stackoverflow.com/questions/16981921/relative-imports-in-python-3
import repackage
import torch as th
from torch import testing

repackage.up()

from openai_code.gaussian_diffusion import (
    get_named_beta_schedule,
    GaussianDiffusion,
    ModelMeanType,
    ModelVarType,
    LossType,
)
from diffusion.diffusion import (
    FixedSmallVarianceGaussianDiffusion,
    cosine_betas,
    LearnedVarianceGaussianDiffusion,
    for_timesteps,
)
from diffusion.sampler import DDIMSampler

import torch as th
from torch import testing

def equiv_test_ddim():
    test_name = "equiv_test_ddim"

    T = 1000
    betas = cosine_betas(T)
    eta = 0.0

    gd = GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.RESCALED_MSE,
        rescale_timesteps=False,
    )

    my_gd = FixedSmallVarianceGaussianDiffusion(betas)
    sampler = DDIMSampler(my_gd, eta=eta)

    N, C, H, W = 2, 3, 64, 64

    x_0 = th.randn((N, C, H, W)).clamp(-1, 1)
    eps = th.randn((N, C, H, W))
    t = th.Tensor([1, 999]).to(dtype=th.int64)
    x_t = gd.q_sample(x_start=x_0, t=t, noise=eps)
    model = lambda *_, r=eps: r

    dbg = {}
    x_t_m1 = gd.ddim_sample(
        model=model,
        x=x_t,
        t=t,
        clip_denoised=True,
        eta=eta,
        dbg=dbg,
    )

    model_out = sampler.diffusion.p_mean_variance(
        model=model,
        x_t=x_t,
        t=t,
        threshold="static"
    )

    print(th.max(model_out.model_eps), th.max(dbg['eps']))

    testing.assert_close(model_out.pred_x_0, dbg['pred_xstart'])
    testing.assert_close(model_out.model_eps, eps)
    testing.assert_close(model_out.model_eps, dbg['eps'])

    alpha_bar, alpha_bar_prev = sampler.alphas(x_t=x_t, t=t)
    sigma = sampler.sigma(alpha_bar=alpha_bar, alpha_bar_prev=alpha_bar_prev)
    # can't use `assert_close` due to different dimensions
    assert (dbg['sigma'] == sigma).all()
    print(f"{test_name}: sigma passed")

    my_x_t_m1 = sampler.sample(
        model_out=model_out,
        x_t=x_t,
        t=t,
        idx=-1
    )

    testing.assert_close(x_t_m1["sample"], my_x_t_m1)

equiv_test_ddim()