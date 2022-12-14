from collections import namedtuple
from abc import ABC, abstractmethod

import torch as th
import torch.nn.functional as F
from einops import rearrange
import math

# TODO: Re-implement these
from .losses import discretized_gaussian_log_likelihood, normal_kl
from .nn import mean_flat


def xor(a, b):
    return bool(a) + bool(b) == 1


# Question list:
# 1. the beta clamp seems unnecessary?
# 2. Why do we set 1 as the 1st alpha value? I notice this makes the 1st Beta be 0
# are the betas indexed s.t betas[0] == represents the image before any diffusion process?
# (this would make sense b/c if the 1st beta is 0, then there would be no variance)

ModelValues = namedtuple(
    "ModelValues", ["mean", "var", "log_var", "pred_x_0", "model_eps"]
)


def cosine_betas(timesteps, s=0.008, max_beta=0.999):
    """
    Get B_t for the cosine schedule (eq 17 in [0])

    :param s: "We use a small offset s to prevent B_t from being too small near t = 0"
    :param max_beta: "In practice, we clip B_t to be no larger than 0.999 to prevent
                      singularities at the end of the diffusion process near t = T"
    """
    # If we add noise twice, then there are 3 total states (0, 1, 2)
    states = timesteps + 1
    t = th.linspace(start=0, end=timesteps, steps=states, dtype=th.float64)
    f_t = th.cos(((t / timesteps) + s) / (1 + s) * (math.pi / 2)) ** 2
    alphas_cumprod = f_t / f_t[0]
    alphas_cumprod_t = alphas_cumprod[1:]
    alphas_cumprod_t_minus_1 = alphas_cumprod[:-1]
    betas = 1 - (alphas_cumprod_t / alphas_cumprod_t_minus_1)
    # TODO: In practice, this clamp just seems to clip the last value from 1 to 0.999
    return betas.clamp(0, max_beta)


def for_timesteps(a, t, broadcast_to):
    """
    Extract values from a for each timestep

    :param a: (timesteps,)
    :param t: (batch size,)
    :param broadcast_shape: (batch size, ...)

    :returns: (batch size, 1, 1, 1) where
              the number of 1s corresponds to len(...)
    """
    batch, *_ = t.shape

    # `a` should always be a 1D tensor of
    # values that exist at every timestep
    assert len(a.shape) == 1, f"{a.shape} is not a 1D tensor"

    # use `t` as an index tensor to extract values
    # at a given timestep
    out = a.to(device=broadcast_to.device).gather(0, t)
    num_nonbatch_dims = len(broadcast_to.shape) - 1
    return out.reshape(batch, *((1,) * num_nonbatch_dims))


def get_eps_and_var(model_output, *, C):
    model_eps, model_v = rearrange(
        model_output, "B (split C) ... -> split B C ...", split=2, C=C
    )
    return model_eps, model_v


class GaussianDiffusion(ABC):
    def __init__(self, betas, f64_debug=False):
        def f32(x):
            return x.to(th.float32) if not f64_debug else x

        # TODO: Get rid of this "check"?
        # It's never once caught a bug ...
        def check(x):
            assert x.shape == (self.n_timesteps,)

        self.n_timesteps = betas.shape[0]
        self.betas = betas
        check(self.betas)

        alphas = 1 - betas
        if f64_debug:
            self.alphas = alphas
        alphas_cumprod = th.cumprod(alphas, dim=0)
        self.alphas_cumprod = f32(alphas_cumprod)
        check(self.alphas_cumprod)

        # TODO(verify): By prepending 1, the 1st beta is 0
        # This represents the initial image, which as a mean but no variance (since it's ground truth)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.alphas_cumprod_prev = f32(alphas_cumprod_prev)

        def setup_q_posterior_mean():
            # (11 in [0])
            self.posterior_mean_coef_x_t = f32(
                (th.sqrt(alphas) * (1 - alphas_cumprod_prev)) / (1 - alphas_cumprod)
            )
            check(self.posterior_mean_coef_x_t)
            self.posterior_mean_coef_x_0 = f32(
                (th.sqrt(alphas_cumprod_prev) * betas) / (1 - alphas_cumprod)
            )
            check(self.posterior_mean_coef_x_0)

        def setup_q_posterior_log_variance():
            # (10 in [0])
            posterior_variance = f32(
                ((1 - alphas_cumprod_prev) / (1 - alphas_cumprod)) * betas
            )
            # clipped to avoid log(0) == -inf b/c posterior variance is 0
            # at start of diffusion chain
            assert alphas_cumprod_prev[0] == 1 and posterior_variance[0] == 0
            self.posterior_variance = f32(posterior_variance)
            check(self.posterior_variance)
            self.posterior_log_variance_clipped = f32(
                th.log(
                    F.pad(posterior_variance[1:], (1, 0), value=posterior_variance[1])
                )
            )
            check(self.posterior_log_variance_clipped)

        def setup_q_sample():
            # (9 in [0]) -- used to go forward in the diffusion process
            self.sqrt_alphas_cumprod = f32(th.sqrt(alphas_cumprod))
            check(self.sqrt_alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod = f32(th.sqrt((1 - alphas_cumprod)))
            check(self.sqrt_one_minus_alphas_cumprod)

        def setup_predict_x0():
            # OpenAI code does:
            #     self.sqrt_recip_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod)
            # which is same as this, since: sqrt(1/a) = 1/sqrt(a)
            self.recip_sqrt_alphas_cumprod = f32(1.0 / th.sqrt(alphas_cumprod))
            check(self.recip_sqrt_alphas_cumprod)
            self.sqrt_recip_alphas_cumprod_minus1 = f32(
                th.sqrt((1 / alphas_cumprod) - 1)
            )
            check(self.sqrt_recip_alphas_cumprod_minus1)

        setup_q_sample()
        setup_q_posterior_mean()
        setup_q_posterior_log_variance()
        setup_predict_x0()

        # Used to calculate the variance from the model prediction
        self.log_betas = f32(th.log(betas))
        check(self.log_betas)

    def q_posterior_mean(self, *, x_0, x_t, t):
        """
        Calculate the mean of the normal distribution q(x_{t-1}|x_t, x_0)
        From (12 in [0])

        Use this to go BACKWARDS 1 step, given the current diffusion step
        All parameters are batches

        :param x_t: The result of the next step in the diffusion process
        :param x_start: The initial image
        :param t: The current timestep
        """
        mean = (
            for_timesteps(self.posterior_mean_coef_x_0, t, x_0) * x_0
            + for_timesteps(self.posterior_mean_coef_x_t, t, x_0) * x_t
        )
        assert mean.shape == x_0.shape and x_0.shape == x_t.shape
        return mean

    def q_sample(self, x_0, t, noise):
        """
        Diffuse the data for a given number of diffusion steps.
        From (9 in [0])

        In other words, sample from q(x_t | x_0)
        Use this to go FORWARDS t steps, given the initial image

        :param x_start: The initial image (batch)
        :param t: The timestep to diffuse to (batch)
        :param noise: The noise (epsilon in the paper)
        """

        N = x_0.shape[0]
        assert t.shape == (N,)

        mean = for_timesteps(self.sqrt_alphas_cumprod, t, x_0) * x_0
        var = for_timesteps(self.sqrt_one_minus_alphas_cumprod, t, x_0)
        assert mean.shape[0] == N and var.shape[0] == N
        return mean + var * noise

    def predict_x0_from_eps(self, *, x_t, t, eps):
        """
        Predict x_0 given epsilon and x_t
        From re-arranging and simplifying (9 in [0])

        The result of this can be used in equation 11 to
        calculate the mean of q(x_{t-1}|x_t,x_0)
        """
        return (
            x_t * for_timesteps(self.recip_sqrt_alphas_cumprod, t, x_t)
            - for_timesteps(self.sqrt_recip_alphas_cumprod_minus1, t, x_t) * eps
        )

    def threshold(self, x_t, threshold):
        if threshold == None:
            return x_t
        elif threshold == "static":
            return x_t.clamp(-1, 1)
        elif threshold == "dynamic":
            raise Exception("Not implemented")

    @abstractmethod
    def p_mean_variance(self, *, model, x_t, t, true_t, threshold) -> ModelValues:
        """
        Get the model's predicted mean and variance for the distribution
        that predicts x_{t-1}
        """
        pass

    def p_sample(self, *, model, x_t, t, threshold):
        out = self.p_mean_variance(model=model, x_t=x_t, t=t, threshold=threshold)

        N = t.shape[0]
        noise = th.randn_like(x_t)
        # no noise when t == 0
        nonzero_mask = (t != 0).float().view(N, *([1] * (len(x_t.shape) - 1)))
        sample = out.mean + nonzero_mask * th.exp(0.5 * out.log_var) * noise
        return sample

    def p_sample_loop_progressive(self, *, model, noise, shape, threshold, device):
        assert xor(
            noise, shape
        ), f"Either noise or shape must be specified, but not both or neither"

        img = N = None
        if noise:
            img = noise
        else:
            img = th.randn(shape, device=device)
        N = img.shape[0]

        for _t in list(range(self.n_timesteps))[::-1]:
            t = th.tensor([_t] * N, device=device)
            with th.no_grad():
                img = self.p_sample(model=model, x_t=img, t=t, threshold=threshold)
                yield img

    def vb_term(self, *, x_0, x_t, t, model):
        true_mean = self.q_posterior_mean(x_0=x_0, x_t=x_t, t=t)
        true_log_var = for_timesteps(self.posterior_log_variance_clipped, t, x_t)
        pred_mean, _, pred_log_var, _, _ = self.p_mean_variance(
            model=model, x_t=x_t, t=t, threshold=None
        )
        kl = normal_kl(true_mean, true_log_var, pred_mean, pred_log_var)
        kl = mean_flat(kl) / math.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_0, means=pred_mean, log_scales=0.5 * pred_log_var
        )
        decoder_nll = mean_flat(decoder_nll) / math.log(2.0)

        # `th.where` selects from tensor 1 if cond is true and tensor 2 otherwise
        return th.where((t == 0), decoder_nll, kl)

    def loss_mse(self, *, model_eps, noise):
        return mean_flat((noise - model_eps) ** 2)

    def loss_vb(self, *, model_output, x_0, x_t, t, vb_stop_grad):
        is_learned = isinstance(self, LearnedVarianceGaussianDiffusion)

        frozen_out = model_output
        if vb_stop_grad:
            assert is_learned, f"Cannot apply stop-gradient to fixed variance diffusion"
            model_eps, model_v = get_eps_and_var(model_output, C=x_t.shape[1])
            frozen_out = th.cat([model_eps.detach(), model_v], dim=1)
        C = x_t.shape[1]
        assert frozen_out.shape[1] == C * 2 if is_learned else C

        vb_loss = self.vb_term(
            x_0=x_0,
            x_t=x_t,
            t=t,
            # TODO: The OpenAI people use kwargs, not sure
            # why not just directly return `frozen_out`
            model=lambda *_, r=frozen_out: r,
        )
        # > For our experiments, we set ?? = 0.001 to prevent L_vlb from
        # > overwhelming L_simple
        # from [0]
        return vb_loss * self.n_timesteps / 1000.0

    def validation_mse(self, *, model_output, noise):
        model_eps = model_output
        if isinstance(self, LearnedVarianceGaussianDiffusion):
            model_eps, _ = get_eps_and_var(model_output, C=noise.shape[1])
        return self.loss_mse(model_eps=model_eps, noise=noise)

    def validation_vb(self, *, model_output, x_0, x_t, t):
        return self.loss_vb(
            model_output=model_output,
            x_0=x_0,
            x_t=x_t,
            t=t,
            # No backprop during validation
            vb_stop_grad=False,
        )

    @abstractmethod
    def losses_training(self, *args, **kwargs):
        pass


class LearnedVarianceGaussianDiffusion(GaussianDiffusion):
    def model_v_to_log_variance(self, v, t):
        """
        Convert the model's v vector to an interpolated variance
        From (15 in [0])
        """

        min_log = for_timesteps(self.posterior_log_variance_clipped, t, v)
        max_log = for_timesteps(self.log_betas, t, v)

        # Model outputs between [-1, 1] for [min_var, max_var]
        frac = (v + 1) / 2
        return frac * max_log + (1 - frac) * min_log

    def p_mean_variance(self, *, model, x_t, t, threshold):
        """
        Get the model's predicted mean and variance for the distribution
        that predicts x_{t-1}

        - Predict x_0 from epsilon
        - Use x_0 and x_t to predict the mean of q(x_{t-1}|x_t,x_0)
        - Turn the model's v vector into a variance
        """
        model_eps, model_v = get_eps_and_var(model(x_t, t), C=x_t.shape[1])
        pred_x_0 = self.predict_x0_from_eps(x_t=x_t, t=t, eps=model_eps)
        pred_x_0 = self.threshold(pred_x_0, threshold)

        pred_mean = self.q_posterior_mean(x_0=pred_x_0, x_t=x_t, t=t)
        pred_log_var = self.model_v_to_log_variance(model_v, t)
        return ModelValues(
            mean=pred_mean,
            var=None,
            log_var=pred_log_var,
            # These are only for DDIM sampling, but I'm not
            # sure if you can use DDIM with learned variance
            pred_x_0=pred_x_0,
            model_eps=model_eps,
        )

    def losses_training(self, *, model_output, noise, x_0, x_t, t):
        model_eps, _ = get_eps_and_var(model_output, C=x_t.shape[1])
        mse_loss = self.loss_mse(model_eps=model_eps, noise=noise)
        vb_loss = self.loss_vb(
            model_output=model_output,
            x_0=x_0,
            x_t=x_t,
            t=t,
            vb_stop_grad=True,
        )
        return {"mse": mse_loss, "vb": vb_loss}


class FixedSmallVarianceGaussianDiffusion(GaussianDiffusion):
    def p_mean_variance(self, *, model, x_t, t, threshold):
        model_variance, model_log_variance = (
            for_timesteps(self.posterior_variance, t, x_t),
            for_timesteps(self.posterior_log_variance_clipped, t, x_t),
        )

        model_output = model(x_t, t)
        pred_x_0 = self.predict_x0_from_eps(x_t=x_t, t=t, eps=model_output)
        pred_x_0 = self.threshold(pred_x_0, threshold)

        model_mean = self.q_posterior_mean(x_0=pred_x_0, x_t=x_t, t=t)
        return ModelValues(
            mean=model_mean,
            var=model_variance,
            log_var=model_log_variance,
            pred_x_0=pred_x_0,
            model_eps=model_output,
        )

    def losses_training(self, *, model_output, noise):
        mse_loss = self.loss_mse(model_eps=model_output, noise=noise)
        return {"mse": mse_loss}
