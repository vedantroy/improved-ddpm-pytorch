import functools

import numpy as np
import torch as th
from torch.optim import AdamW

from . import logger
from .fp16_util import (
    zero_grad,
)
from .resample import UniformSampler


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        lr,
        log_interval,
        schedule_sampler=None,
        weight_decay=0.0,
    ):
        assert isinstance(schedule_sampler, UniformSampler), f"Simple trainer only supports UniformSampler" 

        self.model = model
        self.diffusion = diffusion
        self.schedule_sampler = schedule_sampler
        self.data = data
        self.batch_size = batch_size
        self.lr = lr
        self.log_interval = log_interval

        self.step = 0
        self.resume_step = 0
        # world_size is always 1
        self.global_batch = self.batch_size # * dist.get_world_size()

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=weight_decay)

        assert th.cuda.is_available(), "CUDA is required for training"

    def run_loop(self):
        while True:
            batch, cond = next(self.data)
            assert len(cond.items()) == 0
            self.run_step(batch)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            self.step += 1

    def run_step(self, batch):
        self.forward_backward(batch)
        self.optimize_normal()
        self.log_step()
    
    def optimize_normal(self):
        self._log_grad_norm()
        self.opt.step()

    def forward_backward(self, batch):
        zero_grad(self.model_params)
        batch = batch.cuda()
        cuda = th.device("cuda")
        t, weights = self.schedule_sampler.sample(batch.shape[0], cuda)
        assert (weights == th.Tensor([[1]]).cuda()).all()

        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.model,
            batch,
            t,
        )

        losses = compute_losses()
        loss = (losses["loss"] * weights).mean()
        log_loss_dict(
            self.diffusion, t, {k: v * weights for k, v in losses.items()}
        )
        loss.backward()

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
