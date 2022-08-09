# composer's optimizer is buggy
from torch.optim.adamw import AdamW
from composer.loggers import WandBLogger, FileLogger
from composer import Trainer
from composer.core import Time
from composer.optim.scheduler import CosineAnnealingWithWarmupScheduler
from composer.callbacks import CheckpointSaver, LRMonitor, SpeedMonitor
from callbacks import DiffusionMonitor


def total_batches_and_scheduler_for_time(
    batch_rate, target_time, warmup, cosine_factor=1.2
):
    def intcast(x):
        assert x.is_integer(), f"Invalid int: {x}"
        return int(x)

    total_batches = intcast(target_time * batch_rate)
    t_max = total_batches - warmup

    return total_batches, CosineAnnealingWithWarmupScheduler(
        t_warmup=Time.from_batch(warmup),
        t_max=Time.from_batch(round(t_max * cosine_factor)),
    )


def make_trainer(
    *,
    model,
    train_dl,
    eval_dl,
    grad_accum,
    n_evals,
    n_checkpoints,
    n_diffusion_logs,
    duration_batches,
    schedulers,
    lr,
):
    def get_interval(total, times):
        return Time.from_batch(total // times)

    print(f"Total batches: {duration_batches}")

    checkpoint_interval = get_interval(duration_batches, n_checkpoints)
    print(f"Checkpoint interval: {checkpoint_interval}")

    diffusion_log_interval = get_interval(duration_batches, n_diffusion_logs)
    print(f"Diffusion log interval: {diffusion_log_interval}")

    eval_interval = get_interval(duration_batches, n_evals)
    print(f"Eval interval: {eval_interval}")

    trainer = Trainer(
        model=model,
        train_dataloader=train_dl,
        eval_dataloader=eval_dl,
        eval_interval=eval_interval,
        schedulers=schedulers,
        # default learning rate used by [0]
        optimizers=[AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))],
        max_duration=Time.from_batch(duration_batches),
        device="gpu",
        precision="amp",
        grad_accum=grad_accum,
        loggers=[
            FileLogger(),
            # don't save checkpoints to WandB
            WandBLogger(log_artifacts=False),
        ],
        callbacks=[
            LRMonitor(),
            SpeedMonitor(window_size=10),
            CheckpointSaver(save_interval=checkpoint_interval),
            DiffusionMonitor(interval=diffusion_log_interval),
        ],
    )
    return trainer
