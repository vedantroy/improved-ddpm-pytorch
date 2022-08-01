from composer.optim.decoupled_weight_decay import DecoupledAdamW
from composer.optim.scheduler import (
    CosineAnnealingWithWarmupScheduler,
    CosineAnnealingScheduler,
)
from composer.loggers import WandBLogger, FileLogger
from composer import Trainer
from composer.callbacks import CheckpointSaver, LRMonitor, SpeedMonitor


def make_trainer(model, train_dl, grad_accum, lr=1e-4, duration="1000ep"):
    trainer = Trainer(
        model=model,
        train_dataloader=train_dl,
        eval_dataloader=None,
        schedulers=[CosineAnnealingWithWarmupScheduler(t_warmup="200ba", t_max="1dur")],
        # default learning rate used by [0]
        optimizers=[DecoupledAdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))],
        max_duration=duration,
        device="gpu",
        precision="fp32",
        grad_accum=grad_accum,
        loggers=[
            FileLogger(),
            # don't save checkpoints to WandB
            WandBLogger(log_artifacts=False),
        ],
        callbacks=[LRMonitor(), SpeedMonitor(window_size=10), CheckpointSaver()],
    )
    return trainer
