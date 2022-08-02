# composer's optimizer is buggy
from torch.optim.adamw import AdamW
from composer.loggers import WandBLogger, FileLogger
from composer import Trainer
from composer.callbacks import CheckpointSaver, LRMonitor, SpeedMonitor
from callbacks import DiffusionMonitor


def make_trainer(*, model, train_dl, grad_accum, lr, duration, schedulers):
    trainer = Trainer(
        model=model,
        train_dataloader=train_dl,
        eval_dataloader=None,
        schedulers=schedulers,
        # default learning rate used by [0]
        optimizers=[AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))],
        max_duration=duration,
        device="gpu",
        precision="fp32",
        grad_accum=grad_accum,
        loggers=[
            FileLogger(),
            # don't save checkpoints to WandB
            WandBLogger(log_artifacts=False),
        ],
        callbacks=[
            LRMonitor(),
            SpeedMonitor(window_size=10),
            CheckpointSaver(save_interval="10000ba"),
            DiffusionMonitor(interval="500ba"),
        ],
    )
    return trainer