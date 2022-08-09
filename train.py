import typer
from composer.core import Time

from trainer import make_trainer, total_batches_and_scheduler_for_time
from dataloaders import dataloader
from iddpm import TrainerConfig, IDDPM

MODE = "train"


def main(
    # Files (config, datasets, etc.)
    config_file: str,
    dir_train: str,
    dir_val: str,
    # Time
    target_time: int,
    batch_rate: float,
    # Scheduling
    warmup: int,
    # Batch/micro-batch size
    batch_size: int,
    n_micro_batches: int,
    # Logging / Metrics
    n_checkpoints: int,
    n_diffusion_logs: int,
    n_evals: int,
):
    config = TrainerConfig.create(config_file, None, cli_args=False)
    iddpm = config.initialize_object()

    if MODE == "train":
        total_batches, scheduler = total_batches_and_scheduler_for_time(
            batch_rate=batch_rate, target_time=target_time, warmup=warmup  # 4.5  # 1.0
        )

        train_dl = dataloader("~/dataset/train", batch_size)
        val_dl = dataloader("~/dataset/val", batch_size)
        trainer = make_trainer(
            model=iddpm,
            train_dl=train_dl,
            eval_dl=val_dl,
            grad_accum=1,
            n_evals=50,
            n_checkpoints=10,
            n_diffusion_logs=10,
            duration_batches=total_batches,
            schedulers=[scheduler],
        )
        trainer.fit()
    else:
        raise Exception("Unknown mode")


if __name__ == "__main__":
    typer.run(main)
