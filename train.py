from pathlib import Path
from dataclasses import dataclass

from pytimeparse.timeparse import timeparse
import typer
import yahp as hp

from trainer import make_trainer, total_batches_and_scheduler_for_time
from dataloaders import dataloader
from iddpm import IDDPMConfig


@dataclass
class RunConfig(hp.Hparams):
    target_time: str = hp.required("Target time")
    batch_rate: float = hp.required("Batch rate")

    warmup_batches: int = hp.required("Warmup batches")
    lr: str = hp.required("Learning rate")

    batch_size: int = hp.required("Batch size")
    micro_batches: int = hp.required("Micro-batches")
    precision: str = hp.required("Precision")

    checkpoints: int = hp.required("# Checkpoints")
    diffusion_logs: int = hp.required("# Diffusion logs")
    evals: int = hp.required("# Evals")


def main(
    model_config_file: Path = typer.Option(...),
    run_config_file: Path = typer.Option(...),
    dir_train: Path = typer.Option(...),
    dir_val: Path = typer.Option(...),
):
    def get_dir(p):
        assert p.is_dir, f"{p} is not a directory"
        return str(p)

    dir_train = get_dir(dir_train)
    dir_val = get_dir(dir_val)

    config = IDDPMConfig.create(model_config_file, None, cli_args=False)
    iddpm = config.initialize_object()

    run_config = RunConfig.create(run_config_file, None, cli_args=False)
    c = run_config

    def get_num(expr, typ):
        # gasp!!
        val = eval(expr)
        assert isinstance(val, typ), f"{type(val)} != {typ} (expr = {expr})"
        return val

    lr = get_num(c.lr, float)
    target_time = timeparse(c.target_time)
    assert isinstance(target_time, int), f"Invalid # of secs: {target_time}"

    MODE = "train"
    if MODE == "train":
        total_batches, scheduler = total_batches_and_scheduler_for_time(
            batch_rate=c.batch_rate, target_time=target_time, warmup=c.warmup_batches
        )

        train_dl = dataloader(dir_train, c.batch_size, workers=8)
        val_dl = dataloader(dir_val, c.batch_size, workers=2)
        trainer = make_trainer(
            model=iddpm,
            train_dl=train_dl,
            eval_dl=val_dl,
            grad_accum=c.micro_batches,
            n_evals=c.evals,
            n_checkpoints=c.checkpoints,
            n_diffusion_logs=c.diffusion_logs,
            duration_batches=total_batches,
            schedulers=[scheduler],
            lr=lr,
            precision=c.precision,
        )
        trainer.fit()
    else:
        raise Exception("Unknown mode")


if __name__ == "__main__":
    typer.run(main)
