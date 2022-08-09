from pytimeparse.timeparse import timeparse
import typer

from trainer import make_trainer, total_batches_and_scheduler_for_time
from dataloaders import dataloader
from iddpm import TrainerConfig

# app = typer.Typer(pretty_exceptions_show_locals=False)

#@app.command()
def main(
    # Files (config, datasets, etc.)
    config_file: str = typer.Option(...),
    dir_train: str = typer.Option(...),
    dir_val: str = typer.Option(...),
    # Time
    target_time: str = typer.Option(...),
    batch_rate: float = typer.Option(...),
    # Scheduling
    warmup: int = typer.Option(...),
    lr: str = typer.Option(...),
    # Batch/micro-batch size
    batch_size: int = typer.Option(...),
    micro_batches: int = typer.Option(...),
    # Logging / Metrics
    checkpoints: int = typer.Option(...),
    diffusion_logs: int = typer.Option(...),
    evals: int = typer.Option(...),
):
    config = TrainerConfig.create(config_file, None, cli_args=False)
    iddpm = config.initialize_object()

    def get_num(expr, typ):
        val = eval(expr)
        assert isinstance(val, typ), f"{type(val)} != {typ} (expr = {expr})"
        return val

    # gasp!!
    lr = get_num(lr, float)
    target_time = timeparse(target_time)
    assert isinstance(target_time, int), f"Invalid # of secs: {target_time}"

    MODE = "train"
    if MODE == "train":
        total_batches, scheduler = total_batches_and_scheduler_for_time(
            batch_rate=batch_rate, target_time=target_time, warmup=warmup
        )

        train_dl = dataloader(dir_train, batch_size)
        val_dl = dataloader(dir_val, batch_size)
        trainer = make_trainer(
            model=iddpm,
            train_dl=train_dl,
            eval_dl=val_dl,
            grad_accum=micro_batches,
            n_evals=evals,
            n_checkpoints=checkpoints,
            n_diffusion_logs=diffusion_logs,
            duration_batches=total_batches,
            schedulers=[scheduler],
            lr=lr,
        )
        trainer.fit()
    else:
        raise Exception("Unknown mode")


if __name__ == "__main__":
    #main()
    typer.run(main)
