"""
Train a diffusion model on images.
"""

from params import TrainingHParams
from improved_diffusion import dist_util, logger
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.train_util_v1 import TrainLoop


def main():
    config = TrainingHParams.create("config.yaml", None, cli_args=False)
    model, diffusion, data = config.initialize_object()

    dist_util.setup_dist()
    logger.configure()

    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(
        config.train.schedule_sampler, diffusion
    )

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=config.data.batch_size,
        microbatch=config.train.microbatch,
        lr=config.train.lr,
        ema_rate=config.train.ema_rate,
        log_interval=config.io.log_interval,
        save_interval=config.io.save_interval,
        resume_checkpoint=config.io.resume_checkpoint,
        use_fp16=config.train.use_fp16,
        fp16_scale_growth=config.train.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=config.train.weight_decay,
        lr_anneal_steps=config.train.lr_anneal_steps,
    ).run_loop()


if __name__ == "__main__":
    main()
