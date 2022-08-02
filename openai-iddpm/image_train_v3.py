from data import infinite_dataloader
from improved_diffusion import logger
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.train_util_v3 import TrainLoop

from params import TrainingHParams


def main():
    config = TrainingHParams.create("config_v3.yaml", None, cli_args=False)
    model, diffusion = config.initialize_object(use_custom_data=True)
    dconfig = config.data
    data = infinite_dataloader(dconfig.batch_size, dconfig.image_size, dconfig.data_dir)

    logger.configure()

    model = model.cuda()
    schedule_sampler = create_named_schedule_sampler(
        config.train.schedule_sampler, diffusion
    )

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=config.data.batch_size,
        lr=config.train.lr,
        log_interval=config.io.log_interval,
        schedule_sampler=schedule_sampler,
    ).run_loop()


if __name__ == "__main__":
    main()
