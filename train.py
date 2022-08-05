from composer.core import Time
from composer.optim.scheduler import (
    CosineAnnealingWithWarmupScheduler,
)

from trainer import make_trainer
from dataloaders import dataloader, overfit_dataloader, train_val_loaders
from iddpm import TrainerConfig, IDDPM

# def scan_samples(model: ComposerModel, dl):
#     def unnormalize(x):
#         # Not sure if `.clamp` is necessary
#         return (((x + 1) / 2).clamp(-1, 1) * 255).to(dtype=th.uint8).cpu()
#
#     with th.no_grad():
#         model = model.cuda()
#         model.eval()
#         for idx, batch in enumerate(tqdm(dl)):
#             batch["img"] = batch["img"].cuda()
#             out = model(batch)
#             mse_loss, vb_loss = model.loss(out, batch)
#             if vb_loss.item() > 10:
#                 print(f"High vb loss: {vb_loss}")
#                 t = out.t.cpu().item()
#                 img = unnormalize(out.x_0)[0]
#                 noised_img = unnormalize(out.x_t)[0]
#                 dir = Path("./vb_anomalies") / str(idx)
#                 dir.mkdir(exist_ok=True, parents=True)
#                 torchvision.io.write_png(img, str(dir / f"original_{t}.png"))
#                 torchvision.io.write_png(noised_img, str(dir / f"noised_{t}.png"))

MODE = "train"


def batches_for_time(batch_rate, target_time, warmup, cosine_factor=1.2):
    def intcast(x):
        assert x.is_integer(), f"Invalid int: {x}"
        return int(x)

    def batches(batches_per_sec, secs):
        y = batches_per_sec * secs
        return intcast(y)

    total_batches = batches(batch_rate, target_time)
    t_max = total_batches - warmup

    return Time.from_batch(total_batches), CosineAnnealingWithWarmupScheduler(
        t_warmup=Time.from_batch(warmup), t_max=Time.from_batch(intcast(t_max * cosine_factor))
    )


def run():
    config = TrainerConfig.create("./config/fixed_variance.yaml", None, cli_args=False)
    unet, diffusion = config.initialize_object()
    iddpm = IDDPM(unet, diffusion)

    if MODE == "scan_samples":
        raise Exception("unsupported")
    elif MODE == "overfit":
        batches, batch_size = 1, 2
        micro_batch_size = batch_size // 1
        dl = overfit_dataloader(batches, 16, "./data/parquetx64")
        # manual_train(dl, diffusion, unet)
        trainer = make_trainer(
            iddpm, dl, batch_size // micro_batch_size, lr=1e-4, duration="1000ba"
        )
        trainer.fit()
    elif MODE == "train":
        batch_size = 8
        total_batches, scheduler = batches_for_time(
            batch_rate=4.5,
            target_time=4 * 60 * 60,
            warmup=200,
        )
        print(f"Total batches: {total_batches}")

        dataloaders = train_val_loaders(batch_size, "~/parquetx64", val_batches=8)
        train_dl, val_dl = dataloaders.train, dataloaders.val
        trainer = make_trainer(
            model=iddpm,
            train_dl=train_dl,
            eval_dl=val_dl,
            eval_interval=Time.from_batch(5000),
            grad_accum=1,
            lr=1e-4,
            duration=total_batches,
            schedulers=[scheduler],
        )
        trainer.fit()


if __name__ == "__main__":
    run()
