from composer.core import Time
from composer.optim.scheduler import (
    CosineAnnealingWithWarmupScheduler,
)

from trainer import make_trainer
from dataloaders import dataloader, overfit_dataloader
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


def run():
    config = TrainerConfig.create("./config/fixed_variance.yaml", None, cli_args=False)
    unet, diffusion = config.initialize_object()
    iddpm = IDDPM(unet, diffusion)

    def intcast(x):
        assert x.is_integer(), f"Invalid int: {x}"
        return int(x)

    def batches(batches_per_sec, secs):
        y = batches_per_sec * secs
        return intcast(y)

    if MODE == "scan_samples":
        raise Exception("unsupported")
        return
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
        batches_per_sec = 4.5
        target_time = 4 * 60 * 60
        warmup = 500
        total_batches = batches(batches_per_sec, target_time)

        print(f"Total: {total_batches}")
        print(f"Warmup: {warmup}")
        t_max = (total_batches - warmup)
        print(f"Remaining: {t_max}")
        t_max *= 1.2
        t_max = intcast(t_max)
        print(f"t_max: {t_max}")


        t_max = int(t_max)

        dl = dataloader(batch_size, "~/parquetx64")
        trainer = make_trainer(
            model=iddpm,
            train_dl=dl,
            grad_accum=1,
            lr=1e-4,
            duration=Time.from_batch(total_batches),
            schedulers=[CosineAnnealingWithWarmupScheduler(t_warmup=Time.from_batch(warmup), t_max=Time.from_batch(t_max))]
        )
        trainer.fit()


if __name__ == "__main__":
    run()
