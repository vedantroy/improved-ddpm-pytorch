from trainer import make_trainer
from dataloaders import overfit_dataloader
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

MODE = "overfit"


def run():
    config = TrainerConfig.create("./config/fixed_variance.yaml", None, cli_args=False)
    unet, diffusion = config.initialize_object()
    iddpm = IDDPM(unet, diffusion)

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
        batch_size = 1
        ds = dataset(batch_size, shuffle=True)
        train_dl = dataloader(ds, batch_size)
        trainer = make_trainer(dl, batch_size, lr=1e-4)
        trainer.fit()


if __name__ == "__main__":
    run()
