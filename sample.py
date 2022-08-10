from pathlib import Path
from tqdm import tqdm

import torchvision
import torch as th
import typer

from iddpm import TrainerConfig, IDDPM


def img_to_bytes(img):
    # TODO: Is the clamp necessary?
    return (((img + 1) / 2) * 255).clamp(0, 255).to(th.uint8)


def run(
    config: Path = typer.Option(...),
    out_dir: Path = typer.Option(...),
    checkpoint: Path = typer.Option(...),
    samples: int = typer.Option(...),
):
    assert checkpoint.is_file(), f"Checkpoint file not found: {checkpoint}"

    config = TrainerConfig.create(config, None, cli_args=False)
    iddpm = config.initialize_object()

    out_dir.mkdir(parents=True)

    device = th.device("cuda")
    state_dict = th.load(checkpoint)
    iddpm.load_state_dict(state_dict["state"]["model"])

    iddpm = iddpm.to(device=device)
    iddpm.eval()

    total = iddpm.diffusion.n_timesteps
    for idx, img in enumerate(
        tqdm(
            iddpm.diffusion.p_sample_loop_progressive(
                model=iddpm.model,
                noise=None,
                shape=(samples, 3, 64, 64),
                threshold="static",
                device=device,
            ),
            total=total,
        )
    ):
        step_path = out_dir / f"{idx:04d}"
        step_path.mkdir()

        for sample_idx, sample in enumerate(img):
            torchvision.io.write_png(
                img_to_bytes(sample.cpu()), str(step_path / f"{sample_idx:04d}.png")
            )


if __name__ == "__main__":
    typer.run(run)
