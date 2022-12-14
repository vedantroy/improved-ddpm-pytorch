from pathlib import Path
from tqdm import tqdm
from typing import Optional

import torchvision
import torch as th
import typer
from diffusion.respace import (
    WrappedModel,
    create_map_and_betas,
    space_timesteps,
)
from diffusion.sampler import DDIMSampler, DDPMSampler
from iddpm import IDDPMConfig


def img_to_bytes(img):
    # TODO: Is the clamp necessary?
    return (((img + 1) / 2) * 255).clamp(0, 255).to(th.uint8)


def run(
    config: Path = typer.Option(...),
    out_dir: Path = typer.Option(...),
    checkpoint: Path = typer.Option(...),
    samples: int = typer.Option(...),
    sample_steps: str = typer.Option(default=""),
    save_final_only: bool = typer.Option(False),
    ddim_eta: Optional[float] = typer.Option(default=None),
):
    assert checkpoint.is_file(), f"Checkpoint file not found: {checkpoint}"

    config = IDDPMConfig.create(config, None, cli_args=False)
    iddpm = config.initialize_object()

    if sample_steps == "":
        sample_steps = [iddpm.diffusion.n_timesteps]
    # space_timesteps takes a string or a list of ints
    spacing = space_timesteps(iddpm.diffusion.n_timesteps, sample_steps)

    # We need the original betas to create the spaced betas
    timestep_map, betas = create_map_and_betas(iddpm.diffusion.betas, spacing)
    # TODO: roast mosaic
    # iddpm.model = WrappedModel(iddpm.model, timestep_map)

    iddpm = config.initialize_object(diffusion_kwargs=dict(betas=betas))

    out_dir.mkdir(parents=True)

    device = th.device("cuda")
    state_dict = th.load(checkpoint)
    iddpm.load_state_dict(state_dict["state"]["model"])

    iddpm = iddpm.to(device=device)
    iddpm.eval()
    wrapped_model = WrappedModel(iddpm.model, timestep_map)

    sampler = None
    if ddim_eta != None:
        print(f"Using DDIM with eta: {ddim_eta}")
        sampler = DDIMSampler(iddpm.diffusion, ddim_eta)
    else:
        print("Using DDPM")
        sampler = DDPMSampler(iddpm.diffusion)

    total = iddpm.diffusion.n_timesteps
    for idx, img in enumerate(
        tqdm(
            sampler.sample_loop_progressive(
                model=wrapped_model,
                noise=None,
                shape=(samples, 3, 64, 64),
                threshold="static",
                device=device,
            ),
            total=total,
        )
    ):

        if not save_final_only or idx == total - 1:
            step_path = out_dir / f"{idx:04d}"
            step_path.mkdir()
            for sample_idx, sample in enumerate(img):
                torchvision.io.write_png(
                    img_to_bytes(sample.cpu()), str(step_path / f"{sample_idx:04d}.png")
                )


if __name__ == "__main__":
    typer.run(run)
