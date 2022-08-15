# Use OpenAI's diffusion to sample from the model

from pathlib import Path
from tqdm import tqdm

import torchvision
import torch as th
import typer
from tests.openai_code.gaussian_diffusion import (
    LossType,
    ModelMeanType,
    ModelVarType,
    LossType
)
from tests.openai_code.respace import (
    SpacedDiffusion,
    space_timesteps
)
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
):
    assert checkpoint.is_file(), f"Checkpoint file not found: {checkpoint}"

    config = IDDPMConfig.create(config, None, cli_args=False)
    iddpm = config.initialize_object()

    if sample_steps == "":
        sample_steps = [iddpm.diffusion.n_timesteps]
    # space_timesteps takes a string or a list of ints
    spacing = space_timesteps(iddpm.diffusion.n_timesteps, sample_steps)
    diffusion = SpacedDiffusion(
        use_timesteps=spacing,
        betas=iddpm.diffusion.betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.RESCALED_MSE,
        rescale_timesteps=False
    )

    out_dir.mkdir(parents=True)

    device = th.device("cuda")
    state_dict = th.load(checkpoint)
    iddpm.load_state_dict(state_dict["state"]["model"])

    iddpm = iddpm.to(device=device)
    iddpm.eval()

    total = diffusion.num_timesteps
    for idx, img in enumerate(
        tqdm(
            #diffusion.p_sample_loop_progressive(
            diffusion.ddim_sample_loop_progressive(
                model=iddpm.model,
                noise=None,
                shape=(samples, 3, 64, 64),
                clip_denoised=True,
                device=device,
                eta=0.0
            ),
            total=total,
        )
    ):

        if not save_final_only or idx == total - 1:
            step_path = out_dir / f"{idx:04d}"
            step_path.mkdir()
            for sample_idx, sample in enumerate(img["sample"]):
                torchvision.io.write_png(
                    img_to_bytes(sample.cpu()), str(step_path / f"{sample_idx:04d}.png")
                )


if __name__ == "__main__":
    typer.run(run)
