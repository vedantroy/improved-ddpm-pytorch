from pathlib import Path
from tqdm import tqdm

import torchvision
import torch as th

from iddpm import TrainerConfig, IDDPM
from tests.openai_code.gaussian_diffusion import (
    LossType,
    ModelMeanType,
    ModelVarType,
    GaussianDiffusion,
    get_named_beta_schedule,
)


def img_to_bytes(img):
    # TODO: Is the clamp necessary?
    return (((img + 1) / 2) * 255).clamp(0, 255).to(th.uint8)


def run():
    config = TrainerConfig.create("./config/fixed_variance.yaml", None, cli_args=False)
    unet, diffusion = config.initialize_object()

    betas = get_named_beta_schedule("cosine", 1000)
    diffusion = GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.RESCALED_MSE,
        rescale_timesteps=True,
    )

    iddpm = IDDPM(unet, diffusion)

    OUT_PATH = "./samples"
    CHECKPOINT_PATH = "ep10.checkpoint"
    N_SAMPLES = 10

    out_path = Path(OUT_PATH)
    out_path.mkdir(parents=True)

    device = th.device("cuda")
    state_dict = th.load(CHECKPOINT_PATH)
    iddpm.load_state_dict(state_dict["state"]["model"])

    iddpm = iddpm.to(device=device)
    iddpm.eval()

    total = iddpm.diffusion.n_timesteps
    for idx, img in enumerate(
        tqdm(
            iddpm.diffusion.p_sample_loop_progressive(
                model=iddpm.model,
                noise=None,
                shape=(N_SAMPLES, 3, 64, 64),
                clip_denoised=True,
                # threshold="static",
                device=device,
            ),
            total=total,
        )
    ):
        step_path = out_path / f"{idx:04d}"
        step_path.mkdir()
        img = img["sample"]

        for sample_idx, sample in enumerate(img):
            torchvision.io.write_png(
                img_to_bytes(sample.cpu()), str(step_path / f"{sample_idx:04d}.png")
            )


run()
