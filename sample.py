from pathlib import Path
from tqdm import tqdm

import torchvision
import torch as th

from iddpm import TrainerConfig, IDDPM

def img_to_bytes(img):
    # TODO: Is the clamp necessary?
    return (((img + 1) / 2) * 255).clamp(-1, 1).to(th.unit8)

def run():
    config = TrainerConfig.create("./config/fixed_variance.yaml", None, cli_args=False)
    unet, diffusion = config.initialize_object()
    iddpm = IDDPM(unet, diffusion)

    OUT_PATH = ""
    CHECKPOINT_PATH = ""
    N_SAMPLES = 10

    out_path = Path(OUT_PATH).mkdir(parents=True)

    state_dict = th.load(CHECKPOINT_PATH)
    iddpm.load_state_dict(state_dict["state"]["model"])

    iddpm.eval()

    total = iddpm.diffusion.n_timesteps
    for idx, img in enumerate(tqdm(iddpm.diffusion.p_sample_loop_progressive(
        model=iddpm.model,
        noise=None,
        shape=(N_SAMPLES, 3, 64, 64),
        threshold="static",
        device=th.device("cuda")
    ), total=total)):
        step_path = out_path / f"{idx:04d}"
        step_path.mkdir()

        for sample_idx, sample in img:
            torchvision.io.write_png(img_to_bytes(sample), str(step_path / f"{sample_idx:04d}.png"))