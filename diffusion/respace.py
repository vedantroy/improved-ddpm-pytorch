import torch as th
from torch import nn

from .diffusion import LearnedVarianceGaussianDiffusion

### Start OpenAI Code
def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


### End OpenAI Code

# This has a bug -_-, use OpenAI's code
def simple_space_timesteps(num_timesteps, num_sample_steps):
    assert num_sample_steps >= 2, "num_sample_steps must be at least 2"

    interval = num_timesteps / (num_sample_steps - 1)
    steps = []

    for i in range(num_sample_steps):
        step = min(round(i * interval), num_timesteps - 1)
        steps.append(step)

    assert (
        steps[0] == 0 and steps[-1] == num_timesteps - 1
    ), f"Invalid spacing (len={len(steps)}): {steps}"
    r = set(steps)
    assert len(r) == num_sample_steps, f"{len(r)} != {num_sample_steps}"
    return r


def create_map_and_betas(betas, use_timesteps):
    use_timesteps = set(use_timesteps)

    # Doesn't matter what diffusion we use since the constructor
    # is defined in the base class
    base_diffusion = LearnedVarianceGaussianDiffusion(betas)
    as_t_m_1 = 1

    map_generation_step_to_timestep = []

    new_betas = []
    for i, as_t in enumerate(base_diffusion.alphas_cumprod):
        if i in use_timesteps:
            new_betas.append(1 - (as_t / as_t_m_1))
            as_t_m_1 = as_t
            map_generation_step_to_timestep.append(i)

    assert len(new_betas) == len(map_generation_step_to_timestep)
    return map_generation_step_to_timestep, th.Tensor(new_betas).to(dtype=th.float64)


class WrappedModel:
    def __init__(self, model, timestep_map):
        super().__init__()
        self.model = model
        self.timestep_map = th.Tensor(timestep_map).to(dtype=th.int)


    def __call__(self, x, ts):
        new_ts = self.timestep_map[ts].to(device=ts.device, dtype=ts.dtype)
        return self.model(x, new_ts)
