from typing import List, Optional
from dataclasses import dataclass

import yahp as hp
from improved_diffusion.image_datasets import load_data
from improved_diffusion.script_util import (
    create_gaussian_diffusion,
)
from improved_diffusion.unet import UNetModel

@dataclass
class UNetHParams(hp.Hparams):
    in_channels: int = hp.required("channels in the input tensor")
    model_channels: int = hp.required("base channel count for the model")
    out_channels: int = hp.required("channels in the output tensor")
    num_res_blocks: int = hp.required("number of residual blocks per downsample")
    attention_resolutions: List[int] = hp.required(
        "a collection of downsample rates at which attention will take place"
    )
    dropout: float = hp.required("the dropout probability")
    channel_mult: List[int] = hp.required(
        "channel multiplier for each level of the UNet"
    )
    conv_resample: bool = hp.required(
        "if True, use learned convolutions for upsampling and downsampling"
    )
    dims: int = hp.required("determines if the signal is 1D, 2D, or 3D")
    num_classes: Optional[int] = hp.required(
        "if specified (as an int), then this model will be class-conditional with `num_classes` classes"
    )
    use_checkpoint: bool = hp.required(
        "use gradient checkpointing to reduce memory usage"
    )
    num_heads: int = hp.required(
        "the number of attention heads in each attention layer"
    )
    num_heads_upsample: int = hp.required("num_heads_upsample")
    use_scale_shift_norm: bool = hp.required("use_scale_shift_norm")

    def initialize_object(self):
        return UNetModel(**self.__dict__)


@dataclass
class SpacedGaussianDiffusionHParams(hp.Hparams):
    steps: int = hp.required("diffusion_steps")
    learn_sigma: bool = hp.required("learn_sigma")
    sigma_small: bool = hp.required("sigma_small")
    noise_schedule: str = hp.required("noise_schedule")
    use_kl: bool = hp.required("use_kl")
    predict_xstart: bool = hp.required("predict_xstart")
    rescale_timesteps: bool = hp.required("rescale_timesteps")
    rescale_learned_sigmas: bool = hp.required("rescale_learned_sigmas")
    timestep_respacing: str = hp.required("timestep_respacing")

    def initialize_object(self):
        return create_gaussian_diffusion(**self.__dict__)


@dataclass
class DataHParams(hp.Hparams):
    data_dir: str = hp.required("data_dir")
    batch_size: int = hp.required("batch_size")
    image_size: int = hp.required("image_size")
    class_cond: bool = hp.required("class_cond")

    def initialize_object(self):
        return load_data(**self.__dict__)

@dataclass
class TrainHParams(hp.Hparams):
    lr: float = hp.required("lr")
    ema_rate: float = hp.required("ema_rate")
    use_fp16: bool = hp.required("use_fp16")
    fp16_scale_growth: float = hp.required("fp16_scale_growth")
    weight_decay: float = hp.required("weight_decay")
    lr_anneal_steps: int = hp.required("lr_anneal_steps")
    schedule_sampler: str = hp.required("schedule_sampler")
    microbatch: int = hp.required("microbatch")

@dataclass
class IoHParams(hp.Hparams):
    log_interval: int = hp.required("log_interval")
    save_interval: int = hp.required("save_interval")
    resume_checkpoint: str = hp.required("resume_checkpoint")

@dataclass
class TrainingHParams(hp.Hparams):
    unet: UNetHParams = hp.required("the UNet parameters")
    spaced_diffusion: SpacedGaussianDiffusionHParams = hp.required(
        "the spaced diffusion parameters"
    )
    data: DataHParams = hp.required("the data parameters")
    train: TrainHParams = hp.required("the training parameters")
    io: IoHParams = hp.required("the io parameters")

    def initialize_object(self, use_custom_data=False):
        unet, diffusion = self.unet.initialize_object(), self.spaced_diffusion.initialize_object()
        if not use_custom_data:
            return (
                unet,
                diffusion,
                self.data.initialize_object(),
            )
        else:
            return (unet, diffusion)