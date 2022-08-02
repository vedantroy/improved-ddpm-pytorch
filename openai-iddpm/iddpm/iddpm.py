from typing import List
from dataclasses import dataclass

from composer import ComposerModel
import yahp as hp
from .unet.unet import UNet


@dataclass
class UNetParams(hp.Hparams):
    in_channels: int = hp.required("# input channels")
    out_channels: int = hp.required("# output channels")
    # (C in [0] under Appendix A "Hyperparameters")
    model_channels: int = hp.required("# model channels")
    channel_mult: List[int] = hp.required("the channel multipliers")
    layer_attn: List[bool] = hp.required(
        "whether to use attention between ResNet blocks"
    )
    res_blocks: int = hp.required("# ResNet blocks")
    attention_heads: int = hp.required("# attention heads")

    def initialize_object(self):
        return UNet(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            model_channels=self.model_channels,
            channel_mult=self.channel_mult,
            layer_attn=self.layer_attn,
            num_res_blocks=self.res_blocks,
            num_heads=self.attention_heads,
        )


class StrippedIDDPM(ComposerModel):
    def __init__(self, unet: UNet):
        super().__init__()
        self.model = unet

    def forward():
        pass

    def loss():
        pass
