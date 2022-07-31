from types import SimpleNamespace
from typing import Optional, List
from dataclasses import dataclass

import yahp as hp
import torch as th
from torch import nn

from .layers import (
    AttentionBlock,
    Downsample,
    ResNetBlock,
    Residual,
    TimestepBlock,
    Upsample,
    normalization,
    zero_module,
    timestep_embedding,
)


def split_dict(d, exclude):
    a, b = {}, {}
    for (k, v) in d.items():
        d = b if k in exclude else a
        if v != None:
            d[k] = v
    return a, b


@dataclass
class ResNetBlockHParams(hp.Hparams):
    in_channels: int = hp.required("# input channels")
    out_channels: int = hp.required("# output channels")
    time_emb_channels: int = hp.required("# time embedding channels")

    def initialize_object(self):
        return ResNetBlock(**self.__dict__)


@dataclass
class AttentionBlockHParams(hp.Hparams):
    channels: int = hp.required("# input & output channels")
    heads: int = hp.required("# attention heads")

    def initialize_object(self):
        return AttentionBlock(**self.__dict__)


@dataclass
class UpsampleHParams(hp.Hparams):
    channels: int = hp.required("# input & output channels")

    def initialize_object(self):
        return Upsample(**self.__dict__)


@dataclass
class DownsampleHParams(hp.Hparams):
    channels: int = hp.required("# input & output channels")

    def initialize_object(self):
        return Downsample(**self.__dict__)


@dataclass
class BlockHParams(hp.Hparams):
    ResNet: Optional[ResNetBlockHParams] = hp.optional("ResNet block", default=None)
    Attention: Optional[AttentionBlockHParams] = hp.optional(
        "Attention block", default=None
    )
    Upsample: Optional[UpsampleHParams] = hp.optional("Upsample block", default=None)
    Downsample: Optional[DownsampleHParams] = hp.optional(
        "Downsample block", default=None
    )

    skip: Optional[str] = hp.optional("Skip connection", default=None)
    # skip_push: Optional[bool] = hp.optional("Add to skip connection", default=False)
    # skip_pop: Optional[bool] = hp.optional("Take from skip connection", default=False)
    # Ideally, I'd make this a block, but I can't figure out the circular part
    use_residual: Optional[bool] = hp.optional("Add residual connection", default=False)

    def initialize_object(self):
        block, attrs = split_dict(
            # self.__dict__, ["skip_push", "skip_pop", "use_residual"]
            self.__dict__,
            ["skip", "use_residual"],
        )
        assert (
            len(block.values()) == 1
        ), f"Must specify exactly one block type, found: {block.keys()}"
        return (SimpleNamespace(**attrs), list(block.values())[0].initialize_object())


@dataclass
class UNetHParams(hp.Hparams):
    Blocks: List[BlockHParams] = hp.required("all blocks except for input/output")

    # TODO: We could remove in/out channels, but seems excessive
    model_channels: int = hp.required("model channels")
    in_channels: int = hp.required("in channels")
    out_channels: int = hp.required("out channels")

    def initialize_object(self):
        attrs_and_blocks = [block.initialize_object() for block in self.Blocks]
        _, kwargs = split_dict(
            self.__dict__, ["model_channels", "out_channels", "in_channels"]
        )
        return UNet(attrs_and_blocks, **kwargs)


class UNet(nn.Module):
    def __init__(self, blocks, model_channels, in_channels, out_channels):
        super().__init__()
        self.attrs = [attrs for (attrs, _) in blocks]
        blocks = [block for (_, block) in blocks]
        for idx, block in enumerate(blocks):
            if self.attrs[idx].use_residual:
                blocks[idx] = Residual(block)
        self.blocks = nn.ModuleList(blocks)

        time_emb_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.in_layers = nn.Conv2d(
            in_channels, model_channels, kernel_size=3, padding=1
        )
        self.out_layers = nn.Sequential(
            normalization(model_channels),
            nn.SiLU(),
            zero_module(
                nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)
            ),
        )

    def forward(self, x, timesteps):
        emb = self.time_embed(timestep_embedding(timesteps, dim=self.model_channels))
        x = self.in_layers(x)
        skips = [x]

        for block, attrs in zip(self.blocks, self.attrs):
            input = x
            if attrs.skip == "pop":
                input = th.cat([x, skips.pop()], dim=1)

            if isinstance(block, TimestepBlock):
                x = block(input, emb)
            else:
                x = block(input)

            if attrs.skip == "push":
                skips.append(x)

        assert len(skips) == 0, f"{len(skips)} skips left"
        return self.out_layers(x)
