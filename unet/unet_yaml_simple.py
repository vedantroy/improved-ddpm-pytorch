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


@dataclass
class UNetHParams(hp.Hparams):
    channel_mult: List[int] = hp.required("# channel multipliers")
    attn: List[bool] = hp.required("whether to use attention")
    res_blocks: int = hp.required("# res blocks")

    # TODO: We could remove in/out channels, but seems excessive
    model_channels: int = hp.required("model channels")
    in_channels: int = hp.required("in channels")
    out_channels: int = hp.required("out channels")

    def initialize_object(self):
        return UNet(**self.__dict__)


class UNet(nn.Module):
    def __init__(
        self, channel_mult, attn, res_blocks, model_channels, in_channels, out_channels
    ):
        super().__init__()

        blocks = []
        skip_channels = []

        for idx, mult in range(channel_mult):
            for idx in range(res_blocks):
                blocks.append(ResNetBlock())
                pass
            pass

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
