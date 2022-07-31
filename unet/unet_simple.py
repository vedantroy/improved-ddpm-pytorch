from torch import nn

from .layers import (
    AddSkipConnection,
    AttentionBlock,
    Downsample,
    ResNetBlock,
    Residual,
    TakeFromSkipConnection,
    TimestepEmbedSequential,
    Upsample,
    normalization,
    timestep_embedding,
    zero_module,
)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        model_channels,
        channel_mult,
        layer_attn,
        num_res_blocks,
        num_heads,
    ):
        super().__init__()
        assert len(channel_mult) == len(
            layer_attn
        ), "each layer must specify whether it uses attention between residual blocks"

        time_emb_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )


        skip_channels = []
        attrs = []
        lambda res: ResNetBlock(c_in, c_out, time_emb_dim)
        def res(c_in, c_out):
            return ResNetBlock(c_in, c_out, time_emb_dim)
        res = lambda c_in, c_out: ResNetBlock(c_in, c_out, time_emb_dim)
        attn = lambda c: Residual(AttentionBlock(c))

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