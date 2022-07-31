import torch as th
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
        self.num_res_blocks = num_res_blocks
        self.model_channels = model_channels
        self.num_levels = len(channel_mult)

        time_emb_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.skips = []
        skip_connection_channels = []

        self.downs = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
                )
            ]
        )
        skip_connection_channels.append(model_channels)
        self.ups = nn.ModuleList([])

        for level, mult in enumerate(channel_mult):
            is_bottom_level = level == 0
            prev_layer_out_channels = (
                model_channels
                if is_bottom_level
                else model_channels * channel_mult[level - 1]
            )
            cur_out_channels = model_channels * channel_mult[level]
            use_attn = layer_attn[level]
            is_last_layer_before_middle = level == len(channel_mult) - 1

            for idx in range(num_res_blocks):
                cur_in_channels = (
                    prev_layer_out_channels if idx == 0 else cur_out_channels
                )
                blocks = [ResNetBlock(cur_in_channels, cur_out_channels, time_emb_dim)]
                if use_attn:
                    blocks.append(Residual(AttentionBlock(cur_out_channels, num_heads)))
                self.downs.append(TimestepEmbedSequential(*blocks))
                skip_connection_channels.append(cur_out_channels)
            add_downsample = not is_last_layer_before_middle
            if add_downsample:
                self.downs.append(TimestepEmbedSequential(Downsample(cur_out_channels)))
                skip_connection_channels.append(cur_out_channels)

        middle_channels = model_channels * channel_mult[-1]
        self.middle = TimestepEmbedSequential(
            ResNetBlock(middle_channels, middle_channels, time_emb_dim),
            AttentionBlock(middle_channels, num_heads),
            ResNetBlock(middle_channels, middle_channels, time_emb_dim),
        )

        backwards_channel_mult = list(reversed(channel_mult))
        for level, (mult, use_attn) in enumerate(
            zip(backwards_channel_mult, layer_attn[::-1])
        ):
            is_bottom_level = level == 0
            is_top_level = level == len(channel_mult) - 1
            prev_layer_channels = (
                middle_channels
                if is_bottom_level
                else model_channels * backwards_channel_mult[level - 1]
            )
            cur_out_channels = model_channels * mult

            up = []
            for idx in range(num_res_blocks + 1):
                skip_channels = skip_connection_channels.pop()
                res_block_in_channels = (
                    prev_layer_channels if idx == 0 else cur_out_channels
                ) + skip_channels
                up.append(
                    ResNetBlock(
                        res_block_in_channels,
                        cur_out_channels,
                        time_emb_dim,
                    )
                )
                if use_attn:
                    up.append(Residual(AttentionBlock(cur_out_channels, num_heads)))
                if not is_top_level and idx == num_res_blocks:
                    up.append(Upsample(cur_out_channels))
                self.ups.append(TimestepEmbedSequential(*up))

        self.out_layers = nn.Sequential(
            normalization(model_channels),
            nn.SiLU(),
            zero_module(
                nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)
            ),
        )

    def _print(self, s, indent):
        print(" " * indent + s)

    def _print_layer(self, l, indent=0):
        if isinstance(l, AddSkipConnection):
            self._print("Start Skip Connection:", indent)
            self._print_layer(l.fn, indent + 2)
        elif isinstance(l, TakeFromSkipConnection):
            self._print(
                f"End Skip Connection (channels={l.expected_channels}):", indent
            )
            self._print_layer(l.fn, indent + 2)
        elif isinstance(l, TimestepEmbedSequential):
            for l2 in l:
                self._print_layer(l2, indent + 2)
        elif isinstance(l, nn.Sequential):
            for l2 in l:
                self._print_layer(l2, indent + 2)
        elif isinstance(l, nn.ModuleList):
            for l2 in l:
                self._print_layer(l2, indent)
                print("")
        elif isinstance(l, Residual):
            self._print("Residual:", indent)
            self._print_layer(l.fn, indent + 2)
        elif isinstance(l, ResNetBlock):
            self._print(
                f"ResBlock(in={l.in_channels}, out={l.out_channels}, emb_channels={l.emb_channels})",
                indent,
            )
        elif isinstance(l, AttentionBlock):
            self._print(f"AttentionBlock(in={l.channels}, heads={l.num_heads})", indent)
        elif isinstance(l, nn.Conv2d):
            self._print(f"Conv2d(in={l.in_channels}, out={l.out_channels})", indent)
        elif isinstance(l, Downsample):
            self._print(f"Downsample(in={l.channels})", indent)
        elif isinstance(l, Upsample):
            self._print(f"Upsample(in={l.channels})", indent)
        elif isinstance(l, nn.SiLU):
            self._print("SiLU", indent)
        elif isinstance(l, nn.GroupNorm):
            self._print("GroupNorm", indent)
        else:
            raise Exception(f"Unknown layer type: {type(l)}")

    # This is before I realized Pytorch
    # will print the model architecture ...
    def print_architecture(self):
        print_div = lambda x: print(f"\n===={x}====\n")
        print_div("IN")
        self._print_layer(self.in_conv)
        print_div("DOWN")
        self._print_layer(self.downs)
        print_div("MIDDLE")
        self._print_layer(self.middle)
        print_div("UP")
        self._print_layer(self.ups)
        print_div("OUT")
        self._print_layer(self.out_layers)

    def forward(self, x, timesteps):
        # By using `AddSkipConnection` and `TakeFromSkipConnection`
        # the skip connections are hidden in the forward method, but
        # the network code is more descriptive/declarative in __init__
        # which is a net win

        emb = self.time_embed(timestep_embedding(timesteps, dim=self.model_channels))

        skips = []
        for down in self.downs:
            x = down(x, emb)
            skips.append(x)

        # assert (
        #    len(self.skips) == expected_skips
        # ), f"skips should have {expected_skips} skip connections, but has {len(self.skips)}"

        x = self.middle(x, emb)

        print(f"SKIPS:" f"{len(skips)}")

        for up in self.ups:
            with_skip = th.cat([x, skips.pop()], dim=1)
            x = up(with_skip, emb)

        assert len(skips) == 0, f"{len(skips)} skips left over"
        return self.out_layers(x)
