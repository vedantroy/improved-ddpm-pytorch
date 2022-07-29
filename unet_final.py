import math
from abc import abstractmethod

import torch as th
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange

## Start OpenAI Code
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


## End OpenAI Code


def normalization(channels):
    return nn.GroupNorm(num_groups=32, num_channels=channels)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


# TODO: I'm not sure if this is a good abstraction
class AddSkipConnection(TimestepBlock):
    def __init__(self, fn, skips):
        super().__init__()
        self.fn = fn
        self.skips = skips

    def forward(self, x, emb, **kwargs):
        y = self.fn(x, emb, **kwargs)
        self.skips.append(y)
        return y

class TakeFromSkipConnection(nn.Module):
    def __init__(self, fn, skips, expected_channels):
        super().__init__()
        self.fn = fn
        self.skips = skips
        self.expected_channels = expected_channels

    def forward(self, x, emb, **kwargs):
        skip = self.skips.pop()
        assert (
            skip.shape[1] == self.expected_channels
        ), f"expected skip connection channels {skip.shape[1]} != {self.expected_channels}"
        with_skip = th.cat([skip, x], dim=1)
        y = self.fn(with_skip, emb, **kwargs)
        return y


class QKVAttention(nn.Module):
    def forward(self, qkv):
        N, triple_dim, seq = qkv.shape
        dim = triple_dim // 3
        q, k, v = rearrange(qkv, "b (split dim) s -> split b dim s", split=3)
        scale = 1 / math.sqrt(math.sqrt(dim))
        attn = th.einsum("bcs,bct->bst", q * scale, k * scale)
        attn = F.softmax(attn, dim=2)
        return th.einsum("bst,bdt->bds", attn, v)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.layers = nn.Sequential(
            Rearrange("b c h w -> b c (h w)"),
            normalization(channels),
            nn.Conv1d(channels, channels * 3, kernel_size=1),
            Rearrange("b (heads c) s -> (b heads) c s", heads=num_heads),
            QKVAttention(),
            Rearrange("(b heads) c s -> b (heads c) s", heads=num_heads),
            zero_module(nn.Conv1d(channels, channels, kernel_size=1)),
        )

    def forward(self, x):
        _, _, H, W = x.shape
        return rearrange(self.layers(x), "b c (h w) -> b c h w", h=H, w=W)


class ResNetBlock(TimestepBlock):
    def __init__(self, in_channels: int, out_channels: int, time_emb_channels: int):
        super().__init__()
        self.out_channels = out_channels

        self.in_layers = nn.Sequential(
            normalization(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=time_emb_channels, out_features=2 * out_channels),
            Rearrange("b (split c) -> split b c 1 1", split=2),
        )

        self.out_norm = normalization(out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            zero_module(
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                )
            ),
        )

        if in_channels == out_channels:
            self.skip_projection = nn.Identity()
        else:
            self.skip_projection = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1
            )

    def forward(self, x, emb):
        N, _, H, W = x.shape
        skip = self.skip_projection(x)
        x = self.in_layers(x)
        assert x.shape == (N, self.out_channels, H, W)

        cond_w, cond_b = self.emb_layers(emb)
        assert cond_w.shape == cond_b.shape
        assert cond_w.shape == (N, self.out_channels, 1, 1)

        x = self.out_norm(x) * (1 + cond_w) + cond_b
        x = self.out_layers(x)
        return skip + x


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        assert self.channels == x.shape[1]
        y = F.interpolate(x, scale_factor=2, mode="nearest")
        y = self.conv(y)
        return y


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        assert self.channels == x.shape[1]
        return self.conv(x)


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
        self.num_layers = len(channel_mult)

        time_emb_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        self.skips = []
        skip_connection_channels = []

        for level, mult in enumerate(channel_mult):
            is_top_level = level == 0
            in_channels = model_channels if is_top_level else model_channels * mult
            out_channels = model_channels * channel_mult[level]
            skip_connection_channels.append(out_channels)
            use_attn = layer_attn[level]
            is_last_layer_before_middle = level == len(channel_mult) - 1

            down = []
            for _ in range(num_res_blocks):
                down.append(ResNetBlock(in_channels, out_channels, time_emb_dim))
                if use_attn:
                    down.append(AttentionBlock(out_channels, num_heads))
                down = [AddSkipConnection(TimestepEmbedSequential(down), self.skips)]
            add_downsample = not is_last_layer_before_middle
            if add_downsample:
                down.append(AddSkipConnection(Downsample(out_channels), self.skips))
            self.downs.append(TimestepEmbedSequential(*down))

        middle_channels = model_channels * channel_mult[-1]
        self.middle = TimestepEmbedSequential(
            ResNetBlock(middle_channels, middle_channels, time_emb_dim),
            AttentionBlock(middle_channels, num_heads),
            ResNetBlock(middle_channels, middle_channels, time_emb_dim),
        )

        for level, mult in enumerate(channel_mult[::-1]):
            is_top_level = level == 0
            in_channels = channel_mult[-1]
            out_channels = model_channels * mult
            skip_channels = skip_connection_channels.pop()
            use_attn = layer_attn[len(channel_mult) - level]

            up = []
            for _ in range(num_res_blocks + 1):
                up.append(
                    TakeFromSkipConnection(
                        ResNetBlock(
                            in_channels + skip_channels, out_channels, time_emb_dim
                        ),
                        skips=self.skips,
                        expected_channels=skip_channels,
                    )
                )
                if use_attn:
                    up.append(AttentionBlock(out_channels, num_heads), self.skips)
            if not is_top_level:
                up.append(Upsample(out_channels))
            self.ups.append(TimestepEmbedSequential(*up))

        self.out_layers = nn.Sequential(
            normalization(model_channels),
            nn.SiLU(),
            zero_module(nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)),
        )

    def assert_no_skips_left(self):
            assert len(self.skips) == 0, f"skips should be empty, but has {len(self.skips)} unused skip connections"

    def forward(self, x, timesteps):
        # TODO: The skip connections are hidden
        # in this setup, on one hand this makes the
        # `forward` method more opaque. On the other hand,
        # the __init__ method now describes the architecture
        # very declaratively, which is a net win (I think).
        self.assert_no_skips_left()

        emb = self.time_embed(timestep_embedding(timesteps, dim=self.model_channels))

        x = self.in_conv(x)
        for down in self.downs:
            x = down(x, emb)

        expected_skips = self.num_res_blocks * self.num_layers
        assert len(self.skips) == expected_skips, f"skips should have {expected_skips} skip connections, but has {len(self.skips)}"

        x = self.middle(x, emb)

        for up in self.ups:
            x = up(x, emb)

        self.assert_no_skips_left()
        return self.out(x)