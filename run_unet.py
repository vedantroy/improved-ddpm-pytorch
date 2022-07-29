"""
Simple script to
1. Initialize the UNet
2. Pass a sample input through the UNet
To verify nothing is obviously broken
"""
import torch as th

from unet.unet import UNet

unet = UNet(
    in_channels=3,
    out_channels=3,
    model_channels=128,
    channel_mult=(1, 2, 3, 4),
    layer_attn=(False, False, True, True),
    num_res_blocks=3,
    num_heads=4,
)
unet = unet.cuda()
unet.print_architecture()

batch_size = 2
# This was an actual input
timesteps = th.tensor([472.2500, 217.5000]).cuda()
assert timesteps.shape[0] == batch_size
# This is obviously not
x = th.randn((batch_size, 3, 64, 64)).cuda()

unet.eval()
out = unet(x, timesteps)
assert out.shape == x.shape
