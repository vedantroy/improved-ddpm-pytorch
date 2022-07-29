from unet import UNet

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

