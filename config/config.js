const fs = require("fs")

const model_channels = 128;
const time_emb_channels = 512;
const attention_heads = 4;
const res_blocks = 3;

const has_attn = [false, false, true, true];
const channel_mult = [1, 2, 3, 4];
const max_channels = model_channels * channel_mult[channel_mult.length - 1];
const skip_channels = [];

function Layer(c_in, c_out, res_blocks, attn, down, sample) {
  const up = !down;
  const blocks = [];
  for (let i = 0; i < res_blocks + (1 ? up : 0); i++) {
    const n_skip_channels = (up && (i < res_blocks)) ? skip_channels.pop() : 0;
    blocks.push({
      ResNet: {
        in_channels: (i == 0 ? c_in : c_out) + n_skip_channels,
        out_channels: c_out,
        time_emb_channels,
      },
    });
    if (up) {
      blocks[blocks.length - 1].skip = "pop"
    }
    if (attn) {
      key = "Attention";
      blocks.push({
        Attention: {
          channels: c_out,
        },
        has_residual: true
      });
    }
    if (down) {
      blocks[blocks.length - 1].skip = "push"
      skip_channels.push(c_out);
    }
  }
  if (sample) {
    if (down) {
      blocks.push({
        Downsample: {
          channels: c_out,
        },
      });
      skip_channels.push(c_out);
    } else {
      blocks.push({
        Upsample: {
          channels: c_out,
        },
      });
    }
  }
  return blocks;
}

const layers = [];
for (let i = 0; i < channel_mult.length; ++i) {
  layers.push(
    Layer(
      i == 0 ? model_channels : model_channels * channel_mult[i - 1],
      model_channels * channel_mult[i],
      res_blocks,
      has_attn[i],
      true,
      i < channel_mult.length - 1
    )
  );
}

layers.push([
  {
    ResNet: {
      in_channels: max_channels,
      out_channels: max_channels,
      time_emb_channels,
    },
  },
  {
    Attention: {
      channels: max_channels,
    },
  },
  {
    ResNet: {
      in_channels: max_channels,
      out_channels: max_channels,
      time_emb_channels,
    },
  },
]);

for (let i = 0; i < channel_mult.length; ++i) {
  let r_chan = channel_mult.reverse();
  let r_attn = has_attn.reverse();
  layers.push(
    Layer(
      r_chan[i] * model_channels,
      i == channel_mult.length - 1 ? 1 : r_chan[i + 1] * model_channels,
      res_blocks,
      r_attn[i],
      false
    )
  );
}

const blocks = layers.flat()
fs.writeFileSync("./config/config.json", JSON.stringify(blocks, null, 2));