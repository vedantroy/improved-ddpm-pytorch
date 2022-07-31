# unfinished
local model_channels = 128;
local time_emb_channels = 512;
local attention_heads = 4;
local res_blocks = 3;

local has_attn = [false, false, true, true];
local channel_mult = [1, 2, 3, 4];
local max_channels = model_channels * channel_mult[3];
local skip_chans = [];

#local add_skip(c) = 
#      skip_chans = skip_chans + [c],
#      'skip'

local Layer(c_in, c_out, res_blocks, attn, down) = std.flattenArrays([
    ([
       {
           ResNet: {
              in_channels: if idx == 0 then c_in else c_out,
              out_channels: c_out,
              time_emb_channels: time_emb_channels,
              [if !attn then 'skip']: if down then 'push' else 'pop',
           },
        },
      ] + if attn then [
         {
           Attention: {
             channels: c_out,
             heads: attention_heads,
             [if attn then 'skip']: if down then 'push' else 'pop',
           },
         },
      ] else []) for idx in std.range(0, res_blocks)] + 
      [[{ [if down then 'Downsample' else 'Upsample']: { channels: c_out } }]]);

[
    # Down
    Layer(
          model_channels * if idx > 0 then channel_mult[idx - 1] else 1, 
          model_channels * channel_mult[idx], 
          res_blocks, 
          has_attn[idx], 
          true) for idx in std.range(0, 3) 
] 
+ [
   # Middle
   {
      ResNet: {
         in_channels: max_channels,
         out_channels: max_channels,
         time_emb_channels: time_emb_channels,
      },
   },
   {
      Attention: {
         channels: max_channels,
         heads: attention_heads,
      },
   },
   {
      ResNet: {
         in_channels: max_channels,
         out_channels: max_channels,
         time_emb_channels: time_emb_channels,
      },
   },
]
+ [
    # Up
    Layer(
          model_channels * if idx > 0 then channel_mult[std.length(channel_mult) - 1 - idx] else channel_mult[std.length(channel_mult) - 1], 
          model_channels *  if idx == 3 then 1 else channel_mult[std.length(channel_mult) - 2 - idx], 
          res_blocks, 
          has_attn[std.length(has_attn) - 1 - idx], 
          false) for idx in std.range(0, 3) 
]
