# %%
import gin
import torch
import torch.nn as nn

from .stems import StemConv1D
from .base import LinearProjectionConv1D
from .blocks import ResConv1DBlock

# %%
@gin.configurable()
class RNAConv1dBody(nn.Module):
    def __init__(self, in_chan=4, tower_layers_filters=None, block_layer=ResConv1DBlock, dilation=1.0) -> None:
        super(RNAConv1dBody, self).__init__()

        # stem
        self.stem = StemConv1D(in_chan)

        # tower (of ResBock)
        layer_idx = 0
        self.tower = []
        prev_out_channels = self.stem.out_channels
        if tower_layers_filters is not None:
            for n, filters in tower_layers_filters:
                blocks = []
                for _ in range(n):
                    block = block_layer(prev_out_channels, filters=filters, dilation=dilation**layer_idx)
                    layer_idx += 1 # after, because first round is dilation^0
                    blocks.append(block)
                    prev_out_channels = block.out_channels
                self.tower.append(nn.Sequential(*blocks))
        self.tower = nn.Sequential(*self.tower) # to nn.Module

        # linear projection
        self.linear_projection = LinearProjectionConv1D(prev_out_channels)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.tower(x)
        x = self.linear_projection(x)
        return x