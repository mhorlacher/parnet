# %%
import gin
import torch
import torch.nn as nn

from .base import ResConv1DBlock

# %%
@gin.configurable()
class ResConv1DTower(nn.Module):
    def __init__(self, n_blocks=9, dilation_factor=1):
        super(ResConv1DTower, self).__init__()

        tower = nn.ModuleList([
            
        ])
    
    def forward(self, inputs, **kwargs):
        x = self.tower(inputs)
        return x