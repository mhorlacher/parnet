# %%
import torch
import torch.nn as nn

from .base import Pointwise

# %%
class IndexEmbeddingOutput(nn.Module):
    def __init__(self, num_tasks, dims):
        super(IndexEmbeddingOutput, self).__init__()
    
        self.pointwise = Pointwise(num_tasks, dims)
    
    @property
    def embedding(self):
        return torch.squeeze(self.pointwise.weight, dim=-1)
    
    @property
    def out_channels(self):
        return self.pointwise.out_channels

    def forward(self, x, **kwargs):
        return self.pointwise(x)