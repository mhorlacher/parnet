# %%
import torch
import torch.nn as nn

# %%
from .layers import Conv1DResBlock, IndexEmbeddingOutputHead

# %%
class Network(nn.Module):
    def __init__(self, tasks, nlayers=9):
        super(Network, self).__init__()

        self.tasks = tasks

        self.body = [Conv1DResBlock() for _ in range(nlayers)]
        self.head = IndexEmbeddingOutputHead(len(self.tasks), dims=128)
    
    def forward(self, x, **kwargs):
        x = x['input']

        for layer in self.body:
            x = layer(x)

        return self.head(x)