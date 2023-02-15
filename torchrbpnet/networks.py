# %%
import torch
import torch.nn as nn

# %%
from .layers import Conv1DFirstLayer, Conv1DResBlock, IndexEmbeddingOutputHead

# %%
class MultiRBPNet(nn.Module):
    def __init__(self, tasks, n_layers=9, n_body_filters=256):
        super(MultiRBPNet, self).__init__()

        self.tasks = tasks

        self.body = nn.Sequential(*[Conv1DFirstLayer(4, n_body_filters, 6)]+[(Conv1DResBlock(n_body_filters, n_body_filters, dilation=(2**i))) for i in range(n_layers)])
        self.head = IndexEmbeddingOutputHead(len(self.tasks), dims=n_body_filters)
    
    def forward(self, inputs, **kwargs):
        x = inputs

        for layer in self.body:
            x = layer(x)

        return self.head(x)