# %%
import gin
import torch
import torch.nn as nn

# %%
from .layers import Conv1DFirstLayer, Conv1DResBlock, IndexEmbeddingOutputHead, LinearProjection

# %%
@gin.configurable()
class MultiRBPNet(nn.Module):
    def __init__(self, n_tasks, n_layers=9, n_body_filters=256):
        super(MultiRBPNet, self).__init__()

        self.n_tasks = n_tasks

        self.body = nn.Sequential(*[Conv1DFirstLayer(4, n_body_filters, 6)]+[(Conv1DResBlock(n_body_filters, n_body_filters, dilation=(2**i))) for i in range(n_layers)])
        self.latent_rna_projection = LinearProjection(in_channels=n_body_filters)
        self.head = IndexEmbeddingOutputHead(self.n_tasks, dims=n_body_filters)
    
    def forward(self, inputs, **kwargs):
        x = inputs

        for layer in self.body:
            x = layer(x)

        return self.head(x)