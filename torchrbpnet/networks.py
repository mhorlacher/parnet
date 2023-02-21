# %%
import sys

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
        self.rna_projection = LinearProjection(in_channels=n_body_filters)
        self.head = IndexEmbeddingOutputHead(self.n_tasks, dims=n_body_filters)
    
    def forward(self, inputs, **kwargs):
        x = inputs['sequence']
        for layer in self.body:
            x = layer(x)
        x = self.rna_projection(x)

        return self.head(x)


@gin.configurable()
class ProteinEmbeddingMultiRBPNet(nn.Module):
    def __init__(self, n_layers=9, n_body_filters=256):
        super(ProteinEmbeddingMultiRBPNet, self).__init__()

        # layers RNA
        self.body = nn.Sequential(*[Conv1DFirstLayer(4, n_body_filters, 6)]+[(Conv1DResBlock(n_body_filters, n_body_filters, dilation=(2**i))) for i in range(n_layers)])
        self.rna_projection = nn.Linear(in_features=n_body_filters, out_features=256, bias=False)

        # layers protein
        self.protein_projection = nn.Linear(in_features=1280, out_features=256, bias=False)

    def forward(self, inputs, **kwargs):
        # forward RNA
        x_r = inputs['sequence']
        for layer in self.body:
            x_r = layer(x_r)
        # transpose: # (batch_size, dim, N) --> (batch_size, N, dim)
        x_r = torch.transpose(x_r, dim0=-2, dim1=-1)
        # project: (batch_size, N, dim) --> (batch_size, N, new_dim)
        x_r = self.rna_projection(x_r)
        
        # forward protein
        x_p = inputs['embedding']
        x_p = self.protein_projection(x_p)
        # x_r: (#proteins, dim)

        # transpose representations for matmul
        # x_r = torch.transpose(x_r, dim0=-2, dim1=-1) # (batch_size, N, dim)
        x_p = torch.transpose(x_p, dim0=1, dim1=0) # (dim, #proteins)
        
        try:
            x = torch.matmul(x_r, x_p) # (batch_size, N, #proteins)
        except:
            print('x_r.shape', x_r.shape, file=sys.stderr)
            print('x_p.shape', x_p.shape, file=sys.stderr)
            raise

        return  x