# %%
import sys

import gin
import torch
import torch.nn as nn

# %%
from .layers import StemConv1D, Conv1DTower, LinearProjectionConv1D

# %%
@gin.configurable()
class PanRBPNet(nn.Module):
    def __init__(self, n_tasks, dim=128):
        super(PanRBPNet, self).__init__()

        self.n_tasks = n_tasks
        self.stem = StemConv1D()
        self.body = Conv1DTower(self.stem.out_channels)
        self.output = LinearProjectionConv1D(self.body.out_channels, n_tasks)

    def forward(self, inputs, **kwargs):
        x = inputs['sequence']
        x = self.body(self.stem(x))
        # x.shape: (batch_size, dim, N)

        try: 
            x = self.output(x)
            # x.shape: (batch_size, tasks, N)
        except:
            print(x.shape, x.dtype)
            print(self.output.weight.shape, self.output.weight.dtype)
            raise

        # transpose: # (batch_size, tasks, N) --> (batch_size, N, tasks)
        return torch.transpose(x, dim0=-2, dim1=-1)
        



@gin.configurable()
class ProteinEmbeddingMultiRBPNet(nn.Module):
    def __init__(self, n_layers=9, n_body_filters=256):
        super(ProteinEmbeddingMultiRBPNet, self).__init__()

        # layers RNA
        self.stem = StemConv1D()
        self.body = ConvTower(self.stem.out_channels)

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
        x_p = torch.transpose(x_p, dim0=1, dim1=0) # (dim, #proteins)
        
        try:
            x = torch.matmul(x_r, x_p) # (batch_size, N, #proteins)
        except:
            print('x_r.shape', x_r.shape, file=sys.stderr)
            print('x_p.shape', x_p.shape, file=sys.stderr)
            raise

        return  x