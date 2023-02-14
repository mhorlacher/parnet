# %%
import torch
import torch.nn as nn

# %%
class Conv1DFirstLayer(nn.Module):
    def __init__(self, in_chan, filters=128, kernel_size=12):
        super(Conv1DFirstLayer, self).__init__()

        self.conv1d = nn.Conv1d(in_chan, filters, kernel_size=kernel_size, padding='same')
        self.act = nn.ReLU()
    
    def forward(self, inputs, **kwargs):
        x = self.conv1d(inputs)
        x = self.act(x)
        return x

# %%
class Conv1DResBlock(nn.Module):
    def __init__(self, in_chan, filters=128, kernel_size=3, dropout=0.25, dilation=1, residual=True):
        super(Conv1DResBlock, self).__init__()

        self.conv1d = nn.Conv1d(in_chan, filters, kernel_size=kernel_size, dilation=dilation, padding='same')
        self.batch_norm = nn.BatchNorm1d(filters)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
    
    def forward(self, inputs, **kwargs):
        x = self.conv1d(inputs)
        x = self.batch_norm(x)
        x = self.act(x)
        x = self.dropout(x)
        if self.residual:
            x = inputs + x
        return x