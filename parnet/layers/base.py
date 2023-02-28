# %%
import gin
import torch
import torch.nn as nn

# %%
# @gin.configurable()
# class FirstLayerConv1D(nn.Module):
#     def __init__(self, in_chan, filters=128, kernel_size=12, activation=nn.ReLU()):
#         super(FirstLayerConv1D, self).__init__()

#         self.conv1d = nn.Conv1d(in_chan, filters, kernel_size=kernel_size, padding='same')
#         self.act = activation
    
#     def forward(self, inputs, **kwargs):
#         x = self.conv1d(inputs)
#         x = self.act(x)
#         return x

# %%
@gin.configurable(denylist=['in_chan'])
class UpDownSamplingConv1D(nn.Module):
    def __init__(self, in_chan, filters=128, activation=nn.ReLU(), dropout=None):
        super(UpDownSamplingConv1D, self).__init__()

        self.conv1d = nn.Conv1d(in_chan, filters, kernel_size=1, padding='same')
        self.batch_norm = nn.BatchNorm1d(filters)
        self.act = activation
        self.dropout = nn.Dropout1d(dropout) if dropout is not None else None

        self.out_channels = filters
    
    def forward(self, x, **kwargs):
        x = self.conv1d(x)
        x = self.batch_norm(x)
        x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

# %%
@gin.configurable(denylist=['in_features'])
class LinearProjection(nn.Module):
    def __init__(self, in_features, out_features=128, activation=None) -> None:
        super(LinearProjection, self).__init__()

        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.act = activation
    
    def forward(self, x):
        x = self.linear(x)
        if self.act is not None:
            x = self.act(x)
        return x

# %%
@gin.configurable(denylist=['in_features'])
class LinearProjectionConv1D(nn.Module):
    def __init__(self, in_features, out_features=128) -> None:
        super(LinearProjectionConv1D, self).__init__()

        self.pointwise = nn.Conv1d(in_features, out_features, kernel_size=1, bias=False)

        self.out_channels = out_features
    
    def forward(self, x):
        x = self.pointwise(x)
        return x