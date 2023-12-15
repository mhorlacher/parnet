# %%
import sys

import gin
import torch
import torch.nn as nn

# %%
@gin.configurable()
class StemConv1D(nn.Module):
    """Class to be used as first layer of a model. 

    Applies an ordenary 1D convolution, followed by  batch normalization, activation and dropout. Seperating this 
    layer from the rest of the model allows us to inject hyperparameters for the first layer only via gin-config.
    """

    def __init__(self, filters=128, kernel_size=12, activation=nn.ReLU(), dropout=None):
        """Initializes StemConv1D.

        Args:
            filters (int, optional): Number of convolutional kernels/filters. Defaults to 128.
            kernel_size (int, optional): Size of kernels/filters. Defaults to 12.
            activation (Any, optional): Activation function. Defaults to nn.ReLU().
            dropout (float, optional): Dropout probability. Defaults to None, in which case no dropout will be applied.
        """
        super().__init__()

        self.conv1d = nn.LazyConv1d(filters, kernel_size, padding='same')
        self.batch_norm = nn.BatchNorm1d(filters)
        self.act = activation
        self.dropout = nn.Dropout1d(dropout) if dropout is not None else None
    
    def forward(self, x, **kwargs):
        x = self.conv1d(x)
        x = self.batch_norm(x)
        x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

# %%
@gin.configurable()
class ResConvBlock1D(nn.Module):
    # TODO: Add documentation. 
    def __init__(self, filters=128, kernel_size=3, dropout=0.25, activation=nn.ReLU(), dilation=1, residual=True):
        super().__init__()

        self.conv1d = nn.LazyConv1d(filters, kernel_size=kernel_size, dilation=int(dilation), padding='same')
        self.batch_norm = nn.BatchNorm1d(filters)
        self.act = activation
        self.dropout = nn.Dropout1d(dropout) if dropout is not None else None
        self.residual = residual
    
    def forward(self, inputs, **kwargs):
        x = inputs

        try:
            x = self.conv1d(x)
        except:
            print(x.shape, x.dtype, file=sys.stderr)
            raise

        x = self.batch_norm(x)
        x = self.act(x)
        # dropout
        if self.dropout is not None:
            x = self.dropout(x)

        # residual
        if self.residual:
            x = inputs + x

        return x

# %%
@gin.configurable()
class LinearProjection(nn.Module):
    """Performs a linear projection of the input to the specified number of output channels.

    To be used as an upsampling/downsampling layer.
    """

    def __init__(self, out_features=128, activation=None, bias=False) -> None:
        super().__init__()

        self.pointwise_conv = nn.LazyConv1d(out_features, kernel_size=1, bias=bias)
        self.act = activation
    
    def forward(self, x):
        x = self.pointwise_conv(x)
        if self.act is not None:
            x = self.act(x)
        return x
