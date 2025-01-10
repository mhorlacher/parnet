# %%
import sys
import logging

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(
        self,
        filters=128,
        kernel_size=3,
        dropout=0.25,
        activation=nn.ReLU(),
        dilation=1,
        residual=True,
    ):
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


@gin.configurable()
class LikeBasenji2ConvBlock(nn.Module):
    def __init__(self, filters, kernel_size) -> None:
        super().__init__()

        self.conv1d = nn.LazyConv1d(filters, kernel_size, padding='same')
        self.batch_norm = nn.BatchNorm1d(filters)
        self.act = nn.GELU()

    def forward(self, inputs):
        x = self.conv1d(inputs)
        x = self.batch_norm(x)
        x = self.act(x)
        return x


@gin.configurable()
class LikeBasenji2DilatedResConvBlock(nn.Module):
    # TODO: Add documentation.
    def __init__(
        self,
        filters=256,
        kernel_size=5,
        dropout=0.3,
        activation=nn.GELU(),
        dilation=1.0,
        residual=True,
    ):
        super().__init__()

        self.conv1d = nn.LazyConv1d(int(filters / 2), kernel_size, dilation=int(dilation), padding='same')
        self.conv1d_norm = nn.BatchNorm1d(int(filters / 2))

        self.pointwise = nn.LazyConv1d(filters, kernel_size=1)
        self.pointwise_norm = nn.BatchNorm1d(filters)

        self.act = activation
        self.dropout = nn.Dropout1d(dropout) if dropout > 0.0 else None
        self.residual = residual

    def forward(self, inputs):
        # conv1d
        x = self.conv1d(inputs)
        x = self.conv1d_norm(x)
        x = self.act(x)

        # pointwise
        x = self.pointwise(x)
        x = self.pointwise_norm(x)
        x = self.act(x)

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

    def __init__(self, out_features=128) -> None:
        super().__init__()

        self.pointwise_conv = nn.LazyConv1d(out_features, kernel_size=1, bias=False, padding='same')

    def forward(self, x):
        return self.pointwise_conv(x)


# %%
@gin.configurable()
class SequenceLinearMix(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()

        self.gloabel_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dense = nn.LazyLinear(num_tasks)

    def forward(self, inputs):
        # inputs should have shape [batch, hidden_dim, length]

        x = torch.squeeze(self.gloabel_avg_pool(inputs))  # --> [batch, hidden_dim]
        logging.debug(f'x=torch.squeeze(self.gloabel_avg_pool(inputs)): {x.shape}')

        x = self.dense(x)  # --> [batch, num_tasks]
        logging.debug(f'self.dense(x): {x.shape}')

        return x


@gin.configurable()
class MixCoeffMLP(nn.Module):
    def __init__(self, num_tasks, units=128, act=nn.ReLU()) -> None:
        super().__init__()

        self.gloabel_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dense1 = nn.LazyLinear(units)
        self.act = act
        self.dense2 = nn.LazyLinear(num_tasks)

    def forward(self, inputs):
        # inputs should have shape [batch, hidden_dim, length]

        x = torch.squeeze(self.gloabel_avg_pool(inputs))  # --> [batch, hidden_dim]
        logging.debug(f'x=torch.squeeze(self.gloabel_avg_pool(inputs)): {x.shape}')

        x = self.dense1(x)  # --> [batch, units]
        x = self.act(x)
        logging.debug(f'self.dense(x): {x.shape}')

        x = self.dense2(x)  # --> [batch, num_tasks]

        return x


@gin.configurable()
class NewMixCoeffMLP(nn.Module):
    def __init__(self, num_tasks, units=128, act=nn.ReLU()) -> None:
        super().__init__()

        self.dense1 = nn.LazyLinear(units)
        self.act = act
        self.dense2 = nn.LazyLinear(num_tasks)

    def forward(self, x):
        # x should have shape [batch, hidden_dim, length]

        x = x.mean(-1) # --> [batch, hidden_dim]
        logging.debug(f'x (pooled): {x.shape}')

        x = self.dense1(x)  # --> [batch, units]
        x = self.act(x)
        logging.debug(f'self.dense(x): {x.shape}')

        x = self.dense2(x)  # --> [batch, num_tasks]

        return F.sigmoid(x)


@gin.configurable()
class MixCoeffPenalty(nn.Module):
    def __init__(self, factor=1.0) -> None:
        super().__init__()
        self.factor = factor

    def __call__(self, track_target, track_control, mix_coeff):
        return F.sigmoid(mix_coeff) * self.factor


@gin.configurable()
class NewMixCoeffPenalty(nn.Module):
    def __init__(self, factor=1.0) -> None:
        super().__init__()
        self.factor = factor

    def __call__(self, track_target, track_control, mix_coeff):
        return mix_coeff * self.factor


# %%
@gin.configurable()
class AdditiveMix(nn.Module):
    def __init__(
        self,
        num_tasks,
        head_layer=LinearProjection,
        mix_coeff_layer=SequenceLinearMix,
        penalty_layer=None,
        control_nograd=False,
    ):
        super().__init__()

        self.head_target = head_layer(num_tasks)
        self.head_control = head_layer(num_tasks)
        self.mix_coeff = mix_coeff_layer(num_tasks)
        self.penalty = penalty_layer
        self.control_nograd = control_nograd

    def forward(self, inputs, **kwargs):
        # inputs should have shape [batch, hidden_dim, length]

        # project input feature map to logits for target and control --> [batch, num_tasks, length]
        track_target = self.head_target(inputs)
        track_control = self.head_control(inputs)

        logging.debug(f'track_target.shape: {track_target.shape}, track_control.shape: {track_control.shape}')

        # compute mixing coefficients --> [batch, num_tasks]
        mix_coeff = self.mix_coeff(inputs)
        mix_coeff = torch.unsqueeze(mix_coeff, dim=-1)

        logging.debug(f'mix_coeff.shape: {mix_coeff.shape}')

        # additive mixing of target and control tracks with control track weigthed
        # by the mixing coefficient --> [batch, num_tasks, length]
        track_target = track_target - torch.logsumexp(track_target, dim=-1, keepdim=True)
        track_control = track_control - torch.logsumexp(track_control, dim=-1, keepdim=True)
        track_total = torch.logsumexp(
            torch.stack(
                [
                    mix_coeff + track_target,
                    (track_control.detach() if self.control_nograd else track_control),
                ],
                dim=0,
            ),
            dim=0,
        )

        return_dict = {
            'target': track_target,
            'control': track_control,
            'total': track_total,
            'mix_coeff': mix_coeff,  # TODO: Add this.
        }

        if self.penalty is not None:
            return_dict['penalty_loss'] = self.penalty(track_target, track_control, mix_coeff)

        return return_dict


# %%
@gin.configurable()
class NewAdditiveMix(nn.Module):
    def __init__(
        self,
        num_tasks,
        head_layer=LinearProjection,
        mix_coeff_layer=NewMixCoeffMLP,
        penalty_layer=None,
        control_nograd=False,
    ):
        super().__init__()

        self.head_target = head_layer(num_tasks)
        self.head_control = head_layer(num_tasks)
        self.mix_coeff = mix_coeff_layer(num_tasks)
        self.penalty = penalty_layer
        self.control_nograd = control_nograd

    def forward(self, inputs, **kwargs):
        # inputs should have shape [batch, hidden_dim, length]

        # project input feature map to logits for target and control --> [batch, num_tasks, length]
        target_logit = self.head_target(inputs)
        control_logit = self.head_control(inputs)

        # compute mixing coefficients --> [batch, num_tasks]
        mix_coeff = self.mix_coeff(inputs)
        mix_coeff = torch.unsqueeze(mix_coeff, dim=-1)

        logging.debug(f'mix_coeff.shape: {mix_coeff.shape}')

        # additive mixing of target and control tracks with control track weigthed
        # by the mixing coefficient --> [batch, num_tasks, length]
        target_logprob = target_logit - torch.logsumexp(target_logit, dim=-1, keepdim=True)
        control_logprob = control_logit - torch.logsumexp(control_logit, dim=-1, keepdim=True)

        # use logsumexp trick to avoid numerical instability
        max_logprob = torch.maximum(target_logprob, control_logprob)
        total_logprob = max_logprob + torch.log(
            mix_coeff * torch.exp(target_logprob - max_logprob)
            + (1 - mix_coeff) * torch.exp(control_logprob - max_logprob)
        )

        return_dict = {
            'target': target_logprob,
            'control': control_logprob,
            'total': total_logprob,
            'mix_coeff': mix_coeff.squeeze(-1),  # TODO: Add this.
        }

        if self.penalty is not None:
            return_dict['penalty_loss'] = self.penalty(target_logprob, control_logprob, mix_coeff)

        return return_dict


@gin.configurable()
class ConvBlock(nn.Module):
    def __init__(self, filters=128, kernel_size=3, act=nn.Mish(), dilation=1, batch_norm_size=None) -> None:
        super().__init__()

        self.batch_norm = nn.BatchNorm1d(batch_norm_size if batch_norm_size is not None else filters)
        self.act = act
        self.conv1d = nn.LazyConv1d(filters, kernel_size, padding='same', dilation=int(dilation))

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.act(x)
        x = self.conv1d(x)
        return x


@gin.configurable()
class StemConv(nn.Module):
    def __init__(self, filters=128, filter_firstlayer_factor=1.5, kernel_size=11) -> None:
        super().__init__()

        self.conv = nn.LazyConv1d(int(filters * filter_firstlayer_factor), kernel_size, padding='same')
        self.conv_block = ConvBlock(filters, 1, batch_norm_size=int(filters * filter_firstlayer_factor))

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_block(x)
        return x


@gin.configurable()
class ResConvBlock(nn.Module):
    def __init__(self, filters=128, kernel_size=3, dilation=1, dropout_rate=0.3) -> None:
        super().__init__()

        self.conv_block_1 = ConvBlock(
            int(filters * 1.5), kernel_size, dilation=dilation, batch_norm_size=filters
        )
        self.conv_block_2 = ConvBlock(filters, 1, batch_norm_size=int(filters * 1.5))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        x = self.conv_block_1(inputs)
        x = self.conv_block_2(x)
        x = self.dropout(x)
        x = inputs + x
        return x
