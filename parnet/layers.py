import sys
import logging

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F


class LambdaLayer(torch.nn.Module):
    def __init__(self, fun):
        super(LambdaLayer, self).__init__()
        self.fun = fun

    def forward(self, *args, **kwargs):
        return self.fun(*args, **kwargs)


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
class LinearProjectionHead(nn.Module):
    """Performs a pointwise linear projection of the nucleotide-wise embeddings to logits for each task."""

    def __init__(self, num_tasks=None) -> None:
        super().__init__()
        if num_tasks is None:
            raise ValueError('num_tasks must be specified')
        self.pointwise = nn.LazyConv1d(num_tasks, kernel_size=1, bias=False, padding='same')

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # (B, hidden_dim, L) --> (B, num_tasks, L)
        return self.pointwise(inputs)


@gin.configurable()
class MixCoeffMLP(nn.Module):
    """2-layer MLP that takes a feature map as input and outputs a mixing coefficient for each task."""

    def __init__(self, num_tasks, units=128, act=nn.ReLU()) -> None:
        super().__init__()

        self.dense1 = nn.LazyLinear(units)
        self.act = act
        self.dense2 = nn.LazyLinear(num_tasks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # (B, hidden_dim, L) --> (B, hidden_dim)
        x = inputs.mean(-1)  # --> [batch, hidden_dim]
        logging.debug(f'{x.shape}')

        # (B, hidden_dim) --> (B, units)
        x = self.dense1(x)  # --> [batch, units]
        x = self.act(x)
        logging.debug(f'self.dense(x): {x.shape}')

        x = self.dense2(x)  # --> [batch, num_tasks]

        return F.sigmoid(x)


@gin.configurable()
class MixCoeffPenalty(nn.Module):
    """Simple mixing coefficient penalty that scales the mixing coefficient by a factor.

    This encourages the model to use lower the mixing coefficient, i.e. to increase the contribution of the control
    tack when combining the target and control tracks.
    """

    def __init__(self, factor=1.0) -> None:
        super().__init__()
        self.factor = factor

    def __call__(self, track_target, track_control, mix_coeff):
        # (B, num_tasks) -> (B, num_tasks)
        return mix_coeff * self.factor


@gin.configurable()
class AdditiveMix(nn.Module):
    """Additive mixing of target and control tracks.

    This layer takes a nucleotide-wise embedding, projects it to logits for the target and control tracks for a
    given number of tasks, computes a mixing coefficient for each task, and combines the target and control logits
    using the mixing coefficient.
    """

    def __init__(
        self,
        num_tasks,
        head_layer=LinearProjectionHead,
        mix_coeff_layer=MixCoeffMLP,
        penalty_layer=None,
    ):
        """Initializes AdditiveMix layer.

        Args:
            num_tasks (int): Number of tasks (i.e. eCLIP tracks).
            head_layer (nn.Module, optional): Layer to use for the output head. Defaults to LinearProjectionHead.
            mix_coeff_layer (nn.Module, optional): Layer to use for predicting mixing coefficients. Defaults to MixCoeffMLP.
            penalty_layer (nn.Module, optional): Layer to compute additional penalty added to the loss. Defaults to MixCoeffPenalty.
        """
        super().__init__()
        self.head_target = head_layer(num_tasks)
        self.head_control = head_layer(num_tasks)
        self.mix_coeff = mix_coeff_layer(num_tasks)
        self.penalty = penalty_layer() if penalty_layer is not None else None

    def forward(self, inputs, **kwargs):
        # inputs: (B, hidden_dim, L)

        # project input feature map to logits for target and control tracks
        # (B, hidden_dim, L) -> (B, num_tasks, L)
        target_logit = self.head_target(inputs)
        control_logit = self.head_control(inputs)
        logging.debug(f'{target_logit.shape=}, {control_logit.shape=}')

        # compute mixing coefficients for each task
        # (B, hidden_dim, L) -> (B, num_tasks, 1)
        mix_coeff = self.mix_coeff(inputs)
        mix_coeff = torch.unsqueeze(mix_coeff, dim=-1)
        logging.debug(f'{mix_coeff.shape=}')

        # Additive mixing of target and control tracks, weighted by the mixing coefficient. The logsumexp trick
        # is used to avoid numerical instability.
        target_logprob = target_logit - torch.logsumexp(target_logit, dim=-1, keepdim=True)
        control_logprob = control_logit - torch.logsumexp(control_logit, dim=-1, keepdim=True)

        max_logprob = torch.maximum(target_logprob, control_logprob)
        total_logprob = max_logprob + torch.log(
            mix_coeff * torch.exp(target_logprob - max_logprob)
            + (1 - mix_coeff) * torch.exp(control_logprob - max_logprob)
            + 1e-10  # small constant to avoid numerical issues
        )

        # check for NaN/Inf in total_logprob
        if torch.isnan(total_logprob).any() or torch.isinf(total_logprob).any():
            print(
                f'{target_logprob.min()}, {target_logprob.max()}, {target_logprob.mean()}, ',
                file=sys.stderr,
                flush=True,
            )
            print(
                f'{control_logprob.min()}, {control_logprob.max()}, {control_logprob.mean()}, ',
                file=sys.stderr,
                flush=True,
            )
            print(f'{total_logprob=}, {mix_coeff=}, {target_logprob=}, {control_logprob=}', file=sys.stderr, flush=True)
            raise ValueError('Logits contain NaN or Inf values.')

        return_dict = {
            'target': target_logprob,  # (B, num_tasks, L)
            'control': control_logprob,  # (B, num_tasks, L)
            'total': total_logprob,  # (B, num_tasks, L)
            'mix_coeff': mix_coeff.squeeze(-1),  # (B, num_tasks)
        }

        if self.penalty is not None:
            # add penalty loss (used only during training)
            return_dict['penalty_loss'] = self.penalty(target_logprob, control_logprob, mix_coeff)

        return return_dict
