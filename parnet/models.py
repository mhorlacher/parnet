import os
import logging

import tqdm
import gin
import torch
import torch.nn as nn
import captum

from fla.models.samba import SambaConfig, SambaModel


from parnet.utils import sequence_to_onehot
from parnet.layers import StemConv1D, ResConvBlock1D, AdditiveMix


@gin.configurable()
class RBPNetBackbone(nn.Module):
    def __init__(self, dilation: float = 1.75, layers: int = 9, embedding_dim: int = 128):
        super().__init__()
        self.layers = nn.Sequential(*[ResConvBlock1D(dilation=int(dilation**i)) for i in range(layers)])

        # final linear projection layer to produce sequence embeddings
        self.projection = nn.LazyConv1d(embedding_dim, kernel_size=1, bias=False, padding='same')

    def forward(self, x: torch.Tensor):
        return self.projection(self.layers(x))


@gin.configurable()
class SambaBackbone(SambaModel):
    def __init__(
        self, num_layers: int = 16, hidden_size: int = 128, attn_num_heads: int = 8, attn_window_size: int = 256
    ):
        # create config based on args (so it's assignable via gin)
        assert hidden_size % attn_num_heads == 0, 'hidden_size must be divisible by attn_num_heads'
        config = SambaConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
        )
        config.attn['window_size'] = attn_window_size
        config.attn['num_heads'] = attn_num_heads
        config.attn['num_kv_heads'] = attn_num_heads
        config.max_position_embeddings = attn_window_size

        # init the model with the config
        super().__init__(config)

        # remove the embeddings (unused)
        del self.embeddings

        # set to half-precision (required for flash-attention)
        self.half()

    def forward(self, x):
        # need to switch dimension: (batch_size, num_channels, sequence_length) -> (batch_size, sequence_length, num_channels)
        x_org_device = x.device
        return (
            super()
            .forward(inputs_embeds=x.transpose(-2, -1).half().to(self.device), use_cache=False)
            .last_hidden_state.float()
            .transpose(-2, -1)
            .to(x_org_device)
        )


@gin.configurable()
class RBPNet(nn.Module):
    """Implements the RBPNet model as described in Horlacher et al. (2023), DOI: https://doi.org/10.1186/s13059-023-03015-7."""

    def __init__(
        self,
        num_tasks: int = None,
        stem_layer: nn.Module = StemConv1D,
        backbone: nn.Module = RBPNetBackbone(),
        head_layer: nn.Module = AdditiveMix,
    ):
        """Initializes RBPNet.

        Args:
            num_tasks (int): Number of tasks (i.e. eCLIP tracks).
            layers (int, optional): Number of body layer, e.g. residual blocks. Defaults to 9.
            dilation (float, optional): Dilation coeff. for convolutions in the body layers. The i'th body layer will have a coeff. of floor(dilation**i). Defaults to 1.75.
            head_layer (nn.Module, optional): Layer to use for the output head. Defaults to LinearProjection.
        """
        super().__init__()

        if num_tasks is None:
            # We could infer this from the dataset, but let's keep it explicit for now.
            raise ValueError('num_tasks must be specified in the gin config file.')

        self.stem = stem_layer()
        self.backbone = backbone

        if head_layer is None:
            raise ValueError('head_layer must be specified.')
        self.head = head_layer(num_tasks)

        # Dummy forward pass to initialize weights. Not strictly required, but allows us
        # to print a proper summary of the model with pytorch_lightning and get the correct
        # number of parameters.
        if isinstance(self.backbone, SambaBackbone):
            self.backbone = self.backbone.to('cuda')
        _ = self(torch.zeros(2, 4, 100, dtype=torch.float32))

    def forward(self, sequence: torch.Tensor, to_probs=False) -> dict[str, torch.Tensor]:
        """Performs a forward pass through the model, returning logits for each postion and task.

        Args:
            sequence (torch.Tensor): Batched one-hot encoded sequences of shape (batch_size, num_channels, sequence_length).
            to_probs (bool, optional): Whether to convert logits to probabilities. Defaults to False.

        Returns:
            dict[str, torch.Tensor]: A dictionary logits for total, target, control tracks and the mixing coefficient.
        """
        x = self.stem(sequence)
        x = self.backbone(x)
        x = self.head(x)

        if isinstance(x, torch.Tensor):
            x = {'total': x}

        if to_probs:
            # convert track logits to probabilities
            x['total'] = x['total'].softmax(dim=-1)
            if 'target' in x:
                x['target'] = x['target'].softmax(dim=-1)
            if 'control' in x:
                x['control'] = x['control'].softmax(dim=-1)
        return x

    def predict_from_sequence(self, sequences: str | list[str], to_probs=False):
        """Predicts RBP binding probabilities from a sequence.

        Args:
            sequence (str): Sequence(s) to predict from.
            to_probs (bool, optional): Whether to convert logits to probabilities. Defaults to False.

        Returns:
            torch.Tensor: Predicted binding probabilities.
        """

        if isinstance(sequences, str):
            sequences = [sequences]

        # one-hot encode sequence(s)
        sequence_onehot = torch.stack([sequence_to_onehot(s) for s in sequences]).float()

        return self.forward(sequence_onehot, to_probs=to_probs)

    def embed(self, sequence: torch.Tensor, aggregation: str = None):
        """Returns sequence embeddings.

        Args:
            sequence (torch.Tensor): Batched one-hot encoded sequences.
            aggregation (str, optional): One of 'mean' or 'center'. Defaults to None.
        """
        x = self.stem(sequence)
        x = self.body(x)
        x = self.projection(x)

        if aggregation is None:
            return x

        if aggregation == 'mean':
            return x.mean(dim=-1)
        elif aggregation == 'center':
            return x[:, :, x.shape[-1] // 2]
        else:
            raise ValueError('aggregation must be one of "mean", "center" or None.')

    def embed_from_sequence(self, sequences: str | list[str], aggregation: str = None):
        if isinstance(sequences, str):
            sequences = [sequences]

        # one-hot encode sequence(s)
        sequence_onehot = torch.stack([sequence_to_onehot(s) for s in sequences]).float()

        return self.embed(sequence_onehot, aggregation=aggregation)

    def explain(self, sequence: torch.Tensor, task_idx: int, track='target'):
        """Returns attribution maps for the given task and track.

        Args:
            sequence (torch.Tensor): Batched one-hot encoded sequences of shape (batch_size, num_channels, sequence_length).
            task_idx (int): Index of the task to explain.
            track (str, optional): Track to explain. Defaults to 'target'.
        """
        # NOTE: In the future we should support explaination for all tasks and tracks at once.

        def _explain_forward(inputs):
            pred = self.forward(inputs)[track][:, task_idx, :].softmax(dim=-1)
            return (pred * pred.detach()).sum(dim=-1)

        return captum.attr.InputXGradient(_explain_forward).attribute(sequence)

    def explain_from_sequence(self, sequences, task_idx: int, track='target'):
        if isinstance(sequences, str):
            sequences = [sequences]

        # one-hot encode sequence(s)
        sequence_onehot = torch.stack([sequence_to_onehot(s) for s in sequences]).float()

        return self.explain(sequence_onehot, task_idx, track=track)
