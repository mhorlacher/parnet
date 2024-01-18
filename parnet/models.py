# %%
import sys
import logging

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
from parnet.utils import sequence_to_onehot
from parnet.layers import StemConv1D, LinearProjection, ResConvBlock1D

# %%
@gin.configurable()
class RBPNet(nn.Module):
    """Implements the RBPNet model as described in Horlacher et al. (2023), DOI: https://doi.org/10.1186/s13059-023-03015-7.
    """

    def __init__(self, num_tasks=None, layers=9, dilation=1.75, head_layer=LinearProjection):
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

        self.stem = StemConv1D()
        self.body = nn.Sequential(
            *[ResConvBlock1D(dilation=int(dilation**i)) for i in range(layers)]
        )
        self.head = head_layer(num_tasks)

        # Dummy forward pass to initialize weights. Not strictly required, but allows us 
        # to print a proper summary of the model with pytorch_lightning and get the correct
        # number of parameters. 
        _ = self({'sequence': torch.zeros(2, 4, 100, dtype=torch.float32)})

    def forward(self, inputs, to_probs=False, **kwargs):
        logging.debug(f"Predict on sequence inputs with shape {inputs['sequence'].shape} and dtype {inputs['sequence'].dtype}.")

        x = self.stem(inputs['sequence'])
        x = self.body(x)
        x = self.head(x)

        if isinstance(x, torch.Tensor):
            x = {'total': x}

        # # Convert logits to probabilities if requested. 
        # if to_probs:
        #     if isinstance(x, torch.Tensor):
        #         x = torch.softmax(x, dim=-1)
        #     else:
        #         raise NotImplementedError()

        return x
    
    def predict_from_sequence(self, sequence, alphabet='ACGT', **kwargs):
        """Predicts RBP binding probabilities from a sequence.

        Args:
            sequence (str): Sequence to predict from.
            alphabet (dict, optional): Alphabet to use for encoding the sequence. Defaults to 'ACGT'. 

        Returns:
            torch.Tensor: Predicted binding probabilities.
        """
    
        # One-hot encode sequence, add batch dimension and cast to float. 
        sequence_onehot = sequence_to_onehot(sequence, alphabet=alphabet)
        sequence_onehot = torch.unsqueeze(sequence_onehot, dim=0).float()

        # Predict and remove batch dimension of size 1. 
        return self({'sequence': sequence_onehot}, **kwargs)[0]
