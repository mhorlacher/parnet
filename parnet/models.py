# %%
import sys
import logging

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

# %%
from parnet.utils import sequence_to_onehot
from parnet.layers import StemConv1D, LinearProjection, ResConvBlock1D, LikeBasenji2DilatedResConvBlock, LikeBasenji2ConvBlock

# %%
@gin.configurable()
class RBPNet(nn.Module):
    """Implements the RBPNet model as described in Horlacher et al. (2023), DOI: https://doi.org/10.1186/s13059-023-03015-7."""

    def __init__(
        self,
        num_tasks=None,
        layers=9,
        dilation=1.75,
        body_layer=ResConvBlock1D,
        head_layer=LinearProjection,
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
            raise ValueError("num_tasks must be specified in the gin config file.")

        self.stem = StemConv1D()
        self.body = nn.Sequential(
            *[body_layer(dilation=int(dilation**i)) for i in range(layers)]
        )
        self.head = head_layer(num_tasks)

        # Dummy forward pass to initialize weights. Not strictly required, but allows us
        # to print a proper summary of the model with pytorch_lightning and get the correct
        # number of parameters.
        _ = self({"sequence": torch.zeros(2, 4, 100, dtype=torch.float32)})

    def forward(self, inputs, to_probs=False, **kwargs):
        logging.debug(f"Received inputs of type: {type(inputs)}.")
        logging.debug(
            f"Predict on sequence inputs with shape {inputs['sequence'].shape} and dtype {inputs['sequence'].dtype}."
        )

        x = self.stem(inputs["sequence"])
        x = self.body(x)
        x = self.head(x)

        if isinstance(x, torch.Tensor):
            x = {"total": x}

        # # Convert logits to probabilities if requested.
        # if to_probs:
        #     if isinstance(x, torch.Tensor):
        #         x = torch.softmax(x, dim=-1)
        #     else:
        #         raise NotImplementedError()

        return x

    def predict_from_sequence(self, sequence, alphabet="ACGT", **kwargs):
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
        return self.forward({"sequence": sequence_onehot}, **kwargs)

@gin.configurable()
class RBPNetESM(nn.Module):
    """Implements the RBPNet model as described in Horlacher et al. (2023), DOI: https://doi.org/10.1186/s13059-023-03015-7."""

    def __init__(
        self,
        num_tasks=None,
        head_layer=LinearProjection,
        esm_model_path="facebook/esm2_t12_35M_UR50D",
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
            raise ValueError("num_tasks must be specified in the gin config file.")

        esm_config = transformers.EsmConfig.from_pretrained(esm_model_path)
        esm_config.vocab_size = 6
        esm_config.pad_token_id = 4
        esm_config.mask_token_id = 5

        self.esm = transformers.EsmModel(esm_config)

        self.head = head_layer(num_tasks)

        # Dummy forward pass to initialize weights. Not strictly required, but allows us
        # to print a proper summary of the model with pytorch_lightning and get the correct
        # number of parameters.
        _ = self({"input_ids": torch.randint(0, 6, (1, 100)).long(), "attention_mask": torch.ones(1, 100).float()})

    def forward(self, inputs):
        x = self.esm(
            input_ids=inputs["input_ids"], 
            attention_mask=inputs["attention_mask"]).last_hidden_state # (batch_size, seq_len, hidden_size)
        x = x.transpose(-1, -2) # (batch_size, hidden_size, seq_len)
        x = self.head(x)

        if isinstance(x, torch.Tensor):
            x = {"total": x}

        return x

# %%
@gin.configurable()
class LikeBasenji2(nn.Module):

    def __init__(
        self,
        num_tasks=None,
        C=768,
        L=11,
        dilation=1.75,
        head_layer=LinearProjection,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            *[
                nn.LazyConv1d(int(C*0.5), kernel_size=11, padding="same"),
                nn.BatchNorm1d(int(C*0.5)),
                nn.GELU(),
            ]
        )

        self.conv_tower = nn.Sequential(
            *[LikeBasenji2ConvBlock(filters=C, kernel_size=5) for _ in range(4)]
        )

        self.dilated_tower = nn.Sequential(
            *[LikeBasenji2DilatedResConvBlock(filters=C, kernel_size=3, dilation=int(dilation**i)) for i in range(L)]
        )

        self.projection = nn.LazyConv1d(int(C*1.25), kernel_size=1, padding="same", bias=False)
        # self.projection = nn.LazyLinear(C*1.25, bias=False)
        self.head = head_layer(num_tasks)

        # Dummy forward pass to initialize weights. Not strictly required, but allows us
        # to print a proper summary of the model with pytorch_lightning and get the correct
        # number of parameters.
        _ = self({"sequence": torch.zeros(2, 4, 100, dtype=torch.float32)})

    def forward(self, inputs, to_probs=False, **kwargs):
        logging.debug(f"Received inputs of type: {type(inputs)}.")
        logging.debug(
            f"Predict on sequence inputs with shape {inputs['sequence'].shape} and dtype {inputs['sequence'].dtype}."
        )

        x = self.stem(inputs["sequence"])
        x = self.conv_tower(x)
        x = self.dilated_tower(x)
        x = self.projection(x)
        x = self.head(x)

        # if isinstance(x, torch.Tensor):
        #     x = {"total": x}

        # # Convert logits to probabilities if requested.
        # if to_probs:
        #     if isinstance(x, torch.Tensor):
        #         x = torch.softmax(x, dim=-1)
        #     else:
        #         raise NotImplementedError()

        return x

class NoHeadModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.head = torch.nn.Identity()
        
    def forward(self, x):
        return self.model(x)
    
    def forward_mean_pool(self, x):
        return self.model(x).mean(dim=-1)