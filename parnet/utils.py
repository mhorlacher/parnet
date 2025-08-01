import os

import gin
import torch
import torch.nn.functional as F


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# %%
def sequence_to_onehot(sequence, alphabet='ACGT'):
    """Converts a sequence to one-hot encoding.

    Args:
        sequence (str): Sequence to convert.
        alphabet (str, optional): Alphabet of the sequence. Defaults to 'ACGT'.

    Returns:
        torch.tensor: One-hot encoding of the sequence.
    """

    # Convert sequence to one-hot encoding. We first add an additional dimension for bases not contained in
    # the alphabet. Then, we remove the additional dimension so that the encoding of the unknown bases is a 0-vector.
    alphabet = dict(zip(alphabet, range(len(alphabet))))
    sequence_onehot = F.one_hot(
        torch.tensor([alphabet.get(b, len(alphabet)) for b in sequence]), num_classes=len(alphabet) + 1
    )[:, 0 : len(alphabet)].T

    return sequence_onehot


def to_sparse_tensor_dict(x: torch.Tensor):
    x = x.to_sparse()
    return {'indices': x.indices(), 'values': x.values(), 'size': x.size()}


def sparse_to_dense(indices, values, size):
    return torch.sparse_coo_tensor(indices, values, size).to_dense().to(torch.float32)


# def sample_to_torch_sparse_tensor_dict(example):
#     return {
#         'meta': {'name': example['meta']['name'].numpy()},
#         'inputs': tf.nest.map_structure(lambda x: to_sparse_tensor_dict(torch.tensor(x.numpy())), example['inputs']),
#         'outputs': tf.nest.map_structure(lambda x: to_sparse_tensor_dict(torch.tensor(x.numpy())), example['outputs']),
#     }


def export_checkpoint_to_pt(checkpoint_path: str, config_gin: str, model_pt_path: str):
    gin.parse_config_file(config_gin, skip_unknown=True)
    
