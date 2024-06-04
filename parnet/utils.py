import os

import tensorflow as tf
import torch
import torch.nn.functional as F

def _disable_tensorflow_logs():
    # Disable tensorflow INFO and WARNING log messages. This needs to be 
    # done *before* tensorflow is imported. 
    # TODO: Currently, we are also ignoring erros (levle=3) rather than 
    # just warning (level=2). This could be problematic in the future. 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    # %%
    # Disable absl INFO and WARNING log messages
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)

# %%
def _set_tf_dynamic_memory_growth():
    import tensorflow as tf

    # Make sure that tensorflow doesn't gobble up all GPU memory
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

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
    sequence_onehot = F.one_hot(torch.tensor([alphabet.get(b, len(alphabet)) for b in sequence]), num_classes=len(alphabet)+1)[:, 0:len(alphabet)].T

    return sequence_onehot

def to_sparse_tensor_dict(x: torch.Tensor):
    x = x.to_sparse()
    return {'indices': x.indices(), 'values': x.values(), 'size': x.size()}

def sparse_to_dense(indices, values, size):
    return torch.sparse_coo_tensor(indices, values, size).to_dense().to(torch.float32)

def sample_to_torch_sparse_tensor_dict(example):
    return {
        'meta': {'name': example['meta']['name'].numpy()},
        'inputs': tf.nest.map_structure(lambda x: to_sparse_tensor_dict(torch.tensor(x.numpy())), example['inputs']),
        'outputs': tf.nest.map_structure(lambda x: to_sparse_tensor_dict(torch.tensor(x.numpy())), example['outputs']),
    }