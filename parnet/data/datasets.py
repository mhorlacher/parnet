# %%
import sys

import gin
import torch
import numpy as np
import tensorflow as tf  # TODO: Remove this dependency. See https://www.tensorflow.org/datasets/tfless_tfds#use_with_pytorch.
import tensorflow_datasets as tfds
import datasets


@gin.configurable(denylist=["data_dir", "split"])
class TFDSDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_dir, split, data_name="parnet_dataset", shuffle=None):
        """Dataset wrapper for tfds datasets.

        Given a TFDS dataset, this class wraps it in a torch IterableDataset and
        applies some preprocessing to make it compatible with downstream models (e.g. PanRBPNet).

        Args:
            data_dir (str): Directory where tfds dataset is stored.
            split (str): Split to load, e.g. 'train', 'val', 'test'.
            data_name (str, optional): Name of dataset (is required for loading for some fuckin reason). Defaults to 'parnet_dataset'.
        """
        super(TFDSDataset).__init__()

        self.split = split

        # load tfds dataset to tf.data.Dataset
        self._tf_dataset = tfds.load(
            data_name, data_dir=data_dir, shuffle_files=(shuffle is None)
        )[split]

        # Above we used 'shuffle_files' when loading the data which should give us some decent shuffling without filling up
        # any buffers. To further improve shuffling, users may additionally specify the size of a shuffle buffer (in #samples).
        if shuffle is not None:
            assert shuffle > 0 and isinstance(shuffle, int)
            self._tf_dataset = self._tf_dataset.shuffle(shuffle)

    def _format_example(self, example):
        example = {
            "inputs": {
                # Move channel dim from -1 to -2, i.e. a one-hot encoded sequence of length 100 over
                # over nucleotides A,C,G,T will have an initial shape of [100, 4] but will get
                # converted to [4, 100] as torch convolutions expect the channel to come first.
                "sequence": np.transpose(example["inputs"]["sequence"], axes=[1, 0])
            },
            "outputs": {
                # Outputs stay the same but will be renamed for compatibility.
                # (TODO: Modify upstream code to accept names from dataset as-is)
                "total": example["outputs"]["eCLIP"],
                "control": example["outputs"]["control"],
            },
        }

        # return as tf.Tensor, need to be converted to torch tensors
        return example

    def process_example(self, example):
        return example

    def _example_to_torch(self, example):
        """Converts nested numpy arrays to torch tensors.

        Args:
            example (dict): Nested dictionary of numpy arrays.

        Returns:
            dict: Nested dictionary of torch tensors.
        """

        return tf.nest.map_structure(
            lambda x: torch.tensor(x).to(torch.float32), example
        )

    def __iter__(self):
        for example in self._tf_dataset.as_numpy_iterator():
            # format example (select relevant data, swap axes, etc.)
            example = self._format_example(example)

            # convert numpy arrays (see previous .as_numpy_iterator()) to torch float32 tensors
            example = self._example_to_torch(example)

            # process sample (here just identity mapping, to be overwritten by subclasses for post-processes)
            example = self.process_example(example)

            yield example["inputs"], example["outputs"]


# %%
@gin.configurable(denylist=["data_dir", "split"])
class MaskedTFDSDataset(TFDSDataset):
    def __init__(self, *args, mask_filepaths=[], **kwargs):
        super().__init__(*args, **kwargs)
        self.composite_mask = None
        if mask_filepaths is not None:
            self.composite_mask = self._make_composite_mask(mask_filepaths)

    def _make_composite_mask(self, mask_filepaths):
        """Creates a composite mask from a list of mask filepaths.

        The composite mask is the logical AND of all masks in the list.

        Args:
            mask_filepaths (list): List of mask filepaths.

        Returns:
            torch.Tensor: Composite mask.
        """
        composite_mask = torch.load(mask_filepaths[0])
        for filepath in mask_filepaths[1:]:
            composite_mask = torch.logical_and(composite_mask, filepath)
        return composite_mask

    def _mask(self, structure, mask):
        """Masks a nested structure of tensors along axis 0.

        Currently, the structure is expected to be a dictionary with keys 'eCLIP' and (optional) 'control'.

        Args:
            structure (dict): Nested dictionary of tensors.
            mask (torch.Tensor): Mask to apply to tensors in structure.

        Returns:
            torch.Tensor: Structure with masked tensors.
        """
        try:
            return tf.nest.map_structure(lambda tensor: tensor[mask, :], structure)
        except:
            print(mask.shape, mask.dtype, file=sys.stderr)
            raise

    def process_example(self, example):
        """Overwrites process_example() from TFDSDataset to apply a mask to the outputs.

        Args:
            example (dict): Nested dictionary of tensors.

        Returns:
            dict: Nested dictionary of tensors with masked outputs.
        """
        example["outputs"] = self._mask(example["outputs"], self.composite_mask)
        return example


@gin.configurable(denylist=["hfds_path", "split"])
class HFDSDataset(torch.utils.data.Dataset):
    def __init__(self, hfds_path, split, shuffle=True, keep_in_memory=False, sequence_as_ids=False):
        super(HFDSDataset).__init__()

        self._hfds = datasets.load_from_disk(hfds_path, keep_in_memory=keep_in_memory)[
            split
        ]
        if shuffle:
            self._hfds = self._hfds.shuffle(seed=42, keep_in_memory=keep_in_memory)
        self._hfds.with_format("torch")

        self.sequence_as_ids = sequence_as_ids

    def _format_example(self, example):
        example = {
            "inputs": {
                "sequence": torch.sparse_coo_tensor(**example["inputs"]["sequence"])
                .to_dense()
                .to(torch.float32)
                .T
            },
            "outputs": {
                # Outputs stay the same but will be renamed for compatibility.
                # (TODO: Modify upstream code to accept names from dataset as-is)
                "total": torch.sparse_coo_tensor(**example["outputs"]["eCLIP"])
                .to_dense()
                .to(torch.float32),
                "control": torch.sparse_coo_tensor(**example["outputs"]["control"])
                .to_dense()
                .to(torch.float32),
            },
        }

        # let's turn the one-hot encoded sequence back into a sequence of ids..
        # FIXME: This is a bit of a hack, we should probably write the HFDS to disk with the sequence as ids
        # or better, just as a string of nucleotides. 
        if self.sequence_as_ids:
            x = (1 - torch.sum(example["inputs"]["sequence"], dim=0)) * 4 # we set 4 as the padding ID
            x += torch.argmax(example["inputs"]["sequence"], dim=0) # ..and add the one-hot encoded nucleotide ids
            example["inputs"]["sequence"] = x.long()

        # return as tf.Tensor, need to be converted to torch tensors
        return example

    def process_example(self, example):
        return example

    def __len__(self):
        return len(self._hfds)

    def __getitem__(self, idx):
        example = self._hfds[idx]
        example = self.process_example(self._format_example(example))

        return example["inputs"], example["outputs"]


# %%
# import sys

# import gin
# import torch
# import tensorflow as tf

# from rbpnet import io

# # %%
# @gin.configurable(denylist=['filepath', 'features_filepath'])
# class TFIterableDataset(torch.utils.data.IterableDataset):
#     def __init__(self, filepath, features_filepath=None, batch_size=64, cache=True, shuffle=None):
#         super(TFIterableDataset).__init__()

#         # load tfrecord file and create tf.data pipeline
#         self.dataset = self._load_dataset(filepath, features_filepath, batch_size, cache, shuffle)

#     def _load_dataset(self, filepath, features_filepath=None, batch_size=64, cache=True, shuffle=None):
#         # no not serialize - only after shuffle/cache
#         dataset = io.dataset_ops.load_tfrecord(filepath, deserialize=False)
#         if cache:
#             dataset = dataset.cache()
#         if shuffle:
#             dataset = dataset.shuffle(shuffle)

#         # deserialize proto to example
#         if features_filepath is None:
#             features_filepath = filepath + '.features.json'
#         self.features = io.dataset_ops.features_from_json_file(features_filepath)
#         dataset = io.dataset_ops.deserialize_dataset(dataset, self.features)

#         # batch & prefetch
#         dataset = dataset.batch(batch_size)
#         dataset = dataset.prefetch(tf.data.AUTOTUNE)

#         # format example & prefetch
#         dataset = dataset.map(self._format_example, num_parallel_calls=tf.data.AUTOTUNE)
#         dataset = dataset.prefetch(tf.data.AUTOTUNE)

#         return dataset

#     def _format_example(self, example):
#         # move channel dim from -1 to -2
#         # example['inputs']['input'] = tf.transpose(example['inputs']['input'], perm=[0, 2, 1])
#         # example['outputs']['signal']['total'] = tf.transpose(example['outputs']['signal']['total'], perm=[0, 2, 1])

#         example = {
#             'inputs': {
#                 'sequence': tf.transpose(example['inputs']['input'], perm=[0, 2, 1])},
#             # 'outputs': {
#             #     'total': tf.transpose(example['outputs']['signal']['total'], perm=[0, 2, 1]),
#             #     'control': tf.transpose(example['outputs']['signal']['control'], perm=[0, 2, 1]),
#             # },
#             'outputs': {
#                 'total': example['outputs']['signal']['total'],
#                 'control': example['outputs']['signal']['control'],
#             },
#         }

#         # return (input: Tensor, output: Tensor)
#         return example

#     def process_example(self, example):
#         return example

#     def _to_pytorch_compatible(self, example):
#         return tf.nest.map_structure(lambda x: torch.tensor(x).to(torch.float32), example)

#     def __iter__(self):
#         for example in self.dataset.as_numpy_iterator():
#             processed_pytorch_example = self._to_pytorch_compatible(self.process_example(example))
#             yield processed_pytorch_example['inputs'], processed_pytorch_example['outputs']

# # %%
# @gin.configurable(denylist=['filepath', 'features_filepath'])
# class MaskedTFIterableDataset(TFIterableDataset):
#     def __init__(self, mask_filepaths=None, **kwargs):
#         super(MaskedTFIterableDataset, self).__init__(**kwargs)
#         self.composite_mask = None
#         if mask_filepaths is not None:
#             self.composite_mask = self._make_composite_mask(mask_filepaths)

#     def _make_composite_mask(self, mask_filepaths):
#         composite_mask = torch.load(mask_filepaths[0])
#         for filepath in mask_filepaths[1:]:
#             composite_mask = torch.logical_and(composite_mask, filepath)
#         return composite_mask

#     def mask_structure(self, structure, mask):
#         try:
#             return tf.nest.map_structure(lambda tensor: tensor[:, mask], structure)
#         except:
#             print(mask.shape, mask.dtype, file=sys.stderr)
#             raise


#     def process_example(self, example):
#         example['outputs'] = self.mask_structure(example['outputs'], self.composite_mask)
#         return example

# # %%
# @gin.configurable()
# class MeanESMEmbeddingMaskedTFIterableDataset(MaskedTFIterableDataset):
#     def __init__(self, embedding_matrix_filepath, masks=None, **kwargs):
#         super(MeanESMEmbeddingMaskedTFIterableDataset, self).__init__(masks, **kwargs)
#         self.embedding_matrix = torch.load(embedding_matrix_filepath)

#     def process_example(self, example):
#         # add protein embedding to inputs
#         example['inputs']['embedding'] = self.embedding_matrix[self.composite_mask] if self.composite_mask is not None else self.embedding_matrix
#         if self.composite_mask is not None:
#             example['outputs'] = self.mask_structure(example['outputs'], self.composite_mask)
#         return example

# # %%
# class DummyIterableDataset(torch.utils.data.IterableDataset):
#     def __init__(self, n) -> None:
#         super(DummyIterableDataset).__init__()

#         self.n = n

#     def __iter__(self):
#         for i in range(self.n):
#             yield (torch.rand(16, 4, 101), torch.rand(16, 101, 7))
