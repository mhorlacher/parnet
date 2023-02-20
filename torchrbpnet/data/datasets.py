# %%
import gin
import torch
import tensorflow as tf

from rbpnet import io

# %%
@gin.configurable(denylist=['filepath', 'features_filepath'])
class TFIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, filepath, features_filepath=None, batch_size=64, cache=True, shuffle=None):
        super(TFIterableDataset).__init__()
        
        self.dataset = self._load_dataset(filepath, features_filepath, batch_size, cache, shuffle)

    def _load_dataset(self, filepath, features_filepath=None, batch_size=64, cache=True, shuffle=None):
        dataset = io.dataset_ops.load_tfrecord(filepath, deserialize=False)
        if cache:
            dataset = dataset.cache()
        if shuffle:
            dataset = dataset.shuffle(shuffle)

        # deserialize
        if features_filepath is None:
            features_filepath = filepath + '.features.json'
        self.features = io.dataset_ops.features_from_json_file(features_filepath)
        dataset = io.dataset_ops.deserialize_dataset(dataset, self.features)

        # batch
        dataset = dataset.batch(batch_size)

        # format dataset
        dataset = dataset.map(lambda e: (tf.transpose(e['inputs']['input'], perm=[0, 2, 1]), tf.transpose(e['outputs']['signal']['total'], perm=[0, 2, 1])))
        
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset
    
    def _to_pytorch_compatible(self, example):
        return tf.nest.map_structure(lambda x: torch.tensor(x).to(torch.float32), example)

    def __iter__(self):
        for example in self.dataset.as_numpy_iterator():
            yield self._to_pytorch_compatible(example)

# %%
@gin.configurable(denylist=['filepath', 'features_filepath'])
class MaskedTFIterableDataset(TFIterableDataset):
    def __init__(self, masks=None, **kwargs):
        super(MaskedTFIterableDataset, self).__init__(**kwargs)
        self.composite_mask = None
        if masks is not None:
            self.composite_mask = self._make_composite_mask(masks)

    def _make_composite_mask(self, masks):
        composite_mask = masks[0]
        for mask in masks[1:]:
            composite_mask = torch.logical_and(composite_mask, mask)
        return composite_mask
    
    def _apply_mask_to_outputs(self, example):
        return (example[0], example[1][:, :, self.composite_mask])
    
    def __iter__(self):
        for example in super().__iter__():
            if self.composite_mask is not None:
                example = self._apply_mask_to_outputs(example)
            yield example

# %%
class DummyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, n) -> None:
        super(DummyIterableDataset).__init__()
        
        self.n = n

    def __iter__(self):
        for i in range(self.n):
            yield (torch.rand(16, 4, 101), torch.rand(16, 101, 7))