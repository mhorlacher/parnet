# %%
import torch
import tensorflow as tf

from rbpnet import io

# %%
class TFIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, filepath, features_filepath=None, batch_size=64, cache=True, shuffle=None):
        super(TFIterableDataset).__init__()

        self.dataset = io.dataset_ops.load_tfrecord(filepath, deserialize=False)

        # cache
        if cache:
            self.dataset = self.dataset.cache()

        if shuffle:
            self.dataset = self.dataset.shuffle(shuffle)

        # deserialize
        if features_filepath is None:
            features_filepath = filepath + '.features.json'
        self.features = io.dataset_ops.features_from_json_file(features_filepath)
        self.dataset = io.dataset_ops.deserialize_dataset(self.dataset, self.features)

        # batch
        self.dataset = self.dataset.batch(batch_size)

        # format dataset
        self.dataset = self.dataset.map(lambda e: (tf.transpose(e['inputs']['input'], perm=[0, 2, 1]), tf.transpose(e['outputs']['signal']['total'], perm=[0, 2, 1])))
        
    def __iter__(self):
        for example in self.dataset.as_numpy_iterator():
            yield tf.nest.map_structure(lambda x: torch.tensor(x).to(torch.float32), example)