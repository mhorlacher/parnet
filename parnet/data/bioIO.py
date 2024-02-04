"""Holds classes and functions for reading biological data (FASTA, BED, BigWig, etc.). 

    In general, the classes and functions in this module are only used for constructing the TFDS dataset.
    Users which want to train a model on an existing dataset, or use a pre-trained model, do not need to 
    use this module. Consequently, this module is not imported in the __init__.py file and libraries such 
    as pyBigWig are not required for the installation of parnet. 
"""

import math

import yaml
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd


# %%
# map bases to their integer representation
base2int = {"A": 0, "C": 1, "G": 2, "T": 3}

# map bases to their complement
baseComplement = {"A": "T", "C": "G", "G": "C", "T": "A"}


def reverse_complement(dna_string):
    """Returns the reverse-complement for a DNA string."""

    complement = [baseComplement.get(base, "N") for base in dna_string]
    reversed_complement = reversed(complement)
    return "".join(list(reversed_complement))


def sequence2int(sequence, mapping=base2int):
    """Converts a DNA sequence to a list of integers.

    Args:
        sequence (str): DNA sequence.
        mapping (dict, optional): Character to integer mapping. Defaults to base2int.

    Returns:
        list: List of integers.
    """
    return [
        mapping.get(base, 999) for base in sequence
    ]  # TODO: 999 is a hack, should be replaced by something else (e.g. -1?).


def sequence2onehot(sequence, mapping=base2int):
    """Converts a DNA sequence to a one-hot encoded tf.Tensor.

    Args:
        sequence (str): DNA sequence.
        mapping (dict, optional): Character to integer mapping. Defaults to base2int.

    Returns:
        tf.Tensor: One-hot encoded sequence.
    """
    return tf.one_hot(
        sequence2int(sequence, mapping), depth=4
    )  # Remove tensorflow dependency, ideally this should use just numpy.


def mask_noncanonical_bases(sequence):
    """Masks non-canonical bases (anything except A, C, G, T or whatever mapping is provided) with N.

    Args:
        sequence (str): DNA sequence.

    Returns:
        str: Masked DNA sequence.
    """
    return "".join([base if base in base2int else "N" for base in sequence])


# %%
class Fasta:
    def __init__(self, filepath, mask_noncanonical_bases=True) -> None:
        """
        Initialize a Fasta object.

        Args:
            filepath (str): The path to the FASTA file.
            mask_noncanonical_bases (bool, optional): Whether to mask non-canonical bases with 'N'. Defaults to True.
        """
        try:
            import pysam
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install pysam. See https://github.com/pysam-developers/pysam"
            )

        self.mask_noncanonical_bases = mask_noncanonical_bases
        self._fasta = pysam.FastaFile(filepath)

    def fetch(self, chrom, start, end, strand="+", **kwargs):
        """
        Fetch the sequence from the FASTA file.

        Args:
            chrom (str): The chromosome name.
            start (int): The start position.
            end (int): The end position.
            strand (str, optional): The strand of the sequence. Defaults to '+'.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: The one-hot encoded sequence as a numpy array.
        """
        sequence = self._fasta.fetch(chrom, start, end).upper()

        if self.mask_noncanonical_bases:
            # Everything except A, C, G, T will be mapped to N
            sequence = mask_noncanonical_bases(sequence)

        # Reverse complement if necessary
        if strand == "+":
            pass
        elif strand == "-":
            sequence = "".join(reverse_complement(sequence))
        else:
            raise ValueError(f"Unknown strand: {strand}")

        # Convert to one-hot encoded numpy array. We assume that the alphabet is fairly small (usually 4 or 5 bases).
        return np.array(sequence2onehot(sequence), dtype=np.int8)

    def __call__(self, *args, **kwargs):
        return self.fetch(*args, **kwargs)


# %%
class Bed:
    def __init__(self, filepath) -> None:

        self.bed_df = pd.read_csv(filepath, sep="\t", header=None)
        self.bed_df.columns = ["chrom", "start", "end", "name", "score", "strand"] + [
            str(i) for i in range(6, len(self.bed_df.columns))
        ]

    def __len__(self):
        return len(self.bed_df)

    def __iter__(self):
        for i in range(0, len(self.bed_df)):
            yield self.bed_df.loc[i].to_dict()


# %%
def nan_to_zero(x):
    """Replaces nan's with zeros."""
    return 0 if math.isnan(x) else x


class BigWig:
    def __init__(self, bigwig_filepath) -> None:
        try:
            import pyBigWig  # TODO: For now I prefer to have imports at the top of the file.
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install pyBigWig. See https://github.com/deeptools/pyBigWig"
            )

        self._bigWig = pyBigWig.open(bigwig_filepath)

    def values(self, chrom, start, end, **kwargs):
        chrom, start, end = (
            str(chrom),
            int(start),
            int(end),
        )  # not sure why this is needed, it worked locally with numpy.int32
        values = self._bigWig.values(chrom, start, end)
        values = [nan_to_zero(v) for v in values]
        return np.array(values, dtype=np.float32)

    def __call__(self, *args, **kwargs):
        return self.values(*args, **kwargs)


class StrandedBigWig:
    def __init__(self, bigwig_plus, bigwig_minus, reverse_minus=True) -> None:
        self._bigWig_plus = BigWig(bigwig_plus)
        self._bigWig_minus = BigWig(bigwig_minus)
        self.reverse_minus = reverse_minus

    def values(self, chrom, start, end, strand="+", **kwargs):
        """Returns values for a given range and strand.

        Args:
            chrom  (str): Chromosome (chr1, chr2, ...)
            start  (int): 0-based start position
            end    (int): 0-based end position
            strand (str): Strand ('+' or '-')

        Returns:
            numpy.array: Numpy array of shape (end-start, )
        """

        if strand == "+":
            bigWig = self._bigWig_plus
        elif strand == "-":
            bigWig = self._bigWig_minus
        else:
            raise ValueError(f"Unexpected strand: {strand}")

        values = bigWig.values(chrom, start, end)

        if strand == "-" and self.reverse_minus:
            values = values[::-1]

        return values

    def __call__(self, *args, **kwargs):
        return self.values(*args, **kwargs)


# %%
class DataSpec:
    """Specifies the data layout and assembles file-connectors to generate samples.

    Inputs and outputs are specified in a YAML file. The YAML file should have the following structure:

    inputs:
        sequence: path/to/fasta
        outputs:
        Task_1:
            eCLIP:
                - path/to/eclip/counts/forward
                - path/to/eclip/counts/reverse
            control:
                - path/to/control/counts/forward
                - path/to/control/counts/reverse
        ...
        Task_N:
            eCLIP:
                - path/to/eclip/counts/forward
                - path/to/eclip/counts/reverse
            control:
                - path/to/control/counts/forward
                - path/to/control/counts/reverse

    The YAML specification is then used to initialize Fasta and StrandedBigWig file-connectors, from which
    samples can be fetch in the specified structure. For efficiency (storage and during training), the
    extracted 1D bigWig stracks are stacked to two tensors for eCLIP and control, both of the shape (n_tasks, n_positions).
    """

    def __init__(self, dataspec_yml, control=False):
        self.control = control

        # parse YAML dataspec
        with open(dataspec_yml) as f:
            self._dataspec = yaml.load(f, yaml.FullLoader)
        self.tasks = self._dataspec["outputs"].keys()

        # initialize fasta-file connector
        self._dataspec["inputs"]["sequence"] = Fasta(
            self._dataspec["inputs"]["sequence"]
        )

        # initialize eCLIP bigWig-file connectors, one for each task
        for task in self._dataspec["outputs"]:
            self._dataspec["outputs"][task]["eCLIP"] = StrandedBigWig(
                *self._dataspec["outputs"][task]["eCLIP"]
            )

        # if control counts are available, initialize bigWig-file connectors for controls
        if control:
            for task in self._dataspec["outputs"]:
                self._dataspec["outputs"][task]["control"] = StrandedBigWig(
                    *self._dataspec["outputs"][task]["control"]
                )

    @property
    def tf_signature(self):
        """Returns the features of the data in tf.TensorSpec format (required for tf.data.Dataset.from_generator).

        Returns:
            dict: Nested dictionary of tf.TensorSpecs.
        """

        signature = {
            "meta": {
                "name": tf.TensorSpec(shape=(), dtype=tf.string),
            },
            "inputs": {
                "sequence": tf.TensorSpec(shape=(None, 4), dtype=tf.int8),
            },
            "outputs": {
                "eCLIP": tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            },
        }
        if self.control:
            signature["outputs"]["control"] = tf.TensorSpec(
                shape=(None, None), dtype=tf.float32
            )
        return signature

    @property
    def tfds_features(self):
        """Returns the features of the data in tfds.features.FeaturesDict format.

        Returns:
            tfds.features.FeatureDict: TFDS features describing the data.
        """

        features = {
            "meta": {
                "name": tfds.features.Tensor(shape=(), dtype=tf.string),
            },
            "inputs": {
                "sequence": tfds.features.Tensor(
                    shape=(None, None), dtype=tf.int8, encoding="zlib"
                ),
            },
            "outputs": {
                "eCLIP": tfds.features.Tensor(
                    shape=(None, None), dtype=tf.float32, encoding="zlib"
                ),
            },
        }
        if self.control:
            features["outputs"]["control"] = tfds.features.Tensor(
                shape=(None, None), dtype=tf.float32, encoding="zlib"
            )
        features = tfds.features.FeaturesDict(features)
        return features

    def fetch_sample(self, chrom, start, end, strand, target_size):
        """Fetches a sample from the data.

        Samples are specified by chrom, start, end and strand. The sequence may also be padded to target_size.
        Fetching a sample returns a dictionary with the following structure:

        {
            'meta': {
                'name': Tensor(shape=(), dtype=string),
            },
            'inputs': {
                'sequence': Tensor(shape=(target_size, 4), dtype=int8),
            },
            'outputs': {
                'eCLIP': Tensor(shape=(n_tasks, target_size), dtype=float32),
                'control': Tensor(shape=(n_tasks, target_size), dtype=float32),
            },
        }

        Args:
            chrom (str): Chromosome.
            start (int): Start position (closed).
            end (int): End position (open).
            strand (int): Strand ('+' or '-').
            target_size (int): Target size of the sequence. If the sequence is shorter than target_size, it will be padded.

        Returns:
            dict: Dictionary containing the sample tensors.
        """

        sample = {"meta": {}, "inputs": {}, "outputs": {}}

        # prepare padding
        # TODO: Move this to a separate function.
        size = end - start
        padding_left = int(np.ceil((target_size - size) / 2))
        padding_right = int(np.floor((target_size - size) / 2))

        sample["inputs"]["sequence"] = self._dataspec["inputs"]["sequence"](
            chrom, start, end, strand
        )
        sample["inputs"]["sequence"] = tf.pad(
            sample["inputs"]["sequence"],
            paddings=[[padding_left, padding_right], [0, 0]],
        )

        sample["outputs"]["eCLIP"] = np.stack(
            [
                self._dataspec["outputs"][task]["eCLIP"](chrom, start, end, strand)
                for task in self.tasks
            ]
        )
        sample["outputs"]["eCLIP"] = tf.pad(
            sample["outputs"]["eCLIP"], paddings=[[0, 0], [padding_left, padding_right]]
        )
        if self.control:
            sample["outputs"]["control"] = np.stack(
                [
                    self._dataspec["outputs"][task]["control"](
                        chrom, start, end, strand
                    )
                    for task in self.tasks
                ]
            )
            sample["outputs"]["control"] = tf.pad(
                sample["outputs"]["control"],
                paddings=[[0, 0], [padding_left, padding_right]],
            )

        # meta/name
        sample["meta"]["name"] = tf.constant(
            f"{chrom}:{start}-{end}:{strand}", dtype=tf.string
        )

        # assert padding
        assert (
            target_size
            == sample["outputs"]["eCLIP"].shape[1]
            == sample["inputs"]["sequence"].shape[0]
        )

        return sample

    def __call__(self, *args, **kwargs):
        return self.fetch_sample(*args, **kwargs)
