# parnet

## Installation

parnet can be installed for inference-only via pip:

```
pip install git+https://github.com/mhorlacher/parnet.git
```

In order to train parnet models or to create your own training datasets, additional packages must be installed:

```
# required for loading TFDS datasets
pip install tensorflow>=2.15.0 tensorflow_datasets>=4.9.3

# required for creating TFDS datasets from scratch
pip install pandas pyyaml pyBigWig pysam
```