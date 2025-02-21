# PanRBPNet: A RBP-binding-informed RNA Foundation Model

PanRBPNet - or `parnet` - is a multi-task extension of our previous [RBPNet](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-023-03015-7) model for prediction RNA-protein binding at nucleotide resolution. 

> [!WARNING]  
> This package is under heavy development and while it's API is somewhat stable, it can change at any time and without warning. If you are using this package in your own research, it's highly recommended to either 1) fork the respository or 2) pin the version / commit of the package you are using. 

## Installation

### Using Pip

```
pip install parnet
```

or 

```
pip install git+https://github.com/github.com/mhorlacher/parnet.git@SOME_BRANCH
```

to install from a specific branch (e.g. the latest development version of package version). 

### Using Conda

```
git clone https://github.com/github.com/mhorlacher/parnet
cd parnet
make env
conda activate parnet
```

### Using Docker

First, pull the docker image. 
```
docker pull 
```

Then, run commands as shown below. 
```
docker run parnet --help 
```
To provide input files and capture output file, mount host files and/or directories using `--mount` (see the Docker [docs](https://docs.docker.com/engine/storage/bind-mounts/)). 


## Dataset

Parnet's training datasets are stored in Huggingface's dataset format (HFDS). The primary training dataset of this study, which is composed of 223 eCLIP tracks from the ENCODE project, can be obtained via: 

### Download compressed TFDS dataset
```
wget https://zenodo.org/records/14176118/files/encode.filtered.5.hfds.tar.gz
```

### Unpack TFDS dataset
```
tar -xJf encode.hfds.tar.xz
```

## Usage

```
Usage: parnet [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  build-dataset
  predict
  train
```
