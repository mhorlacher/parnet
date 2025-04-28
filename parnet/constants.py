import importlib

import torch

IDX_TO_EXPERIMENT = torch.load(importlib.resources.files('parnet') / 'assets' / 'ENCODE.idx2symbol-cell.pt')
EXPERIMENT_TO_IDX = {v: k for k, v in IDX_TO_EXPERIMENT.items()}
