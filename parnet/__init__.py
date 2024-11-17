# %%
__version__ = '0.1.1'

# %%
# Import torch and additional external configurables 
import gin.torch.external_configurables
from . import _gin_external_configurables

# %%
# add tensor cores support
import torch
torch.set_float32_matmul_precision('high')

# Set random seeds
from lightning.pytorch import seed_everything

__seed__ = 42
seed_everything(__seed__)