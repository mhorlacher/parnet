# %%
__version__ = '0.1.0'

# %%
# Import torch and additional external configurables 
import gin.torch.external_configurables
from . import _gin_external_configurables

# %%
# add tensor cores support
import torch
torch.set_float32_matmul_precision('high')