# %%
import gin
import torch
import pytorch_lightning

# %%
# Activation Functions
gin.config.external_configurable(torch.nn.ReLU, module='torch.nn')
gin.config.external_configurable(torch.nn.SiLU, module='torch.nn')
gin.config.external_configurable(torch.nn.Mish, module='torch.nn')

# Callbacks
gin.config.external_configurable(pytorch_lightning.callbacks.EarlyStopping, module='pytorch_lightning.callbacks')