# %%
import gin
import torch

# %%
# Activation Functions
gin.config.external_configurable(torch.nn.ReLU, module='torch.nn')
gin.config.external_configurable(torch.nn.SiLU, module='torch.nn')
gin.config.external_configurable(torch.nn.Mish, module='torch.nn')