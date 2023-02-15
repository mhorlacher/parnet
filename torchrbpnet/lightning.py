# %%
import torch
import torch.nn as nn
import pytorch_lightning as pl

from .losses import MultinomialNLLLossFromLogits
from .metrics import batched_pearson_corrcoef

# %%
class Model(pl.LightningModule):
    def __init__(self, network):
        super().__init__()
        self.network = network
        self.loss_fn = MultinomialNLLLossFromLogits()

    def training_step(self, batch, **kwargs):
        x, y = batch
        y_pred = self.network(x)
        loss = self.loss_fn(y, y_pred, dim=-2)
        metrics = self.compute_metrics(y, y_pred)
        self.log_dict(metrics)
        return loss
    
    def compute_metrics(self, y, y_pred):
        pccs = batched_pearson_corrcoef(y, y_pred)
        return {'pcc_mean': torch.mean(pccs), 'pcc_std': torch.std(pccs)}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer