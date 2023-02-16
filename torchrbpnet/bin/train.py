# %%
import datetime
import shutil
from pathlib import Path

import gin
import click
import torch
import pytorch_lightning as pl

# from ..lightning import Model
from ..networks import MultiRBPNet
from ..losses import MultinomialNLLLossFromLogits
from ..metrics import batched_pearson_corrcoef
from ..data import tfrecord_to_dataloader, dummy_dataloader

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# %%
@gin.configurable()
class Model(pl.LightningModule):
    def __init__(self, network, _example_input):
        super().__init__()
        self.network = network
        self.loss_fn = MultinomialNLLLossFromLogits()
        self.metrics = [batched_pearson_corrcoef]
        self.example_input_array = _example_input
    
    def forward(self, *args, **kwargs):
        return self.network(*args, **kwargs)

    def training_step(self, batch, *args, **kwargs):
        x, y = batch
        y_pred = self.network(x)
        loss = self.loss_fn(y, y_pred, dim=-2)
        self.log_dict(self._compute_metrics(y, y_pred, partition='train'), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, *args, **kwargs):
        x, y = batch
        y_pred = self.network(x)
        loss = self.loss_fn(y, y_pred, dim=-2)
        self.log_dict(self._compute_metrics(y, y_pred, partition='val'), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def _compute_loss(self, y, y_pred):
        return self.loss_fn(y, y_pred)

    def _compute_metrics(self, y, y_pred, partition=''):
        results = dict()
        for metric_fn in self.metrics:
            results[f'{partition}/{metric_fn.__name__}'] = metric_fn(y, y_pred)
        results[f'{partition}/loss'] = self._compute_loss(y, y_pred)
        return results

# %%
def _make_callbacks(output_path):
    callbacks = [
        ModelCheckpoint(dirpath=output_path/'checkpoints', every_n_epochs=1, save_last=True),
        EarlyStopping(monitor="val/loss", min_delta=0.00, patience=3, verbose=False, mode="min"),
    ]
    return callbacks

# %%
def _make_loggers(output_path):
    loggers = [
        pl_loggers.TensorBoardLogger(output_path/'tensorboard', name='', version='', log_graph=True),
    ]
    return loggers

# %%
@gin.configurable()
def train(tfrecord, validation_tfrecord, output_path, network=None, **kwargs):
    dataloader_train = dummy_dataloader(5)
    dataloader_val = dummy_dataloader(5)

    trainer = pl.Trainer(
        default_root_dir=output_path, 
        logger=_make_loggers(output_path), 
        callbacks=_make_callbacks(output_path),
        **kwargs,
        )

    model = Model(network, next(iter(dataloader_train))[0])
    trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
    torch.save(model, output_path / 'model.pt')

# %%
@click.command()
@click.argument('tfrecord', required=True, type=str)
@click.option('--config', type=str, default=None)
@click.option('-o', '--output', required=True)
@click.option('--validation-tfrecord', type=str, default=None)
def main(tfrecord, config, output, validation_tfrecord):
    # parse gin-config config
    if config is not None:
        gin.parse_config_file(config)

    output_path = Path(f'{output}/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    output_path.mkdir(parents=True)
    if config is not None:
        shutil.copy(config, str(output_path / 'config.gin'))   

    train(tfrecord, validation_tfrecord, output_path)