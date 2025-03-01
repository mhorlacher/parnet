# disable tensorflow logs and enable dynamic memory growth (tf is only used for data loading via TFDS)
from parnet.utils import _disable_tensorflow_logs, _set_tf_dynamic_memory_growth

_disable_tensorflow_logs()
_set_tf_dynamic_memory_growth()

import datetime
import shutil
from pathlib import Path
import warnings
import logging

import click
import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import MeanMetric

from parnet.data.datasets import TFDSDataset
from parnet.losses import MultinomialNLLLossFromLogits


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        model,
        use_control=False,
        loss_fn=None,
        optimizer=None,
        lr_scheduler_cls=None,
        metrics=None,
        crop_size=None,
    ):
        if loss_fn is None:
            raise ValueError('loss must be specified.')
        if optimizer is None:
            raise ValueError('optimizer must be specified.')

        super().__init__()
        self.model = model
        self.use_control = use_control

        self.crop_size = crop_size

        # # loss
        # self.loss_fn = nn.ModuleDict(
        #     {
        #         'TRAIN_total': loss(),
        #         'VAL_total': loss(),
        #     }
        # )
        # if use_control:
        #     self.loss_fn['TRAIN_control'] = loss()
        #     self.loss_fn['VAL_control'] = loss()
        self.loss_fn = loss_fn

        # penalty loss
        self.penalty_loss = MeanMetric()

        # metrics
        if metrics is None:
            metrics = {}

        self.train_metrics_losses = torchmetrics.MetricCollection(
            {
                'train/loss': torchmetrics.MeanMetric(),
                'train/loss_eCLIP': torchmetrics.MeanMetric(),
                'train/loss_SMI': torchmetrics.MeanMetric(),
                'train/loss_penalty': torchmetrics.MeanMetric(),
            }
        )
        self.train_metrics_eCLIP = torchmetrics.MetricCollection(
            {'train/' + name + '_eCLIP': metric() for name, metric in metrics.items()}
        )
        if self.use_control:
            self.train_metrics_SMI = torchmetrics.MetricCollection(
                {'train/' + name + '_SMI': metric() for name, metric in metrics.items()}
            )

        self.val_metrics_losses = torchmetrics.MetricCollection(
            {
                'val/loss': torchmetrics.MeanMetric(),
                'val/loss_eCLIP': torchmetrics.MeanMetric(),
                'val/loss_SMI': torchmetrics.MeanMetric(),
                'val/loss_penalty': torchmetrics.MeanMetric(),
                'val/mix_coeff': torchmetrics.MeanMetric(),
                'val/mix_coeff_std-over-exp': torchmetrics.MeanMetric(),
                'val/mix_coeff_std-over-batch': torchmetrics.MeanMetric(),
            }
        )
        self.val_metrics_eCLIP = torchmetrics.MetricCollection(
            {'val/' + name + '_eCLIP': metric() for name, metric in metrics.items()}
        )
        if self.use_control:
            self.val_metrics_SMI = torchmetrics.MetricCollection(
                {'val/' + name + '_SMI': metric() for name, metric in metrics.items()}
            )

        # self.metrics = nn.ModuleDict(
        #     {
        #         'TRAIN_total': nn.ModuleDict({name: metric() for name, metric in metrics.items()}),
        #         'VAL_total': nn.ModuleDict({name: metric() for name, metric in metrics.items()}),
        #     }
        # )
        # if use_control:
        #     self.metrics['TRAIN_control'] = nn.ModuleDict(
        #         {name: metric() for name, metric in metrics.items()}
        #     )
        #     self.metrics['VAL_control'] = nn.ModuleDict({name: metric() for name, metric in metrics.items()})

        # self.metrics = nn.ModuleDict({
        #     'TRAIN': nn.ModuleDict({name: metric() for name, metric in metrics.items()}),
        #     'VAL': nn.ModuleDict({name: metric() for name, metric in metrics.items()}),
        # })

        self.optimizer_cls = optimizer
        self.lr_scheduler_cls = lr_scheduler_cls

        # FIXME: Disable for now, because there are some issues with Tensorboard and saving
        # the hyperparameters to the YAML file (i.e. "ValueError: dictionary update sequence
        # element #0 has length 1; 2 is required")
        # self.save_hyperparameters()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(self.parameters())

        if self.lr_scheduler_cls is not None:
            lr_scheduler = {
                'lr_scheduler': {
                    'scheduler': self.lr_scheduler_cls(optimizer),
                    'interval': 'epoch',
                    'monitor': 'val/loss',
                }
            }

        return {'optimizer': optimizer} | (lr_scheduler if self.lr_scheduler_cls is not None else {})

    def _compute_loss(self, y, y_pred, crop_size=None):
        targets = {'total', 'control'}.intersection(y_pred.keys())

        if crop_size is not None:
            # we crop the input and output along the sequence dimension to avoid edge effects
            y = {target: y[target][:, :, crop_size:-crop_size] for target in targets}
            y_pred = {target: y_pred[target][:, :, crop_size:-crop_size] for target in targets}

        loss_eCLIP = self.loss_fn(y['total'], y_pred['total'])

        loss_SMI = torch.tensor(0.0, dtype=torch.float32).to(y_pred['total'].device)
        if 'control' in y_pred:
            loss_SMI = self.loss_fn(y['control'], y_pred['control'])

        loss_penalty = torch.tensor(0.0, dtype=torch.float32).to(y_pred['total'].device)
        if 'penalty_loss' in y_pred:
            loss_penalty = y_pred['penalty_loss'].mean()

        loss = loss_eCLIP + loss_SMI + loss_penalty

        return {
            'loss': loss,
            'loss_eCLIP': loss_eCLIP,
            'loss_SMI': loss_SMI,
            'loss_penalty': loss_penalty,
        }

    def training_step(self, batch, batch_idx=None, **kwargs):
        inputs, y = batch
        y_pred = self.forward(inputs)

        # compute and log losses
        losses = self._compute_loss(y, y_pred, crop_size=self.crop_size)
        for losss_name in ['loss', 'loss_eCLIP', 'loss_SMI', 'loss_penalty']:
            self.train_metrics_losses[f'train/{losss_name}'](losses[losss_name])
        self.log_dict(self.train_metrics_losses, prog_bar=True, on_step=True, on_epoch=True)

        # compute and log metrics
        self.train_metrics_eCLIP.update(y['total'], y_pred['total'])
        if self.use_control:
            self.train_metrics_SMI.update(y['control'], y_pred['control'])

        return losses['loss']

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics_eCLIP.compute(), on_step=False, on_epoch=True)
        self.train_metrics_eCLIP.reset()

        if self.use_control:
            self.log_dict(self.train_metrics_SMI.compute(), on_step=False, on_epoch=True)
            self.train_metrics_SMI.reset()

    def validation_step(self, batch, batch_idx=None, **kwargs):
        inputs, y = batch
        y_pred = self.forward(inputs)

        # compute and log losses
        losses = self._compute_loss(y, y_pred, crop_size=self.crop_size)
        for losss_name in ['loss', 'loss_eCLIP', 'loss_SMI', 'loss_penalty']:
            self.val_metrics_losses[f'val/{losss_name}'].update(losses[losss_name])
        # keep track of mixing coefficients
        self.val_metrics_losses['val/mix_coeff'].update(y_pred['mix_coeff'])

        if y_pred['mix_coeff'].shape[1] > 1:
            # compute std over experiments and then mean over batch (i.e. check how much the mixing coefficients vary across experiments)
            self.val_metrics_losses['val/mix_coeff_std-over-exp'].update(y_pred['mix_coeff'].std(1).mean())

        if y_pred['mix_coeff'].shape[0] > 1:
            # compute std over batch and then mean over experiments (i.e. check how much the mixing coefficients vary across samples)
            self.val_metrics_losses['val/mix_coeff_std-over-batch'].update(y_pred['mix_coeff'].std(0).mean())

        # compute and log metrics
        self.val_metrics_eCLIP.update(y['total'], y_pred['total'])
        if self.use_control:
            self.val_metrics_SMI.update(y['control'], y_pred['control'])

        return losses['loss']

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics_losses.compute(), on_step=False, on_epoch=True)
        self.val_metrics_losses.reset()

        self.log_dict(self.val_metrics_eCLIP.compute(), on_step=False, on_epoch=True)
        self.val_metrics_eCLIP.reset()

        if self.use_control:
            self.log_dict(self.val_metrics_SMI.compute(), on_step=False, on_epoch=True)
            self.val_metrics_SMI.reset()


def _make_loggers(output_path, loggers):
    if loggers is None:
        return []
    return [logger(save_dir=output_path, name='', version='') for logger in loggers]


@gin.configurable()
class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@gin.configurable(denylist=['tfds_filepath', 'output_path'])
def train(
    data_path,
    just_print_model,
    output_path,
    n_devices=1,
    dataset=TFDSDataset,
    model=None,
    loggers=None,
    loss_fn=MultinomialNLLLossFromLogits,
    metrics=None,
    optimizer=torch.optim.AdamW,
    lr_scheduler_cls=None,
    batch_size=128,
    use_control=False,
    crop_size=None,
    callbacks=None,
    **kwargs,
):
    # wrap model in LightningModule
    lightning_model = LightningModel(
        model,
        loss_fn=loss_fn,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler_cls=lr_scheduler_cls,
        use_control=use_control,
        crop_size=crop_size,
    )

    if just_print_model:
        print(model)
        exit()

    train_loader = DataLoader(dataset(data_path, split='train'), batch_size=batch_size)
    val_loader = DataLoader(
        dataset(data_path, split='validation', shuffle=False, keep_in_memory=False), batch_size=batch_size
    )

    trainer = pl.Trainer(
        default_root_dir=output_path,
        logger=_make_loggers(output_path, loggers),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=output_path / 'checkpoints',
                every_n_epochs=1,
                monitor='val/loss',
                mode='min',
                filename='best',
                save_last=True,
                save_top_k=1,
            ),
        ]
        + (callbacks if callbacks is not None else []),
        devices=n_devices,
        **kwargs,
    )

    # write model summary
    with open(str(output_path / 'model.summary.txt'), 'w') as f:
        print(str(lightning_model), file=f)

    # write optimizer summary
    with open(str(output_path / 'optim.summary.txt'), 'w') as f:
        print(str(lightning_model.configure_optimizers()), file=f)

    # fit the model
    trainer.fit(
        lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path='last',  # resume from last checkpoint
    )

    # save the torch (not pytorch-lightning) model
    torch.save(lightning_model.model, output_path / 'model.pt')


@click.command()
@click.argument('data_path', required=False, type=str, default=None)
@click.option('--config', type=str, default=None)
@click.option('--log-level', type=str, default='WARNING')
@click.option('--just-print-model', is_flag=True, default=False)
@click.option('--n-devices', type=int, default=1)
@click.option('-o', '--output', default=None)
def main(data_path, config, log_level, just_print_model, n_devices, output):
    # set log level
    logging.basicConfig(level=log_level)

    # ignore torch warnings
    warnings.filterwarnings('ignore')

    # parse gin-config config
    if config is not None:
        gin.parse_config_file(config)

    # create output directory
    output_path = Path(output)
    if not just_print_model:
        if output_path.exists():
            logging.warning(f'Output path {output_path} already exists. Overwriting.')
        output_path.mkdir(parents=True, exist_ok=True)

        # copy gin config
        if config is not None:
            shutil.copy(config, str(output_path / 'config.gin'))

    # launch training (parameters are configured exclusively via gin)
    train(data_path, just_print_model, output_path, n_devices)
