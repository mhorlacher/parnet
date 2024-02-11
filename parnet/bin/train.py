# %%
# disable tensorflow logs and enable dynamic memory growth (tf is only used for data loading via TFDS)
from parnet.utils import _disable_tensorflow_logs, _set_tf_dynamic_memory_growth

_disable_tensorflow_logs()
_set_tf_dynamic_memory_growth()

# %%
import datetime
import shutil
from pathlib import Path
import warnings
import logging

import click
import gin
import torch
import torch.nn as nn
import pytorch_lightning as pl

from parnet.data.datasets import TFDSDataset
from parnet.losses import MultinomialNLLLossFromLogits


# %%
class LightningModel(pl.LightningModule):
    def __init__(
        self, model, use_control=False, loss=None, optimizer=None, metrics=None
    ):
        if loss is None:
            raise ValueError("loss must be specified.")
        if optimizer is None:
            raise ValueError("optimizer must be specified.")

        super().__init__()
        self.model = model

        # loss
        self.loss_fn = nn.ModuleDict(
            {
                "TRAIN_total": loss(),
                "VAL_total": loss(),
            }
        )
        if use_control:
            self.loss_fn["TRAIN_control"] = loss()
            self.loss_fn["VAL_control"] = loss()

        # metrics
        if metrics is None:
            metrics = {}

        self.metrics = nn.ModuleDict(
            {
                "TRAIN_total": nn.ModuleDict(
                    {name: metric() for name, metric in metrics.items()}
                ),
                "VAL_total": nn.ModuleDict(
                    {name: metric() for name, metric in metrics.items()}
                ),
            }
        )
        if use_control:
            self.metrics["TRAIN_control"] = nn.ModuleDict(
                {name: metric() for name, metric in metrics.items()}
            )
            self.metrics["VAL_control"] = nn.ModuleDict(
                {name: metric() for name, metric in metrics.items()}
            )

        # self.metrics = nn.ModuleDict({
        #     'TRAIN': nn.ModuleDict({name: metric() for name, metric in metrics.items()}),
        #     'VAL': nn.ModuleDict({name: metric() for name, metric in metrics.items()}),
        # })

        # optimizer
        self.optimizer_cls = optimizer

        # save hyperparameters
        self.save_hyperparameters()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx=None, **kwargs):
        inputs, y = batch
        # y = y['total']
        y_pred = self.forward(inputs)

        # log counts
        self.log(
            "log10-1p_total_counts",
            torch.log10(y["total"].sum() + 1),
            on_step=True,
            logger=True,
        )
        if "control" in y:
            self.log(
                "log10-1p_control_counts",
                torch.log10(y["control"].sum() + 1),
                on_step=True,
                logger=True,
            )

        # compute loss across output tracks, i.e. total (+ control, if available)
        loss = self.compute_and_log_loss(y, y_pred, partition="TRAIN")

        self.compute_and_log_metics(y, y_pred, partition="TRAIN")

        return loss

    def validation_step(self, batch, batch_idx=None, **kwargs):
        inputs, y = batch
        # y = y['total']
        y_pred = self.forward(inputs)
        self.compute_and_log_loss(y, y_pred, partition="VAL")
        self.compute_and_log_metics(y, y_pred, partition="VAL")

    def compute_and_log_loss(self, y, y_pred, partition=None):
        # compute loss across output tracks, i.e. total (+ control, if available)
        loss_sum = torch.tensor(0.0, dtype=torch.float32)
        for track_name in set(y_pred.keys()).intersection({"total", "control"}):
            # 1. compute loss
            loss = self.loss_fn[f"{partition}_{track_name}"](
                y[track_name], y_pred[track_name]
            )
            # 2. log loss
            self.log(
                f"loss/{partition}_{track_name}",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )
            # 3. add loss to total loss
            loss_sum += loss

        return loss_sum

    def compute_and_log_metics(self, y, y_pred, partition=None):
        # for name, metric in self.metrics[partition].items():
        #     metric(y, y_pred)
        #     self.log(f'{name}/{partition}', metric, on_step=True, on_epoch=True, prog_bar=False)

        for track_name in set(y_pred.keys()).intersection({"total", "control"}):
            for metric_name, metric in self.metrics[
                f"{partition}_{track_name}"
            ].items():
                metric(y[track_name], y_pred[track_name])
                self.log(
                    f"{metric_name}/{partition}_{track_name}",
                    metric,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                )


# %%
def _make_loggers(output_path, loggers):
    if loggers is None:
        return []
    return [logger(save_dir=output_path, name="", version="") for logger in loggers]


def _make_callbacks(output_path, validation=False):
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=output_path / "checkpoints",
            every_n_epochs=1,
            save_last=True,
            save_top_k=1,
        ),
        pl.callbacks.LearningRateMonitor("step", log_momentum=True),
    ]
    # if validation:
    #     callbacks.append(EarlyStopping('VAL/loss_epoch', patience=15, verbose=True))
    return callbacks


# %%
@gin.configurable(denylist=["tfds_filepath", "output_path"])
def train(
    tfds_filepath,
    output_path,
    dataset=TFDSDataset,
    model=None,
    loggers=None,
    loss=MultinomialNLLLossFromLogits,
    metrics=None,
    optimizer=torch.optim.Adam,
    batch_size=128,
    use_control=False,
    shuffle=None,  # Shuffle is handled by TFDS. Any value >0 will enable 'shuffle_files' in TFDS and call 'ds.shuffle(x)' on the dataset.
    **kwargs,
):

    if shuffle is None:
        logging.warning(
            "'shuffle' is None. This will result in no shuffling of the dataset during training. To shuffle the dataset, set shuffle > 0."
        )
    else:
        logging.info(f"Shuffling dataset with buffer size {shuffle}.")

    # wrap model in LightningModule
    lightning_model = LightningModel(
        model, loss=loss, metrics=metrics, optimizer=optimizer, use_control=use_control
    )

    train_loader = torch.utils.data.DataLoader(
        dataset(tfds_filepath, split="train", shuffle=shuffle), batch_size=batch_size
    )
    val_loader = torch.utils.data.DataLoader(
        dataset(tfds_filepath, split="validation"), batch_size=batch_size
    )

    trainer = pl.Trainer(
        default_root_dir=output_path,
        logger=_make_loggers(output_path, loggers),
        callbacks=_make_callbacks(output_path, validation=(val_loader is not None)),
        **kwargs,
    )

    # write model summary
    with open(str(output_path / "model.summary.txt"), "w") as f:
        print(str(lightning_model), file=f)

    # write optimizer summary
    with open(str(output_path / "optim.summary.txt"), "w") as f:
        print(str(lightning_model.configure_optimizers()), file=f)

    # fit the model
    trainer.fit(
        lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    # save the torch (not pytorch-lightning) model
    torch.save(lightning_model.model, output_path / "model.pt")


# %%
@click.command()
@click.argument("tfds", required=True, type=str)
@click.option("--config", type=str, default=None)
@click.option("--log-level", type=str, default="WARNING")
@click.option("-o", "--output", required=True)
def main(tfds, config, log_level, output):
    # set log level
    logging.basicConfig(level=log_level)

    # ignore torch warnings
    warnings.filterwarnings("ignore")

    # parse gin-config config
    if config is not None:
        gin.parse_config_file(config)

    # create output directory
    output_path = Path(f'{output}/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    if output_path.exists():
        logging.warning(f'Output path {output_path} already exists. Overwriting.')
    output_path.mkdir(parents=True, exist_ok=True)

    # copy gin config
    if config is not None:
        shutil.copy(config, str(output_path / "config.gin"))

    # launch training (parameters are configured exclusively via gin)
    train(tfds, output_path)
