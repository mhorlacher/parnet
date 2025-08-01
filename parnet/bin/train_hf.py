from pathlib import Path
import logging

import torch
import click
import gin
from transformers import Trainer, TrainingArguments
import torchmetrics

METRICS = torchmetrics.MetricCollection({
    'MSE': torchmetrics.MeanSquaredError(),
    'MAE': torchmetrics.MeanAbsoluteError(),
    'R2': torchmetrics.R2Score(),
    # 'Pearson': torchmetrics.PearsonCorrCoef(),
})

from tomix.modeling import TomixModel, TomixConfig


def compute_metrics(eval_pred):
    y_pred, y = eval_pred
    return METRICS(torch.tensor(y_pred).float(), torch.tensor(y).long())


@gin.configurable(denylist=['output', 'model_config'])
def train(
    output,
    model_config_json=None,
    dataset_cls=None,
    model_cls=TomixModel,
    model_config_cls=TomixConfig,
    collate_fn=default_collate_fn,
    batch_size=32,
    mlflow_exp=None,
    **trainer_kwargs,
):
    if dataset_cls is None:
        raise ValueError('dataset_cls must be specified')
    if model_cls is None:
        raise ValueError('model_cls must be specified')

    # create model
    if model_config_json is not None:
        config = model_config_cls.from_pretrained(model_config_json)
    model = model_cls(config)

    # load dataset
    dataset_train = dataset_cls(split='train')  # train
    dataset_valid = dataset_cls(split='valid')  # valid
    # TODO: Are those shuffled with standard Trainer/DataLoader?

    # with torch.no_grad():
    #     # dummy forward pass to init lazy modules (if present)
    #     _ = model(**dataset_train[0])

    print(model)
    print(f'Trainable parameters: {count_trainable_params(model):,}')

    if mlflow_exp is not None:
        mlflow.set_tracking_uri('https://dev.mlflow.powerml.int.bayer.com/')
        mlflow.set_experiment(mlflow_exp)
        mlflow.start_run()

    # load trainer
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output,
            logging_dir=output / 'logs',
            report_to='mlflow' if mlflow_exp is not None else 'tensorboard',
            run_name=output.name,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            remove_unused_columns=False,
            **trainer_kwargs,
        ),
        train_dataset=dataset_train,
        eval_dataset=dataset_valid,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        # TODO: Try different learning rates
        optimizers=(torch.optim.AdamW(model.parameters(), lr=4e-4), None),
    )

    trainer.train()
    # NOTE: The warning "Could not estimate the number of tokens of the input, floating-point operations will not be computed"
    # can be safely ignored, it's simple for computing total FLOPS per training run.
    # See: https://discuss.huggingface.co/t/get-warning-could-not-estimate-the-number-of-tokens-of-the-input-floating-point-operations-will-not-be-computed-when-use-a-customize-trainer-and-customize-data-collator/18517


@click.command()
@click.option('--train-config', type=str, default=None, help='Path to the training configuration file (in gin format).')
@click.option(
    '--model-config', type=str, default=None, help='Path to the huggingface model configuration file (in JSON format).'
)
@click.option('-o', '--output', type=str, required=True, help='Output directory to save checkpoints and logs.')
@click.option('--log-level', type=str, default='WARNING')
def main(train_config: str, model_config: str, output: str, log_level: str):
    output = Path(output)
    output.mkdir(exist_ok=True)

    logging.getLogger().setLevel(log_level)

    # copy config file to output directory
    if train_config is not None:
        (output / 'config.gin').write_text(Path(train_config).read_text())
    # parse config
    gin.parse_config_file(train_config)

    train(output, model_config)


if __name__ == '__main__':
    main()
