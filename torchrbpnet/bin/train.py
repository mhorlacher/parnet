# %%
import gin
import click

# %%
@gin.configurable()
def train():
    pass

# %%
@click.command()
@click.argument('tfrecord', required=True, type=str)
@click.option('--config', type=str, default=None)
@click.option('-o', '--output', required=True)
@click.option('--validation-tfrecord', type=str, default=None)
def main(tfrecord, config, output, validation_tfrecord):
    pass