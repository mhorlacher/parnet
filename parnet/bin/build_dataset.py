import click
import tensorflow as tf
import tensorflow_datasets as tfds

from parnet.data.bioIO import DataSpec, Bed


def build_tf_dataset(bed, dataspec, target_size, min_height):
    ds = tf.data.Dataset.from_generator(
        lambda: (
            dataspec(row["chrom"], row["start"], row["end"], row["strand"], target_size)
            for row in Bed(bed)
        ),
        output_signature=dataspec.tf_signature,
    )

    # filter dataset (require that at least one task has 'min_height' counts at some position along the sequence)
    ds.filter(
        lambda example: tf.reduce_max(example["outputs"]["eCLIP"])
        >= tf.constant(min_height, dtype=tf.float32)
    )

    return ds


@click.command()
# @click.argument('bed', required=True, type=str)
@click.option("--bed-train", type=str, required=True)
@click.option("--bed-val", type=str, default=None)
@click.option("--bed-test", type=str, default=None)
@click.option("--dataspec", type=str, required=True)
@click.option("--target-size", type=int, required=True)
@click.option("--control/--no-control", default=False)
@click.option("-o", "--output-directory", type=str, required=True)
@click.option("--min-height", type=int, default=2)
def main(
    bed_train,
    bed_val,
    bed_test,
    dataspec,
    target_size,
    control,
    output_directory,
    min_height,
):
    # load dataspec
    dataspec = DataSpec(dataspec, control=control)

    split_datasets = {
        "train": build_tf_dataset(bed_train, dataspec, target_size, min_height)
    }
    if bed_val is not None:
        split_datasets["validation"] = build_tf_dataset(
            bed_val, dataspec, target_size, min_height
        )
    if bed_test is not None:
        split_datasets["test"] = build_tf_dataset(
            bed_test, dataspec, target_size, min_height
        )

    # write dataset to disk
    tfds.dataset_builders.store_as_tfds_dataset(
        name="parnet_dataset",
        version="0.0.1",
        split_datasets=split_datasets,
        features=dataspec.tfds_features,
        data_dir=output_directory,
    )
