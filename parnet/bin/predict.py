import pathlib

import tqdm
import click
import torch
from Bio import SeqIO


@click.command()
@click.argument('fasta', type=click.Path(exists=True))
@click.option('-m', '--model', type=click.Path(exists=True))
@click.option('-o', '--output-directory', type=click.Path())
def main(fasta, model, output_directory):
    model = torch.load(model)

    # make output directory
    pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)

    for record in tqdm.tqdm(SeqIO.parse(fasta, 'fasta')):
        pred = model.predict_from_sequence(str(record.seq).upper())

        # save each prediction as a dedicated .pt file
        torch.save(pred, pathlib.Path(output_directory) / f'{record.id}.parnet.pt')


# %%
if __name__ == '__main__':
    main()
