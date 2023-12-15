# %%
import click

# %%
from .bin import train, build_dataset

# %%
@click.group()
def main():
    pass

# %%
main.add_command(train.main, name='train')
main.add_command(build_dataset.main, name='build-dataset')

# %%
if __name__ == '__main__':
    main()