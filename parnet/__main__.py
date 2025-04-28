import click

from .bin import train, predict
# from .bin import build_dataset


@click.group()
def main():
    pass


main.add_command(train.main, name='train')
# main.add_command(build_dataset.main, name='build-dataset')
main.add_command(predict.main, name='predict')

if __name__ == '__main__':
    main()
