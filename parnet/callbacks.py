import torch

from lightning.pytorch.callbacks import Callback


class MixCoeffPenaltyScheduler(Callback):
    def __init__(self, factor=1.0) -> None:
        super().__init__()
        self.penalty_factor = factor

    def setup(self, trainer, pl_module, stage):
        # self._penalty_factor = pl.module.model.head.
        pass

    def on_train_epoch_start(self, trainer, pl_module):
        pass
