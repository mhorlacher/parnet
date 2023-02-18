# %%
import torch
import torchmetrics

from .losses import MultinomialNLLLossFromLogits

# %%
def batched_pearson_corrcoef(y_batch, y_pred_batch, reduction=torch.mean):
    pcc = torch.sum(torch.stack([torchmetrics.functional.pearson_corrcoef(y_batch[i], y_pred_batch[i]) for i in range(y_batch.shape[0])]), dim=0)
    if reduction is not None:
        pcc = reduction(pcc)
    return pcc

# %%
class BatchedPCC(torchmetrics.MeanMetric):
    def __init__(self):
        super(BatchedPCC, self).__init__()

    def update(self, y_pred: torch.Tensor, y: torch.Tensor, ignore_nan=True):
        assert y_pred.shape == y.shape

        values = []
        for i in range(y.shape[0]):
            values.append(torchmetrics.functional.pearson_corrcoef(y[i], y_pred[i]))
        # stack to (batch_size x ...) - at this point the shape should be (batch_size x experiments
        values = torch.stack(values)

        # create boolean tensor of entries that are *not* NaNs
        if ignore_nan:
            values_is_not_nan = torch.logical_not(torch.isnan(values))

        # convert nan's to 0
        values = torch.nan_to_num(values, 0.0)

        if ignore_nan:
            # only divide by #-elements not NaN
            values_mean = torch.sum(values)/torch.sum(values_is_not_nan)
        else:
            values_mean = torch.mean(values)

        # update
        super().update(values_mean)

# %%
class MultinomialNLLFromLogits(torchmetrics.MeanMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = MultinomialNLLLossFromLogits()

    def update(self, y_pred: torch.Tensor, y: torch.Tensor):
        assert y_pred.shape == y.shape

        loss = self.loss_fn(y, y_pred)

        # update (i.e. take mean)
        super().update(loss)

# %%
# class BatchIdx(torchmetrics.MeanMetric):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
    
#     def update(self, batch_idx: torch.Tensor):
#         # update (i.e. take mean)
#         super().update(batch_idx)
