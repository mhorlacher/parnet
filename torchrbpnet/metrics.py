# %%
import torch
import torchmetrics

# %%
def batched_pearson_corrcoef(y_batch, y_pred_batch):
    return torch.sum(torch.stack([torchmetrics.functional.pearson_corrcoef(y_batch[i], y_pred_batch[i]) for i in range(y_batch.shape[0])]), dim=0)