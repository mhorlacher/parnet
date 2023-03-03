# %%
import gin
import torch
import torchmetrics

# %%
def softmax(x, dim=-1):
    return torch.softmax(x, dim=dim)

# %%
def pearson_corrcoef(x, y, dim=-1):
    x = x - torch.unsqueeze(torch.mean(x, dim), dim)
    y = y - torch.unsqueeze(torch.mean(y, dim), dim)
    return torch.sum(x * y, dim) / torch.sqrt(torch.sum(x ** 2, dim) * torch.sum(y ** 2, dim))

# %%
@gin.configurable()
class PearsonCorrCoeff(torchmetrics.MeanMetric):
    def __init__(self, dim=-1, postproc_fn=softmax, reduction=torch.mean, *args, **kwargs):
        super(PearsonCorrCoeff,  self).__init__(*args, **kwargs)

        self.dim = dim
        self.postproc_fn = postproc_fn
        self.reduction = reduction

    def update(self, y: torch.Tensor, y_pred: torch.Tensor):
        assert y.shape == y_pred.shape

        if self.postproc_fn is not None:
            y_pred = self.postproc_fn(y_pred)
        
        pcc = pearson_corrcoef(y, y_pred, dim=self.dim)
        pcc = torch.nan_to_num(pcc, 0.0) # replace nan's with 0 (this might underestimate the pcc)

        reduced_pcc = self.reduction()

        # update (i.e. take mean)
        super().update(reduced_pcc)

# %%
@gin.configurable()
class FilteredPearsonCorrCoeff(torchmetrics.MeanMetric):
    def __init__(self, min_height=2, min_count=2, dim=-1, postproc_fn=softmax, *args, **kwargs):
        super(FilteredPearsonCorrCoeff,  self).__init__(*args, **kwargs)

        self.min_height = min_height
        self.min_count = min_count
        self.dim = dim
        self.postproc_fn = postproc_fn

    def update(self, y: torch.Tensor, y_pred: torch.Tensor):
        assert y.shape == y_pred.shape

        if self.postproc_fn is not None:
            y_pred = self.postproc_fn(y_pred)
        
        pcc = pearson_corrcoef(y, y_pred, dim=self.dim)
        mean_pcc = self.compute_mean(pcc, y)

        # update (i.e. take mean)
        super().update(mean_pcc)

    def compute_mean(self, values: torch.Tensor, y: torch.Tensor):
        # create boolean tensor of entries that are *not* NaNs
        values_is_not_nan_mask = torch.logical_not(torch.isnan(values))
        # convert nan's to 0
        values = torch.nan_to_num(values, 0.0)

        # check if required height is reached per experiment
        if self.min_height is not None:
            # should be shape (batch_size, experiments)
            y_min_height_mask = (torch.max(y, dim=-1).values >= self.min_height)
        
        # check if required count is reached per experiment
        if self.min_count is not None:
            # should be shape (batch_size, experiments)
            y_min_count_mask = (torch.sum(y, dim=-1) >= self.min_count)
        
        # boolean mask indicating which experiment (in each batch) passed nan, heigh and count (and is thus used for the final mean PCC)
        passed_boolean_mask = torch.sum(torch.stack([values_is_not_nan_mask, y_min_height_mask, y_min_count_mask]), dim=0) > 0

        # mask out (i.e. zero) all PCC values that did not pass
        values_masked = torch.mul(values, passed_boolean_mask.to(torch.float32))

        # compute mean by only dividing by #-elements that passed
        values_mean = torch.sum(values_masked)/torch.sum(passed_boolean_mask)

        # if ignore_nan:
        #     # only divide by #-elements not NaN
        #     values_mean = torch.sum(values)/torch.sum(values_is_not_nan)
        # else:
        #     values_mean = torch.mean(values)
        
        return values_mean

# %%
# def batched_pearson_corrcoef(y_batch, y_pred_batch, reduction=torch.mean):
#     pcc = torch.stack([torchmetrics.functional.pearson_corrcoef(y_batch[i], y_pred_batch[i]) for i in range(y_batch.shape[0])])
#     if reduction is not None:
#         pcc = reduction(pcc)
#     return pcc

# %%
# class BatchedPCC(torchmetrics.MeanMetric):
#     def __init__(self):
#         super(BatchedPCC, self).__init__()

#     def update(self, y_pred: torch.Tensor, y: torch.Tensor, ignore_nan=True):
#         if y_pred.shape != y.shape:
#             raise ValueError('shapes y_pred {y_pred.shape} and y {y.shape} are not the same. ')

#         values = []
#         for i in range(y.shape[0]):
#             values.append(torchmetrics.functional.pearson_corrcoef(y[i], y_pred[i]))
#         # stack to (batch_size x ...) - at this point the shape should be (batch_size x experiments
#         values = torch.stack(values)

#         # create boolean tensor of entries that are *not* NaNs
#         if ignore_nan:
#             values_is_not_nan = torch.logical_not(torch.isnan(values))

#         # convert nan's to 0
#         values = torch.nan_to_num(values, 0.0)

#         if ignore_nan:
#             # only divide by #-elements not NaN
#             values_mean = torch.sum(values)/torch.sum(values_is_not_nan)
#         else:
#             values_mean = torch.mean(values)

#         # update
#         super().update(values_mean)

# %%
# @gin.configurable
# class BatchedPearsonCorrCoef(torchmetrics.MeanMetric):
#     def __init__(self, min_height=2, min_count=2):
#         super(BatchedPearsonCorrCoef, self).__init__()

#         self.min_height = min_height
#         self.min_count = min_count

#     def update(self, y_pred: torch.Tensor, y: torch.Tensor):
#         if y_pred.shape != y.shape:
#             raise ValueError('shapes y_pred {y_pred.shape} and y {y.shape} are not the same. ')

#         cc_values = self.compute_cc(y_pred, y)
#         cc_mean = self.compute_mean(cc_values, y)

#         # update
#         super().update(cc_mean)
    
#     def compute_cc(self, y_pred: torch.Tensor, y: torch.Tensor):
#         values = []
#         for i in range(y.shape[0]):
#             values.append(torchmetrics.functional.pearson_corrcoef(y[i], y_pred[i]))
#         # stack to (batch_size x ...) - at this point the shape should be (batch_size x experiments
#         return torch.stack(values)

#     def compute_mean(self, values: torch.Tensor, y: torch.Tensor):
#         # create boolean tensor of entries that are *not* NaNs
#         values_is_not_nan_mask = torch.logical_not(torch.isnan(values))
#         # convert nan's to 0
#         values = torch.nan_to_num(values, 0.0)

#         # check if required height is reached per experiment
#         if self.min_height is not None:
#             # should be shape (batch_size, experiments)
#             y_min_height_mask = (torch.max(y, dim=-2).values >= self.min_height)
#         else:
#             y_min_height_mask = torch.ones(*values.shape) # TODO: Remove! This will lead to y_min_height_mask potentially being on a different device than y. 
        
#         # check if required count is reached per experiment
#         if self.min_count is not None:
#             # should be shape (batch_size, experiments)
#             y_min_count_mask = (torch.sum(y, dim=-2) >= self.min_count) # TODO: Same as above. 
#         else:
#             y_min_count_mask = torch.ones(*values.shape)
        
#         # boolean mask indicating which experiment (in each batch) passed nan, heigh and count (and is thus used for the final mean PCC)
#         passed_boolean_mask = torch.sum(torch.stack([values_is_not_nan_mask, y_min_height_mask, y_min_count_mask]), dim=0) > 0

#         # mask out (i.e. zero) all PCC values that did not pass
#         values_masked = torch.mul(values, passed_boolean_mask.to(torch.float32))

#         # compute mean by only dividing by #-elements that passed
#         values_mean = torch.sum(values_masked)/torch.sum(passed_boolean_mask)

#         # if ignore_nan:
#         #     # only divide by #-elements not NaN
#         #     values_mean = torch.sum(values)/torch.sum(values_is_not_nan)
#         # else:
#         #     values_mean = torch.mean(values)
        
#         return values_mean

# %%
# @gin.configurable()
# class BatchedPearsonCorrCoef(torchmetrics.MeanMetric):
#     def __init__(self, min_height=2, min_count=2):
#         super(BatchedPearsonCorrCoef, self).__init__()

#         self.min_height = min_height
#         self.min_count = min_count

#     def update(self, y_pred: torch.Tensor, y: torch.Tensor):
#         if y_pred.shape != y.shape:
#             raise ValueError('shapes y_pred {y_pred.shape} and y {y.shape} are not the same. ')

#         values = []
#         for i in range(y.shape[0]):
#             values.append(torchmetrics.functional.pearson_corrcoef(y[i], y_pred[i]))
#         # stack to (batch_size x ...) - at this point the shape should be (batch_size x experiments
#         values = torch.stack(values)

#         # create boolean tensor of entries that are *not* NaNs
#         values_is_not_nan_mask = torch.logical_not(torch.isnan(values))
#         # convert nan's to 0
#         values = torch.nan_to_num(values, 0.0)

#         # check if required height is reached per experiment
#         if self.min_height is not None:
#             # should be shape (batch_size, experiments)
#             y_min_height_mask = (torch.max(y, dim=-2).values >= self.min_height)
#         else:
#             y_min_height_mask = torch.ones(*values.shape)
        
#         # check if required count is reached per experiment
#         if self.min_count is not None:
#             # should be shape (batch_size, experiments)
#             y_min_count_mask = (torch.sum(y, dim=-2) >= self.min_count)
#         else:
#             y_min_count_mask = torch.ones(*values.shape)
        
#         # boolean mask indicating which experiment (in each batch) passed nan, heigh and count (and is thus used for the final mean PCC)
#         passed_boolean_mask = torch.sum(torch.stack([values_is_not_nan_mask, y_min_height_mask, y_min_count_mask]), dim=0) > 0
#         # passed_boolean_mask = torch.sum(torch.stack([values_is_not_nan_mask, y_min_height_mask]), dim=0) > 0

#         # mask out (i.e. zero) all PCC values that did not pass
#         values = torch.mul(values, passed_boolean_mask.to(torch.float32))

#         # compute mean by only dividing by #-elements that passed
#         values_mean = torch.sum(values)/torch.sum(passed_boolean_mask)

#         # update
#         super().update(values_mean)


# %%
# @gin.configurable()
# class MultinomialNLLLossFromLogitsMetric(torchmetrics.MeanMetric):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.loss_fn = MultinomialNLLLossFromLogits()

#     def update(self, y_pred: torch.Tensor, y: torch.Tensor):
#         assert y_pred.shape == y.shape

#         loss = self.loss_fn(y, y_pred)

#         # update (i.e. take mean)
#         super().update(loss)

# %%
# class BatchIdx(torchmetrics.MeanMetric):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
    
#     def update(self, batch_idx: torch.Tensor):
#         # update (i.e. take mean)
#         super().update(batch_idx)
