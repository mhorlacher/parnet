import sys
import logging

import gin
import torch
import torch.nn as nn


@gin.configurable()
class MultinomialNLLLoss(nn.Module):
    def __init__(self, min_height=3):
        super().__init__()
        self.min_height = min_height

    def forward(self, counts: torch.Tensor, logits: torch.Tensor):
        # counts: (batch_size, num_tasks, sequence_length)
        # logits: (batch_size, num_tasks, sequence_length)
        logging.debug(f'{counts.shape=}, {logits.shape=}')
        assert counts.shape == logits.shape, 'Shapes of counts and logits do not match.'

        # Check which samples/tasks have a minimum height.
        min_height_map = counts.max(dim=-1).values >= self.min_height

        # check for NaN/Info in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(logits, file=sys.stderr, flush=True)
            torch.save(logits, 'logits_with_NaN_or_Inf.pt')
            raise ValueError('Logits contain NaN or Inf values.')

        # Compute negative log-likelihoods for each batch/task combination.
        # (batch_size * num_tasks, )
        nll = -torch.distributions.Multinomial(logits=logits.view(-1, logits.shape[-1]), validate_args=False).log_prob(
            counts.view(-1, counts.shape[-1])
        )
        logging.debug(f'{nll.shape=}')

        # Reshape to (batch_size, num_tasks)
        nll = nll.view(counts.shape[0], counts.shape[1])
        logging.debug(f'{nll.shape=}')

        # We want to dampen the loss contributions for samples where an excessive number of
        # tasks have signal, as these sequences likely habor strong CLIP biases or come from
        # highly expressed transcripts.
        # For each sample, we average the loss over tasks that reach the minimum height. That is, the
        # loss for each sample is the mean over tasks that have significant amount of signal for that sample.
        # The model with thereby up-weight cases were a single task has a lot of signal.

        # Mask out losses for samples/tasks that do not reach the minimum height.
        # (batch_size, num_tasks)
        nll = nll * min_height_map.float()

        # For each sample, take mean over tasks that reach minimum height.
        # (batch_size, )
        nll = nll.sum(dim=-1) / min_height_map.float().sum(dim=-1).clamp(min=1.0)  # we clamp to avoid division by zero
        logging.debug(f'{nll.shape=}')

        # This weighting also has some issues. For instance, a single task with high signal will on average
        # have higher loss than many tasks with medium signal. Do we really want this?
        # For that reason, we up-weight the loss of each sample by a factors that is proportional to the number of
        # tasks that have a minimum height of at least min_height. We don't want to use linear weighting,
        # as this would lead to a very high loss for samples with many tasks that have a minimum height.
        # Instead, we use a log2-scaling of the factors.
        # (batch_size, )
        nll = nll * torch.log2(min_height_map.float().sum(dim=-1) + 1.0)
        logging.debug(f'{nll.shape=}, {nll.min()=}, {nll.max()=}, {nll.mean()=}, {nll.std()=}')

        # Finally, we take the mean over all samples.
        nll_avg = nll.mean()

        # assert that nll is not NaN or Inf
        if torch.isnan(nll_avg) or torch.isinf(nll_avg):
            print(f'logits: {logits}', file=sys.stderr, flush=True)
            print(f'counts: {counts}', file=sys.stderr, flush=True)
            print(f'min_height_map: {min_height_map}', file=sys.stderr, flush=True)
            print(f'nll: {nll}', file=sys.stderr, flush=True)
            raise ValueError(f'{nll_avg} is NaN or Inf.')

        return nll_avg, min_height_map
