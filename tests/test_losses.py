# %%
import unittest

import torch
from torch.distributions import Multinomial

from torchrbpnet.losses import MultinomialNLLLossFromLogits

# %%
def compute_manual_multinomial_nll(counts, logits):
    nll = []
    for i in range(counts.shape[0]):
        for j in range(counts.shape[2]):
            counts_ij, logits_ij = counts[i, :, j], logits[i, :, j]
            # print(Multinomial(total_count=torch.sum(single_y), logits=single_y_pred))
            nll.append(-Multinomial(int(torch.sum(counts_ij)), logits=logits_ij).log_prob(counts_ij))
    return torch.mean(torch.tensor(nll))

# %%
class TestLosses(unittest.TestCase):
    
    def test_MultinomialNLLLossFromLogits(self):
        # arrange
        y, y_pred = torch.randint(0, 10, size=(4, 42, 7)), torch.rand(4, 42, 7)
        true_nll = compute_manual_multinomial_nll(y, y_pred)

        # act
        nll = MultinomialNLLLossFromLogits(reduction=torch.mean)(y, y_pred, dim=-2)

        # assert
        assert bool(nll == true_nll)
