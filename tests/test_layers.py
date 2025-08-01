import torch

from parnet.layers import LambdaLayer, NewAdditiveMix


def test_NewAdditiveMix():
    # Prepare mix-coeff layer. Provide a constant mixing coefficient and later overwrite the target/control head layers,
    # so that we can return fixed logit-tensors.
    layer = NewAdditiveMix(num_tasks=2)

    # Prepare inputs. We provide log-probs as logits, as they are easily generated from probs.
    layer.head_target = LambdaLayer(lambda _: torch.tensor([[0.1, 0.5, 0.4], [0.2, 0.3, 0.5]]).log().unsqueeze(0))
    layer.head_control = LambdaLayer(lambda _: torch.tensor([[0.2, 0.4, 0.4], [0.1, 0.8, 0.1]]).log().unsqueeze(0))
    layer.mix_coeff = LambdaLayer(lambda _: torch.tensor([0.5, 0.5]).unsqueeze(0))

    total_logprob = layer.forward(None)['total'].softmax(dim=-1)

    assert total_logprob.shape == (1, 2, 3), f'Expected shape (1, 2, 3), got {total_logprob.shape}'
    assert torch.allclose(total_logprob, torch.tensor([[0.15, 0.45, 0.40], [0.15, 0.55, 0.3]]).unsqueeze(0))
