import torch
import math
from torch import nn
from collections.abc import Iterable

def fn_softmax(x: torch.Tensor, dim:int = -1)->torch.Tensor:
    x_max = torch.max(x,dim = dim,keepdim=True).values
    # for numerical stability
    x_stable = x-x_max

    exp_x = torch.exp(x_stable)
    sum_exp = torch.sum(exp_x,dim = dim, keepdim=True)
    return exp_x/sum_exp

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    log_normalizer = torch.logsumexp(logits, dim=-1)
    target_logits = logits.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    losses = log_normalizer - target_logits
    # for a sequence
    loss = losses.mean()
    # perplexity = torch.exp(loss)
    return loss

@torch.no_grad()
def gradient_clipping(
    params: Iterable[nn.Parameter],
    max_grad: float,
    eps = 1e-6,
):
    if max_grad < 0:
        raise ValueError("max_grad must be non-negative")

    params = [p for p in params if p.grad is not None]
    device = params[0].grad.device
    tot_grad_square = torch.zeros(
        (),
        device=device,
        dtype=torch.float32
    )

    for p in params:
        grad = p.grad.detach().float()
        tot_grad_square += grad.pow(2).sum()

    tot_grad_norm = tot_grad_square.sqrt()
    if tot_grad_norm > max_grad:
        clipping = max_grad / (tot_grad_norm + eps)
        for p in params:
            p.grad.mul_(
                clipping.to(
                    device=p.grad.device,
                    dtype=p.grad.dtype,
                )
            )
    return tot_grad_norm
