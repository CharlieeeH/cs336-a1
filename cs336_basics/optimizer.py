from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

def get_lr_cosine_schedule(
    t: int, 
    alpha_max: float,
    alpha_min: float,
    T_w: int,
    T_c: int
):
    if t < T_w:
        return t * alpha_max / T_w
    if t > T_c:
        return alpha_min
    decay_ratio = (t - T_w) / (T_c - T_w)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return alpha_min + coeff * (alpha_max - alpha_min)

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay = 0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        defaults = {
            "lr":lr,
            "betas":betas,
            "eps":eps,
            "weight_decay":weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.no_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("This AdamW does not support sparse gradients")

                if len(state)==0:
                    state['step']=0
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)

                state['step'] += 1
                t = state['step']
                m = state['m']
                v = state['v']

                # Decoupled weight decay
                p.mul_(1 - lr * weight_decay)

                # Update first and second moments
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(
                    grad,
                    grad,
                    value=1 - beta2,
                )

                # Bias correction
                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t

                step_size = lr / bias_correction1

                denominator = (
                    v.sqrt() / math.sqrt(bias_correction2)
                ).add_(eps)

                p.addcdiv_(
                    m,
                    denominator,
                    value=-step_size,
                )
        return loss




# TODO: Tuning the learning rate and observe
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss
