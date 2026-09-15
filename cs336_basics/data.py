import numpy.typing as npt
import numpy as np
import torch

# 模块化， 保证每次采样生成不同的随机数
_rng = np.random.default_rng(42)
def get_batch(
    x: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str
)-> tuple[torch.Tensor,torch.Tensor]:
    n = len(x)

    starts = _rng.integers(
        low=0,
        high=n - context_length,
        size=batch_size,
    )
    inputs_np = np.stack([
        x[i:i+context_length]
        for i in starts
    ])
    targets_np = np.stack([
        x[i+1: i+context_length+1]
        for i in starts
    ])
    inputs = torch.from_numpy(inputs_np).to(
        device=device,
        dtype=torch.long,
    )

    targets = torch.from_numpy(targets_np).to(
        device=device,
        dtype=torch.long,
    )

    return inputs, targets
