import torch
import os
import typing


def save_checkpoint(
    model: torch.nn.Module, 
    optimizer:torch.optim.Optimizer,
    iteration:int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    checkpoint = {
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration':iteration,
    }
    torch.save(checkpoint,out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
    model:torch.nn.Module,
    optimizer:torch.optim.Optimizer,
):
    # 使用 map_location='cpu' 是一个好习惯，可以防止在没有 GPU 的机器上加载时报错
    chekpoint = torch.load(src,map_location="cpu")
    model.load_state_dict(chekpoint['model_state_dict'])
    optimizer.load_state_dict(chekpoint['optimizer_state_dict'])
    return chekpoint['iteration']

    