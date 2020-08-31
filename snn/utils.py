import torch


def threshold(input, thr):
    return torch.where(
        input <= torch.ones_like(input) * thr,
        torch.zeros_like(input),
        torch.ones_like(input),
    )
