import torch


__all__ = ["spcount"]


def spcount(coords: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
    coords = coords.contiguous()
    if coords.device.type == "cuda":
        return torch.ops.torchsparse_ops.count_cuda(coords, num)
    elif coords.device.type == "cpu":
        return torch.ops.torchsparse_ops.count_cpu(coords, num)
    else:
        device = coords.device
        return torch.ops.torchsparse_ops.count_cpu(coords.cpu(), num).to(device)
