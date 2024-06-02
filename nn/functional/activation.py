from torch.nn import functional as F

from torchsparse import SparseTensor
from torchsparse.nn.utils import post_fapply

__all__ = ["relu", "silu", "leaky_relu"]


def relu(input: SparseTensor, inplace: bool = True) -> SparseTensor:
    feats = F.relu(input.feats, inplace=inplace)
    return post_fapply(input, feats)


def silu(input: SparseTensor, inplace: bool = True) -> SparseTensor:
    feats = F.silu(input.feats, inplace=inplace)
    return post_fapply(input, feats)


def leaky_relu(
    input: SparseTensor, negative_slope: float = 0.1, inplace: bool = True
) -> SparseTensor:
    feats = F.leaky_relu(input.feats, negative_slope=negative_slope, inplace=inplace)
    return post_fapply(input, feats)
