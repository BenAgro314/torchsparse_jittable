from typing import Callable

import torch

from torchsparse import SparseTensor

__all__ = ['post_fapply']


def post_fapply(input: SparseTensor, feats: torch.Tensor) -> SparseTensor:
    output = SparseTensor(
        coords=input.coords,
        feats=feats,
        stride=input.stride,
        spatial_range=input.spatial_range,
    )
    output._caches = input._caches
    return output
