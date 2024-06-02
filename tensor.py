from __future__ import annotations
from typing import List, Union, Optional

import torch

from torchsparse.utils import make_nlist, to_dense
from torchsparse.utils.tensor_cache import TensorCache


__all__ = ["SparseTensor"]

_allow_negative_coordinates = False


def get_allow_negative_coordinates():
    return False
    # global _allow_negative_coordinates
    # return _allow_negative_coordinates


def set_allow_negative_coordinates(allow_negative_coordinates):
    raise RuntimeError("Cannot set flag globally.")
    global _allow_negative_coordinates
    _allow_negative_coordinates = allow_negative_coordinates


class SparseTensor:
    def __init__(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        stride: Union[int, List[int], torch.Tensor] = 1,
        spatial_range: Optional[Union[int, List[int], torch.Tensor]] = None,
    ) -> None:
        self.feats = feats
        self.coords = coords
        self.stride = make_nlist(stride, ndim=3)
        self.spatial_range = make_nlist(spatial_range) if spatial_range is not None else None
        self._caches = TensorCache()

    @property
    def F(self) -> torch.Tensor:
        return self.feats

    @F.setter
    def F(self, feats: torch.Tensor) -> None:
        self.feats = feats

    @property
    def C(self) -> torch.Tensor:
        return self.coords

    @C.setter
    def C(self, coords: torch.Tensor) -> None:
        self.coords = coords

    @property
    def s(self) -> List[int]:
        return self.stride

    @property
    def s_str(self) -> str:
        return "_".join(str(k) for k in self.stride)

    @s.setter
    def s(self, stride: Union[int, List[int]]) -> None:
        self.stride = make_nlist(stride, ndim=3)

    def cpu(self):
        self.coords = self.coords.cpu()
        self.feats = self.feats.cpu()
        return self

    def cuda(self):
        self.coords = self.coords.cuda()
        self.feats = self.feats.cuda()
        return self

    def half(self):
        self.feats = self.feats.half()
        return self

    def detach(self):
        self.coords = self.coords.detach()
        self.feats = self.feats.detach()
        return self

    def to(self, device, non_blocking: bool = True):
        self.coords = self.coords.to(device, non_blocking=non_blocking)
        self.feats = self.feats.to(device, non_blocking=non_blocking)
        return self

    def dense(self):
        # assert self.spatial_range is not None
        spatial_range = self.spatial_range
        assert torch.jit.isinstance(spatial_range, List[int])
        return to_dense(self.feats, self.coords, spatial_range)

    def __add__(self, other: SparseTensor):
        output = SparseTensor(
            coords=self.coords,
            feats=self.feats + other.feats,
            stride=self.stride,
            spatial_range=self.spatial_range,
        )
        output._caches = self._caches
        return output
