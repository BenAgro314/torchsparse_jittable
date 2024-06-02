from typing import List, Union, Optional
from functools import lru_cache
import torch

__all__ = ["make_nlist", "make_tensor", "make_divisible"]


def make_nlist(x: Union[int, List[int], torch.Tensor], ndim: Optional[int] = None) -> List[int]:
    if isinstance(x, int):
        assert torch.jit.isinstance(ndim, int)
        x = [x] * ndim
    elif isinstance(x, list):
        pass
    elif isinstance(x, torch.Tensor):
        return x.view(-1).cpu().tolist()

    assert isinstance(x, list), x
    if ndim is not None:
        assert torch.jit.isinstance(ndim, int)
        assert len(x) == ndim, ndim
    return x


# @lru_cache()
def make_tensor(x: List[int], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=dtype, device=device)


def make_divisible(x: int, divisor: int):
    return (x + divisor - 1) // divisor * divisor
