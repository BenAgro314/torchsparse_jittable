from typing import List, Union, List

import torch

from torchsparse.utils import make_nlist

__all__ = ["get_kernel_offsets"]


def get_kernel_offsets(
    size: Union[int, List[int]],
    stride: Union[int, List[int]] = 1,
    dilation: Union[int, List[int]] = 1,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    size = make_nlist(size, ndim=3)
    stride = make_nlist(stride, ndim=3)
    dilation = make_nlist(dilation, ndim=3)

    offsets = [
        (torch.arange(-size[k] // 2 + 1, size[k] // 2 + 1) * stride[k] * dilation[k])
        for k in range(3)
    ]

    offsets_data_t: List[List[int]] = []
    # This condition check is only to make sure that our weight layout is
    # compatible with `MinkowskiEngine`.
    if torch.prod(torch.tensor(size)).item() % 2 == 1:
        for z in offsets[0]:
            for y in offsets[1]:
                for x in offsets[2]:
                    offsets_data_t.append([int(x.item()), int(y.item()), int(z.item())])
    else:
        for x in offsets[0]:
            for y in offsets[1]:
                for z in offsets[2]:
                    offsets_data_t.append([int(x.item()), int(y.item()), int(z.item())])

    return torch.tensor(offsets_data_t, dtype=torch.int, device=device)
