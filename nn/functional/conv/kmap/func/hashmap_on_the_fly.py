from typing import Any, Dict, List, Optional
import torch

import torchsparse.backends
from torchsparse.utils import make_tensor


def build_kmap_implicit_GEMM_hashmap_on_the_fly(
    kmap: Dict[str, Any],
    input_node_num: int,
    _coords: torch.Tensor,
    kernel_size: torch.Tensor,
    stride: torch.Tensor,
    padding: torch.Tensor,
    spatial_range: Optional[List[int]] = None,
    cta_M: int = 128,
    subm: bool = False,
    ifsort: bool = False,
    split_mask_num: int = 1,
) -> Dict[str, Any]:
    kmap["coords"] = _coords
    kmap["spatial_range"] = spatial_range
    # coords = _coords[:, [3, 0, 1, 2]]
    coords = _coords.contiguous()
    if spatial_range is not None:
        coords_max = make_tensor(
            spatial_range, dtype=torch.int, device=coords.device
        ) - 1
    else:
        coords_max = coords.max(0).values
        if not subm:
            coords_max[1:] = (
                coords_max[1:] + 2 * padding - (kernel_size - 1)
            ) // stride

    if torchsparse.tensor.get_allow_negative_coordinates():
        coords_min = coords.min(0).values
        coords_min[1:] = torch.div(
            coords_min[1:] - 2 * padding + (kernel_size - 1), stride
        )
    else:
        coords_min = make_tensor((0, 0, 0, 0), dtype=torch.int, device=coords.device)

    assert (
        torchsparse.backends.hash_rsv_ratio >= 2
    ), f"hash_rsv_ratio should be no less than 2, now {torchsparse.backends.hash_rsv_ratio}."
    hashmap_capacity = max(
        512, int(torchsparse.backends.hash_rsv_ratio * _coords.shape[0])
    )
    to_insert = True
    hashmap_keys = torch.zeros(
        hashmap_capacity, dtype=torch.int64, device=coords.device
    )
    kmap_hashmap_keys = kmap["hashmap_keys"]
    if torch.jit.isinstance(kmap_hashmap_keys, torch.Tensor):
        # assert torch.jit.isinstance(kmap_hashmap_keys, torch.Tensor)
        hashmap_keys = kmap_hashmap_keys
        to_insert = False
    hashmap_vals = torch.zeros(
        hashmap_capacity, dtype=torch.int32, device=coords.device
    )
    kmap_hashmap_vals = kmap["hashmap_vals"]
    if torch.jit.isinstance(kmap_hashmap_vals, torch.Tensor):
        # assert torch.jit.isinstance(kmap_hashmap_vals, torch.Tensor)
        hashmap_vals = kmap_hashmap_vals

    assert torch.jit.isinstance(hashmap_keys, torch.Tensor)
    assert torch.jit.isinstance(hashmap_vals, torch.Tensor)
    hashtable = torch.classes.torchsparse_ops.GPUHashTable(
        hashmap_keys, hashmap_vals
    )

    if subm:
        out = torch.ops.torchsparse_ops.build_kernel_map_subm_hashmap(
            hashtable, coords, coords_min, coords_max, kernel_size, stride, padding, to_insert
        )
    else:
        out = torch.ops.torchsparse_ops.build_kernel_map_downsample_hashmap(
            hashtable, coords, coords_min, coords_max, kernel_size, stride, padding, to_insert
        )

    # update kernel_map
    out_in_map = out[0]
    kmap["out_in_map"] = out_in_map
    if len(out) != 1:
        coords = out[1]
        # coords = coords[:, [1, 2, 3, 0]]
        kmap["coords"] = coords
    kmap["sizes"] = (input_node_num, coords.shape[0])

    if ifsort:
        bitmask = torch.ops.torchsparse_ops.derive_bitmask_from_out_in_map(
            out_in_map, split_mask_num, coords.shape[0]
        )
        sorted_mask, reorder_loc = torch.sort(bitmask, descending=True)
        reorder_loc = reorder_loc.to(torch.int32)
        reorder_out_in_map = torch.ops.torchsparse_ops.reorder_out_in_map_cuda(
            out_in_map, reorder_loc
        )
        reduced_sorted_mask = torch.ops.torchsparse_ops.reduce_bitmask_cuda(
            sorted_mask, cta_M
        )
        kmap["reorder_out_in_map"] = reorder_out_in_map
        kmap["reduced_sorted_mask"] = reduced_sorted_mask
        kmap["reorder_loc"] = reorder_loc
        kmap["sorted_mask"] = sorted_mask

    return kmap


def build_kmap_Gather_Scatter_hashmap_on_the_fly(
    kmap: Dict[str, Any],
    input_node_num: int,
    _coords: torch.Tensor,
    kernel_size: torch.Tensor,
    stride: torch.Tensor,
    padding: torch.Tensor,
    spatial_range: Optional[List[int]] = None,
    cta_M: int = 128,
    subm: bool = False,
) -> Dict[str, Any]:

    kmap = build_kmap_implicit_GEMM_hashmap_on_the_fly(
        kmap,
        input_node_num,
        _coords,
        kernel_size,
        stride,
        padding,
        spatial_range,
        cta_M,
        subm,
        False,
        1,
    )
    out_in_map = kmap["out_in_map"]
    assert torch.jit.isinstance(out_in_map, torch.Tensor)
    results = torch.t(out_in_map).contiguous()
    nbsizes = torch.sum(results != -1, dim=1)
    nbmaps = torch.nonzero(results != -1)
    nbmaps[:, 0] = results.view(-1)[nbmaps[:, 0] * results.size(1) + nbmaps[:, 1]]
    # important for build masks
    nbmaps = nbmaps.contiguous()
    coords = kmap["coords"]
    assert torch.jit.isinstance(coords, torch.Tensor)
    input_mask, output_mask = torch.ops.torchsparse_ops.build_mask_from_kmap(
        _coords.shape[0],
        coords.shape[0],
        nbmaps.int(),
        nbsizes.int()[0 : coords.shape[0]],
    )

    kmap["nbmaps"] = nbmaps
    kmap["nbsizes"] = nbsizes
    kmap["input_mask"] = input_mask
    kmap["output_mask"] = output_mask

    return kmap


def build_kmap_Fetch_on_Demand_hashmap_on_the_fly(
    kmap: Dict[str, Any],
    input_node_num: int,
    _coords: torch.Tensor,
    kernel_size: torch.Tensor,
    stride: torch.Tensor,
    padding: torch.Tensor,
    spatial_range: Optional[List[int]] = None,
    cta_M: int = 128,
    subm: bool = False,
) -> Dict[str, Any]:

    kmap = build_kmap_implicit_GEMM_hashmap_on_the_fly(
        kmap,
        input_node_num,
        _coords,
        kernel_size,
        stride,
        padding,
        spatial_range,
        cta_M,
        subm,
        False,
        1,
    )

    out_in_map = kmap["out_in_map"]
    assert torch.jit.isinstance(out_in_map, torch.Tensor)
    results = torch.t(out_in_map).contiguous()
    nbsizes = torch.sum(results != -1, dim=1).to(torch.int)
    nbmaps = torch.nonzero(results != -1)
    nbmaps[:, 0] = results.view(-1)[nbmaps[:, 0] * results.size(1) + nbmaps[:, 1]]

    kernel_volume = nbsizes.size(0)
    nbaddrs = torch.zeros((kernel_volume + 1), dtype=torch.int, device=nbmaps.device)
    qnbaddrs = torch.zeros((kernel_volume + 1), dtype=torch.int, device=nbmaps.device)

    # Derive quantified arrays
    torch.ops.torchsparse_ops.exclusive_scan_quantified_wrapper(
        kernel_volume, nbsizes, nbaddrs, qnbaddrs
    )

    # nbmaps need to be transposed for Fetch-on-Demand
    kmap["nbmaps"] = nbmaps.transpose(0, 1).int()
    kmap["nbsizes"] = nbsizes

    kmap["nbaddrs"] = nbaddrs
    kmap["qnbaddrs"] = qnbaddrs
    kmap["qmapsize"] = qnbaddrs[-1].cpu().int()

    return kmap
