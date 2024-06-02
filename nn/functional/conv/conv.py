from typing import Any, Dict, List, Union, Optional, Tuple

# import numpy as np
import torch

from torchsparse import SparseTensor
from torchsparse.nn.functional.conv.func.implicit_gemm import forward_implicit_gemm
# from torchsparse.nn.functional.conv.func.fetch_on_demand import forward_fetch_on_demand
# from torchsparse.nn.functional.conv.func.gather_scatter import forward_gather_scatter
from torchsparse.utils import make_nlist
from torchsparse.utils.tensor_cache import TensorCache
from torchsparse.nn import functional as F
from torchsparse.nn.functional.conv.conv_config import ConvConfig
from .func import *

__all__ = ["conv3d"]


def conv3d(
    input: SparseTensor,
    weight: torch.Tensor,
    kernel_size: Union[int, List[int]],
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, List[int]] = 1,
    padding: Union[int, List[int]] = 0,
    dilation: Union[int, List[int]] = 1,
    config: Optional[ConvConfig] = None,
    transposed: bool = False,
    generative: bool = False,
    training: bool = False,
    allow_tf32: bool = False,
    allow_fp16: bool = False,
) -> SparseTensor:

    feats, coords = input.feats, input.coords
    kernel_size = make_nlist(kernel_size, ndim=3)
    # kernel_volume = np.prod(kernel_size)
    stride = make_nlist(stride, ndim=3)
    dilation = make_nlist(dilation, ndim=3)

    conv_mode = F.get_conv_mode()
    if config is None:
        config = F.conv_config.get_default_conv_config(
            conv_mode=conv_mode, training=training
        )

    # TODO: Deal with kernel volume > 32. (Split mask or unsort)

    dataflow = config.dataflow
    kmap_mode = config.kmap_mode

    if dataflow == F.Dataflow.GatherScatter.value or dataflow == F.Dataflow.FetchOnDemand.value:
        config.ifsort = False
    elif (
        dataflow == F.Dataflow.CodedCSR.value
    ):  # Placeholder for PCEngine integration. Mode name can be modified.
        config.ifsort = False
        assert 0, "CodedCSR has not been integrated."

    if kernel_size == [1, 1, 1] and stride == [1, 1, 1] and dilation == [1, 1, 1]:
        feats = feats.matmul(weight)
        if bias is not None:
            feats += bias
        output = SparseTensor(
            coords=coords,
            feats=feats,
            stride=input.stride,
            spatial_range=input.spatial_range,
        )
    elif not transposed:
        kmap = input._caches.kmaps.get(TensorCache.make_kmaps_key((input.stride, kernel_size, stride, dilation)))

        if kmap_mode != "hashmap_on_the_fly":
            hashmap = input._caches.hashmaps.get(TensorCache.make_tuple_key(input.stride))
        else:
            hashmap = input._caches.hashmaps.get(
                TensorCache.make_tuple_key((input.stride[0] * stride[0], input.stride[1] * stride[1], input.stride[2] * stride[2]))
            )
        if hashmap is None:
            hashmap_keys, hashmap_vals = None, None
        else:
            hashmap_keys, hashmap_vals = hashmap

        spatial_range = input.spatial_range

        if kmap is None:
            kmap = F.build_kernel_map(
                coords,
                feats.shape[0],
                kernel_size,
                stride,
                padding,
                hashmap_keys,
                hashmap_vals,
                spatial_range,
                kmap_mode,
                dataflow,
                downsample_mode=config.downsample_mode,
                training=training,
                ifsort=config.ifsort,
                split_mask_num=config.split_mask_num,
                split_mask_num_bwd=config.split_mask_num_bwd,
            )

            kmap_keys = kmap["hashmap_keys"]
            kmap_vals = kmap["hashmap_vals"]
            assert torch.jit.isinstance(kmap_keys, Optional[torch.Tensor])
            assert torch.jit.isinstance(kmap_vals, Optional[torch.Tensor])
            hashmap = (kmap_keys, kmap_vals)

            input._caches.kmaps[TensorCache.make_kmaps_key((input.stride, kernel_size, stride, dilation))] = kmap
            input._caches.hashmaps[TensorCache.make_tuple_key(input.stride)] = hashmap

        if dataflow == F.Dataflow.ImplicitGEMM.value:
            if torch.jit.is_scripting():
                feats = forward_implicit_gemm(feats, weight, kmap, config, transposed, allow_tf32, allow_fp16)
            else:
                feats = ImplicitGEMMConvolutionFuntion.apply(feats, weight, kmap, config, transposed)
        elif dataflow == F.Dataflow.GatherScatter.value:
            raise NotImplementedError("GatherScatterConvolution is not torchscriptable yet")
            # if torch.jit.is_scripting():
            #     feats = forward_gather_scatter(feats, weight, kmap, config, transposed)
            # else:
            #     feats = GatherScatterConvolutionFuntion.apply(feats, weight, kmap, config, transposed)
        elif dataflow == F.Dataflow.FetchOnDemand.value:
            raise NotImplementedError("FetchOnDemand is not torchscriptable yet")
            # if torch.jit.is_scripting():
            #     feats = forward_fetch_on_demand(feats, weight, kmap, config, transposed)
            # else:
            #     feats = FetchOnDemandConvolutionFuntion.apply(feats, weight, kmap, config, transposed)

        if bias is not None:
            feats += bias
        kmap_coords = kmap["coords"]
        assert torch.jit.isinstance(kmap_coords, torch.Tensor)
        kmap_spatial_range = kmap["spatial_range"]
        assert torch.jit.isinstance(kmap_spatial_range, Optional[Union[torch.Tensor, List[int], int]])
        output = SparseTensor(
            coords=kmap_coords,
            feats=feats,
            stride=[input.stride[k] * stride[k] for k in range(3)],
            spatial_range=kmap_spatial_range,
        )
    else:
        tensor_stride = [input.stride[k] // stride[k] for k in range(3)]
        if not generative:
            default: Dict[str, Any] = {}
            kmap = input._caches.kmaps.get(
                TensorCache.make_kmaps_key((tensor_stride, kernel_size, stride, dilation)), default
            )
            kmap = F.transpose_kernel_map(
                kmap,
                config.ifsort,
                training=training,
                split_mask_num=config.split_mask_num,
                split_mask_num_bwd=config.split_mask_num_bwd,
            )

            if dataflow == F.Dataflow.ImplicitGEMM.value:
                if torch.jit.is_scripting():
                    feats = forward_implicit_gemm(feats, weight, kmap, config, transposed, allow_tf32, allow_fp16)
                else:
                    feats = ImplicitGEMMConvolutionFuntion.apply(feats, weight, kmap, config, transposed)
            elif dataflow == F.Dataflow.GatherScatter.value:
                raise NotImplementedError("GatherScatterConvolution is not torchscriptable yet")
                # if torch.jit.is_scripting():
                #     feats = forward_gather_scatter(feats, weight, kmap, config, transposed)
                # else:
                #     feats = GatherScatterConvolutionFuntion.apply(feats, weight, kmap, config, transposed)
            elif dataflow == F.Dataflow.FetchOnDemand.value:
                raise NotImplementedError("GatherScatterConvolution is not torchscriptable yet")
                # if torch.jit.is_scripting():
                #     feats = forward_fetch_on_demand(feats, weight, kmap, config, transposed)
                # else:
                #     feats = FetchOnDemandConvolutionFuntion.apply(feats, weight, kmap, config, transposed)

            if bias is not None:
                feats += bias
            output = SparseTensor(
                coords=input._caches.cmaps[TensorCache.make_tuple_key(tensor_stride)][0],
                feats=feats,
                stride=tensor_stride,
                spatial_range=input._caches.cmaps[TensorCache.make_tuple_key(tensor_stride)][1],
            )
        else:
            hashmap_keys, hashmap_vals = None, None

            spatial_range = input.spatial_range
            kmap = F.build_kernel_map(
                coords,
                feats.shape[0],
                kernel_size,
                stride,
                padding,
                hashmap_keys,
                hashmap_vals,
                spatial_range,
                kmap_mode,
                dataflow,
                downsample_mode=config.downsample_mode,
                training=training,
                ifsort=config.ifsort,
                generative=generative,
            )
            # generate output: logically forced to be not transposed
            if dataflow == F.Dataflow.ImplicitGEMM.value:
                if torch.jit.is_scripting():
                    feats = forward_implicit_gemm(feats, weight, kmap, config, transposed, allow_tf32, allow_fp16)
                else:
                    feats = ImplicitGEMMConvolutionFuntion.apply(feats, weight, kmap, config, transposed)
            elif dataflow == F.Dataflow.GatherScatter.value:
                raise NotImplementedError("GatherScatterConvolution is not torchscriptable yet")
                # if torch.jit.is_scripting():
                #     feats = forward_gather_scatter(feats, weight, kmap, config, transposed)
                # else:
                #     feats = GatherScatterConvolutionFuntion.apply(feats, weight, kmap, config, transposed)
            elif dataflow == F.Dataflow.FetchOnDemand.value:
                raise NotImplementedError("GatherScatterConvolution is not torchscriptable yet")
                # if torch.jit.is_scripting():
                #     feats = forward_fetch_on_demand(feats, weight, kmap, config, transposed)
                # else:
                #     feats = FetchOnDemandConvolutionFuntion.apply(feats, weight, kmap, config, transposed)
            if bias is not None:
                feats += bias
            kmap_coords = kmap["coords"]
            assert torch.jit.isinstance(kmap_coords, torch.Tensor)
            kmap_spatial_range = kmap["spatial_range"]
            assert torch.jit.isinstance(kmap_spatial_range, List[int])
            input._caches.cmaps[TensorCache.make_tuple_key(tensor_stride)] = (
                kmap_coords,
                kmap_spatial_range,
            )
            output = SparseTensor(
                coords=input._caches.cmaps[TensorCache.make_tuple_key(tensor_stride)][0],
                feats=feats,
                stride=tensor_stride,
                spatial_range=input._caches.cmaps[TensorCache.make_tuple_key(tensor_stride)][1],
            )
            hashmap = [kmap["hashmap_keys"], kmap["hashmap_vals"]]
            new_kmaps: Dict[str, Dict[str, Any]] = {}
            new_hashmaps: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {}
            input._caches.kmaps = new_kmaps  # new_kmap
            input._caches.hashmaps = new_hashmaps

    output._caches = input._caches
    output._caches.cmaps.setdefault(
        TensorCache.make_tuple_key(output.stride), (output.coords, output.spatial_range)
    )
    return output
