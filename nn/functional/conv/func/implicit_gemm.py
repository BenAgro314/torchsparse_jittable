from typing import Dict

import torch
from torch.autograd import Function

# from torch.cuda.amp import custom_bwd, custom_fwd

import torchsparse
import torchsparse.backends

import os

from torchsparse.nn.functional.conv.conv_config import ConvConfig
torch.ops.load_library(
    os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(
                            os.path.dirname(
                                __file__
                            )
                        )
                    )
                )
            )
        ),
        "torchsparse_ops.so"
    )
)

__all__ = ["ImplicitGEMMConvolutionFuntion"]

import torch
from torch import nn
from typing import Dict, Any


@torch.jit.script
def forward_implicit_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    kmap: Dict[str, Any],
    config: ConvConfig,
    transposed: bool,
    allow_tf32: bool = False,
    allow_fp16: bool = False,
) -> torch.Tensor:
    sizes = kmap["sizes"]
    if not transposed:
        out_in_map = kmap["out_in_map"]
        reorder_out_in_map = kmap["reorder_out_in_map"]
        reduced_sorted_mask = kmap["reduced_sorted_mask"]
        reorder_loc = kmap["reorder_loc"]
        out_in_map_bwd = kmap["out_in_map_bwd"]
        reorder_out_in_map_bwd = kmap["reorder_out_in_map_bwd"]
        reduced_sorted_mask_bwd_wgrad = kmap["reduced_sorted_mask_bwd_wgrad"]
        reduced_sorted_mask_bwd_dgrad = kmap["reduced_sorted_mask_bwd_dgrad"]
        reorder_loc_bwd = kmap["reorder_loc_bwd"]
    else:
        out_in_map = kmap["out_in_map_t"]
        reorder_out_in_map = kmap["reorder_out_in_map_t"]
        reduced_sorted_mask = kmap["reduced_sorted_mask_t"]
        reorder_loc = kmap["reorder_loc_t"]
        out_in_map_bwd = kmap["out_in_map_bwd_t"]
        reorder_out_in_map_bwd = kmap["reorder_out_in_map_bwd_t"]
        reduced_sorted_mask_bwd_wgrad = kmap["reduced_sorted_mask_bwd_wgrad_t"]
        reduced_sorted_mask_bwd_dgrad = kmap["reduced_sorted_mask_bwd_dgrad_t"]
        reorder_loc_bwd = kmap["reorder_loc_bwd_t"]

    ifsort = config.ifsort

    input = input.contiguous()
    weight = weight.contiguous()

    assert torch.jit.isinstance(sizes, tuple[int, int])
    assert torch.jit.isinstance(out_in_map, torch.Tensor)

    if input.device.type != "cuda":
        if not transposed:
            output = torch.zeros(sizes[1], weight.size(-1), dtype=input.dtype, device=input.device)
        else:
            output = torch.zeros(sizes[0], weight.size(-1), dtype=input.dtype, device=input.device)
    else:
        if torch.float16 in [input.dtype, weight.dtype]:
            input = input.to(torch.float16)
            weight = weight.to(torch.float16)

        num_out_feats = sizes[1] if not transposed else sizes[0]
        num_out_channels = weight.shape[-1]

        if not ifsort:
            output = torch.ops.torchsparse_ops.conv_forward_implicit_gemm_cuda(
                input,
                weight,
                out_in_map,
                num_out_feats,
                num_out_channels,
                allow_tf32,
                allow_fp16,
            )
        else:
            assert torch.jit.isinstance(reorder_out_in_map, torch.Tensor)
            assert torch.jit.isinstance(reduced_sorted_mask, torch.Tensor)
            assert torch.jit.isinstance(reorder_loc, torch.Tensor)
            output = torch.ops.torchsparse_ops.conv_forward_implicit_gemm_sorted_cuda(
                input,
                weight,
                reorder_out_in_map,
                reduced_sorted_mask,
                reorder_loc,
                num_out_feats,
                num_out_channels,
                allow_tf32,
                allow_fp16,
            )
    return output.to(weight.dtype)

def backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    out_in_map_bwd: torch.Tensor,
    reorder_out_in_map_bwd: torch.Tensor,
    reduced_sorted_mask_bwd_wgrad: torch.Tensor,
    reduced_sorted_mask_bwd_dgrad: torch.Tensor,
    reorder_loc_bwd: torch.Tensor,
    transposed: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_output = grad_output.contiguous()
    if grad_output.dtype != weight.dtype:
        grad_output = grad_output.to(weight.dtype)

    kernel_volume, ic, oc = weight.size()

    if grad_output.device.type == "cuda":
        if kernel_volume < 32:  # sort mode
            # dgrad
            grad_input = torch.ops.torchsparse_ops.conv_forward_implicit_gemm_sorted_cuda(
                grad_output,
                weight.transpose(2, 1).contiguous(),
                reorder_out_in_map_bwd,
                reduced_sorted_mask_bwd_dgrad,
                reorder_loc_bwd,
                input.size(0),
                input.size(1),
                torchsparse.backends.allow_tf32,
                torchsparse.backends.allow_fp16,
            )

            # wgrad
            grad_weight = (
                (
                    torch.ops.torchsparse_ops.conv_backward_wgrad_implicit_gemm_sorted_cuda(
                        grad_output,
                        input,
                        reorder_out_in_map_bwd,
                        reduced_sorted_mask_bwd_wgrad,
                        reorder_loc_bwd,
                        32,
                        torchsparse.backends.allow_tf32,
                        torchsparse.backends.allow_fp16,
                    )
                )
                .reshape(kernel_volume, oc, ic)
                .transpose(2, 1)
                .contiguous()
            )

        else:  # unsort mode
            # dgrad
            grad_input = torch.ops.torchsparse_ops.conv_forward_implicit_gemm_cuda(
                grad_output,
                weight.transpose(2, 1).contiguous(),
                out_in_map_bwd,
                input.size(0),
                input.size(1),
                torchsparse.backends.allow_tf32,
                torchsparse.backends.allow_fp16,
            )

            # wgrad
            grad_weight = (
                (
                    torch.ops.torchsparse_ops.conv_backward_wgrad_implicit_gemm_cuda(
                        grad_output,
                        input,
                        out_in_map_bwd,
                        32,
                        torchsparse.backends.allow_tf32,
                        torchsparse.backends.allow_fp16,
                    )
                )
                .reshape(kernel_volume, oc, ic)
                .transpose(2, 1)
                .contiguous()
            )
    else:
        raise NotImplementedError
    return grad_input, grad_weight


class ImplicitGEMMConvolutionFuntion(Function):  # TorchSparse++
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        kmap: Dict[str, torch.Tensor],
        config: Dict[str, bool],
        transposed: bool = False,
    ) -> torch.Tensor:
        ctx.save_for_backward(input, weight, kmap, config, transposed)
        output = forward_implicit_gemm(input, weight, kmap, config, transposed,
                                allow_fp16=torchsparse.backends.allow_fp16,
                                allow_tf32=torchsparse.backends.allow_tf32)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, weight, kmap, config, transposed = ctx.saved_tensors
        grad_input, grad_weight = backward(
            grad_output, input, weight, kmap["out_in_map_bwd"], kmap["reorder_out_in_map_bwd"],
            kmap["reduced_sorted_mask_bwd_wgrad"], kmap["reduced_sorted_mask_bwd_dgrad"],
            kmap["reorder_loc_bwd"], transposed
        )
        return grad_input, grad_weight, None, None, None
