import torch
from torch.autograd import Function

# from torch.cuda.amp import custom_bwd, custom_fwd
from typing import List, Any

from torchsparse.utils.utils import make_tensor
import os

__all__ = ["to_dense"]

torch.ops.load_library(
    os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    __file__
                )
            )
        ),
        "torchsparse_ops.so"
    )
)

@torch.jit.script
def to_dense_forward(
    feats: torch.Tensor,
    coords: torch.Tensor,
    spatial_range: List[int],
):
    feats = feats.contiguous()
    coords = coords.contiguous().int()
    outputs = torch.zeros(
        spatial_range + [feats.size(1)], dtype=feats.dtype, device=feats.device
    )
    spatial_range = make_tensor(spatial_range, dtype=torch.int, device=feats.device)

    if feats.device.type == "cuda":
        torch.ops.torchsparse_ops.to_dense_forward_cuda(
            feats, coords, spatial_range, outputs
        )
    else:
        raise NotImplementedError
    return outputs.to(feats.dtype)

class ToDenseFunction(Function):
    @staticmethod
    # @custom_fwd(cast_inputs=torch.half)
    def forward(
        ctx,
        feats: torch.Tensor,
        coords: torch.Tensor,
        spatial_range: List[int],
    ) -> torch.Tensor:
        outputs = to_dense_forward(feats, coords, spatial_range)
        spatial_range = make_tensor(spatial_range, dtype=torch.int, device=feats.device)
        ctx.for_backwards = (coords, spatial_range)
        return outputs

    @staticmethod
    # @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        coords, spatial_range = ctx.for_backwards
        grad_output = grad_output.contiguous()
        grad_feats = torch.zeros(
            coords.size(0),
            grad_output.size(-1),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )

        if grad_output.device.type == "cuda":
            torch.ops.torchsparse_ops.to_dense_backward_cuda(
                grad_output, coords, spatial_range, grad_feats
            )
        else:
            raise NotImplementedError

        return grad_feats, None, None


def to_dense(
    feats: torch.Tensor, coords: torch.Tensor, spatial_range: List[int] #Any
) -> torch.Tensor:
    if torch.jit.is_scripting():
        return to_dense_forward(feats, coords, spatial_range)
    else:
        return ToDenseFunction.apply(feats, coords, spatial_range)
