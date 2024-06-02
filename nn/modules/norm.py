import torch
from torch import nn

from torchsparse import SparseTensor
from torchsparse.nn.utils import post_fapply

__all__ = ["BatchNorm", "GroupNorm", "InstanceNorm"]


class InstanceNorm(nn.InstanceNorm1d):
    def forward(self, input: SparseTensor) -> SparseTensor:
        feats = super().forward(input.feats)
        return post_fapply(input, feats)


class BatchNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BatchNorm, self).__init__()
        self.net = nn.BatchNorm1d(*args, **kwargs)

    def forward(self, input: SparseTensor) -> SparseTensor:
        feats = self.net(input.feats)
        return post_fapply(input, feats)


class GroupNorm(nn.GroupNorm):
    def forward(self, input: SparseTensor) -> SparseTensor:
        coords, feats, stride = input.coords, input.feats, input.stride

        batch_size = torch.max(coords[:, 0]).item() + 1
        num_channels = feats.shape[1]

        # PyTorch's GroupNorm function expects the input to be in (N, C, *)
        # format where N is batch size, and C is number of channels. "feats"
        # is not in that format. So, we extract the feats corresponding to
        # each sample, bring it to the format expected by PyTorch's GroupNorm
        # function, and invoke it.
        nfeats = torch.zeros_like(feats)
        for k in range(batch_size):
            indices = coords[:, 0] == k
            bfeats = feats[indices]
            bfeats = bfeats.transpose(0, 1).reshape(1, num_channels, -1)
            bfeats = super().forward(bfeats)
            bfeats = bfeats.reshape(num_channels, -1).transpose(0, 1)
            nfeats[indices] = bfeats

        output = SparseTensor(
            coords=coords,
            feats=nfeats,
            stride=stride,
            spatial_range=input.spatial_range,
        )
        output._caches = input._caches
        return output
