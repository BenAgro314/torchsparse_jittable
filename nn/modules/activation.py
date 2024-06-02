from torch import nn

from torchsparse import SparseTensor
from torchsparse.nn.utils import post_fapply

__all__ = ["ReLU", "LeakyReLU", "SiLU"]


class ReLU(nn.ReLU):
    def __init__(self, *args, **kwargs):
        super(ReLU, self).__init__()
        self.net = nn.ReLU(*args, **kwargs)

    def forward(self, input: SparseTensor) -> SparseTensor:
        feats = self.net(input.feats)
        return post_fapply(input, feats)


class LeakyReLU(nn.LeakyReLU):
    def __init__(self, *args, **kwargs):
        super(LeakyReLU, self).__init__()
        self.net = nn.LeakyReLU(*args, **kwargs)

    def forward(self, input: SparseTensor) -> SparseTensor:
        feats = self.net(input.feats)
        return post_fapply(input, feats)


class SiLU(nn.SiLU):
    def __init__(self, *args, **kwargs):
        super(SiLU, self).__init__()
        self.net = nn.SiLU(*args, **kwargs)

    def forward(self, input: SparseTensor) -> SparseTensor:
        feats = self.net(input.feats)
        return post_fapply(input, feats)
