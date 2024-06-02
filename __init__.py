import torchsparse.backends as backends

from .operators import *
from .tensor import *
from .utils.tune import tune
from .version import __version__

import os
import torch
torch.ops.load_library(os.path.join(os.path.dirname(os.path.dirname(__file__)), "torchsparse_ops.so"))

backends.init()
