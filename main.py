from transformer.model import MiniGPT
from torch.tensor import Tensor
import numpy as np

model = MiniGPT(dmodel=128, numheads=8, dk=64, dv=64, dff=512)
