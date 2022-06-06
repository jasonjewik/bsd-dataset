import math
import torch
import torch.nn as nn
from typing import Type, Union, List, Tuple, Optional

__all__ = [
    "AvgPool"
]

class AvgPool(nn.Module):
    def __init__(self, input_shape: List[int], target_shape: List[int]):
        super(AvgPool, self).__init__()
        self.input_shape = input_shape
        self.target_shape = target_shape
        self.ln = nn.LayerNorm((self.input_shape[1], self.input_shape[2]))
        self.avgpool = nn.AdaptiveAvgPool2d(target_shape)

    def forward(self, x, **kwargs):
        x = self.ln(x)
        x = self.avgpool(x)
        return x.squeeze(1)