import math
import torch
import torch.nn as nn
from typing import Type, Union, List, Tuple, Optional

__all__ = [
    "SRCNN"
]

class SRCNN(nn.Module):
    def __init__(self, input_shape: List[int], target_shape: List[int]):
        super(SRCNN, self).__init__()
        self.input_shape = input_shape
        self.target_shape = target_shape
        # self.ln = nn.LayerNorm((self.input_shape[1], self.input_shape[2]))
        self.conv1 = nn.Conv2d(self.input_shape[0], 8, kernel_size = 3, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size = 3, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size = 3, padding = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.ConvTranspose2d(32, 16, kernel_size = 3, stride = 2, padding = 1, output_padding = (1, 1), bias = False)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.ConvTranspose2d(16, 8, kernel_size = 3, stride = 2, padding = 1, output_padding = (1, 1), bias = False)
        self.bn5 = nn.BatchNorm2d(8)
        self.conv6 = nn.ConvTranspose2d(8, 1, kernel_size = 3, stride = 2, padding = 1, output_padding = (1, 1), bias = False)
        self.avgpool = nn.AdaptiveAvgPool2d(target_shape)
        self.relu = nn.ReLU(inplace = True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, **kwargs):
        # x = self.ln(x)
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout(self.relu(self.bn4(self.conv4(x))))
        x = self.dropout(self.relu(self.bn5(self.conv5(x))))
        x = self.avgpool(self.relu(self.conv6(x)))
        return x.squeeze(1)