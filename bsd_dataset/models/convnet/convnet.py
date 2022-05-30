import math
import torch
import torch.nn as nn
from typing import Type, Union, List, Tuple, Optional

__all__ = [
    "ConvNet",
    "GaussianConvNet",
]

class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, stride = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(num_features = out_channels)
        self.conv2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(num_features = out_channels)
        self.conv3 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels * self.expansion, kernel_size = 1, stride = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(num_features = out_channels * self.expansion)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = nn.ModuleList([
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels * self.expansion, kernel_size = 1, stride = stride, bias = False),
            nn.BatchNorm2d(num_features = out_channels * self.expansion)
        ])

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)

        output = self.conv3(output)
        output = self.bn3(output)

        residual = self.downsample[1](self.downsample[0](x))
        output += residual
        output = self.relu(output)

        return output

class BottleneckTranspose(nn.Module):
    expansion: int = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, stride = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(num_features = out_channels)
        self.conv2 = nn.ConvTranspose2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(num_features = out_channels)
        self.conv3 = nn.ConvTranspose2d(in_channels = out_channels, out_channels = out_channels * self.expansion, kernel_size = 1, stride = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(num_features = out_channels * self.expansion)
        self.relu = nn.ReLU(inplace = True)
        self.upsample = nn.ModuleList([
            nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels * self.expansion, kernel_size = 1, stride = stride, bias = False),
            nn.BatchNorm2d(num_features = out_channels * self.expansion)
        ])

    def forward(self, x: torch.Tensor, output_size: Optional[Tuple[int, int]] = None, **kwargs) -> torch.Tensor:
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output, output_size = output_size)
        output = self.bn2(output)
        output = self.relu(output)

        output = self.conv3(output)
        output = self.bn3(output)

        residual = self.upsample[1](self.upsample[0](x, output_size = output_size))
        output += residual
        output = self.relu(output)

        return output

class _ConvNet(nn.Module):
    def __init__(self, input_shape: List[int], target_shape: List[int], num_blocks: List[int]) -> None:
        super().__init__()
        self.ln = nn.LayerNorm((input_shape[1], input_shape[2]))

        self.layer1 = self.make(block = Bottleneck, in_channels = input_shape[0], out_channels = 64, num_blocks = num_blocks[0])
        self.layer2 = self.make(block = Bottleneck, in_channels = 64 * Bottleneck.expansion, out_channels = 128, num_blocks = num_blocks[1], stride = 2)
        self.layer3 = self.make(block = Bottleneck, in_channels = 128 * Bottleneck.expansion, out_channels = 256, num_blocks = num_blocks[2], stride = 2)
        self.layer4 = self.make(block = Bottleneck, in_channels = 256 * Bottleneck.expansion, out_channels = 512, num_blocks = num_blocks[3], stride = 2)
        self.layer5 = self.make(block = BottleneckTranspose, in_channels = 512 * Bottleneck.expansion, out_channels = 256, num_blocks = num_blocks[3], stride = 2)
        self.layer6 = self.make(block = BottleneckTranspose, in_channels = 256 * BottleneckTranspose.expansion, out_channels = 128, num_blocks = num_blocks[2], stride = 2)
        self.layer7 = self.make(block = BottleneckTranspose, in_channels = 128 * BottleneckTranspose.expansion, out_channels = 64, num_blocks = num_blocks[1], stride = 2)
        self.layer8 = self.make(block = BottleneckTranspose, in_channels = 64 * BottleneckTranspose.expansion, out_channels = 64, num_blocks = num_blocks[0])

        count = math.ceil(math.log(max(target_shape[1] / input_shape[1], target_shape[2] / input_shape[2]), 2))
        layers = []
        for _ in range(count): 
            layers.append(self.make(block = BottleneckTranspose, in_channels = 64 * Bottleneck.expansion, out_channels = 64, num_blocks = num_blocks[0], stride = 2))
        self.upsample = nn.ModuleList(layers)
        
        self.conv = nn.Conv2d(in_channels = 64 * Bottleneck.expansion, out_channels = target_shape[0], kernel_size = 1, stride = 1, bias = False)
        self.bn = nn.BatchNorm2d(num_features = target_shape[0])

        self.avgpool = nn.AdaptiveAvgPool2d((target_shape[1], target_shape[2])) 
        
        for module in self.modules():
            if(isinstance(module, (nn.Conv2d, nn.ConvTranspose2d))):
                nn.init.kaiming_normal_(module.weight, mode = "fan_out", nonlinearity = "relu")
            if(isinstance(module, (nn.LayerNorm, nn.BatchNorm2d))):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1)

    def make(self, block: Union[Bottleneck, BottleneckTranspose], in_channels: int, out_channels: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
        head = block(in_channels = in_channels, out_channels = out_channels, stride = stride)
        tail = nn.Sequential(*[block(in_channels = out_channels * block.expansion, out_channels = out_channels) for _ in range(num_blocks - 1)])
        return nn.ModuleList([head, tail])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.ln(x)
        x1 = self.layer1[1](self.layer1[0](x0))
        x2 = self.layer2[1](self.layer2[0](x1))
        x3 = self.layer3[1](self.layer3[0](x2))
        x4 = self.layer4[1](self.layer4[0](x3))
        x5 = self.layer5[1](self.layer5[0](x4, (x3.shape[-2], x3.shape[-1]))) + x3
        x6 = self.layer6[1](self.layer6[0](x5, (x2.shape[-2], x2.shape[-1]))) + x2
        x7 = self.layer7[1](self.layer7[0](x6, (x1.shape[-2], x1.shape[-1]))) + x1
        x  = self.layer8[1](self.layer8[0](x7, (x0.shape[-2], x0.shape[-1])))
        for layer in self.upsample:
            x = layer[1](layer[0](x, (2 * x.shape[-2], 2 * x.shape[-1])))
        x = self.conv(x)
        x = self.bn(x)
        
        x = self.avgpool(x)
        return x.squeeze(1)

def ConvNet(input_shape, target_shape) -> _ConvNet:
    return _ConvNet(input_shape = input_shape, target_shape = [1] + target_shape, num_blocks = [2, 2, 2, 2])

def GaussianConvNet(input_shape, target_shape) -> _ConvNet:
    return _ConvNet(input_shape = input_shape, target_shape = [2] + target_shape, num_blocks = [1, 3, 6, 3])

