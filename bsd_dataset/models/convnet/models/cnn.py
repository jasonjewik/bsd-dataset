import torch
import torch.nn as nn
from typing import Type, Union, List, Optional

__all__ = [
    "XResNet18",
    "XResNet34",
    "XResNet50",
    "XResNet101",
    "XResNet152",
]

def conv1x1(in_channels: int, out_channels: int, stride: int = 1, transpose = False) -> nn.Conv2d:
    return nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, stride = stride, bias = False)

def conv3x3(in_channels: int, out_channels: int, stride: int = 1, transpose = False) -> nn.Conv2d:
    return nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)

def convtranspose1x1(in_channels: int, out_channels: int, stride: int = 1, transpose = False) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, stride = stride, bias = False)

def convtranspose3x3(in_channels: int, out_channels: int, stride: int = 1, transpose = False) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)

class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, transpose: bool = False, sample: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.conv1 = conv1x1(in_channels = in_channels, out_channels = out_channels, transpose = transpose)
        self.bn1 = nn.BatchNorm2d(num_features = out_channels)
        self.conv2 = conv3x3(in_channels = out_channels, out_channels = out_channels, stride = stride, transpose = transpose)
        self.bn2 = nn.BatchNorm2d(num_features = out_channels)
        self.conv3 = conv1x1(in_channels = out_channels, out_channels = out_channels * self.expansion, transpose = transpose)
        self.bn3 = nn.BatchNorm2d(num_features = out_channels * self.expansion)
        self.relu = nn.ReLU(inplace = True)
        self.sample = sample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)

        output = self.conv3(output)
        output = self.bn3(output)

        if(self.sample is not None):
            residual = self.sample(x)

        output += residual
        output = self.relu(output)

        print(output.shape)

        return output

# TODO: Fix upsampling to correct output size
class XResNet(nn.Module):
    def __init__(self, input_shape: List[int], output_shape: List[int],  block: Union[BasicBlock, Bottleneck], num_blocks: List[int]) -> None:
        super().__init__()

        self.in_channels = input_shape[0]

        layers = []
        layers.extend(self.make(block = block, out_channels = 64, num_blocks = num_blocks[0]))
        layers.extend(self.make(block = block, out_channels = 128, num_blocks = num_blocks[1], stride = 2))
        layers.extend(self.make(block = block, out_channels = 256, num_blocks = num_blocks[2], stride = 2))
        layers.extend(self.make(block = block, out_channels = 512, num_blocks = num_blocks[3], stride = 2))
        self.downsample = nn.Sequential(*layers)

        layers = []
        layers.extend(self.make(block = block, out_channels = 512, num_blocks = num_blocks[3], stride = 2, transpose = True, output_padding = (input_shape[1] // 2 // 2) & 1))
        layers.extend(self.make(block = block, out_channels = 256, num_blocks = num_blocks[2], stride = 2, transpose = True, output_padding = (input_shape[1] // 2) & 1))
        layers.extend(self.make(block = block, out_channels = 128, num_blocks = num_blocks[1], stride = 2, transpose = True, output_padding = (input_shape[1]) & 1))
        for _ in range(count): 
            layers.extend(self.make(block = block, out_channels = 64, num_blocks = num_blocks[0], stride = 2, transpose = True))
        layers.extend(self.make(block = block, out_channels = 64, num_blocks = num_blocks[0], transpose = True))
        self.upsample = nn.Sequential(*layers)
        
        self.conv = conv1x1(in_channels = 64 * block.expansion, out_channels = output_shape[0])
        self.bn = nn.BatchNorm2d(num_features = output_shape[0])

        for module in self.modules():
            if(isinstance(module, (nn.Conv2d, nn.ConvTranspose2d))):
                nn.init.kaiming_normal_(module.weight, mode = "fan_out", nonlinearity = "relu")
            if(isinstance(module, nn.BatchNorm2d)):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1)

    def make(self, block: Union[BasicBlock, Bottleneck], out_channels: int, num_blocks: int, stride: int = 1, transpose: bool = False, output_padding: int = 0) -> nn.Sequential:
        sample = None

        if(stride != 1 or self.in_channels != out_channels * block.expansion):
            sample = nn.Sequential(
                conv1x1(in_channels = self.in_channels, out_channels = out_channels * block.expansion, stride = stride, transpose = transpose, output_padding = output_padding),
                nn.BatchNorm2d(num_features = out_channels * block.expansion) 
            )
        
        layers = []

        layers.append(block(in_channels = self.in_channels, out_channels = out_channels, stride = stride, transpose = transpose, sample = sample, output_padding = output_padding))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(in_channels = self.in_channels, out_channels = out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        return x

def XResNet18(input_shape, output_shape) -> XResNet:
    return XResNet(input_shape = input_shape, output_shape = output_shape, block = BasicBlock, num_blocks = [1, 1, 1, 1])

def XResNet34(input_shape, output_shape) -> XResNet:
    return XResNet(input_shape = input_shape, output_shape = output_shape, block = BasicBlock, num_blocks = [3, 4, 6, 3])

def XResNet50(input_shape, output_shape) -> XResNet:
    return XResNet(input_shape = input_shape, output_shape = output_shape, block = Bottleneck, num_blocks = [3, 4, 6, 3])

def XResNet101(input_shape, output_shape) -> XResNet:
    return XResNet(input_shape = input_shape, output_shape = output_shape, block = Bottleneck, num_blocks = [3, 4, 23, 3])

def XResNet152(input_shape, output_shape) -> XResNet:
    return XResNet(input_shape = input_shape, output_shape = output_shape, block = Bottleneck, num_blocks = [3, 8, 36, 3])