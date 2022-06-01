from typing import Any, Callable, Dict, List

import torch

class Compose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor, info: Dict[str, Any]) -> Any:
        for transform in self.transforms:
            x = transform(x, info)
        return x

class ConvertPrecipitation:
    """ Convert kg m-2 s-1 to mm. """

    def __init__(self, var_name: str = None):
        self.var_name = var_name

    def __call__(self, x: torch.Tensor, info: Dict[str, Any]) -> torch.Tensor:
        if self.var_name == None:
            x *= 86400
        else:
            for i, channel in enumerate(info['channels']):
                if channel.endswith(':data'):
                    channel_var = channel.split(':')[1]
                    if self.var_name == channel_var:                        
                        x[i] *= 86400
        return x

class LogTransformPrecipitation:
    """ Apply log transformation. """

    def __init__(self, var_name: str = None, eps: float = 0.1):
        self.var_name = var_name
        self.eps = torch.tensor(eps)

    def __call__(self, x: torch.Tensor, info: Dict[str, Any]) -> torch.Tensor:
        if self.var_name == None:
            x = torch.log(x + self.eps) - torch.log(self.eps)
        else:
            for i, channel in enumerate(info['channels']):
                if channel.endswith(':data'):
                    channel_var = channel.split(':')
                    if self.var_name == channel_var:
                        x[i] = torch.log(x[i] + self.eps) - torch.log(self.eps)
        return x