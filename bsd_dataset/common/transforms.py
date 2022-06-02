from typing import Any, Callable, Dict, List

import torch

class Compose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor, info: Dict[str, Any]) -> Any:
        xx = torch.clone(x)
        for transform in self.transforms:
            xx = transform(xx, info)
        return xx

class ConvertPrecipitation:
    """ Convert kg m-2 s-1 to mm. """

    def __init__(self, var_name: str = None):
        self.var_name = var_name

    def __call__(self, x: torch.Tensor, info: Dict[str, Any]) -> torch.Tensor:
        xx = torch.clone(x)
        if self.var_name == None:
            xx *= 86400
        else:
            for i, channel in enumerate(info['channels']):
                if channel.endswith(':data'):
                    channel_var = channel.split(':')[1]
                    if self.var_name == channel_var:
                        xx[i] *= 86400
        return xx

class LogTransformPrecipitation:
    """ Apply log transformation. """

    def __init__(self, var_name: str = None, eps: float = 0.1):
        self.var_name = var_name
        self.eps = torch.tensor(eps)

    def __call__(self, x: torch.Tensor, info: Dict[str, Any]) -> torch.Tensor:
        xx = torch.clone(x)
        if self.var_name == None:
            xx = torch.log(xx + self.eps) - torch.log(self.eps)
        else:
            for i, channel in enumerate(info['channels']):
                if channel.endswith(':data'):
                    channel_var = channel.split(':')
                    if self.var_name == channel_var:
                        xx[i] = torch.log(xx[i] + self.eps) - torch.log(self.eps)
        return xx