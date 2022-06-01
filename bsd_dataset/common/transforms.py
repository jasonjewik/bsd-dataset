from typing import Any, Callable, Dict, List

import torch

class Compose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor, info: Dict[str, Any]) -> Any:
        for transform in self.transforms:
            x = transform(x, info)
        return x

class VariableSpecificTransform:
    def __init__(self, var_name: str):
        self.var_name = var_name

class ConvertPrecipitation(VariableSpecificTransform):
    def __call__(self, x: torch.Tensor, info: Dict[str, Any]) -> torch.Tensor:
        for i, channel in enumerate(info['channels']):
            if channel.endswith(':data'):
                channel_var = channel.split(':')[1]
                if self.var_name == channel_var:
                    # convert kg m-2 s-1 to mm
                    x[i] *= 86400
        return x

class LogTransformPrecipitation(VariableSpecificTransform):
    def __init__(self, var_name: str, eps: float):
        super().__init__(var_name)
        self.eps = torch.tensor(eps)

    def __call__(self, x: torch.Tensor, info: Dict[str, Any]) -> torch.Tensor:    
        for i, channel in enumerate(info['channels']):
            if channel.endswith(':data'):
                channel_var = channel.split(':')
                if self.var_name == channel_var:
                    x[i] = torch.log(x[i] + self.eps) - torch.log(self.eps)
        return x