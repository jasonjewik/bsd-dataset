from typing import Any, Callable, Dict, Tuple

import skimage.transform
import torch
import torch.nn as nn

import bsd_dataset.common.metrics as metrics

class Interpolator:
    def __init__(self, eval_fns: Dict[str, Callable]):
        self.eval_fns = eval_fns

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        info: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        pool1 = nn.AdaptiveAvgPool2d(y.shape)
        pool2 = nn.AdaptiveAvgPool2d(x.shape[1:])
        xx = pool2(pool1(x.unsqueeze(0)))
        return xx.squeeze()

    def eval(
        self,
        x: torch.Tensor,
        yy: torch.Tensor,
        mask: torch.BoolTensor
    ) -> Dict[str, torch.Tensor]:
        # Run evaluation on the interpolated target data
        result = {}
        for name, fn in self.eval_fns.items():
            result[name] = fn(x, yy, mask)
        return result
