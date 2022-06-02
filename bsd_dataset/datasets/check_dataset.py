from typing import Any, Callable, Dict, Tuple

import skimage.transform
import torch

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
        # Mask nans with zero
        yy = metrics.mask_with_zeros(y, info['y_mask'])
        # Downsample the hi-res target to the shape of the low-res input
        yy = torch.tensor(skimage.transform.resize(yy, x.shape[1:]))
        # Downsample the nan mask
        mask = info['y_mask'].to(torch.int)
        mask = skimage.transform.resize(mask, x.shape[1:])
        mask = torch.tensor(mask).to(bool)
        return yy, mask

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
