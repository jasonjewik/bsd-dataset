import bsd_dataset.common.metrics as metrics
import bsd_dataset.common.visualizations as viz

import numpy as np
import torch

def test_num_std_above_below():
    # Generate some data
    shape = (10, 10)
    y = torch.tensor(np.random.standard_normal(shape))
    means = y * np.random.normal(1, 3, shape)
    variances = np.abs(np.random.normal(50, 10, shape))
    stds = np.sqrt(variances)
    mask = torch.tensor(np.random.standard_gamma(1, shape) < 0)

    # Determine how many standard deviations (up to 3) above (+) 
    # or below (-) the mean each coresponding point in y is
    arr = torch.where(mask, torch.nan, 1.)
    arr = torch.where(y < means, -1 * arr, arr)
    std1 = ((means - 1 * stds) < y) & (y < (means + 1 * stds))
    std2 = ((means - 2 * stds) < y) & (y < (means + 2 * stds))
    std3 = ((means - 3 * stds) < y) & (y < (means + 3 * stds))
    rem = ~(std1 | std2 | std3) & ~mask
    arr = torch.where(std2 & ~std1, arr * 2, arr)
    arr = torch.where(std3 & ~std2 & ~std1, arr * 3, arr)
    arr = torch.where(rem, arr * 4, arr)

    # Use the function we want to test
    arr2 = viz.num_std_above_below(y, mask, means, stds, 3)

    # Mask out the NaNs so that comparison will work
    masked_arr = metrics.mask_with_zeros(arr, mask)
    masked_arr2 = metrics.mask_with_zeros(arr2, mask)

    assert (masked_arr == masked_arr2).all()