from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

import bsd_dataset.common.metrics as metrics
from bsd_dataset.regions import Region


def show_ground_truth(y: torch.Tensor, study_region: Region):
    _, ax = plt.subplots(figsize=(5, 5))

    cmin, cmax = 0, 200
    cmap = mpl.cm.get_cmap(name='Blues').copy()
    cmap.set_bad(color='lightgrey')

    cbar = plt.colorbar(
        mpl.cm.ScalarMappable(cmap=cmap),
        ax=ax,
        label='Precipitation Ground Truth (mm)'
    )
    cbar.set_ticks(np.linspace(0, 1, 5))
    ticks = np.round(np.linspace(
        cmin, cmax, 5, dtype=int
    ))
    cbar.ax.set_yticklabels(ticks)

    lats = study_region.get_latitudes()
    lons = study_region.get_longitudes(180)

    yticks = np.linspace(0, y.shape[1]+1, 8)
    ax.set_yticks(
        ticks=yticks,
        labels=np.linspace(lats[1], lats[0]-1, len(yticks), dtype=int)
    )
    ax.set_ylabel('Latitude')

    xticks = np.linspace(0, y.shape[1]+1, 8)
    ax.set_xticks(
        ticks=xticks,
        labels=np.linspace(lons[0], lons[1]+1, len(xticks), dtype=int)
    )
    ax.set_xlabel('Longitude')

    handles = [
        mpl.patches.Patch(facecolor='lightgrey', label='missing data')]
    ax.legend(handles=handles)
        
    ax.imshow(y, cmap=cmap, vmin=cmin, vmax=cmax)
    ax.invert_yaxis()  # so that the lowest y value is at the bottom
    plt.show()

def show_bias(
    y_pred: torch.Tensor,
    y_true: torch.Tensor, 
    study_region: Region
):
    _, ax = plt.subplots(figsize=(5, 5))

    cmin, cmax = -200, 200
    cmap = mpl.cm.get_cmap(name='bwr').copy()
    cmap.set_bad(color='lightgrey')

    cbar = plt.colorbar(
        mpl.cm.ScalarMappable(cmap=cmap),
        ax=ax,
        label='Precipitation Bias (mm)'
    )
    cbar.set_ticks(np.linspace(0, 1, 5))
    ticks = np.round(np.linspace(
        cmin, cmax, 5, dtype=int
    ))
    cbar.ax.set_yticklabels(ticks)

    lats = study_region.get_latitudes()
    lons = study_region.get_longitudes(180)

    yticks = np.linspace(0, y_true.shape[1]+1, 8)
    ax.set_yticks(
        ticks=yticks,
        labels=np.linspace(lats[1], lats[0]-1, len(yticks), dtype=int)
    )
    ax.set_ylabel('Latitude')

    xticks = np.linspace(0, y_true.shape[1]+1, 8)
    ax.set_xticks(
        ticks=xticks,
        labels=np.linspace(lons[0], lons[1]+1, len(xticks), dtype=int)
    )
    ax.set_xlabel('Longitude')

    handles = [
        mpl.patches.Patch(facecolor='lightgrey', label='missing data')]
    ax.legend(handles=handles)
        
    ax.imshow(y_pred - y_true, cmap=cmap, vmin=cmin, vmax=cmax)
    ax.invert_yaxis()
    plt.show()

def show_rmse(
    y_pred: List[torch.Tensor], 
    y_true: List[torch.Tensor], 
    masks: List[torch.BoolTensor]
):
    rmse_arr = torch.tensor(list(map(metrics.rmse, y_pred, y_true, masks)))
    _, ax = plt.subplots()
    ax.plot(rmse_arr)
    ax.set_xlabel('Day')
    ax.set_ylabel('RMSE (mm)')
    plt.show()

def within_n_std(
    t: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor, 
    n: int
) -> torch.BoolTensor:
    lower_bound = (means - n * stds)
    upper_bound = (means + n * stds)
    return (lower_bound < t) & (t < upper_bound)

def num_std_above_below(
    t: torch.Tensor, 
    mask: torch.BoolTensor, 
    means: torch.Tensor, 
    stds: torch.Tensor, 
    max_std: int
) -> torch.Tensor:
    result = torch.where(mask, torch.nan, 1.)
    result = torch.where(t < means, -1 * result, result)

    within_max_std = torch.zeros_like(t).to(torch.bool)
    not_in_prev_std = torch.ones_like(t).to(torch.bool)    
    
    for n in range(1, max_std+1):
        in_stdn = within_n_std(t, means, stds, n)
        result = torch.where(in_stdn & not_in_prev_std, result * n, result)
        within_max_std |= in_stdn
        not_in_prev_std &= ~in_stdn

    beyond_max_std = ~within_max_std & ~mask
    result = torch.where(beyond_max_std, result * (max_std + 1), result)
    
    return result