from typing import List, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from bsd_dataset.datasets.dataset import BSDD
from bsd_dataset.common.metrics import rmse
from bsd_dataset.regions import Region


def show_ground_truth(y: Union[np.ndarray, torch.Tensor], study_region: Region):
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
    y: Union[np.ndarray, torch.Tensor],
    yy: Union[np.ndarray, torch.Tensor],
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
        
    ax.imshow(yy - y, cmap=cmap, vmin=cmin, vmax=cmax)
    ax.invert_yaxis()
    plt.show()

def show_rmse(y: List[torch.Tensor], y_pred: List[torch.Tensor]):
    rmse_arr = torch.tensor(list(map(rmse, y_pred, y)))
    _, ax = plt.subplots()
    ax.plot(rmse_arr)
    ax.set_xlabel('Day')
    ax.set_ylabel('RMSE (mm)')
    plt.show()