import calendar
from typing import List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import skimage.transform

# 2/29 on leap years needs to be removed because
# CM5A and CM3 data don't record those dates
# even though CHIRPS and CGCM3 do
LEAP_YEAR_DATES = np.array([np.datetime64(f'{year}-02-29')
                            for year in range(1860, 2022)
                            if calendar.isleap(year)])

def irange(start, stop, step=1):
    """
    Inclusive range [start, stop].
    """
    return range(start, stop+1, step)

def fix_dates(xdata: xr.Dataset) -> xr.Dataset:
    """
    Converts date ranges because not all of the datasets
    operate on the same time interval precision, which
    messes with indexing.
    """
    xdata['time'] = pd.date_range(
        xdata['time'][0].values,
        periods=len(xdata['time']),
        freq='1D').floor('D')
    return xdata

def match_image_sizes(arr: List[np.array], shape: Tuple[int, int]) -> np.array:
    """
    Takes a list of images, each of shape (channels, width, height) and resizes 
    them to match the given shape.
    """
    arr = np.array([skimage.transform.resize(a, shape) for a in arr])
    return arr