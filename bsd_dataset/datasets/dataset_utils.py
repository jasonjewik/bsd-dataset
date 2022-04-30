from typing import List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import skimage.transform


def irange(start, stop, step=1):
    """
    Inclusive range [start, stop].
    """
    return range(start, stop+1, step)

def get_shape_of_largest_array(arr: List[np.array]) -> Tuple:
    """
    Takes a list of images, each of the shape (channels, width, height) and
    returns the shape of the one with the largest area (width * height).
    """
    largest_area = 0
    largest_shape = None
    for a in arr:
        area = np.product(a.shape[1:])
        if area > largest_area:
            largest_area = area
            largest_shape = a.shape[1:]
    return largest_shape

def match_array_shapes(arr: List[np.array], shape: Tuple[int, int]) -> List[np.array]:
    """
    Takes a list of images, each of shape (channels, width, height) and resizes 
    them to match the given shape.
    """
    result = []    
    for a in arr:
        channels = a.shape[0]
        resized_shape = (channels,) + shape
        resized_a = skimage.transform.resize(a, resized_shape, order=0, preserve_range=True)
        result.append(resized_a)
    return result

def lon180_to_lon360(lon: float) -> float:
    return (lon + 360) % 360

def lon360_to_lon180(lon: float) -> float:
    return ((lon + 180) % 360) - 180

def get_lon_mask(da: xr.DataArray, lons: np.array) -> xr.DataArray:
    mask = da.values[(da < min(lons)) | (da > max(lons))]
    return mask