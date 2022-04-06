import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from skimage import transform
import xarray as xr

# 2/29 on leap years needs to be removed because
# CM5A and CM3 data don't record those dates
# even though CHIRPS and CGCM3 do
LEAP_YEAR_DATES = np.array([np.datetime64(f'{year}-02-29')
                            for year in range(1984, 2005, 4)])

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

def extract_chirps25_data(src: str, dst: str, lons: np.array, lats: np.array, dates: Tuple[str, str]) -> None:
    """
    Extract high-resolution target data.
    
    Parameters:
      - src: the directory to read from
      - dst: the npy file to write to
      - lons: the longitudinal bounds
      - lats: the latitudinal bounds
      - dates: the temporal range of the data
    """
    ds = xr.open_mfdataset(f'{src}/*.nc')
    xdata = ds.precip.sel(
        time=slice(*dates),
        latitude=slice(*lats), 
        longitude=slice(*lons))
    mask = ~xdata['time'].isin(LEAP_YEAR_DATES)
    xdata = xdata.where(mask, drop=True)
    npdata = xdata.values  # time x lat x lon
    npdata = np.moveaxis(npdata, 1, 2)  # time x lon x lat
    with open(dst, 'wb') as f:
        np.save(f, npdata)

def extract_cmip5_data(src: List[str], dst: str, lons: np.array, lats: np.array, dates: Tuple[str, str]) -> None:
    """
    Extract low resolution input data.
    
    Parameters:
      - src: the directories to read from
      - dst: the file to write to
      - lons: the longitudinal bounds
      - lats: the latitudinal bounds
      - dates: the temporal range of the data
    """
    input_shape = (18, 45)  # shape of the largest input (CM5A)
    output = []
    for direc in src:
        ds = xr.open_mfdataset(f'{direc}/*.nc')
        xdata = ds.prc.sel(
            time=slice(*dates),
            lat=slice(*lats),
            lon=slice(*lons%360))  # convert 360deg to 180deg system
        if direc == 'CGCM3':
            xdata = fix_dates(xdata)
            mask = ~xdata['time'].isin(LEAP_YEAR_DATES)
        else:
            mask = ~xdata.isin(LEAP_YEAR_DATES)
        xdata = xdata.where(mask, drop=True)
        npdata = xdata.values
        npdata = np.moveaxis(npdata, 0, 2)  # lat x lon x time
        npdata = transform.resize(npdata, input_shape)
        npdata *= 86400  # convert kg,m-2,s-1 to mm,day-1
        npdata = np.moveaxis(npdata, 2, 0)  # time x lat x lon
        output.append(npdata)
    output = np.stack(output, 1)  # time x source x lat x lon
    output = np.moveaxis(output, 2, 3)  # time x source x lon x lat
    with open(dst, 'wb') as f:
        np.save(f, output)