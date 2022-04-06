import argparse
import multiprocessing as mp
import os
import threading
from typing import List, Tuple, Union
import queue

import numpy as np
import pandas as pd
from skimage import transform
import xarray as xr
import wget

# 2/29 on leap years needs to be removed because
# CM5A and CM3 data don't record those dates
# even though CHIRPS and CGCM3 do
LEAP_YEAR_DATES = np.array([np.datetime64(f'{year}-02-29')
                            for year in range(1984, 2005, 4)])


def download(urls: Union[List[str], str], out_dir: str):
    """
    Downloads data from a list of URLs.
    
    Parameters:
      - urls: A list of URLs or just one URL.
      - out_dir: The directory to write the downloaded files to. Clears the directory, if it exists.
    """

    # Create the target directory, clearing it if it already exists
    try:
        os.makedirs(out_dir)
    except:
        for x in os.listdir(out_dir):
            os.unlink(os.path.join(out_dir, x))

    # Setup worker threads
    q = queue.Queue()
    def worker():
        while True:
            wget.download(q.get(), bar=None, out=out_dir)
            q.task_done()
    for x in range(5):
        threading.Thread(target=worker, daemon=True).start()

    # Download from requested URLs
    if type(urls) == str:
        urls = [urls]
    for url in urls:
        q.put(url)
    q.join()
            
def download_chirps(out_dir: str):
    """
    Downloads the CHIRPS data from 1981 to 2005.
    https://chc.ucsb.edu/data/chirps
        
    Parameters:
      - out_dir: the directory to download to
    """
    years = range(1981, 2006)
    urls = [f'ftp://anonymous@ftp.chc.ucsb.edu/pub/org/chg/products/CHIRPS-2.0/global_daily/netcdf/p25/chirps-v2.0.{year}.days_p25.nc' for year in years]
    download(urls, out_dir)
    print('Retrieved CHIRPS data')
        
def download_cgcm3(out_dir: str):

    """
    Downloads the CGCM3 precipitation data from 1980 to 2005.
    https://aims2.llnl.gov/metagrid/search
    Data description: 
      - project=CMIP5
      - model=MRI-CGCM3
      - Meteorological Research Institute
      - experiment=historical
      - time_frequency=day
      - modeling realm=atmos
      - ensemble=r1i1p1
      - version=20120701
      
    Parameters:
      - out_dir: the directory to download to
    """
    urls = [
        'http://esgf-data1.diasjp.net/thredds/fileServer/esg_dataroot/cmip5/output1/MRI/MRI-CGCM3/historical/day/atmos/day/r1i1p1/v20120701/prc/prc_day_MRI-CGCM3_historical_r1i1p1_19800101-19891231.nc',
        'http://esgf-data1.diasjp.net/thredds/fileServer/esg_dataroot/cmip5/output1/MRI/MRI-CGCM3/historical/day/atmos/day/r1i1p1/v20120701/prc/prc_day_MRI-CGCM3_historical_r1i1p1_19900101-19991231.nc',
        'http://esgf-data1.diasjp.net/thredds/fileServer/esg_dataroot/cmip5/output1/MRI/MRI-CGCM3/historical/day/atmos/day/r1i1p1/v20120701/prc/prc_day_MRI-CGCM3_historical_r1i1p1_20000101-20051231.nc'
    ]
    download(urls, out_dir)
    print('Retrieved CGCM3 data')
        
def download_cm3(out_dir: str):
    """
    Downloads the CM3 precipitation data from 1980 to 2005.
    https://aims2.llnl.gov/metagrid/search
    Data description:
      - project=CMIP5
      - model=GFDL-CM3
      - Geophysical Fluid Dynamics Laboratory
      - experiment=historical
      - time_frequency=day
      - modeling realm=atmos
      - ensemble=r1i1p1
      - version=20120227
      
    Parameters:
      - out_dir: the directory to download to
    """
    urls = [
        'http://aims3.llnl.gov/thredds/fileServer/css03_data/cmip5/output1/NOAA-GFDL/GFDL-CM3/historical/day/atmos/day/r1i1p1/v20120227/prc/prc_day_GFDL-CM3_historical_r1i1p1_19800101-19841231.nc',
        'http://aims3.llnl.gov/thredds/fileServer/css03_data/cmip5/output1/NOAA-GFDL/GFDL-CM3/historical/day/atmos/day/r1i1p1/v20120227/prc/prc_day_GFDL-CM3_historical_r1i1p1_19850101-19891231.nc',
        'http://aims3.llnl.gov/thredds/fileServer/css03_data/cmip5/output1/NOAA-GFDL/GFDL-CM3/historical/day/atmos/day/r1i1p1/v20120227/prc/prc_day_GFDL-CM3_historical_r1i1p1_19900101-19941231.nc',
        'http://aims3.llnl.gov/thredds/fileServer/css03_data/cmip5/output1/NOAA-GFDL/GFDL-CM3/historical/day/atmos/day/r1i1p1/v20120227/prc/prc_day_GFDL-CM3_historical_r1i1p1_19950101-19991231.nc',
        'http://aims3.llnl.gov/thredds/fileServer/css03_data/cmip5/output1/NOAA-GFDL/GFDL-CM3/historical/day/atmos/day/r1i1p1/v20120227/prc/prc_day_GFDL-CM3_historical_r1i1p1_20000101-20041231.nc',
        'http://aims3.llnl.gov/thredds/fileServer/css03_data/cmip5/output1/NOAA-GFDL/GFDL-CM3/historical/day/atmos/day/r1i1p1/v20120227/prc/prc_day_GFDL-CM3_historical_r1i1p1_20050101-20051231.nc'
    ]
    download(urls, out_dir)
    print('Retrieved CM3 data')
    
def download_cm5a(out_dir: str):
    """
    Downloads the CM5A precipitation data from 1950 to 2005.
    https://aims2.llnl.gov/metagrid/search
    Data description:
      - project=CMIP5
      - model=IPSL-CM5A-LR
      - Institut Pierre-Simon Laplace
      - experiment=historical
      - time_frequency=day
      - modeling realm=atmos
      - ensemble=r1i1p1
      - version=20110909
    """
    url = 'http://aims3.llnl.gov/thredds/fileServer/cmip5_css01_data/cmip5/output1/IPSL/IPSL-CM5A-LR/historical/day/atmos/day/r1i1p1/v20110909/prc/prc_day_IPSL-CM5A-LR_historical_r1i1p1_19500101-20051231.nc'
    download(url, out_dir)
    print('Retrieved CM5A data')
    
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

def extract_all_target_data(src: str, dst: str, lons: np.array, lats: np.array, train_dates: Tuple[str, str], val_dates: Tuple[str, str], test_dates: Tuple[str, str]):
    """
    Extract CHIRPS (HR target) data.
    
    Parameters:
      - src: the directory to read from
      - dst: the directory to write to
      - lons: the longitudinal bounds
      - lats: the latitudinal bounds
      - train_dates: the temporal range of the training data
      - val_dates: the temporal range of the validation data
      - test_dates: the temporal range of the test data
    """
    ds = xr.open_mfdataset(f'{src}/*.nc')
    def write_data(out, time_range):
        xdata = ds.precip.sel(
            time=slice(*time_range),
            latitude=slice(*lats), 
            longitude=slice(*lons))
        mask = ~xdata['time'].isin(LEAP_YEAR_DATES)
        xdata = xdata.where(mask, drop=True)
        npdata = xdata.values  # time x lat x lon
        npdata = np.moveaxis(npdata, 1, 2)  # time x lon x lat
        with open(out, 'wb') as f:
            np.save(f, npdata)
    write_data(os.path.join(dst, 'train_y.npy'), train_dates)
    write_data(os.path.join(dst, 'val_y.npy'), val_dates)
    write_data(os.path.join(dst, 'test_y.npy'), test_dates)


def extract_input_data(src: str, dst: str, lons: np.array, lats: np.array, dates: Tuple[str, str]):
    """
    Extract CMIP5 (LR input) data.
    
    Parameters:
      - src: the directory to read from
      - dst: the file to write to
      - lons: the longitudinal bounds
      - lats: the latitudinal bounds
v      - dates: the temporal range of the data
    """
    input_direcs = ['CM5A', 'CM3', 'CGCM3']
    input_shape = (18, 45)  # shape of the largest input (CM5A)
    output = []
    for direc in input_direcs:
        ds = xr.open_mfdataset(f'{src}/{direc}/*.nc')
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downloads CHIRPS/CMIP5 data and writes them to NumPy files.')
    parser.add_argument('chirps', help='the directory to save CHIRPS data to, e.g. ~/CHIRPS')
    parser.add_argument('cmip5', help='the directory to save CMIP5 data to, e.g. ~/CMIP5')
    parser.add_argument('out', help='the directory to write the processed data as npy files to, e.g. ~/out')
    args = parser.parse_args()

    # Download data
    chirps_direc = args.chirps
    cmip5_direc = args.cmip5
    download_chirps(chirps_direc)
    download_cm3(os.path.join(cmip5_direc, 'CM3'))
    download_cm5a(os.path.join(cmip5_direc, 'CM5A'))
    download_cgcm3(os.path.join(cmip5_direc, 'CGCM3'))

    # Write out train, val, and test npy files
    train_dates = ('1981-01-01', '2003-12-31')
    val_dates = ('2004-01-01', '2004-12-31')
    test_dates = ('2005-01-01', '2005-12-31')
    lons = np.array([-125, -75])
    lats = np.array([30, 50])  # approximately covers the continental United States
    out = args.out
    os.makedirs(out, exist_ok=True)
    extract_all_target_data(chirps_direc, out, lons, lats, train_dates, val_dates, test_dates)
    print('Wrote target data')
    extract_input_data(cmip5_direc, os.path.join(out, 'train_x.npy'), lons, lats, train_dates)
    extract_input_data(cmip5_direc, os.path.join(out, 'val_x.npy'), lons, lats, val_dates)
    extract_input_data(cmip5_direc, os.path.join(out, 'test_x.npy'), lons, lats, test_dates)
    print('Wrote input data')
