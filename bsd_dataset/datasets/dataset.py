import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms
import xarray as xr

from bsd_dataset.regions import Region
from bsd_dataset.datasets.download_utils import download_urls
from bsd_dataset.datasets.dataset_utils import irange, fix_dates, match_image_sizes, LEAP_YEAR_DATES


class BSDDataset(torch.utils.data.Dataset):

    def __init__(
        self, 
        input_datasets: List[str],
        target_dataset: str,
        train_region: Region,
        val_region: Region,
        test_region: Region,
        train_dates: Tuple[str, str],
        val_dates: Tuple[str, str],
        test_dates: Tuple[str, str],
        auxiliary_datasets: List[str] = [],
        variable_dictionary: Dict[str, Any] = {},
        transform: torchvision.transforms = None,
        target_transform: torchvision.transforms = None,
        download: bool = False,
        extract: bool = False,
        root: str = './data'):
        """
        Parameters:
            See get_dataset.py.
        """
        # Save parameters
        self.transform = transform
        self.target_transform = target_transform

        # Define spatial coverage
        def get_lons_lats(region: Region) -> Tuple[np.array, np.array]:
            lons, lats = [0, 0], [0, 0]
            lons[0], lats[0] = region.top_left
            lons[1], lats[1] = region.bottom_right
            lons = np.array(lons)
            lats = np.array(lats)
            return lons, lats
        train_lons, train_lats = get_lons_lats(train_region)
        val_lons, val_lats = get_lons_lats(val_region)
        test_lons, test_lats = get_lons_lats(test_region)

        # Download datasets
        self.root = root
        self.target_direcs = []
        n_workers = 5

        if download:
            
            download_args = {}
            datasets = input_datasets + [target_dataset] + auxiliary_datasets
            train_years = [int(d.split('-')[0]) for d in train_dates]
            val_years = [int(d.split('-')[0]) for d in val_dates]
            test_years = [int(d.split('-')[0]) for d in test_dates]
            
            for ds in datasets:
                dst_direc = os.path.join(root, ds)
                if ds.startswith('chirps'):
                    res = ds.split('_')[1]
                    urls = self._get_chirps_urls(res, train_years)
                    urls += self._get_chirps_urls(res, val_years)
                    urls += self._get_chirps_urls(res, test_years)                    
                elif ds == 'cgcm3':
                    self.target_direcs.append(dst_direc)
                    urls = self._get_cgcm3_urls(train_years)
                    urls += self._get_cgcm3_urls(val_years)
                    urls += self._get_cgcm3_urls(test_years)
                elif ds == 'cm3':
                    self.target_direcs.append(dst_direc)
                    urls = self._get_cm3_urls(train_years)
                    urls += self._get_cm3_urls(val_years)
                    urls += self._get_cm3_urls(test_years)
                elif ds == 'cm5a':
                    self.target_direcs.append(dst_direc)
                    urls = self._get_cm5a_urls(train_years)
                    urls += self._get_cm5a_urls(val_years)
                    urls += self._get_cm5a_urls(test_years)
                elif ds.startswith('gmted2010'):
                    res = ds.split('_')[1]
                    res = float(res[0] + '.' + res[1:])
                    urls = self._get_gmted2010_urls(res)
                else:
                    raise NotImplementedError
                    
                download_args[dst_direc] = list(set(urls))
            
            for dst_direc, urls in download_args.items():
                download_urls(urls, dst_direc, n_workers=n_workers)
        
        # Extract data
        if download or extract:
            # Target data
            fnames = ['train_y.npy', 'val_y.npy', 'test_y.npy']
            all_dates = [train_dates, val_dates, test_dates]
            all_lats = [train_lats, val_lats, test_lats]
            all_lons = [train_lons, val_lons, test_lons]
            if target_dataset.startswith('chirps_25'):
                src = os.path.join(root, target_dataset)
                for fname, dates, lats, lons in zip(fnames, all_dates, all_lats, all_lons):
                    self._extract_chirps_data(src, os.path.join(root, fname), lons, lats, dates)
            else:
                raise NotImplementedError

            # Input data
            src = [os.path.join(root, direc) for direc in input_datasets]
            fnames = ['train_x.npy', 'val_x.npy', 'test_x.npy']
            for fname, dates, lats, lons in zip(fnames, all_dates, all_lats, all_lons):
                self._extract_cmip5_data(src, os.path.join(root, fname), lons, lats, dates)
                
            # Auxiliary data
            fname_templates = ['train', 'val', 'test']
            for ds in auxiliary_datasets:
                src = os.path.join(root, ds)
                if ds.startswith('gmted2010'):
                    fnames = [f'{ft}_{ds}.npy' for ft in fname_templates]
                    for fname, lats, lons in zip(fnames, all_lats, all_lons):
                        self._extract_gmted2010_data(src, os.path.join(root, fname), lons, lats)

    def _get_chirps_urls(self, res: str, years: Tuple[int, int]) -> List[str]:        
        if years[0] < 1981 or years[1] > 2021:
            raise ValueError('Requested CHIRPS data is out of range, must be in 1981-2021')
        urls = [
            'ftp://anonymous@ftp.chc.ucsb.edu/pub/org/chg/products/CHIRPS-2.0/'
            f'global_daily/netcdf/p{res}/chirps-v2.0.{year}.days_p{res}.nc'
            for year in irange(*years)
        ]
        return urls

    def _get_cgcm3_urls(self, years: Tuple[int, int]) -> List[str]:
        if years[0] < 1950 or years[1] > 2005:
            raise ValueError('Requested CGCM3 data is out of range, must be in 1950-2005')
        start = years[0] // 10 * 10
        end = years[1] // 10 * 10 + 10        
        years = range(start, end, 10)
        urls = [
            'http://esgf-data1.diasjp.net/thredds/fileServer/esg_dataroot/'
            'cmip5/output1/MRI/MRI-CGCM3/historical/day/atmos/day/r1i1p1/'
            f'v20120701/prc/prc_day_MRI-CGCM3_historical_r1i1p1_{year}0101'
            f'-{year+9}1231.nc'
            for year in years
        ]
        # Last file from cgcm3 is 2000-2005 instead of 2000-2009
        if urls[-1].endswith('-20091231.nc'):
            urls[-1] = '-'.join(urls[-1].split('-')[:-1]) + '-20051231.nc'
        return urls

    def _get_cm3_urls(self, years: Tuple[int, int]) -> List[str]:
        if years[0] < 1860 or years[1] > 2005:
            raise ValueError('Requested CM3 data is out of range, must be in 1860-2005')
        def round_to_nearest_five(x):
            if x % 10 < 5:
                return x // 10 * 10
            else:
                return x // 10 * 10 + 5
        start = round_to_nearest_five(years[0])
        end = round_to_nearest_five(years[1] + 5)
        years = range(start, end, 5)
        urls = [
            'http://aims3.llnl.gov/thredds/fileServer/css03_data/cmip5/'
            'output1/NOAA-GFDL/GFDL-CM3/historical/day/atmos/day/r1i1p1/'
            f'v20120227/prc/prc_day_GFDL-CM3_historical_r1i1p1_{year}0101'
            f'-{year+4}1231.nc'
            for year in years
        ]
        # Last file from cm3 is 2005-2005 instead of 2005-2009
        if urls[-1].endswith('-20091231.nc'):
            urls[-1] = '-'.join(urls[-1].split('-')[:-1]) + '-20051231.nc'
        return urls

    def _get_cm5a_urls(self, years: Tuple[int, int]) -> List[str]:
        if min(years) < 1850 or max(years) > 2005:
            raise ValueError('Requested CM5A data is out of range, must be in 1850-2005')
        urls = [
            'http://aims3.llnl.gov/thredds/fileServer/cmip5_css01_data/cmip5/'
            'output1/IPSL/IPSL-CM5A-LR/historical/day/atmos/day/r1i1p1/'
            'v20110909/prc/prc_day_IPSL-CM5A-LR_historical_r1i1p1_19500101'
            '-20051231.nc'
        ]
        return urls

    def _get_gmted2010_urls(self, res: float) -> List[str]:
        available_resolutions = [0.0625, 0.125, 0.250, 0.500, 0.750, 1.000]
        if res == 0.0625:
            return ['https://d1qb6yzwaaq4he.cloudfront.net/data/gmted2010/GMTED2010_15n015_00625deg.nc']
        elif res == 0.125:
            return ['https://d1qb6yzwaaq4he.cloudfront.net/data/gmted2010/GMTED2010_15n030_0125deg.nc']
        elif res == 0.250:
            return ['https://d1qb6yzwaaq4he.cloudfront.net/data/gmted2010/GMTED2010_15n060_0250deg.nc']
        elif res == 0.500:
            return ['https://d1qb6yzwaaq4he.cloudfront.net/data/gmted2010/GMTED2010_15n060_0250deg.nc']
        elif res == 0.750:
            return ['https://d1qb6yzwaaq4he.cloudfront.net/data/gmted2010/GMTED2010_15n180_0750deg.nc']
        elif res == 1.000:
            return ['https://d1qb6yzwaaq4he.cloudfront.net/data/gmted2010/GMTED2010_15n240_1000deg.nc']
        else:
            raise ValueError(f'Requested GMTED resolution is unavailable, must be in {available_resolutions}')
            
    def _extract_chirps_data(self, src: str, dst: str, lons: np.array, lats: np.array, dates: Tuple[str, str]) -> None:
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
            latitude=slice(*sorted(lats)), 
            longitude=slice(*sorted(lons)))
        mask = ~xdata['time'].isin(LEAP_YEAR_DATES)
        xdata = xdata.where(mask, drop=True)
        npdata = xdata.values  # time x lat x lon
        npdata = np.moveaxis(npdata, 1, 2)  # time x lon x lat
        with open(dst, 'wb') as f:
            np.save(f, npdata)

    def _extract_cmip5_data(self, src: List[str], dst: str, lons: np.array, lats: np.array, dates: Tuple[str, str]) -> None:
        """
        Extract low resolution input data.

        Parameters:
          - src: the directories to read from
          - dst: the file to write to
          - lons: the longitudinal bounds
          - lats: the latitudinal bounds
          - dates: the temporal range of the data
        """
        output = []
        target_shape = (0, 0, 0)
        for direc in src:
            ds = xr.open_mfdataset(f'{direc}/*.nc')
            lats = sorted([lats[0]%360, lats[1]%360])
            lons = sorted([lons[0]%360, lons[1]%360])
            xdata = ds.prc.sel(
                time=slice(*dates),
                lat=slice(*lats),
                lon=slice(*lons)
            )
            if 'cgcm3' in direc:
                xdata = fix_dates(xdata)
                mask = ~xdata['time'].isin(LEAP_YEAR_DATES)
            else:
                mask = ~xdata['time'].isin(LEAP_YEAR_DATES)
            npdata = xdata.where(mask, drop=True).values
            npdata = np.moveaxis(npdata, 0, 2)  # lat x lon x time            
            npdata *= 86400  # convert kg,m-2,s-1 to mm,day-1
            npdata = np.moveaxis(npdata, 2, 0)  # time x lat x lon
            output.append(npdata)
            area = npdata.shape[1] * npdata.shape[2]
            largest_area = target_shape[1] * target_shape[2]
            if area > largest_area:
                largest_shape = npdata.shape
        output = match_image_sizes(output, largest_shape)
        output = np.stack(output, 1)  # time x source x lat x lon
        output = np.moveaxis(output, 2, 3)  # time x source x lon x lat
        with open(dst, 'wb') as f:
            np.save(f, output)
            
    def _extract_gmted2010_data(self, src: str, dst: str, lons: np.array, lats: np.array) -> None:
        """
        Extract auxiliary data.

        Parameters:
          - src: the directory to read from
          - dst: the npy file to write to
          - lons: the longitudinal bounds
          - lats: the latitudinal bounds
        """
        ds = xr.open_mfdataset(f'{src}/*.nc')
        xdata = ds.elevation.sel(
            latitude=slice(*sorted(lats)), 
            longitude=slice(*sorted(lons)))
        npdata = xdata.values.T  # lon x lat
        with open(dst, 'wb') as f:
            np.save(f, npdata)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        y = self.Y[idx]
        if self.target_transform:
            y = self.target_transform(y)
        mask = np.isnan(y)
        return x, y, mask

    def get_subset(self, split: str):
        """
        Loads the training, validation, or test data.
        Parameters:
            - split: train, val, or test
        """
        def load_XY(Xfile, Yfile):
            with open(Xfile, 'rb') as f:
                self.X = np.load(f)
            with open(Yfile, 'rb') as f:
                self.Y = np.load(f)
        if split == 'train':
            load_XY('./data/train_x.npy', './data/train_y.npy')
        elif split == 'val':
            load_XY('./data/val_x.npy', './data/val_y.npy')
        elif split == 'test':
            load_XY('./data/test_x.npy', './data/test_y.npy')
        else:
            print(f'Split {split} not recognized')
        return self
