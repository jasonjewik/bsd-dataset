from typing import Any, Dict, List, Tuple

import numpy as np
import os
import torch
import torchvision.transforms

from bsd_dataset.regions import RegionCoordinates
from bsd_dataset.datasets.download_utils import download_urls
from bsd_dataset.datasets.dataset_utils import extract_chirps25_data, extract_cmip5_data


class BSDDataset(torch.utils.data.Dataset):

    def __init__(
        self, 
        input_datasets: List[str],
        target_dataset: str,
        train_region: RegionCoordinates,
        val_region: RegionCoordinates,
        test_region: RegionCoordinates,
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
        def get_lons_lats(region: RegionCoordinates) -> Tuple[np.array, np.array]:
            lons, lats = [0, 0], [0, 0]
            lons[0], lats[0] = region.top_left_corner
            lons[1], lats[1] = region.bottom_right_corner
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
            datasets = input_datasets + [target_dataset] + auxiliary_datasets
            for ds in datasets:
                dst_direc = os.path.join(root, ds.upper())
                if ds == 'chirps25':
                    urls = self._get_chirps25_urls(train_dates)
                    urls += self._get_chirps25_urls(val_dates)
                    urls += self._get_chirps25_urls(test_dates)
                elif ds == 'cgcm3':
                    self.target_direcs.append(dst_direc)
                    urls = self._get_cgcm3_urls(train_dates)
                    urls += self._get_cgcm3_urls(val_dates)
                    urls += self._get_cgcm3_urls(test_dates)
                elif ds == 'cm3':
                    self.target_direcs.append(dst_direc)
                    urls = self._get_cm3_urls(train_dates)
                    urls += self._get_cm3_urls(val_dates)
                    urls += self._get_cm3_urls(test_dates)
                elif ds == 'cm5a':
                    self.target_direcs.append(dst_direc)
                    urls = self._get_cm5a_urls(train_dates)
                    urls += self._get_cm5a_urls(val_dates)
                    urls += self._get_cm5a_urls(test_dates)
                else:
                    raise NotImplementedError
                    
                urls = list(set(urls))
                download_urls(urls, dst_direc, n_workers=n_workers)
        
        # Extract data
        if download or extract:
            # Target data
            fnames = ['train_y.npy', 'val_y.npy', 'test_y.npy']
            all_dates = [train_dates, val_dates, test_dates]
            all_lats = [train_lats, val_lats, test_lats]
            all_lons = [train_lons, val_lons, test_lons]
            if target_dataset == 'chirps25':
                src = os.path.join(root, 'CHIRPS25')
                for fname, dates, lats, lons in zip(fnames, all_dates, all_lats, all_lons):
                    extract_chirps25_data(src, os.path.join(root, fname), lons, lats, dates)
            else:
                raise NotImplementedError

            # Input data
            src = [os.path.join(root, direc) for direc in ['CGCM3', 'CM3', 'CM5A']]
            fnames = ['train_x.npy', 'val_x.npy', 'test_x.npy']            
            for fname, dates, lats, lons in zip(fnames, all_dates, all_lats, all_lons):
                extract_cmip5_data(src, os.path.join(root, fname), lons, lats, dates)

    def _get_chirps25_urls(self, dates: Tuple[str, str]) -> List[str]:
        start = int(dates[0].split('-')[0])
        end = int(dates[1].split('-')[0]) + 1
        years = range(start, end)
        urls = [
            'ftp://anonymous@ftp.chc.ucsb.edu/pub/org/chg/products/CHIRPS-2.0/'
            f'global_daily/netcdf/p25/chirps-v2.0.{year}.days_p25.nc'
            for year in years
        ]
        return urls

    def _get_cgcm3_urls(self, dates: Tuple[str, str]) -> List[str]:
        start = int(dates[0].split('-')[0]) // 10 * 10
        end = int(dates[1].split('-')[0]) // 10 * 10 + 10
        years = range(start, end, 10)
        urls = [
            'http://esgf-data1.diasjp.net/thredds/fileServer/esg_dataroot/'
            'cmip5/output1/MRI/MRI-CGCM3/historical/day/atmos/day/r1i1p1/'
            f'v20120701/prc/prc_day_MRI-CGCM3_historical_r1i1p1_{year}0101'
            f'-{year+9}1231.nc'
            for year in years
        ]
        # last file from cgcm3 is 2000-2005 instead of 2000-2009
        for i in range(len(urls)):
            if urls[i].endswith('-20091231.nc'):
                urls[i] = '-'.join(urls[i].split('-')[:-1]) + '-20051231.nc'
        return urls

    def _get_cm3_urls(self, dates: Tuple[str, str]) -> List[str]:
        def round_to_nearest_five(x):
            if x % 10 < 5:
                return x // 10 * 10
            else:
                return x // 10 * 10 + 5
        start = round_to_nearest_five(int(dates[0].split('-')[0]))
        end = round_to_nearest_five(int(dates[1].split('-')[0]) + 5)
        years = range(start, end, 5)
        urls = [
            'http://aims3.llnl.gov/thredds/fileServer/css03_data/cmip5/'
            'output1/NOAA-GFDL/GFDL-CM3/historical/day/atmos/day/r1i1p1/'
            f'v20120227/prc/prc_day_GFDL-CM3_historical_r1i1p1_{year}0101'
            f'-{year+4}1231.nc'
            for year in years
        ]
                # last file from cm3 is 2005-2005 instead of 2005-2009
        for i in range(len(urls)):
            if urls[i].endswith('-20091231.nc'):
                urls[i] = '-'.join(urls[i].split('-')[:-1]) + '-20051231.nc'
        return urls

    def _get_cm5a_urls(self, dates: Tuple[str, str]) -> List[str]:
        urls = [
            'http://aims3.llnl.gov/thredds/fileServer/cmip5_css01_data/cmip5/'
            'output1/IPSL/IPSL-CM5A-LR/historical/day/atmos/day/r1i1p1/'
            'v20110909/prc/prc_day_IPSL-CM5A-LR_historical_r1i1p1_19500101'
            '-20051231.nc'
        ]
        return urls

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        y = self.Y[idx]
        if self.target_transform:
            y = self.target_transform(y)
        return x, y

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
    
