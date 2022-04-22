import os
from typing import Any, Dict, List, Optional, Tuple

import cdsapi
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
        transform: Optional[torchvision.transforms.Compose] = None,
        target_transform: Optional[torchvision.transforms.Compose] = None,
        download: Dict[str, bool] = {},
        extract: Dict[str, bool] = {},
        root: str = './data'):
        """
        Parameters:
            See get_dataset.py.
        """
        # Save parameters
        self.transform = transform
        self.target_transform = target_transform
        self.root = root

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
       
        # Get target data and auxiliary data
        urls, dsts = [], []
        if download.get(target_dataset, False):
            if target_dataset.startswith('chirps'):
                train_years = [int(d.split('-')[0]) for d in train_dates]
                val_years = [int(d.split('-')[0]) for d in val_dates]
                test_years = [int(d.split('-')[0]) for d in test_dates]
                res = target_dataset.split('_')[1]
                chirps_urls = list(set(
                    self._get_chirps_urls(res, train_years) +
                    self._get_chirps_urls(res, val_years) +
                    self._get_chirps_urls(res, test_years)))
                chirps_dst = os.path.join(root, target_dataset)
                urls += chirps_urls
                dsts += [chirps_dst] * len(chirps_urls)     
        for ads in auxiliary_datasets:
            if download.get(ads, False):
                if ads.startswith('gmted2010'):
                    res = ads.split('_')[1]
                    res = float(res[0] + '.' + res[1:])
                    urls += self._get_gmted2010_urls(res)
                    dsts += [os.path.join(root, ads)]
        download_urls(urls, dsts, n_workers=5)

        # Get input data
        c = cdsapi.Client()
        for idata in input_datasets:
            if download.get(idata, False):
                options = variable_dictionary[idata]
                options['format'] = 'tgz'
                output = f'{idata}.tar.gz'
                c.retrieve(idata, options, output)
        
        # Extract target data
        if extract.get(target_dataset, False):            
            if target_dataset.startswith('chirps'):
                fnames = ['train_y.npy', 'val_y.npy', 'test_y.npy']
                all_dates = [train_dates, val_dates, test_dates]
                all_lats = [train_lats, val_lats, test_lats]
                all_lons = [train_lons, val_lons, test_lons]
                src = os.path.join(root, target_dataset)
                for fname, dates, lats, lons in zip(fnames, all_dates, all_lats, all_lons):
                    self._extract_chirps_data(src, os.path.join(root, fname), lons, lats, dates)
        
        # Extract auxiliary data
        for ads in auxiliary_datasets:
            if extract.get(ads, False):                
                if ads.startswith('gmted2010'):
                    src = os.path.join(root, ads)
                    fnames = [f'{f}_{ads}.npy' for f in ['train', 'val', 'test']]
                    all_lats = [train_lats, val_lats, test_lats]
                    all_lons = [train_lons, val_lons, test_lons]
                    for fname, lats, lons in zip(fnames, all_lats, all_lons):
                        self._extract_gmted2010_data(src, os.path.join(root, fname), lons, lats)

        # Extract input data
        for ids in input_datasets:
            if extract.get(ids, False):
                pass

    def _get_chirps_urls(self, res: str, years: Tuple[int, int]) -> List[str]:        
        if years[0] < 1981 or years[1] > 2021:
            raise ValueError('Requested CHIRPS data is out of range, must be in 1981-2021')
        urls = [
            'ftp://anonymous@ftp.chc.ucsb.edu/pub/org/chg/products/CHIRPS-2.0/'
            f'global_daily/netcdf/p{res}/chirps-v2.0.{year}.days_p{res}.nc'
            for year in irange(*years)
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
