from pathlib import Path
import tarfile
from typing import Dict, List, Optional, Tuple, Union
from typing_extensions import Self

import numpy as np
import torch
import torchvision.transforms
import xarray as xr

from bsd_dataset.regions import Region
from bsd_dataset.datasets.download_utils import (
    download_urls, 
    multidownload_from_cds, 
    CDSAPIRequest
)
from bsd_dataset.datasets.dataset_utils import (
    irange, 
    get_shape_of_largest_array, 
    match_array_shapes, 
    lon180_to_lon360, 
    LEAP_YEAR_DATES
)


class BSDDataset(torch.utils.data.Dataset):

    def __init__(
        self, 
        input_datasets: Dict[str, Dict[str, Union[str, List[str]]]],
        target_dataset: str,
        train_region: Region,
        val_region: Region,
        test_region: Region,
        train_dates: Tuple[str, str],
        val_dates: Tuple[str, str],
        test_dates: Tuple[str, str],
        transform: Optional[torchvision.transforms.Compose],
        target_transform: Optional[torchvision.transforms.Compose],
        download: bool,
        extract: bool,
        root: Path
    ) -> Self:
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
       
        # Download data        
        if download:

            # Get CHIRPS and GMTED2010 data
            urls, dsts = [], []
            if target_dataset.startswith('chirps'):
                train_years = [int(d.split('-')[0]) for d in train_dates]
                val_years = [int(d.split('-')[0]) for d in val_dates]
                test_years = [int(d.split('-')[0]) for d in test_dates]
                res = target_dataset.split('_')[1]
                chirps_urls = list(set(
                    self._get_chirps_urls(res, train_years) +
                    self._get_chirps_urls(res, val_years) +
                    self._get_chirps_urls(res, test_years)))
                chirps_dst = root / target_dataset
                urls += chirps_urls
                dsts += [chirps_dst] * len(chirps_urls)
            for ds in input_datasets.keys():
                if ds.startswith('gmted2010'):
                    res = ds.split('_')[1]
                    res = float(res[0] + '.' + res[1:])
                    urls += self._get_gmted2010_urls(res)
                    dsts.append(root / ds)
            download_urls(urls, dsts, n_workers=5)

            # Get CDS data
            cds_api_requests = {}
            for ds, options in input_datasets.items():
                if ds.startswith('cds'):
                    dataset, model = ds.split(':')[1:]
                    options['experiment'] = 'historical'
                    options['format'] = 'tgz'
                    output = root / 'cds' / f'{dataset}.{model}.tar.gz'
                    req = CDSAPIRequest(dataset, options, output)
                    cds_api_requests[ds] = req
            multidownload_from_cds(cds_api_requests, n_workers=5)
        
        # Extract data
        if extract:

            # Get CHIRPS data
            if target_dataset.startswith('chirps'):
                fnames = [root / 'train_y.npy', root / 'val_y.npy', root / 'test_y.npy']
                all_dates = [train_dates, val_dates, test_dates]
                all_lats = [train_lats, val_lats, test_lats]
                all_lons = [train_lons, val_lons, test_lons]
                src = root / target_dataset
                iterzip = zip(fnames, all_dates, all_lats, all_lons)
                for fname, dates, lats, lons in iterzip:
                    npdata = self._extract_chirps_data(src, lons, lats, dates)
                    with open(fname, 'wb') as f:
                        np.save(f, npdata)

            # Extract input
            def extract_input_data(dst: Path, lons: np.array, lats: np.array, dates: Tuple[str, str]) -> None:
                lons = lon180_to_lon360(lons)
                data = []
                for ds in input_datasets.keys():
                    if ds.startswith('gmted2010'):
                        src = root / ds                   
                        npdata = self._extract_gmted2010_data(src, lons, lats, dates)
                        data.append(npdata)
                    if ds.startswith('cds'):
                        src = cds_api_requests[ds].output
                        npdata = self._extract_cds_data(src, lons, lats, dates)
                        data.append(npdata)
                shape = get_shape_of_largest_array(data)
                data = match_array_shapes(data, shape)
                data = np.stack(data)  # channel x time x lon x lat
                with open(dst, 'wb') as f:
                    np.save(f, npdata)

            extract_input_data(root / 'train_x.npy', train_lons, train_lats, train_dates)
            extract_input_data(root / 'val_x.npy', val_lons, val_lats, val_dates)
            extract_input_data(root / 'test_x.npy', test_lons, test_lats, test_dates)


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
        """
        Parameters:
            res: The resolution to download.
        """
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
            
    def _extract_chirps_data(self, src: Path, lons: np.array, lats: np.array, dates: Tuple[str, str]) -> np.array:
        """
        Parameters:
            src: The directory to read from.
            lons: The longitudinal bounds.
            lats: The latitudinal bounds.
            dates: The temporal range of the data.
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
        return npdata
            
    def _extract_gmted2010_data(self, src: Path, lons: np.array, lats: np.array, dates: Tuple[str, str]) -> np.array:
        """
        Parameters:
            src: The directory to read from.
            lons: The longitudinal bounds.
            lats: The latitudinal bounds.
            dates: The dates to cover.
        """
        ds = xr.open_mfdataset(f'{src}/*.nc')
        xdata = ds.elevation.sel(
            latitude=slice(*sorted(lats)), 
            longitude=slice(*sorted(lons)))
        npdata = xdata.values.T  # lon x lat
        npdata = np.expand_dims(npdata, 0)
        n_days = (np.datetime64(dates[1]) - np.datetime64(dates[0])).item().days
        npdata = np.repeat(npdata, n_days, axis=0)  # time x lon x lat
        return npdata

    def _extract_cds_data(self, src: Path, lons: np.array, lats: np.array, dates: Tuple[str, str]) -> np.array:
        """
        Parameters:
            src: The tarfile to read from.
            lons: The longitudinal bounds.
            lats: The latitudinal bounds.
            dates: The dates to cover.
        """
        tar = tarfile.open(src)
        fname = tar.getnames()[0]
        var_name = fname.split('_')[0]
        tar.extract(fname, path=src.parent)
        tar.close()
        ds = xr.open_dataset(src.parent / fname)
        xdata = getattr(ds, var_name).sel(
            time=slice(*dates),
            latitude=slice(*sorted(lats)),
            longitude=slice(*sorted(lons)))
        npdata = xdata.values  # time x lat x lon
        npdata = np.moveaxis(npdata, 1, 2)  # time x lon x lat
        return npdata

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        y = self.Y[idx]
        if self.target_transform:
            y = self.target_transform(y)
        mask = np.isnan(y)
        return x, y, mask

    def get_subset(self, split: str) -> Self:
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
