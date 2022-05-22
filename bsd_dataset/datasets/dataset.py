from pathlib import Path
import os
import tarfile
from typing import List, Optional, Tuple
from typing_extensions import Self

import numpy as np
import torch
import torchvision.transforms
import xarray as xr

from bsd_dataset.regions import Region
from bsd_dataset.datasets.download_utils import (
    CDSAPIRequest,
    DatasetRequest,
    CDSAPIRequestBuilder,
    download_urls, 
    multidownload_from_cds,     
)
from bsd_dataset.datasets.dataset_utils import (
    get_shape_of_largest_array, 
    match_array_shapes, 
    lon180_to_lon360,
    get_lon_mask
)


class BSDDataset(torch.utils.data.Dataset):

    def __init__(
        self, 
        input_datasets: List[DatasetRequest],
        target_dataset: DatasetRequest,
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
    ):
        # Save parameters
        self.input_datasets = input_datasets
        self.target_dataset = target_dataset
        self.train_region = train_region
        self.val_region = val_region
        self.test_region = test_region
        self.train_dates = train_dates
        self.val_dates = val_dates
        self.test_dates = test_dates
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        
        # Check input data
        cds_api_requests = []
        builder = CDSAPIRequestBuilder()
        input_urls, input_dstdirs = [], []
        for ds_req in input_datasets:
            if ds_req.is_cds_req():
                cds_api_req = builder.build(root, ds_req, train_dates, val_dates, test_dates)
                cds_api_requests.append(cds_api_req)
            if ds_req.dataset == 'gmted2010':
                input_urls.append(self.get_gmted2010_url(ds_req))
                input_dstdirs.append(root / 'gmted2010')
    
        # Check target data
        target_urls, target_dstdirs = [], []
        if target_dataset.dataset == 'chirps':
            chirps_urls = self.get_chirps_urls(target_dataset)            
            chirps_dstdirs = [root / target_dataset] * len(chirps_urls)
            target_urls.extend(chirps_urls)
            target_dstdirs.extend(chirps_dstdirs)
       
        # Download data
        if download:
            download_urls(input_urls + target_urls, input_dstdirs + target_dstdirs, n_workers=5)
            multidownload_from_cds(cds_api_requests, n_workers=5)
        
        # Extract data
        if extract:
            
            splits = ['train', 'val', 'test']
            cds_direcs = self.extract_cds_targz(cds_api_requests)
            # Returns a list (split) of lists (data source) of Numpy arrays
            cds_data = list(map(self.extract_cds_data, splits, [cds_direcs] * len(splits)))
            # Returns a list (split) of NumPy arrays
            gmted2010_data = list(map(self.extract_gmted2010_data, splits, [root / 'gmted2010'] * len(splits)))

            # Save separately to save room on disk, scaling and concatenation can happen later
            for split, cdsd, gmtedd in zip(splits, cds_data, gmted2010_data):
                with open(root / f'{split}_x.npz', 'wb') as f:
                    np.savez(f, *cdsd, gmted2010=gmtedd)

            if target_dataset.dataset == 'chirps':
                # Returns a list (split) of NumPy arrays
                target_data = list(map(self.extract_chirps_data, splits, target_dstdirs))
            
            for split, td in zip(splits, target_data):
                with open(root / f'{split}_y.npz', 'wb') as f:
                    np.savez(f, target=td)

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
        if split == 'train':
            self.load_XY('train_x.npz', 'train_y.npz')
        elif split == 'val':
            self.load_XY('val_x.npz', 'val_y.npz')
        elif split == 'test':
            self.load_XY('test_x.npz', 'test_y.npz')
        else:
            print(f'Split {split} not recognized')
        return self

    def load_XY(self, Xfile, Yfile):
        with open(self.root / Xfile, 'rb') as f:
            npzfile = np.load(f)
            arrs = [npzfile[key] for key in npzfile.files if key != 'gmted2010']
            shape = get_shape_of_largest_array(arrs)
            xdata = match_array_shapes(arrs, shape)
            if 'gmted2010' in npzfile.files:
                n_days = xdata.shape[0]
                gmted2010 = npzfile['gmted2010']
                gmted2010 = np.expand_dims(gmted2010, 0)
                gmted2010 = np.repeat(gmted2010, n_days, axis=0)
                xdata = np.concatenate(xdata, gmted2010)
            self.X = xdata
        with open(self.root / Yfile, 'rb') as f:
            npzfile = np.load(f)
            self.Y = npzfile['target']

    def get_lons_lats(self, region: Region) -> Tuple[np.array, np.array]:
        lons, lats = [0, 0], [0, 0]
        lons[0], lats[0] = region.top_left
        lons[1], lats[1] = region.bottom_right
        lons = np.array(lons)
        lats = np.array(lats)
        return lons, lats

    def get_gmted2010_url(self, ds_req: DatasetRequest) -> str:
        resolutions = [0.0625, 0.125, 0.250, 0.500, 0.750, 1.000]
        res = getattr(ds_req, 'resolution', None)
        if res not in resolutions:
            raise AttributeError(
                'gmted2010 dataset has invalid resolution\n'
                f'available resolutions are {resolutions}'
            )
        if res == 0.0625:
            url = 'https://d1qb6yzwaaq4he.cloudfront.net/data/gmted2010/GMTED2010_15n015_00625deg.nc'
        elif res == 0.125:
            url = 'https://d1qb6yzwaaq4he.cloudfront.net/data/gmted2010/GMTED2010_15n030_0125deg.nc'
        elif res == 0.250:
            url = 'https://d1qb6yzwaaq4he.cloudfront.net/data/gmted2010/GMTED2010_15n060_0250deg.nc'
        elif res == 0.500:
            url = 'https://d1qb6yzwaaq4he.cloudfront.net/data/gmted2010/GMTED2010_15n060_0250deg.nc'
        elif res == 0.750:
            url = 'https://d1qb6yzwaaq4he.cloudfront.net/data/gmted2010/GMTED2010_15n180_0750deg.nc'
        elif res == 1.000:
            url = 'https://d1qb6yzwaaq4he.cloudfront.net/data/gmted2010/GMTED2010_15n240_1000deg.nc'
        return url

    def get_chirps_urls(self, ds_req: DatasetRequest) -> List[str]:
        resolutions = [0.05, 0.25]
        res = getattr(ds_req, 'resolution', None)        
        if res not in resolutions:
            raise AttributeError(
                'chirps dataset has invalid resolution,'
                f' available resolutions are {resolutions}'
            )
        
        train_years = [int(d.split('-')[0]) for d in self.train_dates]
        val_years = [int(d.split('-')[0]) for d in self.val_dates]
        test_years = [int(d.split('-')[0]) for d in self.test_dates]
        urls = []
        
        for a, b in [train_years, val_years, test_years]:
            if not (a in range(1981, 2022) and b in range(1981, 2022)):
                raise ValueError(
                    f'requested dates ({a}, {b}) are out of range for CHIRPS,'
                    ' which must be in 1981-2021'
                )
            urls.extend([
                'ftp://anonymous@ftp.chc.ucsb.edu/pub/org/chg/products/CHIRPS-2.0/'
                f'global_daily/netcdf/p{res}/chirps-v2.0.{year}.days_p{res}.nc'
                for year in range(a, b+1)
            ])
        
        return urls

    def extract_cds_targz(self, cds_api_requests: List[CDSAPIRequest]) -> List[str]:
        direcs = []
        for req in cds_api_requests:
            src = req.output
            with tarfile.open(src) as tar:
                dir_name = src.name.split('.')[0]
                direcs.append(dir_name)
                tar.extractall(path=dir_name)
        return direcs

    def extract_cds_data(self, split: str, src_dir: Path) -> np.array:
        region = getattr(self, f'self.{split}_region')
        dates = getattr(self, f'self.{split}_dates')
        lons, lats = self.get_lons_lats(region)
        lons = lon180_to_lon360(lons)
        var_names = os.listdir(src_dir)
        result = []
        for vn in var_names:
            ds = xr.open_mfdataset(f'{src_dir}/{vn}*.nc')
            xdata = getattr(ds, vn).sel(
                time=slice(*dates),
                lat=slice(*sorted(lats)),
                lon=get_lon_mask(ds.lon, lons))  # JUST FOR DEMO EDGE CASE!!
            npdata = xdata.values  # time x lat x lon
            npdata = np.moveaxis(npdata, 1, 2)  # time x lon x lat
            result.append(npdata)
        result = np.stack(result, axis=1)  # time x variable x lon x lat
        return result

    def extract_gmted2010_data(self, split: str, src: Path) -> np.array:
        region = getattr(self, f'self.{split}_region')
        lons, lats = self.get_lons_lats(region)
        ds = xr.open_mfdataset(f'{src}/*.nc')
        xdata = ds.elevation.sel(
            latitude=slice(*sorted(lats)), 
            longitude=slice(*sorted(lons)))
        npdata = xdata.values.T  # lon x lat
        return npdata
            
    def extract_chirps_data(self, split: str, src: Path) -> np.array:
        dates = getattr(self, f'{split}_dates')
        region = getattr(self, f'{split}_region')
        lons, lats = self.get_lons_lats(region)
        ds = xr.open_mfdataset(f'{src}/*.nc')
        xdata = ds.precip.sel(
            time=slice(*dates),
            latitude=slice(*sorted(lats)), 
            longitude=slice(*sorted(lons)))
        # Drop leap days since CDS datasets don't have those
        xdata = xdata.sel(time=~((xdata.time.dt.month == 2) & (xdata.time.dt.day == 29)))
        npdata = xdata.values  # time x lat x lon
        npdata = np.moveaxis(npdata, 1, 2)  # time x lon x lat
        return npdata