from collections import defaultdict
from itertools import product
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
        root: Path
    ):
        # Save arguments
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

        # Build parameters
        self.built_download_requests = False

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

    def build_download_requests(self) -> None:
        cds_api_requests, cds_dstdirs = [], []
        builder = CDSAPIRequestBuilder()
        input_urls, input_dstdirs = [], []
        for ds_req in self.input_datasets:
            if ds_req.is_cds_req():
                cds_api_reqs = builder.build(
                    self.root,
                    ds_req, 
                    self.train_dates, 
                    self.val_dates, 
                    self.test_dates
                )
                for car in cds_api_reqs:
                    direc = car.output.parent / car.output.name.split('.')[1]
                    cds_dstdirs.append(direc)
                cds_api_requests.extend(cds_api_reqs)
            if ds_req.dataset == 'gmted2010':
                input_urls.append(self.get_gmted2010_url(ds_req))
                input_dstdirs.append(self.root / 'gmted2010')       
    
        target_urls, target_dstdirs = [], []
        if self.target_dataset.dataset == 'chirps':
            chirps_urls = self.get_chirps_urls(self.target_dataset)
            target_urls.extend(chirps_urls)
            target_dstdirs = [self.root / 'chirps'] * len(chirps_urls)
                
        self.cds_api_requests = cds_api_requests
        self.cds_dstdirs = list(set(cds_dstdirs))
        self.input_urls = input_urls
        self.input_dstdirs = input_dstdirs
        self.target_urls = target_urls
        self.target_dstdirs = target_dstdirs
        self.built_download_requests = True
       
    def download(self):
        if not self.built_download_requests:
            print('ERROR: download requests not yet built')
            return
        print(
            '========================= WARNING =========================\n'
            'If requesting a lot of data (several GB) from CDS, CDS may\n'
            'take a long while to prepare the data for you. You can\n'
            'check on the status of your request at this link:\n'
            'https://cds.climate.copernicus.eu/cdsapp#!/yourrequests.\n'
            '==========================================================='
        )
        download_urls(
            self.input_urls + self.target_urls, 
            self.input_dstdirs + self.target_dstdirs,
            n_workers=5
        )
        multidownload_from_cds(self.cds_api_requests, n_workers=5)
        self.extract_cds_targz()
        
    def extract(self):
        if not self.built_download_requests:
            print('ERROR: download requests not yet built')
            return
        
        splits = ['train', 'val', 'test']

        # Get the CDS data        
        cds_data = defaultdict(dict)
        for spl, direc in product(splits, self.cds_dstdirs):
            data_src = direc.name
            cds_data[spl][data_src] = self.extract_cds_data(spl, direc)
        # TODO @jasonjewik: include auxiliary information like latitude, longitude, and date

        # Get the GMTED2010 data
        gmted2010_data = dict()
        for spl in splits:
            gmted2010_data[spl] = self.extract_gmted2010_data(spl, self.root / 'gmted2010')

        # Save separately to save room on disk, scaling and concatenation can happen later
        for spl in splits:
            with open(self.root / f'{spl}_x.npz', 'wb') as f:
                np.savez(f, **cds_data[spl], gmted2010=gmted2010_data[spl])

        # Get target data
        target_data = dict()
        dstdir = self.target_dstdirs[0]  # every element of this list is the same
        if self.target_dataset.dataset == 'chirps':
            for spl in splits:
                target_data[spl] = self.extract_chirps_data(spl, dstdir)
        
        # Save to disk
        for spl in splits:
            with open(self.root / f'{spl}_y.npz', 'wb') as f:
                np.savez(f, target=target_data[spl])

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

    def get_gmted2010_url(self, ds_req: DatasetRequest) -> str:
        resolutions = [0.0625, 0.125, 0.25, 0.5, 0.75, 1]
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
        elif res == 0.25:
            url = 'https://d1qb6yzwaaq4he.cloudfront.net/data/gmted2010/GMTED2010_15n060_0250deg.nc'
        elif res == 0.5:
            url = 'https://d1qb6yzwaaq4he.cloudfront.net/data/gmted2010/GMTED2010_15n060_0250deg.nc'
        elif res == 0.75:
            url = 'https://d1qb6yzwaaq4he.cloudfront.net/data/gmted2010/GMTED2010_15n180_0750deg.nc'
        elif res == 1:
            url = 'https://d1qb6yzwaaq4he.cloudfront.net/data/gmted2010/GMTED2010_15n240_1000deg.nc'
        return url

    def get_chirps_urls(self, ds_req: DatasetRequest) -> List[str]:
        resolutions = {0.05: '05', 0.25: '25'}
        res = getattr(ds_req, 'resolution', None)
        str_res = resolutions.get(res, None)
        if str_res is None:
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
                f'global_daily/netcdf/p{str_res}/chirps-v2.0.{year}.days_p{str_res}.nc'
                for year in range(a, b+1)
            ])
        
        return urls

    def extract_cds_targz(self) -> None:
        for req in self.cds_api_requests:
            src = req.output
            with tarfile.open(src) as tar:
                dir_path = src.parent / src.name.split('.')[1]
                tar.extractall(path=dir_path)

    def extract_cds_data(self, split: str, src_dir: Path) -> np.array:
        region = getattr(self, f'{split}_region')
        dates = getattr(self, f'{split}_dates')
        lons = region.get_longitudes(360)
        lats = region.get_latitudes()
        var_names = [x.split('_')[0] for x in os.listdir(src_dir)]
        result = []
        for vn in var_names:
            ds = xr.open_mfdataset(str(src_dir / f'{vn}*.nc'))
            xdata = getattr(ds, vn).sel(
                time=slice(*dates),
                lat=slice(*lats),
                lon=slice(*lons))  # TODO @jasonjewik: investigate
            npdata = xdata.values  # time x lat x lon
            npdata = np.moveaxis(npdata, 1, 2)  # time x lon x lat
            result.append(npdata)
        result = np.stack(result, axis=1)  # time x variable x lon x lat
        return result

    def extract_gmted2010_data(self, split: str, src: Path) -> np.array:
        region = getattr(self, f'{split}_region')
        lons = region.get_longitudes(180)
        lats = region.get_latitudes()
        ds = xr.open_mfdataset(str(src / '*.nc'))
        xdata = ds.elevation.sel(
            latitude=slice(*lats), 
            longitude=slice(*lons))
        npdata = xdata.values.T  # lon x lat
        return npdata
            
    def extract_chirps_data(self, split: str, src: Path) -> np.array:
        dates = getattr(self, f'{split}_dates')
        region = getattr(self, f'{split}_region')
        lons = region.get_longitudes(180)
        lats = region.get_latitudes()
        ds = xr.open_mfdataset(str(src / '*.nc'))
        xdata = ds.precip.sel(
            time=slice(*dates),
            latitude=slice(*lats), 
            longitude=slice(*lons))
        # Drop leap days since CDS datasets don't have those
        xdata = xdata.sel(time=~((xdata.time.dt.month == 2) & (xdata.time.dt.day == 29)))
        npdata = xdata.values  # time x lat x lon
        npdata = np.moveaxis(npdata, 1, 2)  # time x lon x lat
        return npdata