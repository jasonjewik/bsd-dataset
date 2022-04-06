from typing import Any, Dict, List

import numpy as np
import torch

from bsd_dataset.regions import RegionCoordinates
from bsd_dataset.datasets.download_utils import download_urls
from bsd_dataset.datasets.dataset_utils import extract_chirps25_data, extract_cmip5_data


class BSDDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        input_datasets: List[str],
        target_dataset: str,
        region: RegionCoordinates,
        auxiliary_datasets: List[str] = [],
        variable_dictionary: Dict[str, Any] = {},
        download: bool = False):
        """
        Parameters:
            See get_dataset.py.
        """
        # Download datasets
        if download:
            datasets = input_datasets + [target_dataset] + auxiliary_datasets
            for ds in datasets:
                if ds == 'chirps25':
                    years = range(1981, 2006)
                    urls = [f'ftp://anonymous@ftp.chc.ucsb.edu/pub/org/chg/products/CHIRPS-2.0/global_daily/netcdf/p25/chirps-v2.0.{year}.days_p25.nc' for year in years]
                    download_urls(urls, './data/CHIRPS25', n_workers=5)
                if ds == 'cgcm3':
                    urls = [
                        'http://esgf-data1.diasjp.net/thredds/fileServer/esg_dataroot/cmip5/output1/MRI/MRI-CGCM3/historical/day/atmos/day/r1i1p1/v20120701/prc/prc_day_MRI-CGCM3_historical_r1i1p1_19800101-19891231.nc',
                        'http://esgf-data1.diasjp.net/thredds/fileServer/esg_dataroot/cmip5/output1/MRI/MRI-CGCM3/historical/day/atmos/day/r1i1p1/v20120701/prc/prc_day_MRI-CGCM3_historical_r1i1p1_19900101-19991231.nc',
                        'http://esgf-data1.diasjp.net/thredds/fileServer/esg_dataroot/cmip5/output1/MRI/MRI-CGCM3/historical/day/atmos/day/r1i1p1/v20120701/prc/prc_day_MRI-CGCM3_historical_r1i1p1_20000101-20051231.nc'
                    ]
                    download_urls(urls, './data/CGCM3', n_workers=5)
                if ds == 'cm3':
                    urls = [
                        'http://aims3.llnl.gov/thredds/fileServer/css03_data/cmip5/output1/NOAA-GFDL/GFDL-CM3/historical/day/atmos/day/r1i1p1/v20120227/prc/prc_day_GFDL-CM3_historical_r1i1p1_19800101-19841231.nc',
                        'http://aims3.llnl.gov/thredds/fileServer/css03_data/cmip5/output1/NOAA-GFDL/GFDL-CM3/historical/day/atmos/day/r1i1p1/v20120227/prc/prc_day_GFDL-CM3_historical_r1i1p1_19850101-19891231.nc',
                        'http://aims3.llnl.gov/thredds/fileServer/css03_data/cmip5/output1/NOAA-GFDL/GFDL-CM3/historical/day/atmos/day/r1i1p1/v20120227/prc/prc_day_GFDL-CM3_historical_r1i1p1_19900101-19941231.nc',
                        'http://aims3.llnl.gov/thredds/fileServer/css03_data/cmip5/output1/NOAA-GFDL/GFDL-CM3/historical/day/atmos/day/r1i1p1/v20120227/prc/prc_day_GFDL-CM3_historical_r1i1p1_19950101-19991231.nc',
                        'http://aims3.llnl.gov/thredds/fileServer/css03_data/cmip5/output1/NOAA-GFDL/GFDL-CM3/historical/day/atmos/day/r1i1p1/v20120227/prc/prc_day_GFDL-CM3_historical_r1i1p1_20000101-20041231.nc',
                        'http://aims3.llnl.gov/thredds/fileServer/css03_data/cmip5/output1/NOAA-GFDL/GFDL-CM3/historical/day/atmos/day/r1i1p1/v20120227/prc/prc_day_GFDL-CM3_historical_r1i1p1_20050101-20051231.nc'
                    ]
                    download_urls(urls, './data/CM3', n_workers=5)
                if ds == 'cm5a':
                    urls = ['http://aims3.llnl.gov/thredds/fileServer/cmip5_css01_data/cmip5/output1/IPSL/IPSL-CM5A-LR/historical/day/atmos/day/r1i1p1/v20110909/prc/prc_day_IPSL-CM5A-LR_historical_r1i1p1_19500101-20051231.nc']
                    download_urls(urls, './data/CM5A', n_workers=5)

        # Define temporal coverage
        train_dates = ('1981-01-01', '2003-12-31')
        val_dates = ('2004-01-01', '2004-12-31')
        test_dates = ('2005-01-01', '2005-12-31')

        # Define spatial coverage
        lons, lats = [0, 0], [0, 0]
        lons[0], lats[0] = region.top_left_corner
        lons[1], lats[1] = region.bottom_right_corner
        lons = np.array(lons)
        lats = np.array(lats)

        # Extract target data
        if target_dataset == 'chirps25':
            extract_chirps25_data('./data/CHIRPS25', './data/train_y.npy', lons, lats, train_dates)
            extract_chirps25_data('./data/CHIRPS25', './data/val_y.npy', lons, lats, val_dates)
            extract_chirps25_data('./data/CHIRPS25', './data/test_y.npy', lons, lats, test_dates)
        else:
            raise NotImplementedError

        # Extract input data
        src = ['./data/CGCM3', './data/CM3', './data/CM5A']
        extract_cmip5_data(src, './data/train_x.npy', lons, lats, train_dates)
        extract_cmip5_data(src, './data/val_x.npy', lons, lats, val_dates)
        extract_cmip5_data(src, './data/test_x.npy', lons, lats, test_dates)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

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
            
    def eval(self, y_pred, y_true):
        """
        Computes the RMSE between the predicted and ground truth values.
        Parameters:
            - y_pred: model predicted values
            - y_true: ground truth
        """
        mse_loss = torch.nn.MSELoss()

        def nan_to_num(t, mask=None):
            if mask is None:
                mask = torch.isnan(t)
            zeros = torch.zeros_like(t)
            return torch.where(mask, zeros, t)

        y_pred = torch.tensor(y_pred)
        y_true = torch.tensor(y_true)
        y_pred = nan_to_num(y_pred, torch.isnan(y_true))
        y_true = nan_to_num(y_true)
        return torch.sqrt(mse_loss(y_pred, y_true))
    
