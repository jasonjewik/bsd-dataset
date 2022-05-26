import os
from pathlib import Path
import pytest

from bsd_dataset import DatasetRequest, regions
from bsd_dataset.datasets.dataset import BSDDBuilder

@pytest.fixture(scope='function')
def root_dir(tmp_path_factory):
    return Path(tmp_path_factory.mktemp('data_download'))

class TestDownload:

    def test_download_chirps(self, root_dir):
        input_datasets = []
        target_dataset = DatasetRequest(
            dataset='chirps',
            resolution=0.25
        )

        builder = BSDDBuilder(
            input_datasets,
            target_dataset,
            train_region=regions.California,
            val_region=regions.California,
            test_region=regions.California,
            train_dates=('1990-01-01', '1990-12-31'),
            val_dates=('1991-01-01', '1991-12-31'),
            test_dates=('1992-01-01', '1992-12-31'),
            root=root_dir
        )

        builder.prepare_download_requests()
        assert builder.built_download_requests == True
        builder.download()

        f = root_dir / 'chirps' / 'chirps-v2.0.1990.days_p25.nc'
        assert f.is_file()
        f = root_dir / 'chirps' / 'chirps-v2.0.1991.days_p25.nc'
        assert f.is_file()
        f = root_dir / 'chirps' / 'chirps-v2.0.1992.days_p25.nc'
        assert f.is_file()

    def test_download_chirps_oor(self, root_dir):
        input_datasets = []
        target_dataset = DatasetRequest(
            dataset='chirps',
            resolution=0.25
        )

        builder = BSDDBuilder(
            input_datasets,
            target_dataset,
            train_region=regions.California,
            val_region=regions.California,
            test_region=regions.California,
            train_dates=('2300-01-01', '2300-12-31'),
            val_dates=('1991-01-01', '1991-12-31'),
            test_dates=('1992-01-01', '1992-12-31'),
            root=root_dir
        )

        with pytest.raises(ValueError):
            builder.prepare_download_requests()
        assert builder.built_download_requests == False

    def test_download_persianncdr(self, root_dir):
        input_datasets = []
        target_dataset = DatasetRequest(dataset='persiann-cdr')

        builder = BSDDBuilder(
            input_datasets,
            target_dataset,
            train_region=regions.California,
            val_region=regions.California,
            test_region=regions.California,
            train_dates=('1990-01-01', '1990-01-02'),
            val_dates=('1999-10-20', '1999-10-21'),
            test_dates=('2020-06-10', '2020-06-12'),
            root=root_dir
        )

        builder.prepare_download_requests()
        assert builder.built_download_requests == True
        builder.download()
        
        dstdir = root_dir / 'persiann-cdr'
        assert dstdir.is_dir()

    def test_download_persianncdr_oor(self, root_dir):
        input_datasets = []
        target_dataset = DatasetRequest(dataset='persiann-cdr')

        builder = BSDDBuilder(
            input_datasets,
            target_dataset,
            train_region=regions.California,
            val_region=regions.California,
            test_region=regions.California,
            train_dates=('1990-01-01', '1990-01-02'),
            val_dates=('1865-10-20', '1870-10-21'),
            test_dates=('2020-06-10', '2020-06-12'),
            root=root_dir
        )

        with pytest.raises(ValueError):
            builder.prepare_download_requests()
        assert builder.built_download_requests == False