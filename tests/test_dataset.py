import pytest
import shutil

from bsd_dataset import DatasetRequest, regions
from bsd_dataset.datasets.dataset import BSDDataset

@pytest.fixture(scope='function')
def root_dir(tmp_path_factory):
    return tmp_path_factory.mktemp('data_download')

def test_download(root_dir):

    shutil.rmtree(root_dir, ignore_errors=True)

    input_datasets = [
        DatasetRequest(
            dataset='projections-cmip5-daily-single-levels',
            model='gfdl_cm3',
            ensemble_member='r1i1p1',
            variable='mean_precipitation_flux'
        ),
        DatasetRequest(
            dataset='gmted2010',
            resolution=0.25
        )
    ]

    target_dataset = DatasetRequest(
        dataset='chirps',
        resolution=0.25
    )

    dataset = BSDDataset(
        input_datasets,
        target_dataset,
        train_region=regions.California,
        val_region=regions.California,
        test_region=regions.California,
        train_dates=('1990-01-01', '1990-12-31'),
        val_dates=('1991-01-01', '1991-12-31'),
        test_dates=('1992-01-01', '1992-12-31'),
        transform=None,
        target_transform=None,
        root=root_dir
    )

    dataset.build_download_requests()
    assert dataset.built_download_requests == True