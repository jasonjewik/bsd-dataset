import pytest

from pathlib import Path
from bsd_dataset import DatasetRequest, regions
from bsd_dataset.datasets.dataset import BSDDataset

@pytest.fixture(scope='function')
def root_dir(tmp_path_factory):
    return Path(tmp_path_factory.mktemp('data_download'))

def test_download(root_dir):

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

    # Should be a no-op
    dataset.download()

    # Only after building can we download
    dataset.build_download_requests()
    assert dataset.built_download_requests == True
    dataset.download()

    # Check that the correct files were retrieved
    f = root_dir / 'cds' / 'projections-cmip5-daily-single-levels' / 'gfdl_cm3.tar.gz'
    assert f.is_file()
    f = root_dir / 'chirps' / 'chirps-v2.0.1990.days_p25.nc'
    assert f.is_file()
    f = root_dir / 'chirps' / 'chirps-v2.0.1991.days_p25.nc'
    assert f.is_file()
    f = root_dir / 'chirps' / 'chirps-v2.0.1992.days_p25.nc'
    assert f.is_file()
    f = root_dir / 'gmted2010' / 'GMTED2010_15n060_0250deg.nc'
    assert f.is_file()