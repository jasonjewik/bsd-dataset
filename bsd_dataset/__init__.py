from .datasets.download_utils import DatasetRequest
from .get_dataset import get_dataset
from .setup_cdsapi import setup_cdsapi

__version__ = '0.1.0'


input_datasets = [
    'projections-cmip5-daily-single-levels',
    'projections-cmip5-daily-pressure-levels',
    'gmted2010',   
]

target_datasets = [
    'chirps'
]

supported_datasets = input_datasets + target_datasets
