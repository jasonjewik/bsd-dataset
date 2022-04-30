from .setup_cdsapi import setup_cdsapi, CDSAPICredentialHelper, CDSAPIConfig
from .get_dataset import get_dataset

__version__ = '0.1.0'


input_datasets = [
    'cds:cmip5-single-levels:ccsm4',
    'cds:cmip5-single-levels:gfdl_cm3',
    'cds:cmip5-single-levels:ipsl_cm5a_mr',
    'cds:cmip5-single-levels:bnu_esm',
    'cds:cmip5-pressure-levels:ccsm4',
    'cds:cmip5-pressure-levels:gfdl_cm3',
    'cds:cmip5-pressure-levels:ipsl_cm5a_mr',
    'cds:cmip5-pressure-levels:bnu_esm',
    'gmted2010_00625',
    'gmted2010_0125',
    'gmted2010_0250',
    'gmted2010_0500',
    'gmted2010_0750',
    'gmted2010_1000'    
]

target_datasets = [
    'chirps_05',
    'chirps_25'
]

supported_datasets = input_datasets + target_datasets
