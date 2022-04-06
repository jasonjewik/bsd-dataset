from .get_dataset import get_dataset

input_datasets = [
    'ccsm4',
    'cgcm3',
    'cm3',
    'cm5a'
]

target_datasets = [
    'chirps25'
]

auxiliary_datasets = [
]

supported_datasets = input_datasets + target_datasets + auxiliary_datasets
