from .get_dataset import get_dataset

input_datasets = [
    'cgcm3',
    'cm3',
    'cm5a'
]

target_datasets = [
    'chirps05',
    'chirps25'
]

auxiliary_datasets = [
    'gmted2010_00625',
    'gmted2010_0125',
    'gtmed2010_0250',
    'gmted2010_0500',
    'gmted2010_0750',
    'gmted2010_1000'
]

supported_datasets = input_datasets + target_datasets + auxiliary_datasets
