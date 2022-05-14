# bsd-dataset

The Big Statistical Downscaling (BSD) Dataset.

## Using the BSD Dataset package

**Note:** In future versions, the `get_dataset` method will determine which files to download based on the specified `train_dates`, `val_dates`, and `test_dates`. In the meantime, the time ranges need to be explicitly provided in the input dataset options.

```python
from bsd_dataset import get_dataset, regions
from bsd_dataset.common.dataloaders import get_train_loader, get_eval_loader, get_numpy_arrays
from bsd_dataset.common.metrics import rmse

# Specify input dataset options
input_datasets = {
    'cds:cmip5-single-levels:ccsm4': {
        'ensemble_member': 'r1i1p1',
        'variable': 'mean_precipitation_flux',
        'period': [
            '19550101-19891231', '19900101-20051231'
        ]
    },
    'cds:cmip5-single-levels:gfdl_cm3': {
        'ensemble_member': 'r1i1p1',
        'variable': [
            'mean_precipitation_flux',
            'near_surface_specific_humidity'
        ],
        'period': [
            '19800101-19841231', '19850101-19891231',
            '19950101-19991231', '20050101-20051231'
        ]
    },
    'cds:cmip5-single-levels:ipsl_cm5a_mr': {
        'ensemble_member': 'r1i1p1',
        'variable': [
            'mean_precipitation_flux',
            'near_surface_specific_humidity'
        ],
        'period': [
            '19500101-19991231', '20000101-20051231'
        ]
    },
    'cds:cmip5-single-levels:bnu_esm': {
        'ensemble_member': 'r1i1p1',
        'variable': [
            'mean_precipitation_flux',
            'near_surface_specific_humidity'
        ],
        'period': '19500101-20051231'
    },
    'gmted2010_0250': {},
    'cds:cmip5-pressure-levels:ccsm4': {
        'ensemble_member': 'r6i1p1',
        'variable': [
            'u_component_of_wind', 'v_component_of_wind'                
        ],
        'period': [
            '19800101-19841231', '19850101-19891231', 
            '19900101-19941231', '19950101-19991231', 
            '20000101-20051231'
        ],
    },
    'cds:cmip5-pressure-levels:gfdl_cm3': {
        'ensemble_member': 'r1i1p1',
        'variable': [
            'u_component_of_wind', 'v_component_of_wind'
        ],
        'period': [
            '19800101-19841231', '19850101-19891231', 
            '19900101-19941231', '19950101-19991231', 
            '20000101-20041231', '20050101-20051231'
        ]
    },
    'cds:cmip5-pressure-levels:ipsl_cm5a_mr': {
        'ensemble_member': 'r3i1p1',
        'variable': [
            'u_component_of_wind', 'v_component_of_wind'
        ],
        'period': [
            '19800101-19891231', '19900101-19991231', 
            '20000101-20051231'
        ]
    },
    'cds:cmip5-pressure-levels:bnu_esm': {
        'ensemble_member': 'r1i1p1',
        'variable': [
            'u_component_of_wind', 'v_component_of_wind'
        ],
        'period': '19500101-20051231'
    }
}

# Load the full dataset, and download it if necessary
dataset = get_dataset(    
    input_datasets=input_datasets,
    target_dataset='chirps_25',
    train_region=regions.SouthAmerica,
    val_region=regions.SouthAmerica,
    test_region=regions.SouthAmerica,
    train_dates=('1981-01-01', '2003-12-31'),
    val_dates=('2004-01-01', '2004-12-31'),
    test_dates=('2005-01-01', '2005-12-31'),
    download=True,
    extract=True
)

# Get the training set
train_data = dataset.get_subset('train')

# Prepare the train data loader
train_loader = get_train_loader(train_data, batch_size=16)

# Prepare the NumPy arrays, for methods such as BCSD
train_X, train_Y = get_numpy_arrays(train_data)

# Train loop
for x, y in train_loader:
    ...

# Get the test set
test_data = dataset.get_subset('test')

# Prepare the test data loader
test_loader = get_eval_loader(test_data, batch_size=16)

# Accumulate observations in y_true
# and predictions in y_pred
for x, y in test_loader:
    ...

# Evaluate
rmse(y_pred, y_true)
```

## Using a Custom Region

**Note:** Right now, the only available target dataset is CHIRPS, [which covers the Earth from 50 degrees north to 50 degrees south](https://wiki.chc.ucsb.edu/CHIRPS_FAQ#What_are_the_spatial_domain_available_for_CHIRPS.3F).

```python
Spain = Region(
    top_left=(-12, 45),
    bottom_right=(2, 35)
)

dataset = get_dataset(
    train_region=Spain,
    val_region=Spain,
    test_region=Spain,
    ...
)
```

## Running tests

```shell
bsd-dataset$ python -m pytest tests
```