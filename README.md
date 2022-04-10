# bsd-dataset

The Big Statistical Downscaling (BSD) Dataset.

## Using the BSD Dataset package

```python
from bsd_dataset import get_dataset, regions
from bsd_dataset.common.dataloaders import get_train_loader, get_eval_loader, get_numpy_arrays

# Load the full dataset, and download it if necessary
dataset = get_dataset(
    input_datasets=['cm3', 'cgcm3', 'cm5a'],
    target_dataset='chirps25',
    train_region=regions.NorthAmerica,
    val_region=regions.NorthAmerica,
    test_region=regions.NorthAmerica,
    train_dates=('1981-01-01', '2003-12-31'),
    val_dates=('2004-01-01', '2004-12-31'),
    test_dates=('2005-01-01', '2005-12-31'),
    download=True,
    extract=True)

# Get the training set
train_data = dataset.get_subset('train')

# Prepare the standard data loader
train_loader = get_train_loader(train_data, batch_size=16)

# Prepare the NumPy arrays, for methods such as BCSD
train_X, train_Y = get_numpy_arrays(train_data)

# Train loop
for x, y in train_loader:
    ...

# Get the test set
test_data = dataset.get_subset('test')

# Prepare the data loader
test_loader = get_eval_loader(test_data, batch_size=16)

# Get predictions for the full test set
for x, y in test_loader:
    ...

# Evaluate
dataset.eval(y_pred, y_true)
```
