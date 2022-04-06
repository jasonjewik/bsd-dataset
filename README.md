# bsd-dataset

The Big Statistical Downscaling (BSD) Dataset.

## Using the BSD Dataset package

```python
from bsd_dataset import get_dataset, regions
from bsd_dataset.common.dataloaders import get_train_loader, get_eval_loader

# Load the full dataset, and download it if necessary
dataset = get_dataset(
    input_datasets=['cm3', 'cgcm3', 'cm5a'],
    target_dataset='chirps25',
    region=regions.NorthAmerica,
    download=True)

# Get the training set
train_data = dataset.get_subset('train')

# Prepare the standard data loader
train_loader = get_train_loader(train_data, batch_size=16)

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
