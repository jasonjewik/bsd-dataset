from __future__ import annotations
from typing import Any, Dict, List, TYPE_CHECKING

import bsd_dataset

if TYPE_CHECKING:
    from bsd_dataset.regions import RegionCoordinates

def get_dataset(
    input_datasets: List[str],
    target_dataset: str,
    train_region: RegionCoordinates,
    val_region: RegionCoordinates,
    test_region: RegionCoordinates,
    train_dates: Tuple[str, str],
    val_dates: Tuple[str, str],
    test_dates: Tuple[str, str],
    auxiliary_datasets: List[str] = [],
    variable_dictionary: Dict[str, Any] = {},
    transform: torchvision.transforms = None,
    target_transform: torchvision.transforms = None,
    download: bool = False,
    extract: bool = False,
    root: str = './data') -> None:
    """
    Parameters:
        input_datasets: Names of the input datasets.
        target_dataset: Name of the target dataset.
        train_region: Coordinates that define the region to use in training.
        val_region: Coordinates that define the region to use in validation.
        test_region: Coordinates that define the region to use in testing.
        train_dates: A tuple that marks the start and end dates (inclusive) to
            use in training.
        val_dates: A tuple that marks the start and end dates (inclusive) to 
            use in validation.
        test_dates: A tuple that marks the start and end dates (inclusive) to 
            use in testing.
        auxiliary_datasets: Names of the auxiliary datasets.
        variable_dictionary: A dictionary of variables to include besides
            precipitation.
        transform: The transforms to apply to the concatenated input and
            auxiliary datasets.
        target_transform: The transforms to apply to the target dataset.
        download: If true and the requested datasets are not stored locally,
            download them.
        extract: If true, extract the data to NumPy format.
        root: The directory where the raw data is downloaded to and the
            extracted data is stored.
    """
    for dataset in input_datasets:
        if dataset not in bsd_dataset.input_datasets:
            raise ValueError(f'The input dataset {dataset} is not recognized. Must be one of {bsd_dataset.input_datasets}.')
    if target_dataset not in bsd_dataset.target_datasets:
        raise ValueError(f'The target dataset {target_dataset} is not recognized. Must be one of {bsd_dataset.target_datasets}.')
    for dataset in auxiliary_datasets:
        if dataset not in bsd_dataset.auxiliary_datasets:
            raise ValueError(f'The auxiliary dataset {dataset} is not recognized. Must be one of {bsd_dataset.auxiliary_dataset}.')

    from bsd_dataset.datasets.dataset import BSDDataset
    return BSDDataset(
        input_datasets,
        target_dataset,
        train_region,
        val_region,
        test_region,
        train_dates,
        val_dates,
        test_dates,
        auxiliary_datasets,
        variable_dictionary,
        transform,
        target_transform,
        download,
        extract,
        root)
