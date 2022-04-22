from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import bsd_dataset
if TYPE_CHECKING:
    from bsd_dataset.regions import Region

import torchvision.transforms

def get_dataset(
    input_datasets: List[str],
    target_dataset: str,
    train_region: Region,
    val_region: Region,
    test_region: Region,
    train_dates: Tuple[str, str],
    val_dates: Tuple[str, str],
    test_dates: Tuple[str, str],
    auxiliary_datasets: List[str] = [],
    variable_dictionary: Dict[str, Any] = {},
    transform: Optional[torchvision.transforms.Compose] = None,
    target_transform: Optional[torchvision.transforms.Compose] = None,
    download: Dict[str, bool] = {},
    extract: Dict[str, bool] = {},
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
        variable_dictionary: A dictionary that maps dataset names to options to
            pass to the CDS API for downloading input datasets. Do not specify
            format.
        transform: The transforms to apply to the concatenated input and
            auxiliary datasets.
        target_transform: The transforms to apply to the target dataset.
        download: A dictionary that maps dataset names to a boolean indicating
            whether the dataset needs to be downloaded. If a dataset is not
            included in this dictionary, the program assumes said dataset needs
            to be downloaded.
        extract: Similar to download, but for extracting data to NumPy files
            train_x.npy, train_y.npy, val_x.npy, val_y.npy, test_x.npy, and
            test_y.npy.
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
