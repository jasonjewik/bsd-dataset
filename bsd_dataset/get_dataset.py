from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import bsd_dataset
from bsd_dataset.setup_cdsapi import CDSAPICredentialHelper
if TYPE_CHECKING:
    from bsd_dataset.regions import Region

import numpy as np
import torchvision.transforms

def get_dataset(
    input_datasets: Dict[str, Dict[str, Union[str, List[str]]]],
    target_dataset: str,
    train_region: Region,
    val_region: Region,
    test_region: Region,
    train_dates: Tuple[str, str],
    val_dates: Tuple[str, str],
    test_dates: Tuple[str, str],
    transform: Optional[torchvision.transforms.Compose] = None,
    target_transform: Optional[torchvision.transforms.Compose] = None,
    download: bool = False,
    extract: bool = False,
    root: str = './data',
    cds_uid: Optional[str] = None,
    cds_key: Optional[str] = None
) -> None:
    """
    Parameters:
        input_datasets: A dictionary that maps dataset names to the ensemble
            member and variables to use.
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
        transform: The transforms to apply to the concatenated input datasets.
        target_transform: The transforms to apply to the target dataset.
        download: Download all datasets, if True. Default is False.
        extract: Extract datasets, if True. Default is False.
        root: The directory where the raw data is downloaded to and the
            extracted data is stored.
        cds_uid: The UID to pass to the CDS API credential helper.
        cds_key: The API key to pass to the CDS API credential helper.
    """
    # Check for CDS 
    helper = CDSAPICredentialHelper()
    if not helper.parse_config() or cds_uid and cds_key:
        if cds_uid is None and cds_key is None:
            helper.setup_cli()
        else:
            helper.setup(cds_uid, cds_key)

    # Validate datasets
    for dataset in input_datasets:
        if dataset not in bsd_dataset.input_datasets:
            raise ValueError(
                f'The input dataset {dataset} is not recognized.'
                f' Must be one of {bsd_dataset.input_datasets}.'
            )
    if target_dataset not in bsd_dataset.target_datasets:
        raise ValueError(
            f'The target dataset {target_dataset} is not recognized. Must be'
            f' one of {bsd_dataset.target_datasets}.'
        )

    # Validate dates
    dates = train_dates + val_dates + test_dates
    for d in dates:
        np.datetime64(d)

    # Validate root
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        print(f'WARNING: {root} does not exist, attempting to create...')
        root.mkdir()

    # Get dataset
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
        transform,
        target_transform,
        download,
        extract,
        root)
