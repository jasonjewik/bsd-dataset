from __future__ import annotations
from typing import Any, Dict, List, TYPE_CHECKING

import bsd_dataset

if TYPE_CHECKING:
    from bsd_dataset.regions import RegionCoordinates

def get_dataset(
    input_datasets: List[str], 
    target_dataset: str,
    region: RegionCoordinates,
    auxiliary_datasets: List[str] = [],
    variable_dictionary: Dict[str, Any] = {},
    download: bool = True) -> None:
    """
    Parameters:
        input_datasets: Names of the input datasets.
        target_dataset: Name of the target dataset.
        region: Coordinates that define the region of interest.
        auxiliary_datasets: Names of the auxiliary datasets.
        variable_dictionary: A dictionary of variables to include besides
            precipitation.
        download: If true and the requested datasets are not stored locally,
            download them.
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
        region,
        auxiliary_datasets,
        variable_dictionary,
        download)
