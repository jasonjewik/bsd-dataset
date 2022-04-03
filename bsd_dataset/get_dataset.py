import bsd_dataset
from bsd_dataset.datasets import BSDDataset


def get_dataset(input_datasets: List[str], target_dataset: str, auxiliary_datasets: List[str] = [], **dataset_kwargs) -> None:
    """
    Parameters:
        input_datasets: Names of the input datasets.
        target_dataset: Name of the target dataset.
        auxiliary_dataset: Names of the auxiliary datasets.
        dataset_kwargs: Keyword arguments to pass to the dataset constructor.
    """
    for dataset in input_datasets:
        if dataset not in bsd_dataset.input_datasets:
            raise ValueError(f'The input dataset {dataset} is not recognized. Must be one of {bsd_dataset.input_datasets}.')
    if target_dataset not in bsd_dataset.target_datasets:
        raise ValueError(f'The target dataset {target_dataset} is not recognized. Must be one of {bsd_dataset.target_datasets}.')
    for dataset in auxiliary_datasets:
        if dataset not in bsd_dataset.auxiliary_datasets:
            raise ValueError(f'The auxiliary dataset {dataset} is not recognized. Must be one of {bsd_dataset.auxiliary_dataset}.')

    return BSDDataset(input_datasets, target_dataset, auxiliary_datasets, **dataset_kwargs)
