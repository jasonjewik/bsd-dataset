from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from bsd_dataset.datasets.dataset import BSDDataset

def get_train_loader(dataset: BSDDataset, batch_size: int) -> DataLoader:
    """
    Constructs and returns the dataloader for training.
    Parameters:
        - dataset: data
        - batch_size: batch size
    """
    return DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)

def get_eval_loader(dataset: BSDDataset, batch_size: int) -> DataLoader:
    """
    Constructs and returns the dataloader for evaluation.
    Parameters:
        See get_train_loader.
    """
    return DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)
