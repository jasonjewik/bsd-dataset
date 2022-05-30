import os
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from bsd_dataset import get_dataset, regions, DatasetRequest

def get_datasets(options):
    input_datasets = [DatasetRequest(
        dataset = "projections-cmip5-daily-single-levels",
        model="gfdl_cm3",
        ensemble_member="r1i1p1",
        variable=["mean_precipitation_flux", "2m_temperature"],
    ),

    DatasetRequest(
       dataset="gmted2010",
       resolution=0.25
    )
]

    target_dataset = DatasetRequest(dataset="chirps", resolution=0.25)

    dataset = get_dataset(input_datasets, target_dataset, train_region=regions.California, val_region=regions.California, test_region=regions.California, train_dates=("1990-01-01", "1990-12-31"), val_dates=("1991-01-01", "1991-12-31"), test_dates=("1992-01-01", "1992-12-31"), download = False, extract = True) 

    train_dataset = dataset.get_split("train")
    val_dataset = dataset.get_split("val")
    test_dataset = dataset.get_split("test")


def get_dataloaders(options):

    train = (datatype == "train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = options.batch_size, num_workers = options.num_workers, sampler = None)
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return dataloader

def load(options):    
    data = get_dataloaders(options)
    return data