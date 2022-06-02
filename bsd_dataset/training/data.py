import os
import torch
import logging
import configobj
from bsd_dataset.common import  transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from bsd_dataset import get_dataset, regions, DatasetRequest

def fixtypes(config):
    for key, value in config.items():
        if(isinstance(value, dict)):
            fixtypes(value)
        else:
            if(key == "resolution"):
                config[key] = float(value)
            elif(key.endswith("_region")):
                config[key] = eval(value)
            elif(key.endswith("_dates")):
                config[key] = tuple(value)
            elif(key in ["download", "extract"]):
                config[key] = value == "True"

def get_datasets(options):
    if(not os.path.exists(options.data)):
        raise Exception(f"Path does not exist: {options.data}")

    config = dict(configobj.ConfigObj(options.data))
    fixtypes(config)

    input_datasets = [DatasetRequest(**kwargs) for kwargs in config["input_datasets"].values()]
    target_dataset = DatasetRequest(**config["target_dataset"])
    datasets = get_dataset(input_datasets, target_dataset, **config["get_dataset"]) 

    return datasets

def get_dataloaders(options):
    dataloaders = {}

    datasets = get_datasets(options)
    input_shape, target_shape = [1, 1, 1], [1, 1, 1]

    input_transform = transforms.Compose([transforms.ConvertPrecipitation(var_name = "pr"), transforms.LogTransformPrecipitation(var_name = "pr", eps=0.1)])
    target_transform = transforms.LogTransformPrecipitation(eps=0.1)
    
    for split in ["train", "val", "test"]:
        if(eval(f"options.no_{split}")):
            dataloaders[split] = None
            continue

        dataset = datasets.get_split(split, input_transform, target_transform)
        input_shape, target_shape = list(dataset[0][0].shape), list(dataset[0][1].shape)
        sampler = DistributedSampler(dataset) if(options.distributed and split == "train") else None
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = options.batch_size, num_workers = options.num_workers, pin_memory = (split == "train"), sampler = sampler, shuffle = (split == "train") and (sampler is None), drop_last = (split == "train"))
        dataloader.num_samples = options.batch_size * len(dataloader) if (split == "train") else len(dataset)
        dataloader.num_batches = len(dataloader)
        dataloaders[split] = dataloader
    
    return dataloaders, input_shape, target_shape

def load(options):    
    return get_dataloaders(options)