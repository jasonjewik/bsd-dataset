import os
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class MyDataset(Dataset):
    def __init__(self, path, split):
        super().__init__()
        path_x = os.path.join(path, split + "_x.npy")
        path_y = os.path.join(path, split + "_y.npy")
        self.x = torch.from_numpy(np.load(path_x)).float()
        if(len(self.x.shape) == 3):
            self.x = self.x.unsqueeze(1)
        self.y = torch.from_numpy(np.load(path_y)).float()
        self.normalize = torchvision.transforms.Normalize(tuple([0.5] * self.x.shape[1]), tuple([0.5] * self.x.shape[1]))
    
    def __getitem__(self, index):
        mask = torch.zeros(self.y.shape[1:]).bool()
        return self.normalize(self.x[index]), self.y[index], {"y_mask": mask}

    def __len__(self):
        return self.x.shape[0]

def get_dataloaders(options):
    dataloaders = {}
    
    for split in ["train", "val", "test"]:
        if(eval(f"options.no_{split}")):
            dataloaders[split] = None
            continue

        dataset = MyDataset(options.data, split = split)

        input_shape, target_shape = list(dataset[0][0].shape), list(dataset[0][1].shape)
        sampler = DistributedSampler(dataset) if(options.distributed and split == "train") else None
        dataloader = DataLoader(dataset, batch_size = options.batch_size, num_workers = options.num_workers, pin_memory = (split == "train"), sampler = sampler, shuffle = (split == "train") and (sampler is None), drop_last = (split == "train"))
        dataloader.num_samples = options.batch_size * len(dataloader) if (split == "train") else len(dataset)
        dataloader.num_batches = len(dataloader)
        dataloaders[split] = dataloader
    
    return dataloaders, input_shape, target_shape

def load(options):    
    return get_dataloaders(options)