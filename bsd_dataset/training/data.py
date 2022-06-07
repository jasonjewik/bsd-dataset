import os
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class MyDataset(Dataset):
    def __init__(self, path, split, normalize=None):
        super().__init__()
        path_x = os.path.join(path, split + "_x.npy")
        path_y = os.path.join(path, split + "_y.npy")
        self.x = torch.from_numpy(np.load(path_x)).float()
        if(len(self.x.shape) == 3):
            self.x = self.x.unsqueeze(1)
        self.y = torch.from_numpy(np.load(path_y)).float()
        self.mask = self.y.isnan()
        self.normalize = normalize
    
    def __getitem__(self, index):
        if self.normalize is not None:
            return self.normalize(self.x[index]), self.y[index], {"y_mask": self.mask[index]}
        else:
            return self.x[index], self.y[index], {"y_mask": self.mask[index]}

    def __len__(self):
        return self.x.shape[0]

def get_dataloaders(options):
    dataloaders = {}
    
    normalize = None
    for split in ["train", "val", "test"]:
        if(eval(f"options.no_{split}")):
            dataloaders[split] = None
            continue

        if normalize is None:
            unnormalized_dataset = MyDataset(options.data, split = split)
            mean = tuple([torch.mean(unnormalized_dataset.x[:, i]) for i in range(unnormalized_dataset.x.shape[1])])
            std = tuple([torch.std(unnormalized_dataset.x[:, i]) for i in range(unnormalized_dataset.x.shape[1])])
            normalize = torchvision.transforms.Normalize(mean, std)

        dataset = MyDataset(options.data, split = split, normalize=normalize)

        input_shape, target_shape = list(dataset[0][0].shape), list(dataset[0][1].shape)
        sampler = DistributedSampler(dataset) if(options.distributed and split == "train") else None
        dataloader = DataLoader(dataset, batch_size = options.batch_size, num_workers = options.num_workers, pin_memory = (split == "train"), sampler = sampler, shuffle = (split == "train") and (sampler is None), drop_last = (split == "train"))
        dataloader.num_samples = options.batch_size * len(dataloader) if (split == "train") else len(dataset)
        dataloader.num_batches = len(dataloader)
        dataloaders[split] = dataloader
    
    return dataloaders, input_shape, target_shape

def load(options):    
    return get_dataloaders(options)

# dataset = MyDataset('/home/data/BSDD/era-eu-1.4-persiann-0.25/', 'train')
# print (dataset.x.shape)
# print (dataset.y.shape)
# x, y, info = dataset[0]
# print (info['y_mask'].sum())
# print (y[1])
# print (info['y_mask'][1])