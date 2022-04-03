import torch.utils.data

class BSDDataset(torch.utils.data.Dataset):
    """
    Input (x):
        Low resolution climate data over some region and at some time from all of the input datasets. If auxiliary datasets were specified, auxiliary data from the same region is also included.
    Output (y):
        High resolution climate data over the same region and at the same time as the input.
    """
    def __init__(self, input_datasets, target_dataset, auxiliary_datasets, download=False):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass
