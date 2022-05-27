import os
from importlib.machinery import SourceFileLoader
import torch
import random
import numpy as np

def superseed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def load_module(filename):
    module_name = os.path.splitext(os.path.basename(filename))[0]
    return SourceFileLoader(module_name, filename).load_module()

def stack(x, num_samples=None, dim=0):
    return x if num_samples is None \
        else torch.stack([x]*num_samples, dim=dim)