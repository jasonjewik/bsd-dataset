import os
import torch
import random
import numpy as np
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def seeder(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)
        
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    return worker_init_fn, generator