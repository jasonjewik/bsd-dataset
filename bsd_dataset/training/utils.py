import os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def seeder(seed):
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    torch.backends.cudnn.deterministic = True
    
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)
        
    generator = torch.Generator()
    generator.manual_seed(options.seed)
    
    return worker_init_fn, generator