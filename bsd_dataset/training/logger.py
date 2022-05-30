import logging
import torch.multiprocessing as mp
from logging import Filter, Formatter, FileHandler, StreamHandler
from logging.handlers import QueueHandler, QueueListener

class LogFilter(Filter):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def filter(self, record):
        record.msg = f"Rank {self.rank} | {record.msg}"
        return True

class Logger:
    def __init__(self, file):        
        self.queue = mp.Queue()

        formatter = Formatter("%(asctime)s | %(message)s", datefmt = "%H:%M:%S")
        
        file_handler = FileHandler(file, "w+")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        stream_handler = StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)

        self.listener = QueueListener(self.queue, file_handler, stream_handler)
    
    def start(self): 
        self.listener.start()

    def stop(self): 
        self.listener.stop()
    
    def add(self, rank):
        handler = QueueHandler(self.queue)
        handler.addFilter(LogFilter(rank))
        handler.setLevel(logging.INFO)
        handler.flush()

        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)