import os
import threading
from typing import List, Optional
import queue

import numpy as np
import pandas as pd
from skimage import transform
import xarray as xr
import wget


def download_url(url: str, dst: str) -> None:
    """
    Downloads a file from a URL and places it in root.

    Parameters:
        url: URL to download the file from.
        dst: Directory to place downloaded file in.
    """
    dst = os.path.expanduser(dst)
    fname = os.path.basename(url)
    fpath = os.path.join(dst, fname)
    os.makedirs(dst, exist_ok=True)
    if os.path.isfile(fpath):
        os.unlink(fpath)
    try:
        wget.download(url, out=fpath, bar=None)
    except Exception as e:
        print(e)
        print(f'could not download {url}')

def download_urls(urls: List[str], dsts: List[str], n_workers: int = 1) -> None:
    """
    Downloads files from a list of URLs and places them in corresponding destinations.

    Parameters:
        urls: A list of URLs to download files from
        dsts: Directories to place the downloaded files in
        n_workers: If an integer greater than one is specified, that many threads will be used for the downloads. If not specified, no additional threads will be spawned.
    """
    if len(urls) != len(dsts):
        raise ValueError('the number of URLs and output directories must be the same')
    
    if n_workers == 1:
        for url, dst in zip(urls, dsts):
            download_url(url, dst)
    else:
        q = queue.Queue()
        def worker():
            while True:
                url, dst = q.get()
                download_url(url, dst)
                q.task_done()
        for _ in range(n_workers):
            threading.Thread(target=worker, daemon=True).start()
        for url, dst in zip(urls, dsts):
            q.put((url, dst))
        q.join()
