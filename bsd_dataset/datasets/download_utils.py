import os
import threading
from typing import List, Optional
import queue

import numpy as np
import pandas as pd
from skimage import transform
import xarray as xr
import wget


def download_url(url: str, root: str, filename: Optional[str] = None) -> None:
    """
    Downloads a file from a URL and places it in root.

    Parameters:
        url: URL to download the file from.
        root: Directory to place downloaded file in.
        filename: Name to save the file under. If not specified, use the basename of the URL.
    """
    root = os.path.expanduser(root)
    if filename is None:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)
    
    os.makedirs(root, exist_ok=True)
    if os.path.isfile(fpath):
        os.unlink(fpath)

    wget.download(url, out=fpath)

def download_urls(urls: List[str], root: str, filenames: Optional[List[str]] = None, n_workers: Optional[int] = 1) -> None:
    """
    Downloads files from a list of URLs and places them in root.

    Parameters:
        urls: A list of URLs to download files from
        root: Directory to place the downloaded files in
        filenames: A list of names to save the downloaded files under. If specified, must match the length of urls. If not specified, use the basename of each URL.
        n_workers: If an integer greater than one is specified, that many threads will be used for the downloads. If not specified, no additional threads will be spawned.
    """
    if filenames is None:
        filenames = [None] * len(urls)
    elif len(urls) != len(filenames):
        raise ValueError('the number of URLs and file names must be the same')
    
    if n_workers == 1:
        for url, fname in zip(urls, filenames):
            download_url(url, root, fname)
    else:
        q = queue.Queue()
        def worker():
            while True:
                download_url(*q.get())
                q.task_done()
        for _ in range(n_workers):
            threading.Thread(target=worker, daemon=True).start()
        for url in urls:
            q.put(url)
        q.join()
