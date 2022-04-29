import os
from pathlib import Path
import threading
from typing import Dict, List
import queue

import wget
import cdsapi


class CDSAPIRequest:
    def __init__(self, dataset: str, options: Dict[str, str], output: Path):
        if dataset == 'cmip5-single-levels':
            self.dataset = 'projections-cmip5-daily-single-levels'
        if dataset == 'cmip5-pressure-levels':
            self.dataset = 'projections-cmip5-daily-pressure-levels'
        self.options = options
        self.output = output.expanduser().resolve()

    @property
    def options(self):
        return str(self.options)


def download_url(url: str, dst: Path) -> None:
    """
    Downloads a file from a URL and places it in dst.

    Parameters:
        url: URL to download the file from.
        dst: Directory to place downloaded file in.
    """
    dst = dst.expanduser().resolve()
    fname = os.path.basename(url)
    fpath = dst / fname
    dst.mkdir(exist_ok=True)
    fpath.unlink(missing_ok=True)
    try:
        wget.download(url, out=fpath, bar=None)
    except Exception as e:
        print(e)
        print(f'could not download {url}')

def download_urls(urls: List[str], dsts: List[Path], n_workers: int = 1) -> None:
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

def download_from_cds(request: CDSAPIRequest) -> None:
    """
    Downloads from CDS according to the given request.

    Parameters:
        request: A request to pass to the CDS API's retrieve method.
    """
    dst = request.output
    os.makedirs(dst.parent, exist_ok=True)
    dst.unlink(missing_ok=True)
    c = cdsapi.Client()
    c.retrieve(request.dataset, request.options, request.output)

def multidownload_from_cds(requests: List[CDSAPIRequest], n_workers: int = 1) -> None:
    """
    Downloads from CDS according to the passed in requests.

    Parameters:
        requests: A list of requests to pass to the CDS API's retrieve method.
        n_workers: If an integer greater than one is specified, that many threads will be used for the downloads. If not specified, no additional threads will be spawned.
    """
    if n_workers == 1:
        for req in requests:
            download_from_cds(req)
    else:
        q = queue.Queue()
        def worker():
            while True:
                req = q.get()
                download_from_cds(req)
                q.task_done()
        for _ in range(n_workers):
            threading.Thread(target=worker, daemon=True).start()
        for req in requests:
            q.put(req)
        q.join()