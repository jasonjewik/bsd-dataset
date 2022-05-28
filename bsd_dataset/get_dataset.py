from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple

from bsd_dataset.datasets.dataset import BSDDBuilder 
from bsd_dataset.datasets.download_utils import DatasetRequest
from bsd_dataset.setup_cdsapi import CDSAPICredentialHelper
from bsd_dataset.regions import Region

def get_dataset(
    input_datasets: List[DatasetRequest],
    target_dataset: DatasetRequest,
    train_region: Region,
    val_region: Region,
    test_region: Region,
    train_dates: Tuple[str, str],
    val_dates: Tuple[str, str],
    test_dates: Tuple[str, str],
    download: bool = False,
    extract: bool = False,
    root: str = './data',
    cds_uid: Optional[str] = None,
    cds_key: Optional[str] = None,
    device: str = 'cpu'
) -> BSDDBuilder:    

    # Construct builder
    builder = BSDDBuilder(
        input_datasets,
        target_dataset,
        train_region,
        val_region,
        test_region,
        train_dates,
        val_dates,
        test_dates,
        Path(root),
        device=device
    )
    builder.prepare_download_requests()

    if download:
        # Check for CDS        
        helper = CDSAPICredentialHelper()
        if not helper.parse_config() or cds_uid and cds_key:
            if cds_uid is None and cds_key is None:
                helper.setup_cli()
            else:
                helper.setup(cds_uid, cds_key)
        builder.download()
        
    if extract:
        builder.extract()

    return builder
