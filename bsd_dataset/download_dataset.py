import os, sys
import argparse
import bsd_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True,
                        help='The directory where the datasets can be found (or downloaded to, if it does not exist).')
    parser.add_argument('--datasets', nargs='*', default=None,
                        help=f'A space-separated list of datasets to download. If not specified, all of the supported datasets will be downloaded. Available choices are {bsd_dataset.supported_datasets}.')
    args = parser.parse_args()

    if args.datasets is None:
        args.datasets = bsd_dataset.supported_datasets

    input_datasets = []
    target_dataset = None
    auxiliary_datasets = []
    for dataset in args.datasets:
        if dataset in bsd_dataset.input_datasets:
            input_datasets.append(dataset)
        elif dataset in bsd_dataset.target_datasets:
            if target_dataset is None:
                target_dataset = dataset
            else:
                raise ValueError(f'Multiple target datasets specified: {target_dataset} and {dataset}; only one is allowed')
        elif dataset in bsd_dataset.auxiliary_datasets:
            auxiliary_datasets.append(dataset)
        else:
            raise ValueError(f'{dataset} not recognized; must be one of {bsd_dataset.supported_datasets}.')

    print(f'Downloading the following datasets: {args.datasets}')
    bsd_dataset.get_dataset(
        input_datasets,
        target_dataset,
        auxiliary_datasets,
        root_dir=args.root_dir,
        download=True)


if __name__ == '__main__':
    main()
