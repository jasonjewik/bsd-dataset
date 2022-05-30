from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
import threading
from typing import Any, Dict, List, Tuple
import queue

import wget
import cdsapi
import numpy as np


class DatasetRequest:
    def __init__(self, dataset: str, **kwargs: Dict[str, Any]):
        self.dataset = dataset
        for kw, arg in kwargs.items():
            setattr(self, kw, arg)

    def __str__(self) -> str:
        result = f'DatasetRequest({str(vars(self))})'
        return result

    def is_cds_req(self) -> bool:
        cds_datasets = [
            'projections-cmip5-daily-single-levels',
            'projections-cmip5-daily-pressure-levels',
            'projections-cmip6'
        ]
        return self.dataset in cds_datasets

@dataclass
class CDSAPIRequest:
    dataset: str
    options: Dict[str, Any]
    output: Path

class CDSAPIRequestBuilder:
    def build(
        self,
        root: Path, 
        dataset_request: DatasetRequest, 
        train_dates: Tuple[str, str],
        val_dates: Tuple[str, str],
        test_dates: Tuple[str, str]
    ) -> List[CDSAPIRequest]:
        results = []

        dataset = dataset_request.dataset
        try:
            model = getattr(dataset_request, 'model')
        except:
            raise AttributeError(f'CDS request {str(dataset_request)} is missing required parameter "model"')
        try:
            variable = getattr(dataset_request, 'variable')
        except:
            raise AttributeError(f'CDS request {str(dataset_request)} is missing required parameter "variable"')
        if type(variable) == list:
            if len(variable) == 0:
                raise ValueError(f'CDS request {str(dataset_request)} has empty required parameter "variable')            
            for var in variable:
                if var not in self.get_variables(dataset, model):
                    raise ValueError(f'CDS request {str(dataset_request)} has unrecognized variable {var}')
        elif type(variable) == str:
            if variable not in self.get_variables(dataset, model):
                raise ValueError(f'CDS request {str(dataset_request)} has unrecognized variable {variable}')
            variable = [variable]
        else:
            raise TypeError(f'CDS request {str(dataset_request)} "variable" parameter must be str or list type')


        if dataset in ('projections-cmip5-daily-single-levels', 'projections-cmip5-daily-pressure-levels'):
            try:
                ensemble_member = getattr(dataset_request, 'ensemble_member')
            except:
                raise AttributeError(f'CDS request {str(dataset_request)} is missing required parameter "ensemble_member"')
  
            periods = self.get_periods(dataset, model)

            train_periods, success = select_periods(*train_dates, periods)
            if not success:
                raise ValueError(
                    f'the given train dates are not available for dataset {dataset} and model {model}\n'
                    f'available dates are: {self.get_periods(model)}'
                )
            train_periods = self.format_periods(train_periods)

            val_periods, success = select_periods(*val_dates, periods)
            if not success:
                raise ValueError(
                    f'the given val dates are not available for dataset {dataset} and model {model}\n'
                    f'available dates are: {self.get_periods(model)}'
                )
            val_periods = self.format_periods(val_periods)

            test_periods, success = select_periods(*test_dates, periods)
            if not success:
                raise ValueError(
                    f'the given test dates are not available for dataset {dataset} and model {model}\n'
                    f'available dates are: {self.get_periods(model)}'
                )
            test_periods = self.format_periods(test_periods)

            periods = train_periods
            periods.extend(val_periods)
            periods.extend(test_periods)
            periods = sorted(set(periods))
            if len(periods) == 1:
                periods = periods[0]

            base_options = {}
            base_options['experiment'] = 'historical'
            base_options['format'] = 'tgz'
            base_options['model'] = model
            base_options['period'] = periods        
            base_options['ensemble_member'] = ensemble_member            

        if dataset == 'projections-cmip6':
            base_options = {}
            base_options['temporal_resolution'] = 'daily'
            base_options['experiment'] = 'historical'
            base_options['level'] = 'single_levels'
            base_options['model'] = model
            base_options['format'] = 'zip'

            all_dates = [np.datetime64(d) for d in train_dates + val_dates + test_dates]
            base_options['date'] = f'{min(all_dates)}/{max(all_dates)}'

        for var in variable:
            options = deepcopy(base_options)
            options['variable'] = var
            output = root / 'cds' / dataset / f'{var}.{model}.{options["format"]}'
            output = output.expanduser().resolve() 
            results.append(CDSAPIRequest(dataset, options, output))

        return results

    def format_periods(self, periods: List[Tuple[str, str]]) -> List[str]:
        result = []
        for start, stop in periods:
            start = ''.join(start.split('-'))
            stop = ''.join(stop.split('-'))
            result.append(start + '-' + stop)
        return result

    def get_periods(self, dataset: str, model: str) -> List[Tuple[str, str]]:
        if dataset == 'projections-cmip5-daily-single-levels':
            if model == 'ccsm4':
                return [
                    ('1950-01-01', '1989-12-31'),
                    ('1990-01-01', '2005-12-31')
                ]
            if model == 'gfdl_cm3':
                return [
                    ('1980-01-01', '1984-12-31'),
                    ('1985-01-01', '1989-12-31'),
                    ('1990-01-01', '1994-12-31'),
                    ('1995-01-01', '1999-12-31'),
                    ('2000-01-01', '2004-12-31'),
                    ('2005-01-01', '2005-12-31')
                ]
            if model == 'ipsl_cm5a_mr':
                return [
                    ('1950-01-01', '1999-12-31'),
                    ('2000-01-01', '2005-12-31')
                ]
            if model == 'bnu_esm':
                return [
                    ('1950-01-01', '2005-12-31')
                ]
            if model == 'hadcm3':
                return [
                    ('1959-12-01', '1984-11-30'),
                    ('1984-12-01', '2005-12-30')
                ]
        if dataset == 'projections-cmip5-daily-pressure-levels':
            if model == 'ccsm4':
                return [
                    ('1980-01-01', '1984-12-31'),
                    ('1985-01-01', '1989-12-31'),
                    ('1990-01-01', '1994-12-31'),
                    ('1995-01-01', '1999-12-31'),
                    ('2000-01-01', '2005-12-31')
                ]
            if model == 'gfdl_cm3':
                return [
                    ('1980-01-01', '1984-12-31'),
                    ('1985-01-01', '1989-12-31'),
                    ('1990-01-01', '1994-12-31'),
                    ('1995-01-01', '1999-12-31'),
                    ('2000-01-01', '2004-12-31'),
                    ('2005-01-01', '2005-12-31')
                ]
            if model == 'ipsl_cm5a_mr':
                return [
                    ('1980-01-01', '1989-12-31'),
                    ('1990-01-01', '1999-12-31'),
                    ('2000-01-01', '2005-12-31')
                ]
            if model == 'bnu_esm':
                return [
                    ('1950-01-01', '2005-12-31')
                ]
            if model == 'hadcm3':
                return [
                    ('1959-12-01', '1984-11-30'),
                    ('1984-12-01', '2005-12-30')
                ]

    def get_variables(self, dataset: str, model: str) -> List[str]:
        if dataset in ('projections-cmip5-daily-single-levels', 'projections-cmip5-daily-pressure-levels'):
            all_vars = [
                'snowfall',
                '10m_wind_speed',
                '2m_temperature',
                'mean_precipitation_flux', 
                'mean_sea_level_pressure',
                'near_surface_specific_humidity',
                'surface_solar_radiation_downwards',
                'daily_near_surface_relative_humidity',
                'maximum_2m_temperature_in_the_last_24_hours',
                'minimum_2m_temperature_in_the_last_24_hours'            
            ]
            if model == 'ccsm4':
                all_vars.remove('10m_wind_speed')
                all_vars.remove('daily_near_surface_relative_humidity')
            if model == 'gfdl_cm3':
                pass
            if model == 'ipsl_cm5a_mr':
                pass
            if model == 'bnu_esm':
                pass
            if model == 'hadcm3':
                all_vars.remove('snowfall')
                all_vars.remove('surface_solar_radiation_downwards')
                all_vars.remove('daily_near_surface_relative_humidity')
        
        if dataset == 'projections-cmip6':
            all_vars = [
                'daily_maximum_near_surface_air_temperature',
                'daily_minimum_near_surface_air_temperature',
                'near_surface_air_temperature',
                'near_surface_specific_humidity',
                'near_surface_wind_speed',
                'precipitation',
                'sea_level_pressure'
            ]
            if model == 'gfdl_esm4':
                pass
            if model == 'fgoals_f3_l':
                pass
            if model == 'mpi_esm1_2_lr':
                pass
            if model == 'ipsl_cm6a_lr':
                all_vars.remove('near_surface_specific_humidity')
                all_vars.remove('sea_level_pressure')
            if model == 'bcc_esm1':
                all_vars.remove('near_surface_specific_humidity')
            if model == 'hadgem3_gc31_mm':
                all_vars.remove('daily_maximum_near_surface_air_temperature')
        
        return all_vars

def select_periods(start: str, end: str, periods: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], bool]:
    if len(periods) == 0:
        return [], False
    
    selected = []
    success = False
    np_start = np.datetime64(start)
    np_end = np.datetime64(end)
    
    for period_start, period_end in periods:
        np_period_start = np.datetime64(period_start)
        np_period_end = np.datetime64(period_end)

        # start and end engulf the period completely
        if np_start <= np_period_start <= np_period_end <= np_end:
            selected.append((period_start, period_end))
            success = True
        
        # end comes between period start and period end
        elif np_period_start <= np_end <= np_period_end:
            selected.append((period_start, period_end))
            success = True

        # start comes between period start and period end
        elif np_period_start <= np_start <= np_period_end:
            selected.append((period_start, period_end))
            success = True

    return selected, success

def download_url(url: str, dst: Path) -> None:
    try:
        wget.download(url, out=str(dst), bar=None)
    except Exception as e:
        print(e)
        print(f'could not download {url}')

def download_urls(urls: List[str], dsts: List[Path], n_workers: int = 1) -> None:
    if len(urls) != len(dsts):
        raise ValueError('the number of URLs and output directories must be the same')
    
    if n_workers == 1:
        for url, dst in zip(urls, dsts):
            download_url(url, dst)
            print(f'Download {url} to {dst}')
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
            print(f'Downloading {url} to {dst}')
        q.join()

def download_from_cds(request: CDSAPIRequest) -> None:
    """
    Downloads from CDS according to the given request.

    Parameters:
        request: A request to pass to the CDS API's retrieve method.
    """
    dst = request.output
    os.makedirs(dst.parent, exist_ok=True)
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
            print(f'Downloading {req}')
        q.join()