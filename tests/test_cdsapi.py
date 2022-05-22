from copy import deepcopy
import os
from pathlib import Path
import pytest

from bsd_dataset import CDSAPICredentialHelper
from bsd_dataset.datasets.download_utils import DatasetRequest, CDSAPIRequestBuilder


@pytest.fixture(scope='function')
def config_file(tmp_path_factory):
    return tmp_path_factory.mktemp('cdsapi') / 'config'


class TestReading:
    def test_valid_config(self):
        assert CDSAPICredentialHelper('./tests/cdsapi/valid_config').is_valid()

    def test_missing_url_config(self):
        assert not CDSAPICredentialHelper('./tests/cdsapi/missing_url_config_1').is_valid()
        assert not CDSAPICredentialHelper('./tests/cdsapi/missing_url_config_2').is_valid()

    def test_malformed_url_config(self):
        assert not CDSAPICredentialHelper('./tests/cdsapi/malformed_url_config').is_valid()

    def test_missing_uid_config(self):
        assert not CDSAPICredentialHelper('./tests/cdsapi/missing_uid_config').is_valid()

    def test_missing_key_config(self):
        assert not CDSAPICredentialHelper('./tests/cdsapi/missing_key_config').is_valid()

    def test_config_does_not_exist(self):
        assert not CDSAPICredentialHelper('./this/file/does/not/exist').is_valid()


class TestWriting:
    def test_valid_setup(self, config_file):
        assert CDSAPICredentialHelper(config_file).setup('12345', '99999-zzzzz').is_valid()
        assert CDSAPICredentialHelper(config_file).is_valid()

    def test_wrong_types_setup(self, config_file):
        with pytest.raises(TypeError):
            CDSAPICredentialHelper(config_file).setup(12345, '55555-wwwww')
        assert not os.path.isfile(config_file)
        with pytest.raises(TypeError):
            CDSAPICredentialHelper(config_file).setup('09876', 12345)
        assert not os.path.isfile(config_file)
        with pytest.raises(TypeError):
            CDSAPICredentialHelper(config_file).setup(9876, 12345)
        assert not os.path.isfile(config_file)

    def test_empty_setup(self, config_file):
        with pytest.raises(ValueError):
            CDSAPICredentialHelper(config_file).setup('', '55555-wwwww')
        assert not os.path.isfile(config_file)
        with pytest.raises(ValueError):
            CDSAPICredentialHelper(config_file).setup('12345', '')
        assert not os.path.isfile(config_file)
        with pytest.raises(ValueError):
            CDSAPICredentialHelper(config_file).setup('', '')
        assert not os.path.isfile(config_file)


@pytest.fixture(name='builder_fixture', scope='class')
def builder_fixture():
    class BuilderFixture:
        def __init__(self):
            self.builder = CDSAPIRequestBuilder()
            self.root = Path('./data').expanduser().resolve()    
            self.dataset = 'projections-cmip5-daily-single-levels'
            self.model = 'gfdl_cm3'
            self.expected_output = self.root / 'cds' / f'{self.dataset}.{self.model}.tar.gz'
            self.builder_kwargs = {
                'root': self.root,
                'train_dates': ('1981-01-01', '1981-12-31'),
                'val_dates': ('1982-01-01', '1982-12-31'),
                'test_dates': ('1983-01-01', '1983-12-31')
            }

        @property
        def kwargs(self):
            return deepcopy(self.builder_kwargs)

        def build(self, *args, **kwargs):
            return self.builder.build(*args, **kwargs)
    
    return BuilderFixture()


class TestRequestBuilder:
    
    def test_is_cdsreq(self, builder_fixture):
        dataset_request = DatasetRequest(
            builder_fixture.dataset,
            model=builder_fixture.model,
            ensemble_member='r1i1p1',
            variable='mean_precipitation_flux'
        )
        assert dataset_request.is_cds_req()

    def test_valid_request(self, builder_fixture):
        # Single variable
        dataset_request = DatasetRequest(
            builder_fixture.dataset,
            model=builder_fixture.model,
            ensemble_member='r1i1p1',
            variable='mean_precipitation_flux'
        )
        expected_options = {
            'ensemble_member': 'r1i1p1',
            'format': 'tgz',
            'experiment': 'historical',
            'variable': 'mean_precipitation_flux',
            'model': builder_fixture.model,
            'period': '19800101-19841231',
        }
        kwargs = builder_fixture.kwargs
        kwargs['dataset_request'] = dataset_request
        request = builder_fixture.build(**kwargs)

        assert request.dataset == builder_fixture.dataset
        assert request.output == builder_fixture.expected_output
        assert request.options == expected_options

        # List of variables
        dataset_request = DatasetRequest(
            builder_fixture.dataset,
            model=builder_fixture.model,
            ensemble_member='r1i1p1',
            variable=['mean_precipitation_flux', '10m_wind_speed']
        )
        expected_options = {
            'ensemble_member': 'r1i1p1',
            'format': 'tgz',
            'experiment': 'historical',
            'variable': ['mean_precipitation_flux', '10m_wind_speed'],
            'model': builder_fixture.model,
            'period': '19800101-19841231',
        }
        kwargs = builder_fixture.kwargs
        kwargs['dataset_request'] = dataset_request
        request = builder_fixture.build(**kwargs)

        assert request.dataset == builder_fixture.dataset
        assert request.output == builder_fixture.expected_output
        assert request.options == expected_options

    def test_missing_model(self, builder_fixture):
        # kwarg "model" is missing in the dataset request
        dataset_request = DatasetRequest(
            builder_fixture.dataset,
            ensemble_member='r1i1p1',
            variable='mean_precipitation_flux'
        )
        kwargs = builder_fixture.kwargs
        kwargs['dataset_request'] = dataset_request
        with pytest.raises(AttributeError):
            builder_fixture.build(**kwargs)

    def test_missing_variable(self, builder_fixture):
        # kwarg "variable" is missing in the dataset request
        dataset_request = DatasetRequest(
            builder_fixture.dataset,
            model=builder_fixture.model,
            ensemble_member='r1i1p1'
        )
        kwargs = builder_fixture.kwargs
        kwargs['dataset_request'] = dataset_request
        with pytest.raises(AttributeError):            
            builder_fixture.build(**kwargs)

        # kwarg "variable" is set in the dataset request, but empty
        setattr(dataset_request, 'variable', [])
        kwargs = builder_fixture.kwargs
        kwargs['dataset_request'] = dataset_request
        with pytest.raises(ValueError):            
            builder_fixture.build(**kwargs)

    def test_malformed_variable(self, builder_fixture):
        # kwarg "variable" is neither string nor list type
        dataset_request = DatasetRequest(
            builder_fixture.dataset,
            model=builder_fixture.model,
            ensemble_member='r1i1p1',
            variable=2
        )
        kwargs = builder_fixture.kwargs
        kwargs['dataset_request'] = dataset_request
        with pytest.raises(TypeError):
            builder_fixture.builder(**kwargs)

        # kwarg "variable" is unrecognized
        dataset_request = DatasetRequest(
            builder_fixture.dataset,
            model=builder_fixture.model,
            ensemble_member='r1i1p1',
            variable='not_a_climate_variable'
        )
        kwargs = builder_fixture.kwargs
        kwargs['dataset_request'] = dataset_request
        with pytest.raises(ValueError):
            builder_fixture.build(**kwargs)

    def test_missing_ensemble(self, builder_fixture):
        # kwarg "ensemble_member" is missing
        dataset_request = DatasetRequest(
            builder_fixture.dataset,
            model=builder_fixture.model,
            variable='mean_precipitation_flux'
        )
        kwargs = builder_fixture.kwargs
        kwargs['dataset_request'] = dataset_request
        with pytest.raises(AttributeError):
            builder_fixture.build(**kwargs)