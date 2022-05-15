import os
import pytest

from bsd_dataset import CDSAPICredentialHelper


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
        
