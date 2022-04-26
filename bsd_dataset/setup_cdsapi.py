import os
import platform
import string
from typing import Optional


class CDSAPIConfig:
    def __init__(self):
        self.url = ''
        self.uid = ''
        self.apikey = ''

    def is_valid(self) -> bool:
        """Returns true if all fields have been set."""
        return self.url != '' and self.uid != '' and self.apikey != ''


class CDSAPIHelper:
    def __init__(self, fpath: Optional[str] = None):
        system = platform.system()
        if system == 'Linux' or system == 'Darwin':
            root = os.environ['HOME']
        elif system == 'Windows':
            root = os.environ['USERPROFILE']
        else:
            raise NotImplementedError(f'{system} is not a recognized system')
        self.fpath = fpath or os.path.join(root, '.cdsapirc')
        self.config = CDSAPIConfig()
        self.parse_config()

    def is_valid(self) -> bool:
        return self.config.is_valid()

    @property
    def config_path(self) -> str:
        return self.fpath

    def get(self) -> CDSAPIConfig:
        """Returns the configuration."""
        return self.config

    def set(self, uid: str, apikey: str,
            url: str = 'https://cds.climate.copernicus.eu/api/v2'):
        """Sets the configuration."""        
        self.config.uid = uid
        self.config.apikey = apikey
        self.config.url = url
    
    def parse_config(self) -> bool:
        """Return true if a valid configuration can be found."""
        if not os.path.isfile(self.fpath):
            return False
        with open(self.fpath, 'r') as f:
            config_str = f.read()

        reading_url = 0
        reading_uid = 1
        reading_apikey = 2
        done = 3
        state = reading_url

        while state != done:
            if state == reading_url:
                key = 'url: '
                if config_str[:len(key)] == key:
                    nl_idx = config_str.find('\n', len(key))
                    if nl_idx > -1:
                        url = config_str[len(key):nl_idx]
                        if url == '':
                            return False
                        config_str = config_str[nl_idx+1:]
                        state = reading_uid
                    else:
                        return False
                else:
                    return False
            elif state == reading_uid:
                key = 'key: '
                if config_str[:len(key)] == key:
                    colon_idx = config_str.find(':', len(key))
                    if colon_idx > -1:
                        uid = config_str[len(key):colon_idx]
                        if uid == '':
                            return False
                        config_str = config_str[colon_idx+1:]
                        state = reading_apikey
                    else:
                        return False
                else:
                    return False
            elif state == reading_apikey:
                for ch in string.whitespace:
                    if config_str.find(ch) > -1:
                        return False
                apikey = config_str[:]
                if apikey == '':
                    return False
                state = done
        
        self.set(uid, apikey, url)
        return True

    def setup(self, uid: str, key: str):
        """
        Sets up the CDS API v2 credentials using the passed in UID and key.

        If a CDS UID and API key are needed, register at
        https://cds.climate.copernicus.eu/user/register.
        """
        if type(uid) != str or type(key) != str:
            raise TypeError('UID or key is not a string')
        if uid == '' or key == '':
            raise ValueError('UID or key is empty')
        self.set(uid, key)
        with open(self.fpath, 'w') as f:
            f.write(f'url: {self.config.url}\n')
            f.write(f'key: {self.config.uid}:{self.config.apikey}')
        print(f'CDS API credentials written to {self.fpath}')
        return self

    def setup_cli(self):
        """Sets up the CDS API credentials using user prompts in the CLI."""
        docstring = self.setup.__doc__.splitlines()
        for ds in docstring:
            print(ds.strip())
        if self.is_valid():            
            print('\n## Found CDS API credentials ##\n')
            print(f'  URL: {self.config.url}')
            print(f'  UID: {self.config.uid}')
            print(f'  Key: {self.config.apikey}')
            proceed = input('\nProceed (y/[n])? ').lower()
            if proceed != 'y':
                print('\nExiting.')
                if __name__ != '__main__':
                    print()
                return
        print()
        uid = input('CDS UID: ')
        apikey = input('CDS Key: ')
        self.setup(uid, apikey)
        if __name__ != '__main__':
            print()
        return self


def setup_cdsapi(fpath: Optional[str] = None):
    """Wrapper for the CDS API helper."""
    handler = CDSAPIHelper(fpath)
    handler.setup_cli()


if __name__ == '__main__':
    setup_cdsapi()
    