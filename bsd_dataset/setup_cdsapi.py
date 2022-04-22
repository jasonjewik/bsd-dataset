import io
import os
import platform

def setup_cdsapi():
    """
    Sets up the CDS API credentials for downloading climate datasets.
    """    
    system = platform.system()
    if system == 'Linux' or system == 'Darwin':
        root = os.environ['HOME']
    elif system == 'Windows':
        root = os.environ['USERPROFILE']
    else:
        raise NotImplementedError(f'{system} is not a recognized system')

    fpath = os.path.join(root, '.cdsapirc')
    if os.path.isfile(fpath):
        print()
        print(f'CDS API credentials already set up at {fpath}')
        print('===== START FILE CONTENTS =====')
        with open(fpath, 'r') as f:
            print(f.read())
        print('===== END FILE CONTENTS =====')
        print()

    print('If a CDS UID and API Key is needed, register at')
    print('https://cds.climate.copernicus.eu/user/register.')
    print()

    uid = input('CDS UID: ')
    apikey = input('CDS API Key: ')

    with io.open(fpath, 'w') as f:
        f.write('url: https://cds.climate.copernicus.eu/api/v2\n')
        f.write(f'key: {uid}:{apikey}')

    print(f'CDS API credentials written to {fpath}')
    print()

if __name__ == '__main__':
    setup_cdsapi()
    