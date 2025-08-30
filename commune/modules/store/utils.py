
import json
import os


def hash( x, mode: str='sha256',*args,**kwargs) -> str:
    import hashlib
    x = python2str(x)
    if mode == 'keccak':
        y =  import_object('web3.main.Web3').keccak(text=x, *args, **kwargs).hex()
    elif mode == 'ss58':
        y =  import_object('scalecodec.utils.ss58.ss58_encode')(x, *args,**kwargs) 
    elif mode == 'python':
        y =  hash(x)
    elif mode == 'md5':
        y =  hashlib.md5(x.encode()).hexdigest()
    elif mode == 'sha256':
        y =  hashlib.sha256(x.encode()).hexdigest()
    elif mode == 'sha512':
        y =  hashlib.sha512(x.encode()).hexdigest()
    elif mode =='sha3_512':
        y =  hashlib.sha3_512(x.encode()).hexdigest()
    else:
        raise ValueError(f'unknown mode {mode}')
    return mode + ':' + y

def get_json(path, default=None):
    if not path.endswith('.json'):
        path = path + '.json'
    if not os.path.exists(path):
        return default
    try:
        with open(path, 'r') as file:
            data = json.load(file)
    except Exception as e:
        return default
    return data


def put_json(path:str, data:dict, key=None) -> dict:
    if not path.endswith('.json'):
        path = path + '.json'
    data = json.dumps(data) if not isinstance(data, str) else data
    dirpath = os.path.dirname(path)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    with open(path, 'w') as file:
        file.write(data)
    return {'success': True, 'path': f'{path}', 'size': len(data)*8}

def rm_folder(path):
    import shutils
    if not os.path.exists(path):
        return {'success': False, 'path': path}
    if not os.path.isdir(path):
        return {'success': False, 'path': path}
    shutils.rmtree(path)
    return {'success': True, 'path': path}

def rm_file(path):
    if os.path.exists(path):
        os.remove(path)
    assert not os.path.exists(path), f'Failed to remove {path}'
    return {'success': False, 'path': path}

def rm(path):
    if os.path.isdir(path):
        return rm_folder(path)
    elif os.path.isfile(path):
        return rm_file(path)
    else:
        raise ValueError(f'Path {path} does not exist or is neither a file nor a directory')

