# for lost functions that dont know where to go

from typing import *
import random 
def copy( data: Any) -> Any:
    import copy
    return copy.deepcopy(data)

def tqdm(*args, **kwargs):
    from tqdm import tqdm
    return tqdm(*args, **kwargs)

def find_word( word:str, path='./')-> str:
    import commune as c
    path = os.path.abspath(path)
    files = get_files(path)
    progress = c.tqdm(len(files))
    found_files = {}
    for f in files:
        try:
            text = c.get_text(f)
            if word not in text:
                continue
            lines = text.split('\n')
        except Exception as e:
            continue
        
        line2text = {i:line for i, line in enumerate(lines) if word in line}
        found_files[f[len(path)+1:]]  = line2text
        progress.update(1)
    return found_files


def bytes2str( data: bytes, mode: str = 'utf-8') -> str:
    
    if hasattr(data, 'hex'):
        return data.hex()
    else:
        if isinstance(data, str):
            return data
        return bytes.decode(data, mode)


def python2str( input):
    from copy import deepcopy
    import json
    input = deepcopy(input)
    input_type = type(input)
    if input_type == str:
        return input
    if input_type in [dict]:
        input = json.dumps(input)
    elif input_type in [bytes]:
        input = bytes2str(input)
    elif input_type in [list, tuple, set]:
        input = json.dumps(list(input))
    elif input_type in [int, float, bool]:
        input = str(input)
    return input

def bytes2dict(data: bytes) -> str:
    import json
    data = bytes2str(data)
    return json.loads(data)

def str2bytes( data: str, mode: str = 'hex') -> bytes:
    if mode in ['utf-8']:
        return bytes(data, mode)
    elif mode in ['hex']:
        return bytes.fromhex(data)



def chunk(sequence:list = [0,2,3,4,5,6,6,7],
        chunk_size:int=4,
        num_chunks:int= None):
    assert chunk_size != None or num_chunks != None, 'must specify chunk_size or num_chunks'
    if chunk_size == None:
        chunk_size = len(sequence) / num_chunks
    if chunk_size > len(sequence):
        return [sequence]
    if num_chunks == None:
        num_chunks = int(len(sequence) / chunk_size)
    if num_chunks == 0:
        num_chunks = 1
    chunks = [[] for i in range(num_chunks)]
    for i, element in enumerate(sequence):
        idx = i % num_chunks
        chunks[idx].append(element)
    return chunks

def is_class(module: Any) -> bool:
    return type(module).__name__ == 'type' 

def reverse_map(x:dict)->dict:
    '''
    reverse a dictionary
    '''
    return {v:k for k,v in x.items()}

def stdev( x:list= [0,1,2,3,4,5,6,7,8,9,10], p=2):
    if not isinstance(x, list):
        x = list(x)
    mean = mean(x)
    return (sum([(i - mean)**p for i in x]) / len(x))**(1/p)
std = stdev


emojis = {
    'smile': '😊',
    'sad': '😞',
    'heart': '❤️',
    'star': '⭐',
    'fire': '🔥',
    'check': '✅',
    'cross': '❌',
    'warning': '⚠️',
    'info': 'ℹ️',
    'question': '❓',
    'exclamation': '❗',
    'plus': '➕',
    'minus': '➖',
}

def emoji( name:str):
    return emojis.get(name, '❓')


def torch():
    import torch
    return torch

def tensor( *args, **kwargs):
    from torch import tensor
    return tensor(*args, **kwargs)

def mean(x:list=[0,1,2,3,4,5,6,7,8,9,10]):
    if not isinstance(x, list):
        x = list(x)
    return sum(x) / len(x)

def median(x:list=[0,1,2,3,4,5,6,7,8,9,10]):
    if not isinstance(x, list):
        x = list(x)
    x = sorted(x)
    n = len(x)
    if n % 2 == 0:
        return (x[n//2] + x[n//2 - 1]) / 2
    else:
        return x[n//2]

def df( x, **kwargs):
    from pandas import DataFrame
    return DataFrame(x, **kwargs)



def retry(fn, trials:int = 3, verbose:bool = True):
    # if fn is a self method, then it will be a bound method, and we need to get the function
    if hasattr(fn, '__self__'):
        fn = fn.__func__
    def wrapper(*args, **kwargs):
        for i in range(trials):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if verbose:
                    print(f'Retrying {fn.__name__} {i+1}/{trials}')

    return wrapper



def shuffle( x:list)->list:
    if len(x) == 0:
        return x
    random.shuffle(x)
    return x



def merge(  from_obj, 
                    to_obj ,
                    include_hidden:bool=True, 
                    allow_conflicts:bool=True, 
                    verbose: bool = False):
    
    '''
    Merge the functions of a python object into the current object (a)
    '''
    
    for fn in dir(from_obj):
        if fn.startswith('_') and not include_hidden:
            continue
        if hasattr(to_obj, fn) and not allow_conflicts:
            continue
        if verbose:
            print(f'Adding {fn}')
        setattr(to_obj, fn, getattr(from_obj, fn))
        
    return to_obj

def obj2typestr( obj):
    return str(type(obj)).split("'")[1]
    
def check_word( word:str)-> str:
    import commune as c
    files = c.glob('./')
    progress = c.tqdm(len(files))
    for f in files:
        try:
            text = c.get_text(f)
        except Exception as e:
            continue
        if word in text:
            return True
        progress.update(1)
    return False


def colors():
    return ['black', 'red', 'green', 
            'yellow', 'blue', 'magenta', 
            'cyan', 'white', 'bright_black', 
            'bright_red', 'bright_green', 
            'bright_yellow', 'bright_blue', 
            'bright_magenta', 'bright_cyan', 
            'bright_white']

colours = colors

def random_color():
    return random.choice(colors())
randcolor = randcolour = colour = color = random_colour = random_color




    
def hash( x, mode: str='sha256',*args,**kwargs) -> str:
    import hashlib
    x = python2str(x)
    if mode == 'keccak':
        return import_object('web3.main.Web3').keccak(text=x, *args, **kwargs).hex()
    elif mode == 'ss58':
        return import_object('scalecodec.utils.ss58.ss58_encode')(x, *args,**kwargs) 
    elif mode == 'python':
        return hash(x)
    elif mode == 'md5':
        return hashlib.md5(x.encode()).hexdigest()
    elif mode == 'sha256':
        return hashlib.sha256(x.encode()).hexdigest()
    elif mode == 'sha512':
        return hashlib.sha512(x.encode()).hexdigest()
    elif mode =='sha3_512':
        return hashlib.sha3_512(x.encode()).hexdigest()
    else:
        raise ValueError(f'unknown mode {mode}')



def num_words( text):
    return len(text.split(' '))


def random_word( *args, n=1, seperator='_', **kwargs):
    import commune as c
    random_words = c.module('key').generate_mnemonic(*args, **kwargs).split(' ')[0]
    random_words = random_words.split(' ')[:n]
    if n == 1:
        return random_words[0]
    else:
        return seperator.join(random_words.split(' ')[:n])

def choice( options:Union[list, dict])->list:
    from copy import deepcopy
    options = deepcopy(options) # copy to avoid changing the original
    if len(options) == 0:
        return None
    if isinstance(options, dict):
        options = list(options.values())
    assert isinstance(options, list),'options must be a list'
    return random.choice(options)

def sample( options:list, n=2):
    if isinstance(options, int):
        options = list(range(options))
    options = shuffle(options)
    return options[:n]



def round_decimals( x:Union[float, int], decimals: int=6, small_value: float=1.0e-9):
    
    import math
    """
    Rounds x to the number of {sig} digits
    :param x:
    :param sig: signifant digit
    :param small_value: smallest possible value
    :return:
    """
    x = float(x)
    return round(x, decimals)




def munch2dict( x:'Munch', recursive:bool=True)-> dict:
    from munch import Munch

    '''
    Turn munch object  into dictionary
    '''
    if isinstance(x, Munch):
        x = dict(x)
        for k,v in x.items():
            if isinstance(v, Munch) and recursive:
                x[k] = munch2dict(v)
    return x 
to_dict = munch2dict



required_libs = []

def ensure_libs(libs: List[str] = None, verbose:bool=False):
    results = []
    for lib in libs:
        results.append(ensure_lib(lib, verbose=verbose))
    return results

def version( lib:str=None):
    import commune as c
    lib = lib or c.repo_name
    lines = [l for l in c.cmd(f'pip3 list', verbose=False).split('\n') if l.startswith(lib)]
    if len(lines)>0:
        return lines[0].split(' ')[-1].strip()
    else:
        return f'No Library Found {lib}'
    
def pip_exists(lib:str):
    return bool(lib in pip_libs())

def ensure_lib( lib:str, verbose:bool=False):
    if  pip_exists(lib):
        return {'lib':lib, 'version':version(lib), 'status':'exists'}
    elif pip_exists(lib) == False:
        pip_install(lib, verbose=verbose)
    return {'lib':lib, 'version':version(lib), 'status':'installed'}

def pip_install(lib:str= None,
                upgrade:bool=True ,
                verbose:str=True,
                ):
    import commune as c
    if lib in c.modules():
        c.print(f'Installing {lib} Module from local directory')
        lib = c.resolve_module(lib).dirpath()
    if lib == None:
        lib = c.libpath

    if c.path_exists(lib):
        cmd = f'pip install -e'
    else:
        cmd = f'pip install'
        if upgrade:
            cmd += ' --upgrade'
    return c.cmd(cmd, verbose=verbose)


# JUPYTER NOTEBOOKS
def enable_jupyter():
    import commune as c
    c.nest_asyncio()

jupyter = enable_jupyter

def pip_list(lib=None):
    import commune as c
    lib = lib or c.repo_name
    pip_list =  c.cmd(f'pip list', verbose=False, bash=True).split('\n')
    if lib != None:
        pip_list = [l for l in pip_list if l.startswith(lib)]
    return pip_list

def is_mnemonic(s: str) -> bool:
    import re
    # Match 12 or 24 words separated by spaces
    return bool(re.match(r'^(\w+ ){11}\w+$', s)) or bool(re.match(r'^(\w+ ){23}\w+$', s))

def is_private_key(s: str) -> bool:
    import re
    # Match a 64-character hexadecimal string
    pattern = r'^[0-9a-fA-F]{64}$'
    return bool(re.match(pattern, s))


def jsonable( value):
    import json
    try:
        json.dumps(value)
        return True
    except:
        return False
        
## STORAGE
def dict2hash( d:dict) -> str:
    for k in d.keys():
        assert jsonable(d[k]), f'{k} is not jsonable'
    return hash(d)
def locals2hash(kwargs:dict = {'a': 1}, keys=['kwargs']) -> str:
    kwargs.pop('cls', None)
    kwargs.pop('self', None)
    return dict2hash(kwargs)


def random_int(start_value=100, end_value=None):
    if end_value == None: 
        end_value = start_value
        start_value, end_value = 0 , start_value
    assert start_value != None, 'start_value must be provided'
    assert end_value != None, 'end_value must be provided'
    return random.randint(start_value, end_value)

def random_float(min=0, max=1):
    return random.uniform(min, max)

def random_ratio_selection( x:list, ratio:float = 0.5)->list:
    if type(x) in [float, int]:
        x = list(range(int(x)))
    assert len(x)>0
    if ratio == 1:
        return x
    assert ratio > 0 and ratio <= 1
    random.shuffle(x)
    k = max(int(len(x) * ratio),1)
    return x[:k]

def is_int( value) -> bool:
    o = False
    try :
        int(value)
        if '.' not in str(value):
            o =  True
    except:
        pass
    return o

def is_float( value) -> bool:
    o =  False
    try :
        float(value)
        if '.' in str(value):
            o = True
    except:
        pass

    return o 

def dict2munch( x:dict, recursive:bool=True)-> 'Munch':
    from munch import Munch
    '''
    Turn dictionary into Munch
    '''
    if isinstance(x, dict):
        for k,v in x.items():
            if isinstance(v, dict) and recursive:
                x[k] = dict2munch(v)
        x = Munch(x)
    return x 

def munch2dict( x:'Munch', recursive:bool=True)-> dict:
    from munch import Munch
    if isinstance(x, Munch):
        x = dict(x)
        for k,v in x.items():
            if isinstance(v, Munch) and recursive:
                x[k] = munch2dict(v)
    return x 

def munch( x:Dict) -> 'Munch':
    return dict2munch(x)

def time(  t=None) -> float:
    from time import time
    return time()
def timestamp(  t=None) -> float:
    return int(time())
def time2datetime( t:float):
    import commune as c
    return c.util('time.time2datetime')(t)

time2date = time2datetime

def datetime2time( x:str):
    import datetime
    return datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timestamp()

def search_dict(d:dict = 'k,d', search:str = {'k.d': 1}) -> dict:
    search = search.split(',')
    new_d = {}
    for k,v in d.items():
        if search in k.lower():
            new_d[k] = v
    return new_d
