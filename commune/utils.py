# Standard Library Imports
import os
import sys
import time
import random
import subprocess
import shutil
import glob
import socket
import urllib
import json
import re
import itertools
from contextlib import contextmanager
import requests
import psutil
import netaddr
from loguru import logger
from typing import Any, Optional, List, Dict, Tuple, Union
import gc
import asyncio
asyncio.BaseEventLoop.__del__ = lambda self: None

def path_exists(path:str):
    return os.path.exists(path)

def exists(path:str):
    return os.path.exists(path)

def import_module( import_path:str ) -> 'Object':
    from importlib import import_module
    return import_module(import_path)

def import_object( key:str, splitters=['/', '::', '.'], **kwargs)-> Any:
    from importlib import import_module
    ''' Import an object from a string with the format of {module_path}.{object}'''
    module_path = None
    object_name = None
    for splitter in splitters:
        key = key.replace(splitter, '.')
    module_path = '.'.join(key.split('.')[:-1])
    object_name = key.split('.')[-1]
    
    if isinstance(key, str) and key.endswith('.py') and path_exists(key):
        key = c.path2objectpath(key)
    assert module_path != None and object_name != None, f'Invalid key {key}'
    module_obj = import_module(module_path)
    return  getattr(module_obj, object_name)

def shlex_split(s):
    result = []
    current = ''
    in_single_quote = False
    in_double_quote = False
    escape = False

    for char in s:
        if escape:
            current += char
            escape = False
        elif char == '\\':
            escape = True
        elif char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif char.isspace() and not in_single_quote and not in_double_quote:
            if current:
                result.append(current)
                current = ''
        else:
            current += char

    if current:
        result.append(current)

    return result

def is_success( x):
    # assume that if the result is a dictionary, and it has an error key, then it is an error
    if isinstance(x, dict):
        if 'error' in x:
            return False
        if 'success' in x and x['success'] == False:
            return False
    return True

def is_error( x:Any):
    """
    The function checks if the result is an error
    The error is a dictionary with an error key set to True
    """
    if isinstance(x, dict):
        if 'error' in x and x['error'] == True:
            return True
        if 'success' in x and x['success'] == False:
            return True
    return False

# As a context manager
@contextmanager
def print_load(message="Loading", duration=5):
    spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    factor = 4
    
    # Create a flag to control the animation
    stop_animation = False
    
    def animate():
        start_time = time.time()
        try:
            while not stop_animation:
                for frame in spinner:
                    if stop_animation:
                        break
                    loading_text = f"\r{CYAN}{frame*factor}{message}({int(time.time() - start_time)}s){frame*factor}"
                    sys.stdout.write(loading_text)
                    sys.stdout.flush()
                    time.sleep(0.1)
        except KeyboardInterrupt:
            sys.stdout.write("\r" + " " * (len(message) + 10))
            sys.stdout.write(f"\r{CYAN}Loading cancelled!{RESET}\n")
    
    # Start animation in a separate thread
    import threading
    thread = threading.Thread(target=animate)
    thread.start()
    
    try:
        yield
    finally:
        # Stop the animation
        stop_animation = True
        thread.join()
        sys.stdout.write("\r" + " " * (len(message) + 10))
        sys.stdout.write(f"\r{CYAN}✨ {message} complete!{RESET}\n")

def test_loading_animation():
    with print_load("Testing", duration=3):
        time.sleep(3)

def get_console( console = None, **kwargs):
    import logging
    from rich.logging import RichHandler
    from rich.console import Console
    logging.basicConfig( handlers=[RichHandler()])   
    return Console()

def print_console( *text:str, 
            color:str=None, 
            verbose:bool = True,
            console: 'Console' = None,
            flush:bool = False,
            buffer:str = None,
            **kwargs):
            
    if not verbose:
        return 
    if color == 'random':
        color = random_color()
    if color:
        kwargs['style'] = color
    
    if buffer != None:
        text = [buffer] + list(text) + [buffer]

    console = get_console(console)
    try:
        if flush:
            console.print(**kwargs, end='\r')
        console.print(*text, **kwargs)
    except Exception as e:
        print(e)

def success( *args, **kwargs):
    logger = resolve_logger()
    return logger.success(*args, **kwargs)

def error( *args, **kwargs):
    logger = resolve_logger()
    return logger.error(*args, **kwargs)


def debug( *args, **kwargs):
    logger = resolve_logger()
    return logger.debug(*args, **kwargs)

def warning( *args, **kwargs):
    logger = resolve_logger()
    return logger.warning(*args, **kwargs)

def status( *args, **kwargs):
    console = get_console()
    return console.status(*args, **kwargs)

def log( *args, **kwargs):
    console = get_console()
    return console.log(*args, **kwargs)

### LOGGER LAND ###

def resolve_logger( logger = None):
    if not hasattr('logger'):
        from loguru import logger
        logger = logger.opt(colors=True)
    if logger is not None:
        logger = logger
    return logger


def critical( *args, **kwargs):
    console = get_console()
    return console.critical(*args, **kwargs)

def echo(x):
    return x


# for lost functions that dont know where to go

def copy( data: Any) -> Any:
    from copy import deepcopy
    return deepcopy(data)

def tqdm(*args, **kwargs):
    from tqdm import tqdm
    return tqdm(*args, **kwargs)

def find_word( word:str, path='./')-> str:
    import os
    import commune as c
    path = os.path.abspath(path)
    files = c.files(path)
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

def random_color():
    colors = ['black', 'red', 'green', 
            'yellow', 'blue', 'magenta', 
            'cyan', 'white', 'bright_black', 
            'bright_red', 'bright_green', 
            'bright_yellow', 'bright_blue', 
            'bright_magenta', 'bright_cyan', 
            'bright_white']
    return random.choice(colors)
    

def get_hash( x, mode: str='sha256',*args,**kwargs) -> str:
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
    return  y



def num_words( text):
    return len(text.split(' '))

def random_word( *args, n=1, seperator='_', **kwargs):
    import commune as c
    random_words = c.mod('key').generate_mnemonic(*args, **kwargs).split(' ')[0]
    random_words = random_words.split(' ')[:n]
    if n == 1:
        return random_words[0]
    else:
        return seperator.join(random_words.split(' ')[:n])

def chown(path:str, user:str='root', group:str='root'):
    cmd = f'chown {user}:{group} {path}'
    return os.system(cmd)


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
        lib = c.lib_path

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

def munch( x:'Munch', recursive:bool=True)-> dict:
    from munch import Munch
    if isinstance(x, Munch):
        x = dict(x)
        for k,v in x.items():
            if isinstance(v, Munch) and recursive:
                x[k] = munch2dict(v)
    return x 

def munch2dict( x:'Munch', recursive:bool=True)-> dict:
    from munch import Munch
    if isinstance(x, Munch):
        x = dict(x)
        for k,v in x.items():
            if isinstance(v, Munch) and recursive:
                x[k] = munch2dict(v)
    return x 

def timestamp(t=None) -> float:
    return int(time.time())

def time2date(self, x:float=None):
    import datetime
    return datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S')


import sys
import time
import threading
from contextlib import contextmanager

@contextmanager
def spinner(message="Working"):
    """A context manager for displaying a spinner while waiting."""
    spinner_chars = ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷']
    stop_spinner = False
    
    def spin():
        i = 0
        while not stop_spinner:
            sys.stdout.write(f'\r{message} {spinner_chars[i % len(spinner_chars)]}')
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1
    
    spinner_thread = threading.Thread(target=spin)
    spinner_thread.daemon = True
    spinner_thread.start()
    
    try:
        yield
    finally:
        stop_spinner = True
        sys.stdout.write('\r' + ' ' * (len(message) + 10) + '\r')
        sys.stdout.flush()

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

def module2size(search=None, depth=10000, **kwargs):
    import commune as c
    module2size = {}
    module2code = c.module2code(search=search, depth=depth, **kwargs)
    for k,v in module2code.items():
        module2size[k] = len(v)
    module2size = dict(sorted(module2size.items(), key=lambda x: x[1], reverse=True))
    return module2size

def port_available(port:int, ip:str ='0.0.0.0'):
    return not port_used(port=port, ip=ip)

def resolve_ip(ip=None, external:bool=True) -> str:
    if ip == None:
        if external:
            ip = external_ip()
        else:
            ip = '0.0.0.0'
    assert isinstance(ip, str)
    return ip

def get_available_ports(port_range: List[int] = None , ip:str =None) -> int:
    import commune as c
    port_range = c.resolve_port_range(port_range)
    ip = ip if ip else '0.0.0.0'
    available_ports = []
    # return only when the port is available
    for port in range(*port_range): 
        if not c.port_used(port=port, ip=ip):
            available_ports.append(port)         
    return available_ports
available_ports = get_available_ports


def resolve_port(port:int=None, **kwargs):
    '''
    Resolves the port and finds one that is available
    '''
    if port == None or port == 0:
        port = free_port(port, **kwargs)
        
    if port_used(port):
        port = free_port(port, **kwargs)
        
    return int(port)



def ip(max_age=None, update:bool = False, **kwargs) -> str:
    
    try:
        import commune as c
        path = c.get_path('ip')
        ip = c.get(path, None, max_age=max_age, update=update)
        if ip == None:
            ip = external_ip()
            c.put(path, ip)
    except Exception as e:
        print('Error while getting IP')
        return '0.0.0.0'
    return ip


@contextmanager
def timer(name=None):
    operation_name = name or "Operation"
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        print(f"{operation_name} took {elapsed_time:.6f} seconds to complete")

def has_free_ports(n:int = 1, **kwargs):
    return len(free_ports(n=n, **kwargs)) > 0

def external_ip( default_ip='0.0.0.0') -> str:
    import commune as c
    
    r""" Checks CURL/URLLIB/IPIFY/AWS for your external ip.
        Returns:
            external_ip  (:obj:`str` `required`):
                Your routers external facing ip as a string.

        Raises:
            Exception(Exception):
                Raised if all external ip attempts fail.
    """
    ip = None
    try:
        ip = c.cmd('curl -s ifconfig.me')
        assert isinstance(c.ip_to_int(ip), int)
    except Exception as e:
        print(e)

    if is_valid_ip(ip):
        return ip
    try:
        ip = requests.get('https://api.ipify.org').text
        assert isinstance(c.ip_to_int(ip), int)
    except Exception as e:
        print(e)

    if is_valid_ip(ip):
        return ip
    # --- Try AWS
    try:
        ip = requests.get('https://checkip.amazonaws.com').text.strip()
        assert isinstance(c.ip_to_int(ip), int)
    except Exception as e:
        print(e)

    if is_valid_ip(ip):
        return ip
    # --- Try myip.dnsomatic 
    try:
        process = os.popen('curl -s myip.dnsomatic.com')
        ip  = process.readline()
        assert isinstance(c.ip_to_int(ip), int)
        process.close()
    except Exception as e:
        print(e)  

    if is_valid_ip(ip):
        return ip
    # --- Try urllib ipv6 
    try:
        ip = urllib.request.urlopen('https://ident.me').read().decode('utf8')
        assert isinstance(c.ip_to_int(ip), int)
    except Exception as e:
        print(e)

    if is_valid_ip(ip):
        return ip
    # --- Try Wikipedia 
    try:
        ip = requests.get('https://www.wikipedia.org').headers['X-Client-IP']
        assert isinstance(c.ip_to_int(ip), int)
    except Exception as e:
        print(e)

    if is_valid_ip(ip):
        return ip

    return default_ip

def kill_port(port, timeout=10):
    try:
        # Check operating system
        operating_system = sys.platform
        
        if operating_system == "windows":
            # Windows command
            command = f"for /f \"tokens=5\" %a in ('netstat -aon ^| find \":{port}\"') do taskkill /F /PID %a"
            subprocess.run(command, shell=True)
        
        elif operating_system in ["linux", "darwin"]:  # Linux or MacOS
            # Unix command
            command = f"lsof -i tcp:{port} | grep LISTEN | awk '{{print $2}}' | xargs kill -9"
            subprocess.run(command, shell=True)
        t0 = time.time()
        while port_used(port):
            if time.time() - t0 > timeout:
                raise Exception(f'Timeout for killing port {port}')
        
        print(f"Process on port {port} has been killed")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False
    


def kill_ports(ports = None, *more_ports):
    import commune as c
    ports = ports or used_ports()
    if isinstance(ports, int):
        ports = [ports]
    if '-' in ports:
        ports = list(range([int(p) for p in ports.split('-')]))
    ports = list(ports) + list(more_ports)
    for p in ports:
        kill_port(p)
    return used_ports()

def is_port_public(port:int, ip:str=None, timeout=0.5):
    import socket
    ip = ip or ip()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((ip, port)) == 0
        
def public_ports(timeout=1.0):
    import commune as c
    futures = []
    for port in free_ports():
        c.print(f'Checking port {port}')
        futures += [c.submit(is_port_public, {'port':port}, timeout=timeout)]
    results =  c.wait(futures, timeout=timeout)
    results = list(map(bool, results))
    return results

def free_ports(n=10, random_selection:bool = False, **kwargs ) -> List[int]:
    free_ports = []
    avoid_ports = kwargs.pop('avoid_ports', [])
    for i in range(n):
        try:
            free_ports += [free_port(  random_selection=random_selection, 
                                        avoid_ports=avoid_ports, **kwargs)]
        except Exception as e:
            print(f'Error: {e}')
            break
        avoid_ports += [free_ports[-1]]
 
    return free_ports

def random_port(*args, **kwargs):
    import commune as c
    return c.choice(c.free_ports(*args, **kwargs))


def free_port(ports = None,
                port_range: List[int] = None , 
                ip:str =None, 
                avoid_ports = None,
                random_selection:bool = True) -> int:
    import commune as c
    
    '''
    
    Get an availabldefe port within the {port_range} [start_port, end_poort] and {ip}
    '''
    avoid_ports = avoid_ports if avoid_ports else []
    
    if ports == None:
        port_range = c.get_port_range(port_range)
        ports = list(range(*port_range))
    ip = ip if ip else '0.0.0.0'
    if random_selection:
        ports = c.shuffle(ports)
    port = None
    for port in ports: 
        if port in avoid_ports:
            continue
        if c.port_available(port=port, ip=ip):
            return port
    raise Exception(f'ports {port_range[0]} to {port_range[1]} are occupied, change the port_range to encompase more ports')

get_available_port = free_port



def used_ports(ports:List[int] = None, ip:str = '0.0.0.0', port_range:Tuple[int, int] = None):
    import commune as c
    '''
    Get availabel ports out of port range
    
    Args:
        ports: list of ports
        ip: ip address
    
    '''
    port_range = resolve_port_range(port_range=port_range)
    if ports == None:
        ports = list(range(*port_range))
    
    async def check_port(port, ip):
        return port_used(port=port, ip=ip)
    
    used_ports = []
    jobs = []
    for port in ports: 
        jobs += [check_port(port=port, ip=ip)]
            
    results = gather(jobs)
    for port, result in zip(ports, results):
        if isinstance(result, bool) and result:
            used_ports += [port]
        
    return used_ports



def port_free(*args, **kwargs) -> bool:
    return not port_used(*args, **kwargs)


def get_port(port:int = None)->int:
    port = port if port is not None and port != 0 else free_port()
    while port_used(port):
        port += 1   
    return port 
    
def ports() -> List[int]:
    
    return list(range(*get_port_range()))

def resolve_port_range(port_range: list = None) -> list:
    return get_port_range(port_range)

def set_port_range(*port_range: list):
    import commune as c
    if '-' in port_range[0]:
        port_range = list(map(int, port_range[0].split('-')))
    if len(port_range) ==0 :
        port_range = c.port_range
    elif len(port_range) == 1:
        if port_range[0] == None:
            port_range = c.port_range
    assert len(port_range) == 2, 'Port range must be a list of two integers'        
    for port in port_range:
        assert isinstance(port, int), f'Port {port} range must be a list of integers'
    assert port_range[0] < port_range[1], 'Port range must be a list of integers'
    c.put('port_range', port_range)
    return port_range

def ip_to_int(str_val: str) -> int:
    r""" Maps an ip-string to a unique integer.
        arg:
            str_val (:tyep:`str`, `required):
                The string representation of an ip. Of form *.*.*.* for ipv4 or *::*:*:*:* for ipv6

        Returns:
            int_val  (:type:`int128`, `required`):
                The integer representation of an ip. Must be in the range (0, 3.4028237e+38).

        Raises:
            netaddr.core.AddrFormatError (Exception):
                Raised when the passed str_val is not a valid ip string value.
    """
    return int(netaddr.IPAddress(str_val))


def ip_version(str_val: str) -> int:
    r""" Returns the ip version (IPV4 or IPV6).
        arg:
            str_val (:tyep:`str`, `required):
                The string representation of an ip. Of form *.*.*.* for ipv4 or *::*:*:*:* for ipv6

        Returns:
            int_val  (:type:`int128`, `required`):
                The ip version (Either 4 or 6 for IPv4/IPv6)

        Raises:
            netaddr.core.AddrFormatError (Exception):
                Raised when the passed str_val is not a valid ip string value.
    """
    import netaddr
    return int(netaddr.IPAddress(str_val).version)

def ip__str__(ip_type:int, ip_str:str, port:int):
    """ Return a formatted ip string
    """
    return "/ipv%i/%s:%i" % (ip_type, ip_str, port)

def get_port_range(port_range: list = None) -> list:
    import commune as c
    port_range = c.get('port_range', [])
    if isinstance(port_range, str):
        port_range = list(map(int, port_range.split('-')))
    if len(port_range) == 0:
        port_range = c.port_range
    print(c.port_range)
    port_range = list(port_range)
    assert isinstance(port_range, list), 'Port range must be a list'
    assert isinstance(port_range[0], int), 'Port range must be a list of integers'
    assert isinstance(port_range[1], int), 'Port range must be a list of integers'
    return port_range

def is_valid_ip(ip:str) -> bool:
    import netaddr
    r""" Checks if an ip is valid.
        Args:
            ip  (:obj:`str` `required`):
                The ip to check.

        Returns:
            valid  (:obj:`bool` `required`):
                True if the ip is valid, False otherwise.
    """
    try:
        netaddr.IPAddress(ip)
        return True
    except Exception as e:
        return False

def check_used_ports(start_port = 8501, end_port = 8600, timeout=5):
    import commune as c
    port_range = [start_port, end_port]
    used_ports = {}
    for port in range(*port_range):
        used_ports[port] = c.port_used(port)
    return used_ports

def port_used( port: int, ip: str = '0.0.0.0', timeout: int = 1):
    import socket
    if not isinstance(port, int):
        return False
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Set the socket timeout
        sock.settimeout(timeout)

        # Try to connect to the specified IP and port
        try:
            port=int(port)
            sock.connect((ip, port))
            return True
        except socket.error:
            return False

def int_to_ip(int_val: int) -> str:
    r""" Maps an integer to a unique ip-string 
        Args:
            int_val  (:type:`int128`, `required`):
                The integer representation of an ip. Must be in the range (0, 3.4028237e+38).

        Returns:
            str_val (:tyep:`str`, `required):
                The string representation of an ip. Of form *.*.*.* for ipv4 or *::*:*:*:* for ipv6

        Raises:
            netaddr.core.AddrFormatError (Exception):
                Raised when the passed int_vals is not a valid ip int value.
    """
    import netaddr
    return str(netaddr.IPAddress(int_val))


def is_url( address:str) -> bool:
    import commune as c
    if not isinstance(address, str):
        return False
    if '://' in address:
        return True
    conds = []
    conds.append(isinstance(address, str))
    conds.append(':' in address)
    conds.append(c.is_int(address.split(':')[-1]))
    return all(conds)


def osname():
    return os.name

def get_pid():
    return os.getpid()

## SYSTEM INFO
def system_info():
    return {
        'os': osname(),
        'cpu': cpu_info(),
        'memory': memory_info(),
        'disk': disk_info(),
    }

def is_mac():
    return sys.platform == 'darwin'

def run_command(command:str):
    import subprocess
    process = subprocess.run(shlex_split(command), 
                        stdout=subprocess.PIPE, 
                        universal_newlines=True)
    return process

def check_pid(pid):        
    """ Check For the existence of a unix pid. """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True

def ensure_path(path):
    """
    ensures a dir_path exists, otherwise, it will create it 
    """

    dir_path = os.path.dirname(path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    return path


def cpu_count():
    return os.cpu_count()

num_cpus = cpu_count

def set_cwd(path:str):
    return os.chdir(path)

def memory_usage_info(fmt='gb'):
    import psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    response = {
        'rss': memory_info.rss,
        'vms': memory_info.vms,
        'pageins' : memory_info.pageins,
        'pfaults': memory_info.pfaults,
    }


    for key, value in response.items():
        response[key] = format_data_size(value, fmt=fmt)

    return response

def cpu_type():
    import platform
    return platform.processor()

def cpu_info():
    return {
        'cpu_count': cpu_count(),
        'cpu_type': cpu_type(),
    }

def format_data_size(x: Union[int, float], fmt:str='b', prettify:bool=False):
    assert type(x) in [int, float, str], f'x must be int or float, not {type(x)}'
    x = float(x)
    fmt2scale = {
        'b': 1,
        'kb': 1000,
        'mb': 1000**2,
        'gb': 1000**3,
        'GiB': 1024**3,
        'tb': 1000**4,
    }
        
    assert fmt in fmt2scale.keys(), f'fmt must be one of {fmt2scale.keys()}'
    scale = fmt2scale[fmt] 
    x = x/scale 
    
    return x

def hardware(fmt:str='gb'):
    return {
        'cpu': cpu_info(),
        'memory': memory_info(fmt=fmt),
        'disk': disk_info(fmt=fmt),
        'gpu': gpu_info(fmt=fmt),
    }

def sys_path():
    return sys.path

def gc():
    gc.collect()
    return {'success': True, 'msg': 'garbage collected'}


def nest_asyncio():
    import nest_asyncio
    nest_asyncio.apply()

def memory_usage(fmt='gb'):
    fmt2scale = {'b': 1e0, 'kb': 1e1, 'mb': 1e3, 'gb': 1e6}
    import psutil
    process = psutil.Process()
    scale = fmt2scale.get(fmt)
    return (process.memory_info().rss // 1024) / scale

def num_gpus():
    import torch
    return torch.cuda.device_count()

def gpu_memory():
    import torch
    return torch.cuda.memory_allocated()
def gpus():
    return list(range(num_gpus()))

def gpu_info( fmt='gb') -> Dict[int, Dict[str, float]]:
    import torch
    gpu_info = {}
    for gpu_id in gpus():
        mem_info = torch.cuda.mem_get_info(gpu_id)
        gpu_info[int(gpu_id)] = {
            'name': torch.cuda.get_device_name(gpu_id),
            'free': mem_info[0],
            'used': (mem_info[1]- mem_info[0]),
            'total': mem_info[1], 
            'ratio': mem_info[0]/mem_info[1],
        }

    gpu_info_map = {}

    skip_keys =  ['ratio', 'total', 'name']

    for gpu_id, gpu_info in gpu_info.items():
        for key, value in gpu_info.items():
            if key in skip_keys:
                continue
            gpu_info[key] = format_data_size(value, fmt=fmt)
        gpu_info_map[gpu_id] = gpu_info
    return gpu_info_map

def disk_info( path:str = '/', fmt:str='gb'):
    import shutil
    response = shutil.disk_usage(path)
    response = {
        'total': response.total,
        'used': response.used,
        'free': response.free,
    }
    for key, value in response.items():
        response[key] = format_data_size(value, fmt=fmt)
    return response

def memory_info(fmt='gb'):
    import psutil
    """
    Returns the current memory usage and total memory of the system.
    """
    # Get memory statistics
    memory_stats = psutil.virtual_memory()

    # Total memory in the system
    response = {
        'total': memory_stats.total,
        'available': memory_stats.available,
        'used': memory_stats.total - memory_stats.available,
        'free': memory_stats.available,
        'active': memory_stats.active,
        'inactive': memory_stats.inactive,
        'percent': memory_stats.percent,
        'ratio': memory_stats.percent/100,
    }

    for key, value in response.items():
        if key in ['percent', 'ratio']:
            continue
        response[key] = format_data_size(value, fmt=fmt)    

    return response

def virtual_memory_available():
    import psutil
    return psutil.virtual_memory().available

def virtual_memory_total():
    import psutil
    return psutil.virtual_memory().total

def virtual_memory_percent():
    import psutil
    return psutil.virtual_memory().percent

def cpu_usage():
    import psutil
    # get the system performance data for the cpu
    cpu_usage = psutil.cpu_percent()
    return cpu_usage

def add_rsa_key(b=2048, t='rsa'):
    return cmd(f"ssh-keygen -b {b} -t {t}")

def kill_process(process):
    import signal
    process_id = process.pid
    process.stdout.close()
    process.send_signal(signal.SIGINT)
    process.wait()
    return {'success': True, 'msg': 'process killed', 'pid': process_id}

def seed_everything(seed: int) -> None:
    import torch, random, numpy
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def set_env(key:str, value:str):
    os.environ[key] = value
    return {'success': True, 'key': key, 'value': value}

def get_env(key:str):
    return os.environ.get(key)

def cwd():
    return os.getcwd()



# import re
# import subprocess

def proc(command:str,  *extra_commands, verbose:bool = False, **kwargs):
    process = subprocess.Popen((shlex_split(command), *extra_commands), 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, 
                                universal_newlines=True, **kwargs)
    streamer = stream_output(process, verbose=verbose)
    return streamer

def process(command:str,  *extra_commands, verbose:bool = False, **kwargs):
    process = subprocess.Popen((shlex_split(command), *extra_commands), 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, 
                                universal_newlines=True, **kwargs)
    streamer = stream_output(process, verbose=verbose)
    return streamer






def determine_type( x):
    x_type = type(x)
    x_type_name = x_type.__name__.lower()
    return x_type_name

def detailed_error(e) -> dict:
    import traceback
    tb = traceback.extract_tb(e.__traceback__)
    file_name = tb[-1].filename
    line_no = tb[-1].lineno
    line_text = tb[-1].line
    response = {
        'success': False,
        'error': str(e),
        'file_name': file_name.replace(os.path.expanduser('~'), '~'),
        'line_no': line_no,
        'line_text': line_text
    }   
    return response

def error(e) -> dict:
    import traceback
    tb = traceback.extract_tb(e.__traceback__)
    file_name = tb[-1].filename
    line_no = tb[-1].lineno
    line_text = tb[-1].line
    response = {
        'success': False,
        'error': str(e),
        'file_name': file_name.replace(os.path.expanduser('~'), '~'),
        'line_no': line_no,
        'line_text': line_text
    }   
    return response

def getcwd():
    return os.getcwd()

def cuda_available() -> bool:
    import commune as c
    return c.util('hardware.cuda_available')

def free_gpu_memory():
    gpu_info = gpu_info()
    return {gpu_id: gpu_info['free'] for gpu_id, gpu_info in gpu_info.items()}

def most_used_gpu():
    most_used_gpu = max(free_gpu_memory().items(), key=lambda x: x[1])[0]
    return most_used_gpu

def most_used_gpu_memory():
    most_used_gpu = max(free_gpu_memory().items(), key=lambda x: x[1])[1]
    return most_used_gpu

def least_used_gpu():
    least_used_gpu = min(free_gpu_memory().items(), key=lambda x: x[1])[0]
    return least_used_gpu

def least_used_gpu_memory():
    least_used_gpu = min(free_gpu_memory().items(), key=lambda x: x[1])[1]
    return least_used_gpu

def argv( include_script:bool = False):
    args = sys.argv
    if include_script:
        return args
    else:
        return args[1:]

    processes = get_processes_on_ports()
    if port in processes:
        try:
            process = psutil.Process(processes['pid'])
            process.kill()
            print(f"Successfully killed process on port {port}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            print(f"Could not kill process on port {port}")
    else:
        print(f"No process found on port {port}")


def listdir(path:str='./'):
    return os.listdir(path)


def file2hash(path='./'):
    import commune as c
    file2hash = {}
    import commune as c
    for k,v in c.file2text(path).items():
        file2hash[k] = c.hash(v)
    return file2hash

def locals2kwargs(locals_dict:dict, kwargs_keys=['kwargs'], remove_arguments=['cls','self']) -> dict:
    locals_dict = locals_dict or {}
    kwargs = locals_dict or {}
    for k in remove_arguments:
        kwargs.pop(k, None)
    assert isinstance(kwargs, dict), f'kwargs must be a dict, got {type(kwargs)}'
    # These lines are needed to remove the self and cls from the locals_dict
    for k in kwargs_keys:
        kwargs.update( locals_dict.pop(k, {}) or {})
    return kwargs

def walk(path='./', depth=2):
    import commune as c
    results = []
    if depth == 0:
        return results
    path = os.path.abspath(path)
    # break when it gets past 3 depths from the path file

    for subpath in c.ls(path):
        try:
            if os.path.isdir(subpath):
                results += walk(subpath, depth=depth-1)
            else:
                results += [subpath]
        except Exception as e:
            pass
    return results

def put_text(path:str, text:str) -> dict:
    # Get the absolute path of the file
    dirpath = os.path.dirname(path)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    # Write the text to the file
    with open(path, 'w') as file:
        file.write(text)
    # get size
    return {'success': True, 'path': f'{path}', 'size': len(text)*8}

def get_text( path: str, default=None, **kwargs ) -> str:
    # Get the absolute path of the file
    try:
        with open(path, 'r') as file:
            content = file.read()
    except Exception as e:
        print(f'ERROR IN GET_TEXT --> {e}')
        content = default 
    return content

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
def get_num_files( directory):
    num_files = 0
    for root, _, files in os.walk(directory):
        num_files += len(files)
    return num_files
    
def sizeof( obj):
    import sys
    result = 0
    if isinstance(obj, dict):
        for k,v in obj.items():
            result +=  sizeof(k) + sizeof(v)
    elif isinstance(obj, list):
        for v in obj:
            result += sizeof(v)
    else:
        result += sys.getsizeof(obj)
            
    return result


def file2chars( path='./', fmt='b') -> int:
    import commune as c
    files = c.glob(path)
    file2size = {}
    file2size = dict(sorted(file2size.items(), key=lambda item: item[1]))
    return file2size

def find_largest_folder(directory: str = '~/'):
    directory = get_path(directory)
    """Find the largest folder in the given directory."""
    largest_size = 0
    largest_folder = ""

    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        if os.path.isdir(folder_path):
            folder_size = get_folder_size(folder_path)
            if folder_size > largest_size:
                largest_size = folder_size
                largest_folder = folder_path

    return largest_folder, largest_size

def file2size( path='./', fmt='b') -> int:
    import commune as c
    files = c.glob(path)
    file2size = {}
    for file in files:
        file2size[file] = format_data_size(filesize(file), fmt)
    file2size = dict(sorted(file2size.items(), key=lambda item: item[1]))
    return file2size


def get_folder_size( folder_path:str='/'):
    folder_path = os.path.abspath(folder_path)
    """Calculate the total size of all files in the folder."""
    total_size = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if not os.path.islink(file_path):
                total_size += os.path.getsize(file_path)
    return total_size

def mv( path1, path2):
    import shutil
    assert os.path.exists(path1), path1
    shutil.move(path1, path2)
    assert os.path.exists(path2), path2
    assert not os.path.exists(path1), path1
    return {'success': True, 'msg': f'Moved {path1} to {path2}'}

def rmtree( path):
    import shutil
    assert os.path.isdir(path), f'{path} is not a directory'
    return shutil.rmtree(path)
rmdir = rmtree 

def isdir( path):
    path = os.path.abspath(path=path)
    return os.path.isdir(path)
    
def isfile( path):
    path = os.path.abspath(path=path)
    return os.path.isfile(path)
    
def makedirs( *args, **kwargs):
    return os.makedirs(*args, **kwargs)



async def async_write(path, data,  mode ='w'):
    import aiofiles
    async with aiofiles.open(path, mode=mode) as f:
        await f.write(data)




def get_line(module, idx):
    import commune as c
    code = c.code(module)
    lines = code.split('\n')
    assert idx < len(lines), f'idx {idx} is out of range for {len(lines)}'  
    line =  lines[max(idx, 0)]
    return line

def find_lines(text:str, search:str) -> List[str]:
    """
    Finds the lines in text with search
    """
    found_lines = []
    lines = text.split('\n')
    for line in lines:
        if search in line:
            found_lines += [line]
    
    return found_lines
def file2lines(path:str='./')-> List[str]:
    result = file2text(path)
    return {f: text.split('\n') for f, text in result.items()}

def file2n(path:str='./')-> List[str]:
    result = file2text(path)
    return {f: len(text.split('\n')) for f, text in result.items()}

def munch( x:dict, recursive:bool=True)-> 'Munch':
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

dict2munch = munch

def put_yaml( path:str,  data: dict) -> Dict:
    import yaml
    from munch import Munch
    from copy import deepcopy
    import pandas as pd
    '''
    Loads a yaml file
    '''
    dirpath = os.path.dirname(path)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    # Directly from dictionary
    data_type = type(data)
    if data_type in [pd.DataFrame]:
        data = data.to_dict()
    if data_type in [Munch]:
        data = munch2dict(deepcopy(data))
    if data_type in [dict, list, tuple, set, float, str, int]:
        yaml_str = yaml.dump(data)
    else:
        raise NotImplementedError(f"{data_type}, is not supported")
    with open(path, 'w') as file:
        file.write(yaml_str)
    return {'success': True, 'msg': f'Wrote yaml to {path}'}


def get_yaml( path:str=None, default={}, **kwargs) -> Dict:
    '''f
    Loads a yaml file
    '''
    import yaml
    path = os.path.abspath(path)
    with open(path, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data
load_yaml = get_yaml
save_yaml = put_yaml 


def search_files(path:str='./', search:str='__pycache__') -> List[str]:
    import commune as c
    path = os.path.abspath(path)
    files = c.glob(path)
    return list(filter(lambda x: search in x, files))

def lsdir( path:str='./') -> List[str]:
    path = os.path.abspath(path)
    return os.listdir(path)

def lsd(path:str='./') -> List[str]:
    path = os.path.abspath(path)
    return os.listdir(path)

def tilde_path():
    return os.path.expanduser('~')


def hidden_files(path:str='./')-> List[str]:
    import commune as c
    path = os.path.abspath(path)
    files = [f[len(path)+1:] for f in  c.glob(path)]
    hidden_files = [f for f in files if f.startswith('.')]
    return hidden_files


def wordinfolder( word:str, path:str='./')-> bool:
    import commune as c
    path = c.os.path.abspath(path)
    files = c.glob(path)
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

def get_file_size( path:str):
    path = os.path.abspath(path)
    return os.path.getsize(path)
    
def get_files( path ='./', files_only:bool = True, recursive:bool=True, avoid_terms = ['__pycache__', '.git', '.ipynb_checkpoints', 'package.lock', 'egg-info', 'Cargo.lock', 'artifacts', 'yarn.lock', 'cache/', 'target/debug', 'node_modules']):
    import glob
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(path) and not path.endswith('**'):
        path = os.path.join(path, '**')
    paths = glob.glob(path, recursive=recursive)
    if files_only:
        paths =  list(filter(lambda f:os.path.isfile(f), paths))
    if avoid_terms:
        paths = list(filter(lambda f: not any([term in f for term in avoid_terms]), paths))
    return sorted(paths)


def type2files( path:str='./', **kwargs):
    files = get_files(path, **kwargs)
    type2files = {}
    for f in files:
        if '.' in f:
            f_type = f.split('.')[-1]
            if f_type not in type2files:
                type2files[f_type] = []
            type2files[f_type].append(f)
    return type2files

def type2filecount( path:str='./', **kwargs):
    return {k: len(v) for k,v in type2files(path, **kwargs).items()}

def abspath(path:str):
    return os.path.abspath(os.path.expanduser(path))

def file2text(path = './', 
              avoid_terms = ['__pycache__', '.git', '.ipynb_checkpoints', 'package.lock','egg-info', 'Cargo.lock', 'artifacts', 'yarn.lock','cache/','target/debug','node_modules'],
                avoid_paths = ['~', '/tmp', '/var', '/proc', '/sys', '/dev'],
                relative=False, 
                 **kwargs):
    
    path = abspath(path)
    assert all([not abspath(k) == path for k in avoid_paths]), f'path {path} is in {avoid_paths}'
    file2text = {}
    for file in get_files(path, recursive=True, avoid_terms=avoid_terms , **kwargs):
        
        if os.path.isdir(file):
            continue
        try:
            with open(file, 'r') as f:
                content = f.read()
                file2text[file] = content
        except Exception as e:
            continue
    if relative:
        home_path = os.path.abspath(os.path.expanduser('~'))

        results = {}
        for k,v in file2text.items():
            if k.startswith(home_path):
                k = '~'+path[len(home_path):] 
                results[k] = v
        return results

    return file2text

def cp( path1:str, path2:str, refresh:bool = False):
    import shutil
    # what if its a folder?
    assert os.path.exists(path1), path1
    if refresh == False:
        assert not os.path.exists(path2), path2
    
    path2_dirpath = os.path.dirname(path2)
    if not os.path.isdir(path2_dirpath):
        os.makedirs(path2_dirpath, exist_ok=True)
        assert os.path.isdir(path2_dirpath), f'Failed to create directory {path2_dirpath}'

    if os.path.isdir(path1):
        shutil.copytree(path1, path2)

    elif os.path.isfile(path1):
        
        shutil.copy(path1, path2)
    else:
        raise ValueError(f'path1 is not a file or a folder: {path1}')
    return path2


def path2text( path:str, relative=False):
    import glob
    path = os.path.abspath(path)
    assert os.path.exists(path), f'path {path} does not exist'
    if os.path.isdir(path):
        filepath_list = glob.glob(path + '/**')
    else:
        assert os.path.exists(path), f'path {path} does not exist'
        filepath_list = [path] 
    path2text = {}
    for filepath in filepath_list:
        try:
            path2text[filepath] = get_text(filepath)
        except Exception as e:
            pass
    if relative:
        pwd = pwd()
        path2text = {os.path.relpath(k, pwd):v for k,v in path2text.items()}
    return path2text

def textsize( path: str = './', **kwargs ) -> str:
    return len(str(cls.text(path)))

def num_files(path='./',  **kwargs) -> List[str]: 
    return len(cls.files(path))


def osname():
    return os.name

def get_pid():
    return os.getpid()

## SYSTEM INFO
def system_info():
    return {
        'os': osname(),
        'cpu': cpu_info(),
        'memory': memory_info(),
        'disk': disk_info(),
    }

def is_mac():
    return sys.platform == 'darwin'

def run_command(command:str):
    process = subprocess.run(shlex_split(command), 
                        stdout=subprocess.PIPE, 
                        universal_newlines=True)
    return process

def check_pid(pid):        
    """ Check For the existence of a unix pid. """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True

def ensure_path(path):
    """
    ensures a dir_path exists, otherwise, it will create it 
    """

    dir_path = os.path.dirname(path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    return path


def cpu_count():
    return os.cpu_count()

num_cpus = cpu_count

def set_cwd(path:str):
    return os.chdir(path)

def memory_usage_info(fmt='gb'):
    import psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    response = {
        'rss': memory_info.rss,
        'vms': memory_info.vms,
        'pageins' : memory_info.pageins,
        'pfaults': memory_info.pfaults,
    }


    for key, value in response.items():
        response[key] = format_data_size(value, fmt=fmt)

    return response

def cpu_type():
    import platform
    return platform.processor()

def cpu_info():
    return {
        'cpu_count': cpu_count(),
        'cpu_type': cpu_type(),
    }

def format_data_size(x: Union[int, float], fmt:str='b', prettify:bool=False):
    assert type(x) in [int, float, str], f'x must be int or float, not {type(x)}'
    x = float(x)
    fmt2scale = {
        'b': 1,
        'kb': 1000,
        'mb': 1000**2,
        'gb': 1000**3,
        'GiB': 1024**3,
        'tb': 1000**4,
    }
        
    assert fmt in fmt2scale.keys(), f'fmt must be one of {fmt2scale.keys()}'
    scale = fmt2scale[fmt] 
    x = x/scale 
    
    return x


def hardware(fmt:str='gb'):
    return {
        'cpu': cpu_info(),
        'memory': memory_info(fmt=fmt),
        'disk': disk_info(fmt=fmt),
        'gpu': gpu_info(fmt=fmt),
    }

def sys_path():
    return sys.path

def gc():
    gc.collect()
    return {'success': True, 'msg': 'garbage collected'}


def nest_asyncio():
    import nest_asyncio
    nest_asyncio.apply()

def memory_usage(fmt='gb'):
    fmt2scale = {'b': 1e0, 'kb': 1e1, 'mb': 1e3, 'gb': 1e6}
    import psutil
    process = psutil.Process()
    scale = fmt2scale.get(fmt)
    return (process.memory_info().rss // 1024) / scale

def num_gpus():
    import torch
    return torch.cuda.device_count()

def gpu_memory():
    import torch
    return torch.cuda.memory_allocated()
def gpus():
    return list(range(num_gpus()))

def gpu_info( fmt='gb') -> Dict[int, Dict[str, float]]:
    import torch
    gpu_info = {}
    for gpu_id in gpus():
        mem_info = torch.cuda.mem_get_info(gpu_id)
        gpu_info[int(gpu_id)] = {
            'name': torch.cuda.get_device_name(gpu_id),
            'free': mem_info[0],
            'used': (mem_info[1]- mem_info[0]),
            'total': mem_info[1], 
            'ratio': mem_info[0]/mem_info[1],
        }

    gpu_info_map = {}

    skip_keys =  ['ratio', 'total', 'name']

    for gpu_id, gpu_info in gpu_info.items():
        for key, value in gpu_info.items():
            if key in skip_keys:
                continue
            gpu_info[key] = format_data_size(value, fmt=fmt)
        gpu_info_map[gpu_id] = gpu_info
    return gpu_info_map

def disk_info( path:str = '/', fmt:str='gb'):
    import shutil
    response = shutil.disk_usage(path)
    response = {
        'total': response.total,
        'used': response.used,
        'free': response.free,
    }
    for key, value in response.items():
        response[key] = format_data_size(value, fmt=fmt)
    return response

def memory_info(fmt='gb'):
    import psutil
    """
    Returns the current memory usage and total memory of the system.
    """
    # Get memory statistics
    memory_stats = psutil.virtual_memory()

    # Total memory in the system
    response = {
        'total': memory_stats.total,
        'available': memory_stats.available,
        'used': memory_stats.total - memory_stats.available,
        'free': memory_stats.available,
        'active': memory_stats.active,
        'inactive': memory_stats.inactive,
        'percent': memory_stats.percent,
        'ratio': memory_stats.percent/100,
    }

    for key, value in response.items():
        if key in ['percent', 'ratio']:
            continue
        response[key] = format_data_size(value, fmt=fmt)    

    return response

def virtual_memory_available():
    import psutil
    return psutil.virtual_memory().available

def virtual_memory_total():
    import psutil
    return psutil.virtual_memory().total

def virtual_memory_percent():
    import psutil
    return psutil.virtual_memory().percent

def cpu_usage():
    import psutil
    # get the system performance data for the cpu
    cpu_usage = psutil.cpu_percent()
    return cpu_usage

def add_rsa_key(b=2048, t='rsa'):
    return cmd(f"ssh-keygen -b {b} -t {t}")

def kill_process(process):
    import signal
    process_id = process.pid
    process.stdout.close()
    process.send_signal(signal.SIGINT)
    process.wait()
    return {'success': True, 'msg': 'process killed', 'pid': process_id}

def seed_everything(seed: int) -> None:
    import torch, random, numpy
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def set_env(key:str, value:str):
    os.environ[key] = value
    return {'success': True, 'key': key, 'value': value}

def get_env(key:str):
    return os.environ.get(key)

def cwd():
    return os.getcwd()



def proc(command:str,  *extra_commands, verbose:bool = False, **kwargs):
    process = subprocess.Popen(re.split(command, *extra_commands), 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, 
                                universal_newlines=True, **kwargs)
    streamer = stream_output(process, verbose=verbose)
    return streamer



def cmd(
    command: Union[str, list],
    *args,
    verbose: bool = False,
    env: Dict[str, str] = None,
    sudo: bool = False,
    password: str = None,
    bash: bool = False,
    return_process: bool = False,
    stream: bool = False,
    color: str = 'white',
    cwd: str = None,
    **kwargs
) -> 'subprocess.Popen':
    """
    Execute a shell command with various options and handle edge cases.
    """
    import commune as c
    def stream_output(process, verbose=verbose):
        """Stream output from process pipes."""
        try:
            modes = ['stdout', 'stderr']
            for mode in modes:
                pipe = getattr(process, mode)
                if pipe is None:
                    continue
                
                # Read bytewise
                while True:
                    ch = pipe.read(1)
                    if not ch:
                        break
                    try:
                        ch_decoded = ch.decode('utf-8')
                        if verbose:
                            print(ch_decoded, end='', flush=True)
                        yield ch_decoded
                    except UnicodeDecodeError:
                        continue
        finally:
            kill_process(process)

    try:
        # Handle command construction
        if isinstance(command, list):
            command = ' '.join(command)
        
        if args:
            command = ' '.join([command] + list(map(str, args)))

        # Handle sudo
        if password is not None:
            sudo = True
        if sudo:
            command = f'sudo {command}'
        # Handle bash execution
        if bash:
            command = f'bash -c "{command}"'
        # Handle working directory
        cwd = os.getcwd() if cwd is None else abspath(cwd)
        # Handle environment variables
        if env is None:
            env = {}
        env = {**os.environ, **env}


        # Create process
        process = subprocess.Popen(
            shlex_split(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            env=env,
            **kwargs
        )

        if return_process:
            return process

        # Handle output streaming
        streamer = stream_output(process)

        if stream:
            return streamer
        else:
            # Collect all output
            text = ''
            for ch in streamer:
                text += ch
            return text

    except Exception as e:
        if verbose:
            print(f"Error executing command: {str(e)}")
        raise


    
def loadenv():
    from dotenv import load_dotenv
    load_dotenv(verbose=True)
    env = os.environ
    return env

def env_path():
    from dotenv import find_dotenv
    env_path = find_dotenv()
    return env_path

def determine_type( x):
    x_type = type(x)
    x_type_name = x_type.__name__.lower()
    return x_type_name

def detailed_error(e) -> dict:
    import traceback
    tb = traceback.extract_tb(e.__traceback__)
    file_name = tb[-1].filename
    line_no = tb[-1].lineno
    line_text = tb[-1].line
    response = {
        'success': False,
        'error': str(e),
        'file_name': file_name.replace(os.path.expanduser('~'), '~'),
        'line_no': line_no,
        'line_text': line_text
    }   
    return response

def getcwd():
    return os.getcwd()

def cuda_available() -> bool:
    import commune as c
    return c.util('hardware.cuda_available')

def free_gpu_memory():
    gpu_info = gpu_info()
    return {gpu_id: gpu_info['free'] for gpu_id, gpu_info in gpu_info.items()}

def most_used_gpu():
    most_used_gpu = max(free_gpu_memory().items(), key=lambda x: x[1])[0]
    return most_used_gpu

def most_used_gpu_memory():
    most_used_gpu = max(free_gpu_memory().items(), key=lambda x: x[1])[1]
    return most_used_gpu

def least_used_gpu():
    least_used_gpu = min(free_gpu_memory().items(), key=lambda x: x[1])[0]
    return least_used_gpu

def least_used_gpu_memory():
    least_used_gpu = min(free_gpu_memory().items(), key=lambda x: x[1])[1]
    return least_used_gpu

def argv( include_script:bool = False):
    args = sys.argv
    if include_script:
        return args
    else:
        return args[1:]

    processes = get_processes_on_ports()
    if port in processes:
        try:
            process = psutil.Process(processes['pid'])
            process.kill()
            print(f"Successfully killed process on port {port}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            print(f"Could not kill process on port {port}")
    else:
        print(f"No process found on port {port}")


def listdir(path:str='./'):
    return os.listdir(path)


thread_map = {}
def wait(futures:list, timeout:int = None, generator:bool=False, return_dict:bool = True) -> list:
    import commune as c
    is_singleton = bool(not isinstance(futures, list))

    futures = [futures] if is_singleton else futures

    if len(futures) == 0:
        return []
    if is_coroutine(futures[0]):
        return gather(futures, timeout=timeout)
    
    future2idx = {future:i for i,future in enumerate(futures)}

    if timeout == None:
        if hasattr(futures[0], 'timeout'):
            timeout = futures[0].timeout
        else:
            timeout = 30

    if generator:
        def get_results(futures):
            import concurrent 
            try: 
                for future in concurrent.futures.as_completed(futures, timeout=timeout):
                    if return_dict:
                        idx = future2idx[future]
                        yield {'idx': idx, 'result': future.result()}
                    else:
                        yield future.result()
            except Exception as e:
                yield None
            
    else:
        def get_results(futures):
            import concurrent
            results = [None]*len(futures)
            try:
                for future in concurrent.futures.as_completed(futures, timeout=timeout):
                    idx = future2idx[future]
                    results[idx] = future.result()
                    del future2idx[future]
                if is_singleton: 
                    results = results[0]
            except Exception as e:
                unfinished_futures = [future for future in futures if future in future2idx]
                print(f'Error: {e}, {len(unfinished_futures)} unfinished futures with timeout {timeout} seconds')
            return results

    return get_results(futures)


def executor(self,  max_workers=8, mode='thread'):
    if mode == 'process':
        from concurrent.futures import ProcessPoolExecutor
        executor =  ProcessPoolExecutor(max_workers=max_workers)
    elif mode == 'thread':
        from concurrent.futures import ThreadPoolExecutor
        executor =  ThreadPoolExecutor(max_workers=max_workers)
    elif mode == 'async':
        from commune.core.api.src.async_executor import AsyncExecutor
        executor = AsyncExecutor(max_workers=max_workers)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'thread', 'process' or 'async'.")

    return executor

def as_completed(futures:list, timeout:int=10, **kwargs):
    import concurrent   
    results = []
    try:
        for x in  concurrent.futures.as_completed(futures, timeout=timeout):
            results.append(x)
    except TimeoutError:
        print(f'TimeoutError: {timeout} seconds')
        pass
    return results

def is_coroutine(future):
    """
    is this a thread coroutine?
    """
    return hasattr(future, '__await__')

def thread(fn: Union['callable', str],  
                args:list = None, 
                kwargs:dict = None, 
                daemon:bool = True, 
                name = None,
                tag = None,
                start:bool = True,
                tag_seperator:str='::', 
                **extra_kwargs):
    import threading
    import commune as c
    
    if isinstance(fn, str):
        fn = c.get_fn(fn)
    if args == None:
        args = []
    if kwargs == None:
        kwargs = {}

    assert callable(fn), f'target must be callable, got {fn}'
    assert  isinstance(args, list), f'args must be a list, got {args}'
    assert  isinstance(kwargs, dict), f'kwargs must be a dict, got {kwargs}'
    
    # unique thread name
    if name == None:
        name = fn.__name__
        cnt = 0
        while name in thread_map:
            cnt += 1
            if tag == None:
                tag = ''
            name = name + tag_seperator + tag + str(cnt)
    
    if name in thread_map:
        thread_map[name].join()

    t = threading.Thread(target=fn, args=args, kwargs=kwargs, **extra_kwargs)
    # set the time it starts
    t.daemon = daemon
    if start:
        t.start()
    thread_map[name] = t
    return t


def threads(search:str = None):
    threads =  list(thread_map.keys())
    if search != None:
        threads = [t for t in threads if search in t]
    return threads

def cancel(futures): 
    for f in futures:
        f.cancel()
    return {'success': True, 'msg': 'cancelled futures'}

async def async_read(path, mode='r'):
    import aiofiles
    async with aiofiles.open(path, mode=mode) as f:
        data = await f.read()
    return data

def get_new_event_loop(nest_asyncio:bool = False):
    if nest_asyncio:
        set_nest_asyncio()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop

def sync_wrapper(fn):
    
    def wrapper_fn(*args, **kwargs):
        if 'loop'  in kwargs:
            loop = kwargs['loop']
        else:
            loop = get_event_loop()
        return loop.run_until_complete(fn(*args, **kwargs))
    return  wrapper_fn

def new_event_loop(nest_asyncio:bool = True) -> 'asyncio.AbstractEventLoop':
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    if nest_asyncio:
        set_nest_asyncio()
    return loop

def set_event_loop(self, loop=None, new_loop:bool = False) -> 'asyncio.AbstractEventLoop':
    import asyncio
    try:
        if new_loop:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        else:
            loop = loop if loop else asyncio.get_event_loop()
    except RuntimeError as e:
        self.new_event_loop()
        
    self.loop = loop
    return self.loop

def get_event_loop(nest_asyncio:bool = True) -> 'asyncio.AbstractEventLoop':
    try:
        loop = asyncio.get_event_loop()
    except Exception as e:
        loop = new_event_loop(nest_asyncio=nest_asyncio)
    return loop

def set_nest_asyncio():
    import nest_asyncio
    nest_asyncio.apply()
    

def gather(jobs:list, timeout:int = 20, loop=None)-> list:

    if loop == None:
        loop = get_event_loop()

    if not isinstance(jobs, list):
        singleton = True
        jobs = [jobs]
    else:
        singleton = False

    assert isinstance(jobs, list) and len(jobs) > 0, f'Invalid jobs: {jobs}'
    # determine if we are using asyncio or multiprocessing

    # wait until they finish, and if they dont, give them none

    # return the futures that done timeout or not
    async def wait_for(future, timeout):
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            result = {'error': f'TimeoutError: {timeout} seconds'}
        return result
    
    jobs = [wait_for(job, timeout=timeout) for job in jobs]
    future = asyncio.gather(*jobs)
    results = loop.run_until_complete(future)

    if singleton:
        return results[0]
    return results

def rsa() -> str:
    """
    Generate an RSA key pair if it does not exist, and return the public key.
    """ 
    path = os.path.expanduser('~/.ssh/id_rsa.pub')
    if not os.path.exists(path):
        cmd('ssh-keygen -t rsa -b 4096 -C ')
    return cmd(f'cat {path}')

def sumtext( text, split_size=100000) -> List[str]:
    text_size = len(text)
    if text_size < split_size:
        return [text]
    else:   
        return [text[i:i+split_size] for i in range(0, text_size, split_size)]
    future2idx = {}
    futures = []
    for idx, chunk in enumerate(chunks):
        future = c.submit(c.cond, [chunk])
        future2idx[future] = idx
    for f in c.as_completed(future2idx):
        idx = future2idx.pop(f)
        chunks[idx] = f.result()
    return chunks

def sumpath(path='./', split_size=100000) -> List[str]:
    text_size = len(text)
    if text_size < split_size:
        return [text]
    else:   
        return [text[i:i+split_size] for i in range(0, text_size, split_size)]
    future2idx = {}
    futures = []
    for idx, chunk in enumerate(chunks):
        future = c.submit(c.cond, [chunk])
        future2idx[future] = idx
    for f in c.as_completed(future2idx):
        idx = future2idx.pop(f)
        chunks[idx] = f.result()
    return chunks


def str2python(x):
    x = str(x)
    if isinstance(x, str) :
        if x.startswith('py(') and x.endswith(')'):
            try:
                return eval(x[3:-1])
            except:
                return x
    if x.lower() in ['null'] or x == 'None':  # convert 'null' or 'None' to None
        return None 
    elif x.lower() in ['true', 'false']: # convert 'true' or 'false' to bool
        return bool(x.lower() == 'true')
    elif x.startswith('[') and x.endswith(']'): # this is a list
        try:
            list_items = x[1:-1].split(',')
            # try to convert each item to its actual type
            x =  [str2python(item.strip()) for item in list_items]
            if len(x) == 1 and x[0] == '':
                x = []
            return x
        except:
            # if conversion fails, return as string
            return x
    elif x.startswith('{') and x.endswith('}'):
        # this is a dictionary
        if len(x) == 2:
            return {}
        try:
            dict_items = x[1:-1].split(',')
            # try to convert each item to a key-value pair
            return {key.strip(): str2python(value.strip()) for key, value in [item.split(':', 1) for item in dict_items]}
        except:
            # if conversion fails, return as string
            return x
    else:
        # try to convert to int or float, otherwise return as string
        
        for type_fn in [int, float]:
            try:
                return type_fn(x)
            except ValueError:
                pass
    return x

def module2hash(search = None, max_age = None, **kwargs):
    infos = self.infos(search=search, max_age=max_age, **kwargs)
    return {i['name']: i['hash'] for i in infos if 'name' in i}

def getsourcelines( module = None, search=None, *args, **kwargs) -> Union[str, Dict[str, str]]:
    import commune as c
    if module != None:
        if isinstance(module, str) and '/' in module:
            fn = module.split('/')[-1]
            module = '/'.join(module.split('/')[:-1])
            module = getattr(c.mod(module), fn)
        else:
            module = cls.resolve_module(module)
    else: 
        module = cls
    return inspect.getsourcelines(module)


def round(x, sig=6, small_value=1.0e-9):
    import math
    """
    rounds a number to a certain number of significant figures
    """
    return round(x, sig - int(math.floor(math.log10(max(abs(x), abs(small_value))))) - 1)




def get_args_kwargs(params={},  args:List = [], kwargs:dict = {}, ) -> Tuple:
    """
    resolve params as args 
    """
    params = params or {}
    args = args or []
    kwargs = kwargs or {}
    if isinstance(params, list):
        args = params
    elif isinstance(params, dict):
        if 'args' in params and 'kwargs' in params and len(params) == 2:
            args = params['args']
            kwargs = params['kwargs']
        else:
            kwargs = params
    return args, kwargs