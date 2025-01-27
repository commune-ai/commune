
import os
import urllib
import requests
from loguru import logger
from typing import *
import netaddr
import os
import shutil
import subprocess
import shlex
import sys
from typing import *

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

def kill_process(pid):
    import signal
    if isinstance(pid, str):
        pid = int(pid)
    os.kill(pid, signal.SIGKILL)

kill_pid = kill_process

def run_command(command:str):
    process = subprocess.run(shlex.split(command), 
                        stdout=subprocess.PIPE, 
                        universal_newlines=True)
    return process

def path_exists(path:str):
    return os.path.exists(path)

def check_pid(pid):        
    """ Check For the existence of a unix pid. """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True

def kill_process(pid):
    import signal
    if isinstance(pid, str):
        pid = int(pid)
    os.kill(pid, signal.SIGKILL)

def path_exists(path:str):
    return os.path.exists(path)

def ensure_path(path):
    """
    ensures a dir_path exists, otherwise, it will create it 
    """

    dir_path = os.path.dirname(path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    return path

def seed_everything(seed: int) -> None:
    import torch, random
    import numpy as np
    "seeding function for reproducibility"
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def cpu_count():
    return os.cpu_count()

num_cpus = cpu_count


def get_env(key:str):
    return os.environ.get(key)



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


def cpu_type():
    import platform
    return platform.processor()


def cpu_info():
    return {
        'cpu_count': cpu_count(),
        'cpu_type': cpu_type(),
    }

def cpu_usage(self):
    import psutil
    # get the system performance data for the cpu
    cpu_usage = psutil.cpu_percent()
    return cpu_usage


def gpu_memory():
    import torch
    return torch.cuda.memory_allocated()

def num_gpus():
    import torch
    return torch.cuda.device_count()

def gpus():
    return list(range(num_gpus()))

def add_rsa_key(b=2048, t='rsa'):
    return cmd(f"ssh-keygen -b {b} -t {t}")
    
def kill_process(process):
    import signal
    pid = process.pid
    process.stdout.close()
    process.send_signal(signal.SIGINT)
    process.wait()
    return {'success': True, 'msg': 'process killed', 'pid': pid}
    # sys.exit(0)


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

def disk_info(path:str = '/', fmt:str='gb'):
    path = resolve_path(path)
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


def cuda_available() -> bool:
    import torch
    return torch.cuda.is_available()

def hardware(fmt:str='gb'):
    return {
        'cpu': cpu_info(),
        'memory': memory_info(fmt=fmt),
        'disk': disk_info(fmt=fmt),
        'gpu': gpu_info(fmt=fmt),
    }

def getcwd(*args,  **kwargs):
    return os.getcwd(*args, **kwargs)

def argv(include_script:bool = False):
    import sys
    args = sys.argv
    if include_script:
        return args
    else:
        return args[1:]


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


from typing import *
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

def most_used_gpu():
    most_used_gpu = max(free_gpu_memory().items(), key=lambda x: x[1])[0]
    return most_used_gpu

def most_used_gpu_memory():
    most_used_gpu = max(free_gpu_memory().items(), key=lambda x: x[1])[1]
    return most_used_gpu
    

def least_used_gpu():
    least_used_gpu = min(free_gpu_memory().items(), key=lambda x: x[1])[0]
    return least_used_gpu

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

def stream_output(process, verbose=True):
    try:
        modes = ['stdout', 'stderr']
        for mode in modes:
            pipe = getattr(process, mode)
            if pipe == None:
                continue
            # print byte wise
            for ch in iter(lambda: pipe.read(1), b''):
                ch = ch.decode('utf-8')
                if verbose:
                    print(ch, end='')
                yield ch
    except Exception as e:
        print(e)
    finally:
        kill_process(process)

def proc(command:str,  *extra_commands, verbose:bool = False, **kwargs):
    process = subprocess.Popen(shlex.split(command, *extra_commands), 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, 
                                universal_newlines=True, **kwargs)
    streamer = stream_output(process, verbose=verbose)
    return streamer

def cmd(command:Union[str, list],
                *args,
                verbose:bool = False , 
                env:Dict[str, str] = {}, 
                sudo:bool = False,
                password: bool = None,
                bash : bool = False,
                return_process: bool = False,
                stream: bool =  False,
                color : str = 'white',
                cwd : str = None,
                **kwargs) -> 'subprocess.Popen':


   
    import commune as c
    if len(args) > 0:
        command = ' '.join([command] + list(args))

    sudo = bool(password != None)

    if sudo:
        command = f'sudo {command}'

    if bash:
        command = f'bash -c "{command}"'
    cwd = c.resolve_path(c.pwd() if cwd == None else cwd)
    env = {**os.environ, **env}
    process = subprocess.Popen(shlex.split(command),
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.STDOUT,
                                cwd = cwd,
                                env=env, **kwargs)
    if return_process:
        return process
    streamer = stream_output(process)
    if stream:
        return streamer
    else:
        text = ''
        for ch in streamer:
            text += ch
    return text



import random
import os
import sys
from typing import *
import glob 
import requests
import json

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

def get_files( path ='./', 
              search=None,
              avoid_terms = None,
              include_terms = None,
               recursive:bool = True, files_only:bool = True):
    import glob
    path = os.path.abspath(path)
    if os.path.isdir(path):
        path = os.path.join(path, '**')
    paths = glob.glob(path, recursive=recursive)
    if files_only:
        paths =  list(filter(lambda f:os.path.isfile(f), paths))
    if avoid_terms != None:
        paths = [p for p in paths if not any([term in p for term in avoid_terms])]
    if include_terms != None:
        paths = [p for p in paths if any([term in p for term in include_terms])]
    if search != None:
        paths = [p for p in paths if search in p]
    return paths

def abspath(path:str):
    return os.path.abspath(os.path.expanduser(path))

def file2text(path = './', avoid_terms = ['__pycache__', 
                                '.git', 
                                '.ipynb_checkpoints', 
                                'package.lock',
                                'egg-info',
                                'Cargo.lock',
                                'artifacts',
                                'yarn.lock',
                                'cache/',
                                'target/debug',
                                'node_modules'],
                avoid_paths = ['~', '/tmp', '/var', '/proc', '/sys', '/dev'],
                relative=True,  **kwargs):
    
    path = os.path.abspath(os.path.expanduser(path))
    assert all([not os.path.abspath(k) in path for k in avoid_paths]), f'path {path} is in avoid_paths'
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
        return {k[len(path)+1:]:v for k,v in file2text.items()}
    return file2text

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


def hash_modes():
    return ['keccak', 'ss58', 'python', 'md5', 'sha256', 'sha512', 'sha3_512']

def get_cwd():
    return os.getcwd()

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

def go(path=None):
    import commune as c
    path = os.path.abspath('~/'+str(path or c.reponame))
    return c.cmd(f'code {path}')

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
