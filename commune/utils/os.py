
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
import random
import os
import sys
from typing import *
import glob 
import requests
import json


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
