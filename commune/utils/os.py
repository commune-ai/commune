
import os
import shutil
import subprocess
import shlex
import sys
from typing import *

def jsonable( value):
    import json
    try:
        json.dumps(value)
        return True
    except:
        return False

def osname():
    return os.name

def check_pid( pid):        
    """ Check For the existence of a unix pid. """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True

def gc():
    import gc
    gc.collect()
    return {'success': True, 'msg': 'garbage collected'}


def get_pid():
    return os.getpid()


def get_file_size( path:str):
    path = os.path.abspath(path)
    return os.path.getsize(path)


def resolve_path(path):
    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    return path

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

kill_pid = kill_process

def run_command(command:str):
    process = subprocess.run(shlex.split(command), 
                        stdout=subprocess.PIPE, 
                        universal_newlines=True)
    
    return process

def path_exists(path:str):
    return os.path.exists(path)

def seed_everything(seed: int) -> None:
    import numpy as np
    import torch
    import random
    "seeding function for reproducibility"
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True




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


def set_env(key:str, value:str):
    os.environ[key] = value
    return {'success': True, 'key': key, 'value': value}


def get_cwd():
    return os.getcwd()


def set_cwd(path:str):
    return os.chdir(path)

def get_pid():
    return os.getpid()

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



def stream_output(process, verbose=False):
    try:
        modes = ['stdout', 'stderr']
        for mode in modes:
            pipe = getattr(process, mode)
            if pipe == None:
                continue
            for line in iter(pipe.readline, b''):
                line = line.decode('utf-8')
                if verbose:
                    print(line[:-1])
                yield line
    except Exception as e:
        print(e)
        pass

    kill_process(process)


def kill_process(process):
    import signal
    process_id = process.pid
    process.stdout.close()
    process.send_signal(signal.SIGINT)
    process.wait()
    return {'success': True, 'msg': 'process killed', 'pid': process_id}
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

    

def mv(path1, path2):
    assert os.path.exists(path1), path1
    if not os.path.isdir(path2):
        path2_dirpath = os.path.dirname(path2)
        if not os.path.isdir(path2_dirpath):
            os.makedirs(path2_dirpath, exist_ok=True)
    shutil.move(path1, path2)
    assert os.path.exists(path2), path2
    assert not os.path.exists(path1), path1
    return path2

def cp(path1:str, path2:str, refresh:bool = False):
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

def cuda_available() -> bool:
    import torch
    return torch.cuda.is_available()

def free_gpu_memory():
    gpu_info = gpu_info()
    return {gpu_id: gpu_info['free'] for gpu_id, gpu_info in gpu_info.items()}

def most_used_gpu(self):
    most_used_gpu = max(self.free_gpu_memory().items(), key=lambda x: x[1])[0]
    return most_used_gpu

def most_used_gpu_memory(self):
    most_used_gpu = max(self.free_gpu_memory().items(), key=lambda x: x[1])[1]
    return most_used_gpu

def least_used_gpu(self):
    least_used_gpu = min(self.free_gpu_memory().items(), key=lambda x: x[1])[0]
    return least_used_gpu

def least_used_gpu_memory(self):
    least_used_gpu = min(self.free_gpu_memory().items(), key=lambda x: x[1])[1]
    return least_used_gpu


def hardware(fmt:str='gb'):
    return {
        'cpu': cpu_info(),
        'memory': memory_info(fmt=fmt),
        'disk': disk_info(fmt=fmt),
        'gpu': gpu_info(fmt=fmt),
    }

def get_folder_size(folder_path:str='/'):
    folder_path = resolve_path(folder_path)
    """Calculate the total size of all files in the folder."""
    total_size = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if not os.path.islink(file_path):
                total_size += os.path.getsize(file_path)
    return total_size

def filesize( filepath:str):
    filepath = os.path.abspath(filepath)
    return os.path.getsize(filepath)

def file2size( path='./', fmt='b') -> int:
    import commune as c
    files = c.glob(path)
    file2size = {}
    for file in files:
        file2size[file] = format_data_size(filesize(file), fmt)
    file2size = dict(sorted(file2size.items(), key=lambda item: item[1]))
    return file2size
def file2chars( path='./', fmt='b') -> int:
    import commune as c
    files = c.glob(path)
    file2size = {}
    file2size = dict(sorted(file2size.items(), key=lambda item: item[1]))
    return file2size

def find_largest_folder(directory: str = '~/'):
    directory = resolve_path(directory)
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

def getcwd(*args,  **kwargs):
    return os.getcwd(*args, **kwargs)

def argv(include_script:bool = False):
    import sys
    args = sys.argv
    if include_script:
        return args
    else:
        return args[1:]

def mv(path1, path2):
    assert os.path.exists(path1), path1
    if not os.path.isdir(path2):
        path2_dirpath = os.path.dirname(path2)
        if not os.path.isdir(path2_dirpath):
            os.makedirs(path2_dirpath, exist_ok=True)
    shutil.move(path1, path2)
    assert os.path.exists(path2), path2
    assert not os.path.exists(path1), path1
    return {'success': True, 'msg': f'Moved {path1} to {path2}'}

def sys_path():
    return sys.path

def gc():
    gc.collect()
    return {'success': True, 'msg': 'garbage collected'}

def get_pid():
    return os.getpid()


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
    

def cuda_available() -> bool:
    import torch
    return torch.cuda.is_available()

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


@staticmethod
def get_pid():
    return os.getpid()

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

def cpu_info():
    
    return {
        'cpu_count': cpu_count(),
        'cpu_type': cpu_type(),
    }

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

def cmd( command:Union[str, list],
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

    cwd = c.pwd() if cwd == None else cwd

    env = {**os.environ, **env}

    process = subprocess.Popen(shlex.split(command),
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.STDOUT,
                                cwd = cwd,
                                env=env, **kwargs)
    if return_process:
        return process
    streamer = stream_output(process, verbose=verbose)
    if stream:
        return streamer
    else:
        text = ''
        for ch in streamer:
            text += ch
    return text








# hey/thanks bittensor
import os
import urllib
import requests
from loguru import logger
from typing import *
import netaddr

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
    import netaddr
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
    port_range = c.get('port_range', c.default_port_range)
    if isinstance(port_range, str):
        port_range = list(map(int, port_range.split('-')))
    if len(port_range) == 0:
        port_range = c.default_port_range
    port_range = list(port_range)
    assert isinstance(port_range, list), 'Port range must be a list'
    assert isinstance(port_range[0], int), 'Port range must be a list of integers'
    assert isinstance(port_range[1], int), 'Port range must be a list of integers'
    return port_range

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
    import netaddr
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
    return int(netaddr.IPAddress(str_val).version)

def ip__str__(ip_type:int, ip_str:str, port:int):
    """ Return a formatted ip string
    """
    return "/ipv%i/%s:%i" % (ip_type, ip_str, port)

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


def unreserve_port(port:int, 
                    var_path='reserved_ports'):
    import commune as c
    reserved_ports =  c.get(var_path, {}, root=True)
    
    port_info = reserved_ports.pop(port,None)
    if port_info == None:
        port_info = reserved_ports.pop(str(port),None)
    
    output = {}
    if port_info != None:
        c.put(var_path, reserved_ports, root=True)
        output['msg'] = 'port removed'
    else:
        output['msg'] =  f'port {port} doesnt exist, so your good'

    output['reserved'] =  c.reserved_ports()
    return output

def unreserve_ports(*ports, var_path='reserved_ports' ):
    import commune as c
    reserved_ports =  c.get(var_path, {})
    if len(ports) == 0:
        # if zero then do all fam, tehe
        ports = list(reserved_ports.keys())
    elif len(ports) == 1 and isinstance(ports[0],list):
        ports = ports[0]
    ports = list(map(str, ports))
    reserved_ports = {rp:v for rp,v in reserved_ports.items() if not any([p in ports for p in [str(rp), int(rp)]] )}
    c.put(var_path, reserved_ports)
    return c.reserved_ports()

def kill_port(port:int):
    r""" Kills a process running on the passed port.
        Args:
            port  (:obj:`int` `required`):
                The port to kill the process on.
    """
    try:
        os.system(f'kill -9 $(lsof -t -i:{port})')
    except Exception as e:
        print(e)
        return False
    return True

def kill_ports(ports = None, *more_ports):
    ports = ports or used_ports()
    if isinstance(ports, int):
        ports = [ports]
    if '-' in ports:
        ports = list(range([int(p) for p in ports.split('-')]))
    ports = list(ports) + list(more_ports)
    for port in ports:
        kill_port(port)
    return check_used_ports()

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


def resolve_port(port:int=None, **kwargs):
    '''
    Resolves the port and finds one that is available
    '''
    if port == None or port == 0:
        port = free_port(port, **kwargs)
        
    if port_used(port):
        port = free_port(port, **kwargs)
        
    return int(port)

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

def ip(max_age=None, update:bool = False, **kwargs) -> str:
    try:
        import commune as c
        path = 'ip'
        ip = c.get(path, None, max_age=max_age, update=update)
        if ip == None:
            ip = external_ip()
            c.put(path, ip)
    except Exception as e:
        print('Error while getting IP')
        return '0.0.0.0'
    return ip

def resolve_ip(ip=None, external:bool=True) -> str:
    if ip == None:
        if external:
            ip = external_ip()
        else:
            ip = '0.0.0.0'
    assert isinstance(ip, str)
    return ip

def port_free( *args, **kwargs) -> bool:
    return not port_used(*args, **kwargs)

def port_available(port:int, ip:str ='0.0.0.0'):
    return not port_used(port=port, ip=ip)

def used_ports(ports:List[int] = None, ip:str = '0.0.0.0', port_range:Tuple[int, int] = None):
    '''
    Get availabel ports out of port range
    
    Args:
        ports: list of ports
        ip: ip address
    
    '''
    import commune as c
    port_range = resolve_port_range(port_range=port_range)
    if ports == None:
        ports = list(range(*port_range))
    
    async def check_port(port, ip):
        return port_used(port=port, ip=ip)
    
    used_ports = []
    jobs = []
    for port in ports: 
        jobs += [check_port(port=port, ip=ip)]
            
    results = c.wait(jobs)
    for port, result in zip(ports, results):
        if isinstance(result, bool) and result:
            used_ports += [port]
        
    return used_ports

def resolve_port(port:int=None, **kwargs):
    '''
    Resolves the port and finds one that is available
    '''
    if port == None or port == 0:
        port = free_port(port, **kwargs)
    if port_used(port):
        port = free_port(port, **kwargs)
    return int(port)

get_used_ports = used_ports

def has_free_ports(n:int = 1, **kwargs):
    return len(free_ports(n=n, **kwargs)) > 0

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
            
    results = c.gather(jobs)
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

def port_range():
    return get_port_range()

def ports() -> List[int]:
    
    return list(range(*get_port_range()))

def resolve_port_range(port_range: list = None) -> list:
    return get_port_range(port_range)

def set_port_range(*port_range: list):
    import commune as c
    if '-' in port_range[0]:
        port_range = list(map(int, port_range[0].split('-')))
    if len(port_range) ==0 :
        port_range = c.default_port_range
    elif len(port_range) == 1:
        if port_range[0] == None:
            port_range = c.default_port_range
    assert len(port_range) == 2, 'Port range must be a list of two integers'        
    for port in port_range:
        assert isinstance(port, int), f'Port {port} range must be a list of integers'
    assert port_range[0] < port_range[1], 'Port range must be a list of integers'
    c.put('port_range', port_range)
    return port_range
