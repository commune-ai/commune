
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
    '''
    Turn munch object  into dictionary
    '''
    if isinstance(x, Munch):
        x = dict(x)
        for k,v in x.items():
            if isinstance(v, Munch) and recursive:
                x[k] = munch2dict(v)
    return x 



def munch( x:Dict) -> 'Munch':
    '''
    Converts a dict to a munch
    '''
    return dict2munch(x)

  
def time(  t=None) -> float:
    from time import time
    return time()

  
def timestamp(  t=None) -> float:
    return int(time())



def time2datetime( t:float):
    import commune as c
    return c.get_util('time.time2datetime')(t)

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

def dict2str(cls, data: str) -> str:
    import json
    return json.dumps(data)

def bytes2dict(data: bytes) -> str:
    import json
    data = bytes2str(data)
    return json.loads(data)

def str2bytes( data: str, mode: str = 'hex') -> bytes:
    if mode in ['utf-8']:
        return bytes(data, mode)
    elif mode in ['hex']:
        return bytes.fromhex(data)

def bytes2str( data: bytes, mode: str = 'utf-8') -> str:
    
    if hasattr(data, 'hex'):
        return data.hex()
    else:
        if isinstance(data, str):
            return data
        return bytes.decode(data, mode)

def determine_type( x):
    x_type = type(x)
    x_type_name = x_type.__name__.lower()
    return x_type_name

def str2python(input)-> dict:
    import json
    assert isinstance(input, str), 'input must be a string, got {}'.format(input)
    try:
        output_dict = json.loads(input)
    except json.JSONDecodeError as e:
        return input

    return output_dict


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


def hash_modes():
    return ['keccak', 'ss58', 'python', 'md5', 'sha256', 'sha512', 'sha3_512']



def critical( *args, **kwargs):
    console = resolve_console()
    return console.critical(*args, **kwargs)


def resolve_console( console = None, **kwargs):
    import logging
    from rich.logging import RichHandler
    from rich.console import Console
    logging.basicConfig( handlers=[RichHandler()])   
        # print the line number
    console = Console()
    console = console
    return console

def print( *text:str, 
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

    console = resolve_console(console)
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
    console = resolve_console()
    return console.status(*args, **kwargs)

def log( *args, **kwargs):
    console = resolve_console()
    return console.log(*args, **kwargs)

### LOGGER LAND ###

def resolve_logger( logger = None):
    if not hasattr('logger'):
        from loguru import logger
        logger = logger.opt(colors=True)
    if logger is not None:
        logger = logger
    return logger


def echo(x):
    return x



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
    path = os.path.abspath('~/'+str(path or c.libname))
    return c.cmd(f'code {path}')

def cuda_available() -> bool:
    return c.get_util('hardware.cuda_available')


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

def argv( include_script:bool = False):
    args = sys.argv
    if include_script:
        return args
    else:
        return args[1:]

def mv( path1, path2):
    import shutil
    assert os.path.exists(path1), path1
    if not os.path.isdir(path2):
        path2_dirpath = os.path.dirname(path2)
        if not os.path.isdir(path2_dirpath):
            os.makedirs(path2_dirpath, exist_ok=True)
    shutil.move(path1, path2)
    assert os.path.exists(path2), path2
    assert not os.path.exists(path1), path1
    return {'success': True, 'msg': f'Moved {path1} to {path2}'}

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


def cancel(futures):
    for f in futures:
        f.cancel()
    return {'success': True, 'msg': 'cancelled futures'}




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


def dict2hash( d:dict) -> str:
    for k in d.keys():
        assert jsonable(d[k]), f'{k} is not jsonable'
    return hash(d)
def locals2hash(kwargs:dict = {'a': 1}, keys=['kwargs']) -> str:
    kwargs.pop('cls', None)
    kwargs.pop('self', None)
    return dict2hash(kwargs)




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
    
    path = os.path.abspath(path)
    files = c.glob(path)
    return list(filter(lambda x: search in x, files))


def lsdir( path:str) -> List[str]:
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

def is_address( address:str) -> bool:
    if not isinstance(address, str):
        return False
    if '://' in address:
        return True
    conds = []
    conds.append(isinstance(address, str))
    conds.append(':' in address)
    conds.append(is_int(address.split(':')[-1]))
    return all(conds)

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


def shuffle( x:list)->list:
    if len(x) == 0:
        return x
    random.shuffle(x)
    return x

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



def df( x, **kwargs):
    from pandas import DataFrame
    return DataFrame(x, **kwargs)

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


def walk(path='./', depth=2):
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

@staticmethod
def is_class(module: Any) -> bool:
    return type(module).__name__ == 'type' 

def path2functions(self, path=None):
    path = path or (self.root_path + '/utils')
    paths = self.ls(path)
    path2functions = {}        
    for p in paths:

        functions = []
        if os.path.isfile(p) == False:
            continue
        text = self.get_text(p)
        if len(text) == 0:
            continue
        
        for line in text.split('\n'):
            if 'def ' in line and '(' in line:
                functions.append(line.split('def ')[1].split('(')[0])
        replative_path = p[len(path)+1:]
        path2functions[replative_path] = functions
    return path2functions

@staticmethod
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


def task( fn, timeout=1, mode='asyncio'):
    
    if mode == 'asyncio':
        import asyncio
        assert callable(fn)
        future = asyncio.wait_for(fn, timeout=timeout)
        return future
    else:
        raise NotImplemented
    

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

def pip_libs(cls):
    return list(cls.lib2version().values())

required_libs = []

def ensure_libs(libs: List[str] = None, verbose:bool=False):
    results = []
    for lib in libs:
        results.append(ensure_lib(lib, verbose=verbose))
    return results

def install(cls, libs: List[str] = None, verbose:bool=False):
    return cls.ensure_libs(libs, verbose=verbose)

def ensure_env(cls):
    cls.ensure_libs(cls.libs)

def pip_exists(cls, lib:str, verbose:str=True):
    return bool(lib in cls.pip_libs())

def version(cls, lib:str=None):
    import commune as c
    lib = lib or c.libname
    lines = [l for l in cls.cmd(f'pip3 list', verbose=False).split('\n') if l.startswith(lib)]
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

    if c.exists(lib):
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
    lib = lib or c.libname
    pip_list =  c.cmd(f'pip list', verbose=False, bash=True).split('\n')
    if lib != None:
        pip_list = [l for l in pip_list if l.startswith(lib)]
    return pip_list

def is_mnemonic(s: str) -> bool:
    import re
    # Match 12 or 24 words separated by spaces
    return bool(re.match(r'^(\w+ ){11}\w+$', s)) or bool(re.match(r'^(\w+ ){23}\w+$', s))

def file2functions(self, path):
    path = os.path.abspath(path)
    
    return functions

def is_private_key(s: str) -> bool:
    import re
    # Match a 64-character hexadecimal string
    pattern = r'^[0-9a-fA-F]{64}$'
    return bool(re.match(pattern, s))


def get_folder_contents_advanced(url='commune-ai/commune.git', 
                                 host_url = 'https://github.com/',
                                 auth_token=None):
    try:
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'Python Script'
        }
        if not url.startswith(host_url):
            url = host_url + url
        
        if auth_token:
            headers['Authorization'] = f'token {auth_token}'
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse JSON response
        content = response.json()
        
        # If it's a GitHub API response, it will be a list of files/folders
        if isinstance(content, list):
            return json.dumps(content, indent=2)
        return response.text
        
    except Exception as e:
        print(f"Error: {e}")
        return None
    
