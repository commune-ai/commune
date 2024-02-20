
import os
import commune as c
from typing import Dict, List, Optional, Union



class OsModule(c.Module):
    @staticmethod
    def check_pid(pid):        
        """ Check For the existence of a unix pid. """
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        else:
            return True
    @staticmethod
    def kill_process(pid):
        import signal
        if isinstance(pid, str):
            pid = int(pid)
        
        os.kill(pid, signal.SIGKILL)

    @staticmethod
    def run_command(command:str):
        import subprocess
        import shlex
        process = subprocess.run(shlex.split(command), 
                            stdout=subprocess.PIPE, 
                            universal_newlines=True)
        
        return process



    @staticmethod
    def path_exists(path:str):
        return os.path.exists(path)

    @staticmethod
    def ensure_path( path):
        """
        ensures a dir_path exists, otherwise, it will create it 
        """

        dir_path = os.path.dirname(path)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        return path


    @staticmethod
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

    @staticmethod
    def cpu_count():
        return os.cpu_count()

    num_cpus = cpu_count
    
    @staticmethod
    def get_env(key:str):
        return os.environ.get(key)
    
    @staticmethod
    def set_env(key:str, value:str):
        os.environ[key] = value
        return {'success': True, 'key': key, 'value': value}

    @staticmethod
    def get_cwd():
        return os.getcwd()
    
    @staticmethod
    def set_cwd(path:str):
        return os.chdir(path)
    

    @staticmethod
    def get_pid():
        return os.getpid()
    
    @classmethod
    def memory_usage_info(cls, fmt='gb'):
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
            response[key] = cls.format_data_size(value, fmt=fmt)

        return response



    @classmethod
    def memory_info(cls, fmt='gb'):
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
            response[key] = cls.format_data_size(value, fmt=fmt)    
  
        return response

    @classmethod
    def virtual_memory_available(cls):
        import psutil
        return psutil.virtual_memory().available
    
    @classmethod
    def virtual_memory_total(cls):
        import psutil
        return psutil.virtual_memory().total
    
    @classmethod
    def virtual_memory_percent(cls):
        import psutil
        return psutil.virtual_memory().percent
    
    @classmethod
    def cpu_type(cls):
        import platform
        return platform.processor()
    
    @classmethod
    def cpu_info(cls):
        
        return {
            'cpu_count': cls.cpu_count(),
            'cpu_type': cls.cpu_type(),
        }
    

    def cpu_usage(self):
        import psutil
        # get the system performance data for the cpu
        cpu_usage = psutil.cpu_percent()
        c.print(c.fn_schema(psutil.cpu_percent))
        return cpu_usage
    

    
    @classmethod
    def gpu_memory(cls):
        import torch
        return torch.cuda.memory_allocated()
    
    @classmethod
    def num_gpus(cls):
        import torch
        return torch.cuda.device_count()
    
    
    @classmethod
    def cmd(cls, 
                    command:Union[str, list],
                    verbose:bool = True, 
                    env:Dict[str, str] = {}, 
                    sudo:bool = False,
                    password: bool = None,
                    color: str = 'white',
                    bash : bool = False,
                    **kwargs) -> 'subprocess.Popen':
        '''
        Runs  a command in the shell.
        
        '''
        if isinstance(command, list):
            kwargs = c.locals2kwargs(locals())
            for idx,cmd in enumerate(command):
                assert isinstance(cmd, str), f'command must be a string, not {type(cmd)}'
                kwargs['command'] = cmd
                response = cls.cmd(**kwargs)
            return response

        import subprocess
        import shlex
        
        def kill_process(process):
            import signal
            process.stdout.close()
            process.send_signal(signal.SIGINT)
            process.wait()
            # sys.exit(0)
            
        if password != None:
            sudo = True
            
        if sudo:
            command = f'sudo {command}'
            
            
        if bash:
            command = f'bash -c "{command}"'
        process = subprocess.Popen(shlex.split(command),
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.STDOUT,
                                    env={**os.environ, **env}, **kwargs)
        
        new_line = b''
        stdout_text = ''
        line_count_idx = 0
        try:
            for ch in iter(lambda: process.stdout.read(1), b""):
                if  ch == b'\n':
                    stdout_text += (new_line + ch).decode()
                    line_count_idx += 1
                    if verbose:
                        c.print(new_line.decode(), color='cyan')
                    new_line = b''
                    continue
                new_line += ch
        except Exception as e:
            c.print(e)
            kill_process(process)
        finally:
             kill_process(process)
        
       

        return stdout_text


    @staticmethod
    def format_data_size(x: Union[int, float], fmt:str='b', prettify:bool=False):
        assert type(x) in [int, float], f'x must be int or float, not {type(x)}'
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
        

    @classmethod
    def disk_info(cls, path:str = '/', fmt:str='gb'):
        path = c.resolve_path(path)
        import shutil
        response = shutil.disk_usage(path)
        response = {
            'total': response.total,
            'used': response.used,
            'free': response.free,
        }
        for key, value in response.items():
            response[key] = cls.format_data_size(value, fmt=fmt)
        return response


        
    @classmethod
    def mv(cls, path1, path2):
        import shutil
        path1 = cls.resolve_path(path1)
        path2 = cls.resolve_path(path2)
        assert os.path.exists(path1), path1
        if not os.path.isdir(path2):
            path2_dirpath = os.path.dirname(path2)
            if not os.path.isdir(path2_dirpath):
                os.makedirs(path2_dirpath, exist_ok=True)
        shutil.move(path1, path2)
        assert os.path.exists(path2), path2
        assert not os.path.exists(path1), path1
        return path2

 
    @classmethod
    def cp(cls, path1:str, path2:str, refresh:bool = False):
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
    
    
    @classmethod
    def cuda_available(cls) -> bool:
        import torch
        return torch.cuda.is_available()
    @classmethod
    def gpu_info(cls, fmt='gb') -> Dict[int, Dict[str, float]]:
        import torch
        gpu_info = {}
        for gpu_id in cls.gpus():
            mem_info = torch.cuda.mem_get_info(gpu_id)
            gpu_info[int(gpu_id)] = {
                'name': torch.cuda.get_device_name(gpu_id),
                'free': mem_info[0],
                'used': (mem_info[1]- mem_info[0]),
                'total': mem_info[1], 
                'ratio': mem_info[0]/mem_info[1],
            }

        for gpu_id, gpu_info in gpu_info.items():
            for key, value in gpu_info.items():
                if key in ['ratio', 'total']:
                    continue
                gpu_info[key] = cls.format_data_size(value, fmt=fmt)
        return gpu_info

    gpu_map =gpu_info

    @classmethod
    def hardware(cls, fmt:str='gb'):
        return {
            'cpu': cls.cpu_info(),
            'memory': cls.memory_info(fmt=fmt),
            'disk': cls.disk_info(fmt=fmt),
            'gpu': cls.gpu_info(fmt=fmt),
        }

    @classmethod
    def gpu_total_map(cls) -> Dict[int, Dict[str, float]]:
        import torch
        return {k:v['total'] for k,v in c.gpu_info().items()}

    

    @classmethod
    def resolve_device(cls, device:str = None, verbose:bool=True, find_least_used:bool = True) -> str:
        
        '''
        Resolves the device that is used the least to avoid memory overflow.
        '''
        import torch
        if device == None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            assert torch.cuda.is_available(), 'Cuda is not available'
            gpu_id = 0
            if find_least_used:
                gpu_id = cls.most_free_gpu()
                
            device = f'cuda:{gpu_id}'
        
            if verbose:
                device_info = cls.gpu_info(gpu_id)
                c.print(f'Using device: {device} with {device_info["free"]} GB free memory', color='yellow')
        return device  
        
    @classmethod
    def get_folder_size(cls, folder_path:str='/'):
        folder_path = c.resolve_path(folder_path)
        """Calculate the total size of all files in the folder."""
        total_size = 0
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                if not os.path.islink(file_path):
                    total_size += os.path.getsize(file_path)
        return total_size

    @classmethod
    def find_largest_folder(cls, directory: str = '~/'):
        directory = c.resolve_path(directory)
        """Find the largest folder in the given directory."""
        largest_size = 0
        largest_folder = ""

        for folder_name in os.listdir(directory):
            folder_path = os.path.join(directory, folder_name)
            if os.path.isdir(folder_path):
                folder_size = cls.get_folder_size(folder_path)
                if folder_size > largest_size:
                    largest_size = folder_size
                    largest_folder = folder_path

        return largest_folder, largest_size