
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
        return cpu_usage
    

    
    @classmethod
    def gpu_memory(cls):
        import torch
        return torch.cuda.memory_allocated()
    
    @classmethod
    def num_gpus(cls):
        import torch
        return torch.cuda.device_count()
    
    
    def add_rsa_key(self, b=2048, t='rsa'):
        return c.cmd(f"ssh-keygen -b {b} -t {t}")
    
    @classmethod
    def cmd(cls, 
                    command:Union[str, list],
                    *args,
                    verbose:bool = False , 
                    env:Dict[str, str] = {}, 
                    sudo:bool = False,
                    password: bool = None,
                    bash : bool = False,
                    return_process: bool = False,
                    generator: bool =  False,
                    color : str = 'white',
                    cwd : str = None,
                    **kwargs) -> 'subprocess.Popen':
        
        '''
        Runs  a command in the shell.
        
        '''
        import subprocess
        import shlex
        
        if len(args) > 0:
            command = ' '.join([command] + list(args))

        
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
                                    cwd = cwd,
                                    env={**os.environ, **env}, **kwargs)
        
        if return_process:
            return process

        def stream_output(process):
            pipe = process.stdout
            for ch in iter(lambda: pipe.read(1), b""):
                # if the the terminal is stuck and needs to enter
                process.poll() 
                try:
                    yield ch.decode()
                except Exception as e:
                    pass       
            kill_process(process)



        if generator:
            return stream_output(process)

        else:
            text = ''
            new_line = ''
            
            for ch in stream_output(process):
                
                text += ch
                # only for verbose
                if verbose:
                    new_line += ch

                    if ch == '\n':
                        c.print(new_line[:-1], color=color)
                        new_line = ''

        return text


    @staticmethod
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

        gpu_info_map = {}

        skip_keys =  ['ratio', 'total', 'name']

        for gpu_id, gpu_info in gpu_info.items():
            for key, value in gpu_info.items():
                if key in skip_keys:
                    continue
                gpu_info[key] = cls.format_data_size(value, fmt=fmt)
            gpu_info_map[gpu_id] = gpu_info
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
    

    @classmethod
    def getcwd(*args,  **kwargs):
        return os.getcwd(*args, **kwargs)
    

    @classmethod
    def argv(cls, include_script:bool = False):
        import sys
        args = sys.argv
        if include_script:
            return args
        else:
            return args[1:]




    @classmethod
    def free_gpu_memory(cls, 
                     max_gpu_ratio: float = 1.0 ,
                     reserved_gpus: bool = False,
                     buffer_memory: float = 0,
                     fmt = 'b') -> Dict[int, float]:
        import torch
        free_gpu_memory = {}
        
        buffer_memory = c.resolve_memory(buffer_memory)
        
        gpu_info = cls.gpu_info_map()
        gpus = [int(gpu) for gpu in gpu_info.keys()] 
        
        if  reserved_gpus != False:
            reserved_gpus = reserved_gpus if isinstance(reserved_gpus, dict) else cls.copy(cls.reserved_gpus())
            assert isinstance(reserved_gpus, dict), 'reserved_gpus must be a dict'
            
            for r_gpu, r_gpu_memory in reserved_gpus.items():
                gpu_info[r_gpu]['total'] -= r_gpu_memory
               
        for gpu_id, gpu_info in gpu_info.items():
            if int(gpu_id) in gpus or str(gpu_id) in gpus:
                gpu_memory = max(gpu_info['total']*max_gpu_ratio - gpu_info['used'] - buffer_memory, 0)
                if gpu_memory <= 0:
                    continue
                free_gpu_memory[gpu_id] = c.format_data_size(gpu_memory, fmt=fmt)
        
        assert sum(free_gpu_memory.values()) > 0, 'No free memory on any GPU, please reduce the buffer ratio'

                
        return cls.copy(free_gpu_memory)
    

    free_gpus = free_gpu_memory


    
    @classmethod
    def get_text(cls, 
                 path: str, 
                 tail = None,
                 start_byte:int = 0,
                 end_byte:int = 0,
                 start_line :int= None,
                 end_line:int = None ) -> str:
        # Get the absolute path of the file
        path = c.resolve_path(path)

        # Read the contents of the file
        with open(path, 'rb') as file:

            file.seek(0, 2) # this is done to get the fiel size
            file_size = file.tell()  # Get the file size
            if start_byte < 0:
                start_byte = file_size - start_byte
            if end_byte <= 0:
                end_byte = file_size - end_byte 
            if end_byte < start_byte:
                end_byte = start_byte + 100
            chunk_size = end_byte - start_byte + 1

            file.seek(start_byte)

            content_bytes = file.read(chunk_size)

            # Convert the bytes to a string
            try:
                content = content_bytes.decode()
            except UnicodeDecodeError as e:
                if hasattr(content_bytes, 'hex'):
                    content = content_bytes.hex()
                else:
                    raise e

            if tail != None:
                content = content.split('\n')
                content = '\n'.join(content[-tail:])
    
            elif start_line != None or end_line != None:
                
                content = content.split('\n')
                if end_line == None or end_line == 0 :
                    end_line = len(content) 
                if start_line == None:
                    start_line = 0
                if start_line < 0:
                    start_line = start_line + len(content)
                if end_line < 0 :
                    end_line = end_line + len(content)
                content = '\n'.join(content[start_line:end_line])
            else:
                content = content_bytes.decode()
        return content
    

    @classmethod
    def mv(cls, path1, path2):
        assert os.path.exists(path1), path1
        if not os.path.isdir(path2):
            path2_dirpath = os.path.dirname(path2)
            if not os.path.isdir(path2_dirpath):
                os.makedirs(path2_dirpath, exist_ok=True)
        shutil.move(path1, path2)
        assert os.path.exists(path2), path2
        assert not os.path.exists(path1), path1
        return {'success': True, 'msg': f'Moved {path1} to {path2}'}

    
    