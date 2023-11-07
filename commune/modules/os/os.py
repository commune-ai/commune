
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
    
    def memory_info(self):
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info()

    def virtual_memory_available(self):
        import psutil
        return psutil.virtual_memory().available
    
    def virtual_memory_total(self):
        import psutil
        return psutil.virtual_memory().total
    
    def virtual_memory_percent(self):
        import psutil
        return psutil.virtual_memory().percent
    

    def cpu_type(self):
        import platform
        return platform.processor()
    

    def info(self):
        return {
            'cpu_count': self.cpu_count(),
            'cpu_type': self.cpu_type(),
            'free_memory': self.virtual_memory_available(),
            'total_memory': self.virtual_memory_total(),
            'num_gpus': self.num_gpus(),
            'gpu_memory': self.gpu_memory(),
            
        }
    
    def gpu_memory(self):
        import torch
        return torch.cuda.memory_allocated()
    
    def num_gpus(self):
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



