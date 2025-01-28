
import os
import commune as c
import commune as c
from typing import *
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import uvicorn
import os
import json
import asyncio
from .network import Network

class Manager:
    description = 'Process manager manages processes using pm2'
    pm2_dir = os.path.expanduser('~/.pm2')

    def __init__(self, network='local', **kwargs):
        self.net = Network(network=network)
        self.ensure_env()

        attrs = ['add_server', 'rm_server', 'namespace', 'modules']
        self.add_server = self.net.add_server
        self.rm_server = self.net.rm_server
        self.namespace = self.net.namespace
    
    def kill(self, name:str, verbose:bool = True, **kwargs):
        try:
            if name == 'all':
                return self.kill_all(verbose=verbose)
            c.cmd(f"pm2 delete {name}", verbose=False)
            self.rm_logs(name)
            result =  {'message':f'Killed {name}', 'success':True}
        except Exception as e:
            result =  {'message':f'Error killing {name}', 'success':False, 'error':e}
        c.rm_server(name)
        return result


    def namespace(self, search=None):
        from .network import Network
        network = Network()
        return network.namespace(search=search)

    def namespace(self, search=None):
        return self.net.namespace(search=search)
    
    
    def kill_all(self, verbose:bool = True, timeout=20):
        servers = self.processes()
        futures = [c.submit(self.kill, kwargs={'name':s, 'update': False}, return_future=True) for s in servers]
        results = c.wait(futures, timeout=timeout)
        return results
    
    def killall(self, **kwargs):
        return self.kill_all(**kwargs)
    
    def logs_path_map(self, name=None):
        logs_path_map = {}
        for l in c.ls(f'{self.pm2_dir}/logs/'):
            key = '-'.join(l.split('/')[-1].split('-')[:-1]).replace('-',':')
            logs_path_map[key] = logs_path_map.get(key, []) + [l]
        for k in logs_path_map.keys():
            logs_path_map[k] = {l.split('-')[-1].split('.')[0]: l for l in list(logs_path_map[k])}
        if name != None:
            return logs_path_map.get(name, {})
        return logs_path_map

    
    def rm_logs( self, name):
        logs_map = self.logs_path_map(name)
        for k in logs_map.keys():
            c.rm(logs_map[k])

    def logs(self, module:str,  tail: int =100,   mode: str ='cmd', **kwargs):
        if mode == 'local':
            text = ''
            for m in ['out','error']:
                # I know, this is fucked 
                path = f'{self.pm2_dir}/logs/{module.replace("/", "-")}-{m}.log'.replace(':', '-').replace('_', '-')
                try:
                    text +=  c.get_text(path, tail=tail)
                except Exception as e:
                    c.print('ERROR GETTING LOGS -->' , e)
                    continue
            return text
        elif mode == 'cmd':
            return c.cmd(f"pm2 logs {module}")
        else:
            raise NotImplementedError(f'mode {mode} not implemented')
        
    
    def kill_many(self, search=None, verbose:bool = True, timeout=10):
        futures = []
        for name in c.servers(search=search):
            f = c.submit(c.kill, dict(name=name, verbose=verbose), timeout=timeout)
            futures.append(f)
        return c.wait(futures)
    
    
    def start(self, 
                  fn: str = 'serve',
                   module:str = None,  
                   name:Optional[str]=None, 
                   args : list = None,
                   kwargs: dict = None,
                   interpreter:str='python3', 
                   autorestart: bool = True,
                   verbose: bool = False , 
                   force:bool = True,
                   run_fn: str = 'run_fn',
                   cwd : str = None,
                   env : Dict[str, str] = None,
                   refresh:bool=True , 
                   **extra_kwargs):
        env = env or {}
        if '/' in fn:
            module, fn = fn.split('/')
        module = module or self.module_name()
        name = name or module
        if refresh:
            self.kill(name)
        cmd = f"pm2 start {c.filepath()} --name {name} --interpreter {interpreter}"
        cmd = cmd  if autorestart else ' --no-autorestart' 
        cmd = cmd + ' -f ' if force else cmd
        kwargs =  {
                    'module': module ,  
                    'fn': fn, 
                    'args': args or [],  
                    'kwargs': kwargs or {} 
                    }

        kwargs_str = json.dumps(kwargs).replace('"', "'")
        cmd = cmd +  f' -- --fn {run_fn} --kwargs "{kwargs_str}"'
        stdout = c.cmd(cmd, env=env, verbose=verbose, cwd=cwd)
        return {'success':True, 'msg':f'Launched {module}',  'cmd': cmd, 'stdout':stdout}

    def restart(self, name:str):
        assert name in self.processes()
        c.print(f'Restarting {name}', color='cyan')
        c.cmd(f"pm2 restart {name}", verbose=False)
        self.rm_logs(name)  
        return {'success':True, 'message':f'Restarted {name}'}
    
    def processes(self, search=None,  **kwargs) -> List[str]:
        output_string = c.cmd('pm2 status', verbose=False)
        module_list = []
        for line in output_string.split('\n')[3:]:
            if  line.count('│') > 2:
                name = line.split('│')[2].strip()
                module_list += [name]
        if search != None:
            module_list = [m for m in module_list if search in m]
        module_list = sorted(list(set(module_list)))
        return module_list
    
    def procs(self, **kwargs):
        return self.processes(**kwargs)
        
    def exists(self, name:str, **kwargs) -> bool:
        return name in self.processes(**kwargs)

    def ensure_env(self,**kwargs):
        '''ensure that the environment variables are set for the process'''
        is_pm2_installed = bool( '/bin/pm2' in c.cmd('which pm2', verbose=False))
        if not is_pm2_installed:
            c.cmd('npm install -g pm2')
            c.cmd('pm2 update')
        return {'success':True, 'message':f'Ensured env '}
