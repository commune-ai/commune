
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

class Network:
    description = 'Process manager manages processes using pm2'
    pm2_dir = os.path.expanduser('~/.pm2')

    def __init__(self, network='local', process_prefix='server', **kwargs):

        self.set_network(network)
        self.process_prefix = process_prefix + '/' + network + '/'
        print(f'Process prefix: {self.process_prefix}')
        self.ensure_env()
    
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
    
    def kill_all(self, verbose:bool = True, timeout=20):
        servers = self.processes()
        futures = [c.submit(self.kill, kwargs={'name':s, 'update': False}) for s in servers]
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


    def servers(self, search=None,  **kwargs) -> List[str]:
        return list(self.namespace(search=search, **kwargs).keys())

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

        name = self.process_prefix + name
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
        output_string = c.cmd('pm2 status')
        processes = []
        tag = ' default '
        for line in output_string.split('\n'):
            if  tag in line:
                name = line.split(tag)[0].strip()
                name = name.split(' ')[-1]
                processes += [name]
        if search != None:
            search = self.resolve_name(search)
            processes = [m for m in processes if search in m]
        processes = sorted(list(set(processes)))
        return processes

    def procs(self, **kwargs):
        return self.processes(**kwargs)

    def resolve_name(self, name:str, **kwargs) -> str:
        if name == None:
            return name
        return self.process_prefix + name
        
    def exists(self, name:str, **kwargs) -> bool:
        name = self.resolve_name(name)
        return name in self.processes(**kwargs)

    def ensure_env(self,**kwargs):
        '''ensure that the environment variables are set for the process'''
        is_pm2_installed = bool( '/bin/pm2' in c.cmd('which pm2', verbose=False))
        if not is_pm2_installed:
            c.cmd('npm install -g pm2')
            c.cmd('pm2 update')
        return {'success':True, 'message':f'Ensured env '}


    min_stake = 0
    block_time = 8 
    endpoints = ['namespace']

    def resolve_path(self, path:str) -> str:
        return c.resolve_path('~/.commune/network/' + path)
    def set_network(self, 
                    network:str, 
                    tempo:int=60, 
                    n=100, 
                    path=None,
                    **kwargs):
        self.network = network 
        self.tempo = tempo
        self.n = n 
        self.network_path = self.resolve_path(path or f'{self.network}')
        self.modules_path =  f'{self.network_path}/modules'
        return {'network': self.network, 
                'tempo': self.tempo, 
                'n': self.n,
                'network_path': self.network_path}
    
    def params(self,*args,  **kwargs):
        return { 'network': self.network, 'tempo' : self.tempo,'n': self.n}

    def modules(self, 
                search=None, 
                max_age=None, 
                update=False, 
                features=['name', 'url', 'key'], 
                timeout=8, 
                **kwargs):
        modules = c.get(self.modules_path, max_age=max_age or self.tempo, update=update)
        if modules == None:
            modules = []
            urls = ['0.0.0.0'+':'+str(p) for p in c.used_ports()]
            futures  = [c.submit(c.call, [s + '/info'], timeout=timeout) for s in urls]
            try:
                for f in c.as_completed(futures, timeout=timeout):
                    data = f.result()
                    if all([k in data for k in features]):
                        modules.append({k: data[k] for k in features})
            except Exception as e:
                c.print('Error getting modules', e)
                modules = []
            c.put(self.modules_path, modules)
        if search != None:
            modules = [m for m in modules if search in m['name']]
        return modules

    def namespace(self, search=None,  max_age:int = None, update:bool = False, **kwargs) -> dict:
        processes = self.processes(search=search, **kwargs)
        modules = self.modules(search=search, max_age=max_age, update=update, **kwargs)
        processes = [ p.replace(self.process_prefix, '') for p in processes if p.startswith(self.process_prefix)]
        namespace = {m['name']: m['url'] for m in modules if m['name'] in processes}
        return namespace

    def add_server(self, name:str, url:str, key:str) -> None:
        modules = self.modules()
        modules.append( {'name': name, 'url': url, 'key': key})
        c.put(self.modules_path, modules)
        return {'success': True, 'msg': f'Block {name}.'}
    
    def rm_server(self, name:str, features=['name', 'key', 'url']) -> Dict:
        modules = self.modules()
        modules = [m for m in modules if not any([m[f] == name for f in features])]
        c.put(self.modules_path, modules)

    def resolve_network(self, network:str) -> str:
        return network or self.network
    
    def server_exists(self, name:str, **kwargs) -> bool:
        servers = self.servers(**kwargs)
        return bool(name in servers)