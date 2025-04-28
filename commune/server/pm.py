import commune as c 
from typing import *
import json
import os
class ProcessManager:

    def __init__(self, prefix='proc/', **kwargs):
        self.prefix = prefix
        self.process_manager_path = c.abspath('~/.pm2')
        self.sync_env()

    def get_name(self, name:str, **kwargs) -> str:
        if  name != None and not name.startswith(self.prefix):
            name = self.prefix + name
        return name
        
    def exists(self, name:str, **kwargs) -> bool:
        name = self.get_name(name)
        return name in self.procs(**kwargs)

    def procs(self, search=None,  **kwargs) -> List[str]:
        output_string = c.cmd('pm2 status')
        procs = []
        tag = ' default '
        for line in output_string.split('\n'):
            if  tag in line:
                name = line.split(tag)[0].strip()
                name = name.split(' ')[-1]
                procs += [name]
        if search != None:
            search = self.get_name(search)
            procs = [m for m in procs if search in m]
        procs = sorted(list(set(procs)))
        return procs

    def run(self, 
                  fn: str = 'serve',
                   name:str = None, 
                   module:str = 'server',  
                   params: dict = None,
                   network:str = 'local',
                   interpreter:str='python3', 
                   verbose: bool = False , 
                   cwd : str = None,
                    max_age = 10,
                    trials:int=3,
                    trial_backoff:int=1,
                    refresh:bool=True ):
        """
        run a process with pm2

        Args:
            fn (str, optional): The function to run. Defaults to 'serve'.
            name (str, optional): The name of the proc. Defaults to None.
            module (str, optional): The module to run. Defaults to 'server'.
            params (dict, optional): The parameters for the function. Defaults to None.
            interpreter (str, optional): The interpreter to use. Defaults to 'python3'.
            verbose (bool, optional): Whether to print the output. Defaults to False.
            wait_for_server (bool, optional): Whether to wait for the server to start. Defaults to True.
            cwd (str, optional): The current working directory. Defaults to None.
            refresh (bool, optional): Whether to refresh the environment. Defaults to True.
        Returns:
            dict: The result of the command
         
        """
        self.sync_env()
        params['remote'] = False
        name = name or module
        if '/' in fn:
            module, fn = fn.split('/')
        params_str = json.dumps({'fn': module +'/' + fn, 'params': params or {}}).replace('"','\\"')
        proc_name = self.get_name(name)
        if self.exists(proc_name):
            self.kill(proc_name, rm_server=False)
        cmd = f"pm2 start {c.filepath()} --name {proc_name} --interpreter {interpreter} -f --no-autorestart -- --fn run --params \"{params_str}\""
        c.cmd(cmd, verbose=verbose, cwd=c.lib_path)
        return {'success':True, 'message':f'Running {proc_name}'}

    def kill(self, name:str, verbose:bool = True, rm_server=True, **kwargs):
        proc_name = self.get_name(name)
        try:
            c.cmd(f"pm2 delete {proc_name}", verbose=False)
            for m in ['out', 'error']:
                os.remove(self.get_logs_path(name, m))
            result =  {'message':f'Killed {proc_name}', 'success':True}
        except Exception as e:
            result =  {'message':f'Error killing {proc_name}', 'success':False, 'error':e}
        return result
    
    def kill_all(self, verbose:bool = True, timeout=20):
        servers = self.procs()
        futures = [c.submit(self.kill, kwargs={'name':s, 'update': False}) for s in servers]
        results = c.wait(futures, timeout=timeout)
        return results
    
    def killall(self, **kwargs):
        return self.kill_all(**kwargs)

    def get_logs_path(self, name:str, mode='out')->str:
        assert mode in ['out', 'error'], f'Invalid mode {mode}'
        name = self.get_name(name)
        return f'{self.process_manager_path}/logs/{name.replace("/", "-")}-{mode}.log'.replace(':', '-').replace('_', '-') 

    def logs(self, module:str, top=None, tail: int =None , stream=True, **kwargs):
        module = self.get_name(module)
        if tail or top:
            stream = False
        if stream:
            return c.cmd(f"pm2 logs {module}", verbose=True)
        else:
            text = ''
            for m in ['out', 'error']:
                # I know, this is fucked 
                path = self.get_logs_path(module, m)
                try:
                    text +=  c.get_text(path)
                except Exception as e:
                    c.print('ERROR GETTING LOGS -->' , e)
            if top != None:
                text = '\n'.join(text.split('\n')[:top])
            if tail != None:
                text = '\n'.join(text.split('\n')[-tail:])
        return text

    def sync_env(self,**kwargs):
        '''ensure that the environment variables are set for the proc'''
        is_pm2_installed = bool( '/bin/pm2' in c.cmd('which pm2', verbose=False))
        if not is_pm2_installed:
            c.cmd('npm install -g pm2')
            c.cmd('pm2 update')
        return {'success':True, 'message':f'Ensured env '}
  
   