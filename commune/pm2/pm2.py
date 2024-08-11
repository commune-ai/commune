import commune as c


import os
from typing import *
import json

class PM2(c.Module):
    dir = os.path.expanduser('~/.pm2')
   
    @classmethod
    def restart(cls, name:str, verbose:bool = False, prefix_match:bool = True):
        list = cls.servers()
        if name in list:
            rm_list = [name]
        else:
            if prefix_match:
                rm_list = [ p for p in list if p.startswith(name)]
            else:
                raise Exception(f'pm2 process {name} not found')

        if len(rm_list) == 0:
            return []
        for n in rm_list:
            c.print(f'Restarting {n}', color='cyan')
            c.cmd(f"pm2 restart {n}", verbose=False)
            cls.rm_logs(n)  
        return {'success':True, 'message':f'Restarted {name}'}
       

    @classmethod
    def kill(cls, name:str, verbose:bool = False, **kwargs):
        if name == 'all':
            return cls.kill_all(verbose=verbose)
        c.cmd(f"pm2 delete {name}", verbose=False)
        # remove the logs from the pm2 logs directory
        cls.rm_logs(name)
        return {'success':True, 'message':f'Killed {name}'}
    
    @classmethod
    def status(cls, verbose=False):
        stdout = c.cmd(f"pm2 status", verbose=False)
        if verbose:
            c.print(stdout,color='green')
        return stdout


    dir = os.path.expanduser('~/.pm2')
    @classmethod
    def logs_path_map(cls, name=None):
        logs_path_map = {}
        for l in c.ls(f'{cls.dir}/logs/'):
            key = '-'.join(l.split('/')[-1].split('-')[:-1]).replace('-',':')
            logs_path_map[key] = logs_path_map.get(key, []) + [l]

    
        for k in logs_path_map.keys():
            logs_path_map[k] = {l.split('-')[-1].split('.')[0]: l for l in list(logs_path_map[k])}

        if name != None:
            return logs_path_map.get(name, {})

        return logs_path_map
   
    @classmethod
    def rm_logs( cls, name):
        logs_map = cls.logs_path_map(name)

        for k in logs_map.keys():
            c.rm(logs_map[k])

    @classmethod
    def logs(cls, 
                module:str, 
                tail: int =100, 
                verbose: bool=True ,
                mode: str ='cmd',
                **kwargs):
        
        if mode == 'local':
            text = ''
            for m in ['out','error']:
                # I know, this is fucked 
                path = f'{cls.dir}/logs/{module.replace("/", "-")}-{m}.log'.replace(':', '-').replace('_', '-')
                try:
                    text +=  c.get_text(path, tail=tail)
                except Exception as e:
                    c.print(e)
                    continue
            
            return text
        elif mode == 'cmd':
            return c.cmd(f"pm2 logs {module}", verbose=True)
        else:
            raise NotImplementedError(f'mode {mode} not implemented')
   
    @classmethod
    def kill_many(cls, search=None, verbose:bool = True, timeout=10):
        futures = []
        for name in cls.servers(search=search):
            f = c.submit(cls.kill, dict(name=name, verbose=verbose), return_future=True, timeout=timeout)
            futures.append(f)
        return c.wait(futures)
    
    @classmethod
    def kill_all(cls, verbose:bool = True, timeout=10, trials=10):
        while len(cls.servers()) > 0:
            results = cls.kill_many(search=None, verbose=verbose, timeout=timeout)
            trials -= 1
            assert trials > 0, 'Failed to kill all processes'

                
    @classmethod
    def servers(cls, search=None,  verbose:bool = False) -> List[str]:
        output_string = c.cmd('pm2 status', verbose=False)
        module_list = []
        for line in output_string.split('\n'):
            if  line.count('│') > 2:
                server_name = line.split('│')[2].strip()
                if 'errored' in line:
                    cls.kill(server_name, verbose=True)
                    continue

                module_list += [server_name]
            
        if search != None:
            search_true = lambda x: any([s in x for s in search])
            module_list = [m for m in module_list if search_true(m)]

        module_list = sorted(list(set(module_list)))
                
        return module_list
    
    pm2ls = servers
    

    # commune.run_command('pm2 status').stdout.split('\n')[5].split('    │')[0].split('  │ ')[-1]commune.run_command('pm2 status').stdout.split('\n')[5].split('    │')[0].split('  │ ')[-1] 
    
    @classmethod
    def exists(cls, name:str) -> bool:
        return bool(name in cls.servers())
    
    @classmethod
    def start(cls, 
                path:str , 
                  name:str,
                  cmd_kwargs:str = None, 
                  refresh: bool = True,
                  verbose:bool = True,
                  force : bool = True,
                  current_dir: str = True,
                  interpreter : str = None,
                  **kwargs):
        
        if cls.exists(name) and refresh:
            cls.kill(name, verbose=verbose)
            
        cmd = f'pm2 start {path} --name {name}'

        if force:
            cmd += ' -f'
            
        if interpreter != None:
            cmd += f' --interpreter {interpreter}'
            
        if cmd_kwargs != None:
            cmd += f' -- '

            if isinstance(cmd_kwargs, dict):
                for k, v in cmd_kwargs.items():
                    cmd += f'--{k} {v}'
            elif isinstance(cmd_kwargs, str):
                cmd += f'{cmd_kwargs}'
                
        c.print(f'[bold cyan]Starting (PM2)[/bold cyan] [bold yellow]{name}[/bold yellow]', color='green')

        if current_dir:
            kwargs['cwd'] = c.dirpath(path)

        return c.cmd(cmd, verbose=verbose, **kwargs)

    @classmethod
    def restart_many(cls, search:str = None, network = None, **kwargs):
        t1 = c.time()
        servers = cls.servers(search)
        futures = [c.submit(c.restart, kwargs={"name": m, **kwargs}) for m in servers]
        results = []
        for f in c.as_completed(futures):
            result = f.result()
            results.append(result)
        return results
