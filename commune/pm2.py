import commune as c
import os
from typing import *
import json

class PM2(c.Module):
    dir = os.path.expanduser('~/.pm2')
   
    def restart(self, name:str,prefix_match:bool = True):
        assert name in self.servers()
        c.print(f'Restarting {name}', color='cyan')
        c.cmd(f"pm2 restart {name}", verbose=False)
        self.rm_logs(name)  
        return {'success':True, 'message':f'Restarted {name}'}

    def kill(self, name:str, verbose:bool = True, **kwargs):
        if name == 'all':
            return self.kill_all(verbose=verbose)
        c.cmd(f"pm2 delete {name}", verbose=False)
        self.rm_logs(name)
        return {'success':True, 'message':f'Killed {name}'}
    
    
    def status(self, verbose=False):
        stdout = c.cmd(f"pm2 status", verbose=False)
        if verbose:
            c.print(stdout,color='green')
        return stdout

    dir = os.path.expanduser('~/.pm2')
    
    def logs_path_map(self, name=None):
        logs_path_map = {}
        for l in c.ls(f'{self.dir}/logs/'):
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

    
    def logs(self, 
                module:str, 
                tail: int =100, 
                mode: str ='cmd',
                **kwargs):
        
        if mode == 'local':
            text = ''
            for m in ['out','error']:
                # I know, this is fucked 
                path = f'{self.dir}/logs/{module.replace("/", "-")}-{m}.log'.replace(':', '-').replace('_', '-')
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
   
    
    def kill_many(self, search=None, verbose:bool = True, timeout=10):
        futures = []
        for name in self.servers(search=search):
            f = c.submit(self.kill, dict(name=name, verbose=verbose), return_future=True, timeout=timeout)
            futures.append(f)
        return c.wait(futures)
    
    
    def kill_all(self, verbose:bool = True, timeout=10, trials=10):
        while len(self.servers()) > 0:
            print(self.kill_many(search=None, verbose=verbose, timeout=timeout))
            trials -= 1
            assert trials > 0, 'Failed to kill all processes'
        return {'success':True, 'message':f'Killed all processes'}

                
    
    def servers(self, search=None,  verbose:bool = False) -> List[str]:
        output_string = c.cmd('pm2 status', verbose=False)
        module_list = []
        for line in output_string.split('\n')[3:]:
            if  line.count('│') > 2:
                server_name = line.split('│')[2].strip()
                if 'errored' in line:
                    self.kill(server_name, verbose=True)
                    continue
                module_list += [server_name]
            
        if search != None:
            search_true = lambda x: any([s in x for s in search])
            module_list = [m for m in module_list if search_true(m)]

        module_list = sorted(list(set(module_list)))
                
        return module_list
    
    pm2ls = servers
    
    # commune.run_command('pm2 status').stdout.split('\n')[5].split('    │')[0].split('  │ ')[-1]commune.run_command('pm2 status').stdout.split('\n')[5].split('    │')[0].split('  │ ')[-1] 
    
    
    def exists(self, name:str) -> bool:
        return bool(name in self.servers())
    
    
    def start(self, 
                path:str , 
                  name:str,
                  cmd_kwargs:str = None, 
                  refresh: bool = True,
                  verbose:bool = True,
                  force : bool = True,
                  current_dir: str = True,
                  interpreter : str = None,
                  **kwargs):
        
        if self.exists(name) and refresh:
            self.kill(name, verbose=verbose)
            
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

    
    def restart_many(self, search:str = None, network = None, **kwargs):
        t1 = c.time()
        servers = self.servers(search)
        futures = [c.submit(c.restart, kwargs={"name": m, **kwargs}) for m in servers]
        results = []
        for f in c.as_completed(futures):
            result = f.result()
            results.append(result)
        return results



    
    def launch(self, 
                   module:str = None,  
                   fn: str = 'serve',
                   name:Optional[str]=None, 
                   tag : str = None,
                   args : list = None,
                   kwargs: dict = None,
                   device:str=None, 
                   interpreter:str='python3', 
                   autorestart: bool = True,
                   verbose: bool = False , 
                   force:bool = True,
                   meta_fn: str = 'module_fn',
                   tag_seperator:str = '::',
                   cwd = None,
                   refresh:bool=True ):
        import commune as c

        if hasattr(module, 'module_name'):
            module = module.module_name()
            
        # avoid these references fucking shit up
        args = args if args else []
        kwargs = kwargs if kwargs else {}

        # convert args and kwargs to json strings
        kwargs =  {
            'module': module ,
            'fn': fn,
            'args': args,
            'kwargs': kwargs 
        }

        kwargs_str = json.dumps(kwargs).replace('"', "'")

        name = name or module
        if refresh:
            self.pm2_kill(name)
        module = c.module()
        # build command to run pm2
        filepath = c.filepath()
        cwd = cwd or module.dirpath()
        command = f"pm2 start {filepath} --name {name} --interpreter {interpreter}"

        if not autorestart:
            command += ' --no-autorestart'
        if force:
            command += ' -f '
        command = command +  f' -- --fn {meta_fn} --kwargs "{kwargs_str}"'
        env = {}
        if device != None:
            if isinstance(device, int):
                env['CUDA_VISIBLE_DEVICES']=str(device)
            if isinstance(device, list):
                env['CUDA_VISIBLE_DEVICES']=','.join(list(map(str, device)))
        if refresh:
            self.pm2_kill(name)  
        
        cwd = cwd or module.dirpath()
        
        stdout = c.cmd(command, env=env, verbose=verbose, cwd=cwd)
        return {'success':True, 'message':f'Launched {module}', 'command': command, 'stdout':stdout}

    
    def remote_fn(self, 
                    fn: str='train', 
                    module: str = None,
                    args : list = None,
                    kwargs : dict = None, 
                    name : str =None,
                    refresh : bool =True,
                    cwd = None,
                    **extra_launch_kwargs
                    ):
        import commune as c
        
        kwargs = c.locals2kwargs(kwargs)
        if 'remote' in kwargs:
            kwargs['remote'] = False
        if '/' in fn:
            module = '.'.join(fn.split('.')[:-1])
            fn = fn.split('.')[-1]
            
        kwargs = kwargs if kwargs else {}
        args = args if args else []
        if 'remote' in kwargs:
            kwargs['remote'] = False

        cwd = cwd or self.dirpath()
        kwargs = kwargs or {}
        args = args or []
        module = self.resolve_object(module)
        name = self.resolve_name(module)

        # resolve the name
        if name == None:
            # if the module has a module_path function, use that as the name
            if hasattr(module, 'module_name'):
                name = module.module_name()
            else:
                name = module.__name__.lower() 

        assert fn != None, 'fn must be specified for pm2 launch'
    
        return  self.launch( module=module, 
                            fn = fn,
                            name=name, 
                            args = args,
                            kwargs = kwargs,
                            refresh=refresh,
                            **extra_launch_kwargs)
