
import commune as c 
from typing import *
class ServerManager:
   
    @classmethod
    def kill(cls, 
             module,
             mode:str = 'pm2',
             verbose:bool = False,
             update : bool = True,
             prefix_match = False,
             network = 'local', # local, dev, test, main
             **kwargs):

        kill_fn = getattr(cls, f'{mode}_kill')
        delete_modules = []

        try:
            killed_module =kill_fn(module, verbose=verbose,prefix_match=prefix_match, **kwargs)
        except Exception as e:
            return {'error':str(e)}
        if isinstance(killed_module, list):
            delete_modules.extend(killed_module)
        elif isinstance(killed_module, str):
            delete_modules.append(killed_module)
        else:
            delete_modules.append(killed_module)
        # update modules
        c.deregister_server(module, network=network)

        assert c.server_exists(module, network=network) == False, f'module {module} still exists'

        servers = c.servers()
        for m in delete_modules:
            if m in servers:
                c.deregister_server(m, network=network)

        return {'server_killed': delete_modules, 'update': update}
    
    

    @classmethod
    def kill_prefix(cls, prefix:str, **kwargs):
        servers = c.servers(network='local')
        killed_servers = []
        for s in servers:
            if s.startswith(prefix):
                c.kill(s, **kwargs)
                killed_servers.append(s)
        return {'success':True, 'message':f'Killed servers with prefix {prefix}'}
    


    @classmethod
    def kill_many(cls, servers, search:str = None, network='local',  timeout=10, **kwargs):
        servers = c.servers(network=network)
        servers = [s for s in servers if  search in s]
        futures = []
        for s in servers:
            c.print(f'Killing {s}', color='red')
            future = c.submit(c.kill, kwargs={'module':s, **kwargs}, imeout=timeout)
            futures.append(future)
        results = []
        for r in c.as_completed(futures, timeout=timeout):
            results += [r.result()]
        c.print(f'Killed {len(results)} servers', color='red')
        return results
    

    @classmethod
    def fleet(cls, module, n=5, timeout=10):
        futures = []
        if '::'  not in module:
            module = f'{module}::'

        
        for i in range(n):
            module_name = f'{module}{i}'
            future = c.submit(cls.serve, kwargs=dict(module=module_name), timeout=timeout)
            futures.append(future)
        results = []
        for future in c.as_completed(futures, timeout=timeout):
            result = future.result()
            results.append(result)

        return results


    @classmethod
    def serve_many(cls, modules:list, **kwargs):

        if isinstance(modules[0], list):
            modules = modules[0]
        
        futures = []
        for module in modules:
            future = c.submit(c.serve, kwargs={'module': module, **kwargs})
            futures.append(future)
            
        results = []
        for future in c.as_completed(futures):
            result = future.result()
            results.append(result)
        return results
    serve_batch = serve_many


    @classmethod
    def wait_for_server(cls,
                          name: str ,
                          network: str = 'local',
                          timeout:int = 600,
                          sleep_interval: int = 1, 
                          verbose:bool = False) -> bool :
        
        time_waiting = 0
        while time_waiting < timeout:
            namespace = c.namespace(network=network)
            if name in namespace:
                c.print(f'{name} is ready', color='green')
                return True
            time_waiting += sleep_interval
            c.print(f'Waiting for {name} for {time_waiting} seconds', color='red')
            c.sleep(sleep_interval)
        raise TimeoutError(f'Waited for {timeout} seconds for {name} to start')
    


    @staticmethod
    def kill_all_servers( *args, **kwargs):
        '''
        Kill all of the servers
        '''
        for module in c.servers(*args, **kwargs):
            c.kill(module)

        # c.update(network='local')
            
    @classmethod
    def kill_all(cls, network='local', timeout=20, verbose=True):
        futures = []
        servers = c.servers(network=network)
        n = len(servers)
        progress = c.tqdm(n)
        for s in servers:
            c.print(f'Killing {s}', color='red')
            futures += [c.submit(c.kill, kwargs={'module':s, 'update': False}, return_future=True)]
        results_list = []
        for f in c.as_completed(futures, timeout=timeout):
            result = f.result()
            print(result)
            progress.update(1)
            results_list += [result]
        namespace = c.namespace(network=network, update=True)
        new_n = len(servers)
        c.print(f'Killed {n - new_n} servers, with {n} remaining {servers}', color='red')
        return {'success':True, 'old_n':n, 'new_n':new_n, 'servers':servers, 'namespace':namespace}



    @classmethod
    def serve(cls, 
              module:Any = None,
              kwargs:dict = None,  # kwargs for the module
              params = None, # kwargs for the module
              tag:str=None,
              server_network = 'local', # network to run the server
              port :int = None, # name of the server if None, it will be the module name
              server_name:str=None, # name of the server if None, it will be the module name
              name = None, # name of the server if None, it will be the module name
              refresh:bool = True, # refreshes the server's key
              remote:bool = True, # runs the server remotely (pm2, ray)
              tag_seperator:str='::',
              max_workers:int = None,
              free: bool = False,
              mnemonic = None, # mnemonic for the server
              key = None,
              **extra_kwargs
              ):
        module = module or c.module_name()
        if module.endswith('.py'):
            module = module[:-3]
        if tag_seperator in str(module):
            module, tag = module.split(tag_seperator)
        kwargs = {**(params or kwargs or {}), **extra_kwargs}
        name = name or server_name or module
        if tag_seperator in name:
            module, tag = name.split(tag_seperator)
        else:
            if tag != None:
                name = f'{name}{tag_seperator}{tag}'

        if port == None:
            # now if we have the server_name, we can repeat the server
            address = c.get_address(name, network=server_network)
            try:
                port = int(address.split(':')[-1])
                if c.port_used(port):
                    c.kill_port(port)
            except Exception as e:
                port = c.free_port()
        # RESOLVE THE PORT FROM THE ADDRESS IF IT ALREADY EXISTS

        # # NOTE REMOVE is FROM THE KWARGS REMOTE
        if remote:
            remote_kwargs = c.locals2kwargs(locals())  # GET THE LOCAL KWARGS FOR SENDING TO THE REMOTE
            remote_kwargs['remote'] = False  # SET THIS TO FALSE TO AVOID RECURSION
            for _ in ['extra_kwargs', 'address']:
                remote_kwargs.pop(_, None) # WE INTRODUCED THE ADDRES
            response = cls.remote_fn('serve', name=name, kwargs=remote_kwargs)
            if response['success'] == False:
                return response
            return {'success':True, 
                    'name': name, 
                    'address':c.ip() + ':' + str(remote_kwargs['port']), 
                    'kwargs':kwargs, 
                    'module':module
                    } 

        module_class = c.module(module)
        kwargs.update(extra_kwargs)
        module = module_class(**kwargs)
        cls(module=module, 
                                          name=name, 
                                          port=port, 
                                          network=server_network, 
                                          max_workers=max_workers, 
                                          mnemonic = mnemonic,
                                          free=free, 
                                          key=key)

        return  {'success':True, 
                     'address':  f'{c.default_ip}:{port}' , 
                     'name':name, 
                     'kwargs': kwargs,
                     'module':module}
