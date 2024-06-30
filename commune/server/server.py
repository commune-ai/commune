import commune as c
import pandas as pd
from typing import *
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from .middleware import ServerMiddleware





class Server(c.Module):
    def __init__(
        self,
        module: Union[c.Module, object] = None,
        name: str = None,
        network:str = 'local',
        port: Optional[int] = None,
        key = None,
        protocal = 'server.protocal',
        save_history:bool= True,
        history_path:str = None , 
        nest_asyncio = True,
        new_loop = True,
        max_bytes = 10 * 1024 * 1024,  # 1 MB limit
        mnemonic = None,
        **kwargs
        ) -> 'Server':
        """
        
        params:
        - module: the module to serve
        - name: the name of the server
        - port: the port of the serve
        
        """

        if new_loop:
            c.new_event_loop(nest_asyncio=nest_asyncio)
        self.set_module(module=module, 
                        name=name, 
                        port=port, 
                        key=key, 
                        protocal=protocal, 
                        save_history=save_history,
                        history_path=history_path,
                        mnemonic=mnemonic,
                        network=network, **kwargs

                        )
        self.set_api(max_bytes=max_bytes)

    def set_module(self, 
                   module: Union[c.Module, object] = None, 
                   name: str = None, 
                   port: Optional[int] = None, 
                   key = None, 
                   protocal = 'server.protocal', 
                   save_history:bool= False, 
                   history_path:str = None, 
                   network:str = 'local',
                     mnemonic = None,
                   **kwargs
                   ):
        
        self.protocal = c.module(protocal)(module=module,     
                                            history_path=self.resolve_path(history_path  or name),
                                            name = name,
                                            port=port,
                                            key=key,
                                            save_history = save_history,
                                            mnemonic = mnemonic,
                                             **kwargs)
        self.module = self.protocal.module 
 
        response = {'name':self.module.name, 'address': self.module.address, 'port':self.module.port, 'key':self.module.key.ss58_address, 'network':self.module.network, 'success':True}
        return response

    def add_fn(self, name:str, fn: str):
        assert callable(fn), 'fn not callable'
        setattr(self.module, name, fn)
        return {'success':True, 'message':f'Added {name} to {self.name} module'}
           
    def set_api(self, 
                max_bytes=1024 * 1024,
                allow_origins = ["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
                ):

        self.app = FastAPI()
        # add the middleware
        self.app.add_middleware(ServerMiddleware, max_bytes=max_bytes)    
        self.app.add_middleware(
                CORSMiddleware,
                allow_origins=allow_origins,
                allow_credentials=allow_credentials,
                allow_methods=allow_methods,
                allow_headers=allow_headers,
            )
        
        @self.app.post("/{fn}")
        def forward_api(fn:str, input:dict):
            return self.protocal.forward(fn=fn, input=input)
        
        # start the server
        try:
            c.print(f' Served(name={self.module.name}, address={self.module.address}, key=🔑{self.module.key}🔑 ) 🚀 ', color='purple')
            c.register_server(name=self.module.name, address = self.module.address, network=self.module.network)
            uvicorn.run(self.app, host='0.0.0.0', port=self.module.port, loop="asyncio")
        except Exception as e:
            c.print(e, color='red')
        finally:
            c.deregister_server(self.name, network=self.module.network)

    def info(self) -> Dict:
        return {
            'name': self.module.name,
            'address': self.module.address,
            'key': self.module.key.ss58_address,
            'network': self.module.network,
            'port': self.module.port,
            'whitelist': self.module.whitelist,
            'blacklist': self.module.blacklist,            
        }

    def __del__(self):
        c.deregister_server(self.name)
    

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
            c.print(result)
            results.append(result)
        return results
    serve_batch = serve_many



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
        
        if tag_seperator in str(module):
            module, tag = module.split('::')
        module = module or c.module_name()
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

        # # NOTE REMOVE THIS FROM THE KWARGS REMOTE
        if remote:
            remote_kwargs = c.locals2kwargs(locals())  # GET THE LOCAL KWARGS FOR SENDING TO THE REMOTE
            remote_kwargs['remote'] = False  # SET THIS TO FALSE TO AVOID RECURSION
            for _ in ['extra_kwargs', 'address']:
                remote_kwargs.pop(_, None) # WE INTRODUCED THE ADDRES
            response = c.remote_fn('serve', name=name, kwargs=remote_kwargs)
            if response['success'] == False:
                return response
            return {'success':True, 
                    'name': name, 
                    'address':c.ip() + ':' + str(remote_kwargs['port']), 
                    'kwargs':kwargs, 
                    'module':module
                    } 

        module_class = c.module(module)
        print('Serving', module_class, kwargs, module)
        kwargs.update(extra_kwargs)
        cls(module=module_class(**kwargs), 
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

    @classmethod
    def history(cls, **kwargs):
        return c.ls(c.resolve_path('history'))
    

    @classmethod
    def fleet(cls, module, n=5, timeout=10):
        futures = []
        if '::'  not in module:
            module = f'{module}::'

        
        for i in range(n):
            module_name = f'{module}{i}'
            print(f'Serving {module_name}')
            future = c.submit(cls.serve, kwargs=dict(module=module_name), timeout=timeout)
            futures.append(future)
        results = []
        for future in c.as_completed(futures, timeout=timeout):
            result = future.result()
            c.print(result)
            results.append(result)

        return results

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
            c.print(result, verbose=verbose)
            progress.update(1)
            results_list += [result]
        servers = c.servers(network=network, update=True)
        new_n = len(servers)
        c.print(f'Killed {n - new_n} servers, with {n} remaining {servers}', color='red')
        return results_list
   
   
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
        
    killpre = kill_prefix


    @classmethod
    def kill_many(cls, servers, search:str = None, network='local', parallel=True,  timeout=10, **kwargs):


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
        
    delete = kill_server = kill
    
    


Server.run(__name__)