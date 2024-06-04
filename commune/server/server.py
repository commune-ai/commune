import commune as c
import pandas as pd
from typing import *
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


class Server(c.Module):
    def __init__(
        self,
        module: Union[c.Module, object] = None,
        name: str = None,
        network:str = 'local',
        port: Optional[int] = None,
        key = None,
        protocal = 'server.protocal',
        save_history:bool= False,
        history_path:str = None , 
        nest_asyncio = True,
        new_loop = True,
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
        module = module or 'module'
        if isinstance(module, str):
            module = c.module(module)()
        # RESOLVE THE WHITELIST AND BLACKLIST
        module.whitelist = list(set((module.whitelist if hasattr(module, 'whitelist') else [] ) + c.whitelist))
        module.blacklist = list(set((self.blacklist if hasattr(self, 'blacklist') else []) + c.blacklist))
        self.name = module.server_name = name or module.server_name
        port = port or c.free_port()
        while c.port_used(port):
            port =  c.free_port()
        self.port = module.port = port
        
        self.ip = module.ip = c.ip()
        self.address = module.address = f"{module.ip}:{module.port}"
        self.network = module.network = network
        self.schema = module.schema() if hasattr(module, 'schema') else c.get_schema(module)
        module.key = c.get_key(key or self.name, create_if_not_exists=True)
        self.protocal = c.module(protocal)(module=module, 
                                           history_path=self.resolve_path(history_path or f'history/{self.name}'),
                                           save_history = save_history,
                                             **kwargs)
        self.ip = module.ip
        self.port = module.port
        self.module = module 

        self.set_api()

    def add_fn(self, name:str, fn: str):
        assert callable(fn), 'fn not callable'
        setattr(self.module, name, fn)
        return {'success':True, 'message':f'Added {name} to {self.name} module'}
           
    def set_api(self):

        self.app = FastAPI()
        self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        @self.app.post("/{fn}")
        def forward_api(fn:str, input:dict):
            return self.protocal.forward(fn=fn, input=input)
        
        # start the server
        try:
            c.print(f' Served(name={self.name}, address={self.address}, key=🔑{self.key}🔑 ) 🚀 ', color='purple')
            c.register_server(name=self.name, address = self.address, network=self.network)
            uvicorn.run(self.app, host='0.0.0.0', port=self.port, loop="asyncio")
        except Exception as e:
            c.print(e, color='red')
        finally:
            c.deregister_server(self.name, network=self.network)

    def info(self) -> Dict:
        return {
            'name': self.name,
            'address': self.module.address,
            'key': self.module.key.ss58_address,
            'network': self.module.network,
            'port': self.module.port,
            'whitelist': self.module.whitelist,
            'blacklist': self.module.blacklist,            
        }

    def wait_for_server(self, timeout=10):
        return c.wait_for_server(self.name, timeout=timeout)

    
    def __del__(self):
        c.deregister_server(self.name)
    
    @classmethod
    def serve(cls, 
              module:Any ,
              kwargs:dict = None,  # kwargs for the module
              tag:str=None,
              server_network = 'local',
              port :int = None, # name of the server if None, it will be the module name
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
        
        # RESOLVE THE NAME 
        name = cls.resolve_server_name(module=module, name=name, tag=tag, tag_seperator=tag_seperator)
        if tag_seperator in name:
            module, tag = name.split(tag_seperator)
        # RESOLVE TE KWARGS
        kwargs = kwargs or {}
        kwargs.update(extra_kwargs or {})

        module_class = c.module(module)
        kwargs.update(extra_kwargs)
        if mnemonic != None:
            c.add_key(name, mnemonic)

        module = module_class(**kwargs)
        module.server_name = name
        module.tag = tag
        address = c.get_address(name, network=server_network)
        if address != None and ':' in address:
            port = address.split(':')[-1]   

        if c.server_exists(name, network=server_network) and not refresh: 
            return {'success':True, 'message':f'Server {name} already exists'}
        
        server = c.module(f'server')(module=module, 
                            name=name,  
                            port=port, 
                            network=server_network, 
                            max_workers=max_workers, 
                            free=free, 
                            key=key)

        return  server.info()


    @classmethod
    def resolve_server_name(cls, 
                            module:str = None, 
                            tag:str=None, 
                            name:str = None,  
                            tag_seperator:str='::', 
                            **kwargs):
        """
        Resolves the server name
        """
        # if name is not specified, use the module as the name such that module::tag
        if name == None:
            module = cls.module_path() if module == None else module

            # module::tag
            if tag_seperator in module:
                module, tag = module.split(tag_seperator)
            if tag_seperator in module: 
                module, tag = module.split(tag_seperator)
            name = module
            if tag in ['None','null'] :
                tag = None
            if tag != None:
                name = f'{name}{tag_seperator}{tag}'

        # ensure that the name is a string
        assert isinstance(name, str), f'Invalid name {name}'
        return name

    
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

Server.run(__name__)