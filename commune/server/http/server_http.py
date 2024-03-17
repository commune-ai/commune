
from typing import Dict, List, Optional, Union
import commune as c


class ServerHTTP(c.module('server')):
    def __init__(
        self,
        module: Union[c.Module, object] = None,
        name: str = None,
        network:str = 'local',
        ip = '0.0.0.0',
        port: Optional[int] = None,
        sse: bool = True,
        chunk_size: int = 1000,
        max_request_staleness: int = 60, 
        key = None,
        verbose: bool = False,
        timeout: int = 256,
        access_module: str = 'server.access',
        public: bool = False,
        serializer: str = 'serializer',
        save_history:bool= True,
        history_path:str = None , 
        nest_asyncio = True,
        new_loop = True,
        **kwargs
        
        ) -> 'Server':

        if new_loop:
            self.loop = c.new_event_loop(nest_asyncio=nest_asyncio)
   
        self.serializer = c.module(serializer)()
        self.set_address(ip=ip, port=port)
        self.max_request_staleness = max_request_staleness
        self.network = network
        self.verbose = verbose
        self.sse = sse
        self.save_history = save_history
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.public = public
        
        # name 
        if isinstance(module, str):
            module = c.module(module)()
        if name == None:
            if hasattr(module, 'server_name'):
                name = module.server_name
            else:
                name = module.__class__.__name__
        self.name = name


        self.schema = {}
        if hasattr(module, 'schema'):
            self.schema = module.schema()

        module.ip = self.ip
        module.port = self.port
        module.address  = self.address
        self.module = module 
        self.set_key(key)
        self.access_module = c.module(access_module)(module=self.module)  
        self.set_history_path(history_path)
        self.set_api(ip=self.ip, port=self.port)

