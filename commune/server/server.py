import commune as c
import pandas as pd
from typing import *
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from .middleware import ServerMiddleware
from .manager import ServerManager
from sse_starlette.sse import EventSourceResponse

class Server(ServerManager, c.Module):

    def __init__(
        self,
        module: Union[c.Module, object] = None,
        name: str = None,
        network:str = 'local',
        port: Optional[int] = None,
        key:str = None, # key for the server (str)
        endpoints:Optional[List[str]] = None, # list of endpoints
        max_request_staleness:int = 5,
        loop = None,
        max_bytes:int = 10 * 1024 * 1024,  # 1 MB limit
        nest_asyncio:bool = True, # whether to use nest asyncio
        access_module:Optional[c.Module] = 'server.access',
        **kwargs
        ) -> 'Server':
        
        if  nest_asyncio:
            c.new_event_loop(nest_asyncio=nest_asyncio)
        self.loop = c.get_event_loop() if loop == None else loop
        self.max_request_staleness = max_request_staleness
        self.serializer = c.module('serializer')()
        self.network = network
        self.set_module(module=module, 
                        name=name, 
                        port=port, 
                        key=key, 
                        endpoints=endpoints, 
                        access_module=access_module,
                        network=network)
        self.set_api(max_bytes=max_bytes, **kwargs)

    def set_module(self, 
                   module:Union[c.Module, object],
                   name:Optional[str]=None,
                   port:Optional[int]=None,
                   key:Optional[str]=None,
                   endpoints:Optional[List[str]] = None,
                     access_module:Optional[str] = 'server.access',
                    network:str='local'):
        if isinstance(module, str):
            module = c.module(module)()
        if not  hasattr(module, 'get_endpoints'):
            module.get_endpoints = lambda : dir(module)
        endpoints = endpoints or module.get_endpoints()
        module.endpoints = endpoints
        module.name = module.server_name = name or module.server_name
        module.port = port if port not in ['None', None] else c.free_port()
        module.address = f"{c.ip()}:{module.port}"
        module.network = network
        module.key  = c.get_key(key or module.name, create_if_not_exists=True)
        self.module = module
        self.access_module = c.module(access_module)(module=self.module)
        return {'success':True, 'message':f'Set {self.module.name} module'}
    def add_fn(self, name:str, fn: str):
        assert callable(fn), 'fn not callable'
        setattr(self.module, name, fn)
        return {'success':True, 'message':f'Added {name} to {self.name} module'}

    def forward(self, fn,  request: Request):
        headers = dict(request.headers.items())
        # STEP 1 : VERIFY THE SIGNATURE AND STALENESS OF THE REQUEST TO MAKE SURE IT IS AUTHENTIC
        key_address = headers.get('key', headers.get('address', None))
        assert key_address, 'No key or address in headers'
        request_staleness = c.timestamp() - int(headers['timestamp'])
        assert  request_staleness < self.max_request_staleness, f"Request is too old ({request_staleness}s > {self.max_request_staleness}s (MAX)" 
        data = self.loop.run_until_complete(request.json())
        data = self.serializer.deserialize(data) 
        signature_data = {'data': headers['hash'], 'timestamp': headers['timestamp']}
        assert c.verify(auth=signature_data, signature=headers['signature'], address=key_address)
        self.access_module.verify(fn=fn, address=key_address)

        # STEP 2 : PREPARE THE DATA FOR THE FUNCTION CALL
        if 'params' in data:
            data['kwargs'] = data['params']
        # if the data is just key words arguments
        if not 'args' in data and not 'kwargs' in data:
            data = {'kwargs': data, 'args': []}
        data['args'] =  list(data.get('args', []))
        data['kwargs'] = dict(data.get('kwargs', {}))
        assert isinstance(data['args'], list), 'args must be a list'
        assert isinstance(data['kwargs'], dict), 'kwargs must be a dict'
        # STEP 3 : CALL THE FUNCTION FOR THE RESPONSE
        fn_obj = getattr(self.module, fn)
        response = fn_obj(*data['args'], **data['kwargs']) if callable(fn_obj) else fn_obj
        latency = c.round(c.time() - int(headers['timestamp']), 3)
        correct_emoji = '✅' 
        
        info_str = f"fn={fn} from={headers['key'][:4]}..."
        msg = f"<{correct_emoji}Response({info_str} latency={latency}s){correct_emoji}>"
        c.print(msg, color='green')
        # STEP 4 : SERIALIZE THE RESPONSE AND RETURN SSE IF IT IS A GENERATOR AND JSON IF IT IS A SINGLE OBJECT
        #TODO WS: ADD THE SSE RESPONSE
        if c.is_generator(response):
            def generator_wrapper(generator):
                for item in generator:
                    yield self.serializer.serialize(item)
            return EventSourceResponse(generator_wrapper(response))
        else:
            return self.serializer.serialize(response)

    def wrapper_forward(self, fn:str):
        def fn_forward(request: Request):
            try:
                output = self.forward(fn, request)
            except Exception as e:
                output =  c.detailed_error(e)
            return output
        return fn_forward

    def set_api(self, 
                max_bytes=1024 * 1024,
                allow_origins = ["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
                **kwargs
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

        # add all of the whitelist functions in the module
        for fn in self.module.endpoints:
            c.print(f'Adding {fn} to the server')
            # make a copy of the forward function
            self.app.post(f"/{fn}")(self.wrapper_forward(fn))

        # start the server
        try:
            c.print(f' Served(name={self.module.name}, address={self.module.address}, key=🔑{self.module.key}🔑 ) 🚀 ', color='purple')
            c.register_server(name=self.module.name, 
                              address = self.module.address, 
                              network=self.module.network)
            uvicorn.run(self.app, host='0.0.0.0', port=self.module.port, loop="asyncio")
        except Exception as e:
            c.print(e, color='red')
        finally:
            c.deregister_server(self.name, network=self.module.network)

    def __del__(self):
        c.deregister_server(self.name)


Server.run(__name__)