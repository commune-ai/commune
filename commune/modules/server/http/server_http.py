
from typing import Dict, List, Optional, Union
import commune as c
import torch 
import traceback
import json




class ServerHTTP(c.Module):
    def __init__(
        self,
        module: Union[c.Module, object],
        name: str = None,
        network:str = 'local',
        port: Optional[int] = None,
        sse: bool = True,
        chunk_size: int = 42_000,
        max_request_staleness: int = 60, 
        max_workers: int = None,
        mode:str = 'thread',
        verbose: bool = False,
        timeout: int = 256,
        public: bool = True
        ) -> 'Server':
        
        self.serializer = c.module('serializer')()
        self.ip = c.default_ip # default to '0.0.0.0'
        self.port = int(c.resolve_port(port))
        self.address = f"{self.ip}:{self.port}"
        self.max_request_staleness = max_request_staleness
        self.chunk_size = chunk_size
        self.network = network
        self.verbose = verbose

        # executro 
        self.sse = sse
        if self.sse == False:
            self.max_workers = max_workers
            self.mode = mode
            self.executor = c.executor(max_workers=max_workers, mode=mode)
        self.timeout = timeout
        self.public = public

        if name == None:
            if hasattr(module, 'server_name'):
                name = module.server_name
            else:
                name = module.__class__.__name__
        

        self.module = module 
        c.print(self.module, type(self.module), module.key)
        self.key = module.key      
        # register the server
        self.name = name
        module.ip = self.ip
        module.port = self.port
        module.address  = self.address
        self.set_api(ip=self.ip, port=self.port)



    def set_api(self, ip = None, port = None):
        ip = self.ip if ip == None else ip
        port = self.port if port == None else port
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware

        self.app = FastAPI()
        self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )


        @self.app.post("/{fn}")
        async def forward_api(fn:str, input:dict):
            try:

                input['fn'] = fn

                input = self.process_input(input)

                data = input['data']
                args = data.get('args',[])
                kwargs = data.get('kwargs', {})
                
                input_kwargs = dict(fn=fn, args=args, kwargs=kwargs)
                fn_name = f"{self.name}::{fn}"
                c.print(f'🚀 Forwarding {input["address"]} --> {fn_name} 🚀\033', color='yellow')
                c.print(input_kwargs)
                if self.sse:
                    result = self.forward(**input_kwargs)
                else: 
                    result = self.executor.submit(self.forward, kwargs=input_kwargs, timeout=self.timeout)
                    result = result.result()
                # if the result is a future, we need to wait for it to finish
                if isinstance(result, dict) and 'error' in result:
                    success = False 
                success = True
            except Exception as e:
                success = False
                result = c.detailed_error(e)

            if success:
                c.print(f'✅ Success: {self.name}::{fn} --> {input["address"]}... ✅\033 ', color='green')
            else:
                c.print(f'🚨 Error: {self.name}::{fn} --> {input["address"]}... 🚨\033', color='red')
                

            result = self.process_result(result)
            return result
        
        self.serve()
        
        

    def state_dict(self) -> Dict:
        return {
            'ip': self.ip,
            'port': self.port,
            'address': self.address,
        }


    def test(self):
        r"""Test the HTTP server.
        """
        # Test the server here if needed
        c.print(self.state_dict(), color='green')
        return self

    
    def process_input(self,input: dict) -> bool:
        assert 'data' in input, f"Data not included"
        assert 'signature' in input, f"Data not signed"
        # you can verify the input with the server key class
        if self.public:
            pass
        else:
            assert self.key.verify(input), f"Data not signed with correct key"
            # deserialize the data
            input['data'] = self.serializer.deserialize(input['data'])
            # here we want to verify the data is signed with the correct key
            request_staleness = c.timestamp() - input['data'].get('timestamp', 0)

            # verifty the request is not too old
            assert request_staleness < self.max_request_staleness, f"Request is too old, {request_staleness} > MAX_STALENESS ({self.max_request_staleness})  seconds old"


        return input


    def process_result(self,  result):
        if self.sse:
            # for sse we want to wrap the generator in an eventsource response
            from sse_starlette.sse import EventSourceResponse
            result = self.generator_wrapper(result)
            return EventSourceResponse(result)
        else:
            # if we are not
            if c.is_generator(result):
                result = list(result)
            result = self.serializer.serialize({'data': result, 'public': self.public})
            if self.public:
                result = self.key.sign(result, return_json=True)
            return result
        
    
    def generator_wrapper(self, generator):
        if not c.is_generator(generator):   
            generator = [generator]
            
        for item in generator:
            # we wrap the item in a json object, just like the serializer does
            item = self.serializer.serialize({'data': item})
            item = self.key.sign(item, return_json=True)
            item = json.dumps(item)
            item_size = c.sizeof(item)
            if item_size > self.chunk_size:
                # if the item is too big, we need to chunk it
                item_hash = c.hash(item)
                chunks =[f'CHUNKSTART:{item_hash}'] + [item[i:i+self.chunk_size] for i in range(0, item_size, self.chunk_size)] + [f'CHUNKEND:{item_hash}']
                # we need to yield the chunks in a format that the eventsource response can understand
                for chunk in chunks:
                    yield chunk

            yield item


    def serve(self, **kwargs):
        import uvicorn

        try:
            c.print(f'\033🚀 Serving {self.name} on {self.address} 🚀\033')
            c.register_server(name=self.name, address = self.address, network=self.network)
            c.print(f'\033🚀 Registered {self.name} on {self.ip}:{self.port} 🚀\033')
            uvicorn.run(self.app, host=c.default_ip, port=self.port)
        except Exception as e:
            c.print(e, color='red')
            c.deregister_server(self.name, network=self.network)
        finally:
            c.deregister_server(self.name, network=self.network)
        

    def forward(self, fn: str, args: List = None, kwargs: Dict = None, **extra_kwargs):
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        obj = getattr(self.module, fn)
        if callable(obj):
            response = obj(*args, **kwargs)
        else:
            response = obj

        return response


    def __del__(self):
        c.deregister_server(self.name)

