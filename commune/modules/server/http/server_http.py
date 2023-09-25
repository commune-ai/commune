
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
        ip: Optional[str] = None,
        port: Optional[int] = None,
        address: Optional[str] = None,
        sse: bool = True,
        chunk_size: int = 42_000,
        max_request_staleness: int = 60,
    ) -> 'Server':
        self.sse = sse
        
        self.serializer = c.module('serializer')()

        # RESOLVE THE IP AND PORT -> ADDRESS
        self.ip = c.ip()  # default to '0.0.0.0'
        self.port = c.resolve_port(port)
        self.address = f"0.0.0.0:{self.port}" if address == None else address
        self.max_request_staleness = max_request_staleness
        assert self.address != None, f"Address not set"

        # ensure that the module has a name

        if isinstance(module, str):
            module = c.module(module)()
        elif isinstance(module, type):
            module = module()
        # RESOLVE THE NAME OF THE SERVER
        self.name = module.server_name =  name if name != None else module.server_name
        self.module = module 
        self.key = module.key      
        # register the server
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
            address_abbrev = None
            try:

                input['fn'] = fn

                input = self.process_input(input)

                data = input['data']
                args = data.get('args',[])
                kwargs = data.get('kwargs', {})
                
                result = self.forward(fn=fn,
                                    args=args,
                                    kwargs=kwargs,
                                    )

                success = True
            except Exception as e:
                result = c.detailed_error(e)
                success = False
            
            result = self.process_result(result)
            

            if success:
                
                c.print(f'\033[32m Success: {self.name}::{fn} --> {input["address"][:5]}... 🎉\033 ')
            else:
                c.print(result)
                c.print(f'\033🚨 Error: {self.name}::{fn} --> {input["address"][:5]}... 🚨\033')

            # send result to
            #  client
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
        assert self.key.verify(input), f"Data not signed with correct key"
        # deserialize the data
        input['data'] = self.serializer.deserialize(input['data'])
        # here we want to verify the data is signed with the correct key
        request_staleness = c.timestamp() - input['data'].get('timestamp', 0)

        # verifty the request is not too old
        assert request_staleness < self.max_request_staleness, f"Request is too old, {request_staleness} > MAX_STALENESS ({self.max_request_staleness})  seconds old"

        # verify the input with the access module
        input = self.module.access_module.verify(input)

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
            result = self.serializer.serialize({'data': result})
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
            # # get size of chunk
            # item_size = len(item)
            # if item_size > chunk_size:
            #     # if the item is too big, we need to chunk it
            #     item_hash = c.hash(item)
            #     chunks =[f'CHUNKSTART:{item_hash}'] + [item[i:i+chunk_size] for i in range(0, item_size, chunk_size)] + [f'CHUNKEND:{item_hash}']
            #     # we need to yield the chunks in a format that the eventsource response can understand
            #     for chunk in chunks:
            #         yield chunk
            # else:
            yield item


    def serve(self, **kwargs):
        import uvicorn

        try:
            c.print(f'\033🚀 Serving {self.name} on {self.ip}:{self.port} 🚀\033')
            c.register_server(name=self.name, ip=self.ip, port=self.port)

            c.print(f'\033🚀 Registered {self.name} on {self.ip}:{self.port} 🚀\033')

            uvicorn.run(self.app, host=c.default_ip, port=self.port)
        except Exception as e:
            c.print(e, color='red')
            c.deregister_server(self.name)
        finally:
            c.deregister_server(self.name)
        

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

