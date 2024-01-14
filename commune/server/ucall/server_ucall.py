
from typing import Dict, List, Optional, Union
import commune as c
import torch 
import traceback
import json




class ServerUcall(c.Module):
    def __init__(
        self,
        module: Union[c.Module, object],
        name: str = None,
        network:str = 'local',
        port: Optional[int] = None,
        sse: bool = False,
        chunk_size: int = 42_000,
        max_request_staleness: int = 60, 
        verbose: bool = False,
        timeout: int = 256,
        access_module: str = 'server.access',
        public: bool = False,
        serializer: str = 'serializer',
        new_event_loop:bool = True
        ) -> 'Server':

        self.serializer = c.module(serializer)()
        self.ip = c.default_ip # default to '0.0.0.0'
        self.port = int(port) if port != None else c.free_port()
        self.address = f"{self.ip}:{self.port}"
        self.max_request_staleness = max_request_staleness
        self.network = network
        self.verbose = verbose
        self.sse = sse
        assert self.sse == False, f"SSE not implemented yet"
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.public = public
        self.module = module 
        if new_event_loop:
            c.new_event_loop()

        # name 
        if name == None:
            if hasattr(self.module, 'server_name'):
                name = self.module.server_name
            else:
                name = self.module.__class__.__name__
        self.name = name


        self.key = module.key      
        # register the server
        module.ip = self.ip
        module.port = self.port
        module.address  = self.address
        self.access_module = c.module(access_module)(module=self.module)  


        self.set_api(ip=self.ip, port=self.port)


    def set_api(self, ip:str = '0.0.0.0', port:int = 8888):
        ip = self.ip if ip == None else ip
        port = self.port if port == None else port


        from ucall.posix import Server
        # from ucall.uring import Server on 5.19+
        self.app = Server(port=self.port)
        @self.app
        def remote_call(fn:str, input:dict):
            """
            THE ULTIMATE RPC CALL

            fn (str): the function to call
            input (dict): the input to the function
                data (dict): the data to pass to the function
                    kwargs (dict): the keyword arguments to pass to the function
                    args (list): the positional arguments to pass to the function
                    timestamp (int/float): the timestamp of the request
                signature (str): the signature of the data request 

            """
            input['fn'] = fn
            input = self.process_input(input)
            data = input['data']
            args = data.get('args',[])
            kwargs = data.get('kwargs', {})
            
            input_kwargs = dict(fn=fn, 
                                args=args, 
                                kwargs=kwargs)
            fn_name = f"{self.name}::{fn}"
            c.print(f'🚀 Forwarding {input["address"]} --> {fn_name} 🚀\033', color='yellow')


            try:
                result = self.forward(**input_kwargs)
                # if the result is a future, we need to wait for it to finish
            except Exception as e:
                result = c.detailed_error(e)
                
            if isinstance(result, dict) and 'error' in result:
                success = False 
            success = True


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


    
    def process_input(self,input: dict) -> bool:

        """
        INPUT
        {
            'data': {
                'args': [],
                'kwargs': {},
                'timestamp': 0,
            },
        }
        
        
        """
        assert 'data' in input, f"Data not included"

        # you can verify the input with the server key class
        if not self.public:
            assert 'signature' in input, f"Data not signed"
            assert self.key.verify(input), f"Data not signed with correct key"
            input['data'] = self.serializer.deserialize(input['data'])

        if self.verbose:
            # here we want to verify the data is signed with the correct key
            request_staleness = c.timestamp() - input['data'].get('timestamp', 0)
            # verifty the request is not too old
            assert request_staleness < self.max_request_staleness, f"Request is too old, {request_staleness} > MAX_STALENESS ({self.max_request_staleness})  seconds old"
            self.access_module.verify(input)
        

        return input


    def process_result(self,  result):
        if self.sse:
            # from sse_starlette.sse import EventSourceResponse
            # # for sse we want to wrap the generator in an eventsource response
            # result = self.generator_wrapper(result)
            # return EventSourceResponse(result)
            assert False, f"SSE not implemented yet"
        else:
            # if we are not using sse, then we can do this with json
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
            c.print(f'\033🚀 Registered {self.name} --> {self.ip}:{self.port} 🚀\033')
            self.app.run()
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


    @classmethod
    def test(cls):
        self = cls(module=c.module("module")())
        


    @classmethod
    def install(cls):
        return c.cmd("pip3 install ucall")
