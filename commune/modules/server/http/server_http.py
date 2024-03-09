
from typing import Dict, List, Optional, Union
import commune as c
import torch 
import traceback
import json 

class ServerHTTP(c.Module):
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
        max_workers: int = None,
        mode:str = 'thread',
        verbose: bool = False,
        timeout: int = 256,
        access_module: str = 'server.access',
        public: bool = False,
        serializer: str = 'serializer',
        save_history:bool= True,
        history_path:str = None , 
        nest_asyncio = True,
        new_loop = True,
        
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
        self.key = module.key      
        module.ip = self.ip
        module.port = self.port
        module.address  = self.address
        self.module = module 
        self.access_module = c.module(access_module)(module=self.module)  
        self.set_history_path(history_path)
        self.set_api(ip=self.ip, port=self.port)






    def set_address(self,ip='0.0.0.0', port:int=None):
        if '://' in ip:
            assert ip.startswith('http'), f"Invalid ip {ip}"
            ip = ip.split('://')[1]
            
        self.ip = ip
        self.port = int(port) if port != None else c.free_port()
        while c.port_used(self.port):
            self.port = c.free_port()
        self.address = f"http://{self.ip}:{self.port}"
    def forward(self, fn:str, input:dict):
        """
        fn (str): the function to call
        input (dict): the input to the function
            data: the data to pass to the function
                kwargs: the keyword arguments to pass to the function
                args: the positional arguments to pass to the function
                timestamp: the timestamp of the request
                address: the address of the caller

            signature: the signature of the request
   
        """
        user_info = None
        try:
            input['fn'] = fn
            # you can verify the input with the server key class
            if not self.public:
                assert self.key.verify(input), f"Data not signed with correct key"


            if 'args' in input and 'kwargs' in input:
                input['data'] = {'args': input['args'], 
                                 'kwargs': input['kwargs'], 
                                 'timestamp': input['timestamp'], 
                                 'address': input['address']}
            input['data'] = self.serializer.deserialize(input['data'])
            # here we want to verify the data is signed with the correct key
            request_staleness = c.timestamp() - input['data'].get('timestamp', 0)
            # verifty the request is not too old
            assert request_staleness < self.max_request_staleness, f"Request is too old, {request_staleness} > MAX_STALENESS ({self.max_request_staleness})  seconds old"
            
            # verify the access module
            user_info = self.access_module.verify(input)
            if not user_info['passed']:
                return user_info
            assert 'args' in input['data'], f"args not in input data"

            data = input['data']
            args = data.get('args',[])
            kwargs = data.get('kwargs', {})

            fn_name = f"{self.name}::{fn}"

            c.print(f'🚀 Forwarding {input["address"]} --> {fn_name} 🚀\033', color='yellow')
            
            fn_obj = getattr(self.module, fn)
            
            if callable(fn_obj):
                result = fn_obj(*args, **kwargs)
            else:
                result = fn_obj

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
        

        if self.save_history:

            output = {
            'module': self.name,
            'fn': fn,
            'timestamp': input['data']['timestamp'],
            'address': input['address'],
            'args': input['data']['args'],
            'kwargs': input['data']['kwargs'],
            'result': None if self.sse else result,
            'user': user_info,

            }
            output.update(output.pop('data', {}))
            output['latency'] = c.time() - output['timestamp']
            self.add_history(output)

        return result


    def set_api(self, ip:str = '0.0.0.0', port:int = 8888):
        ip = self.ip if ip == None else ip
        port = self.port if port == None else port
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        import uvicorn
        
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
            return self.forward(fn=fn, input=input)
        
        try:
            c.print(f'\033🚀 Serving {self.name} on {self.address} 🚀\033')
            c.register_server(name=self.name, address = self.address, network=self.network)
            c.print(f'\033🚀 Registered {self.name} --> {self.ip}:{self.port} 🚀\033')
            uvicorn.run(self.app, host=c.default_ip, port=self.port, loop="asyncio")
        except Exception as e:
            c.print(e, color='red')
            c.deregister_server(self.name, network=self.network)
        finally:
            c.deregister_server(self.name, network=self.network)
        

    @classmethod
    def history_paths(cls, server=None, history_path='history', n=100, key=None):
        if server == None:
            dirpath  = f'{history_path}'
            paths =  cls.glob(dirpath)
        else:
            
            dirpath  = f'{history_path}/{server}'
            paths =  cls.ls(dirpath)
        paths = sorted(paths, reverse=True)[:n]
        return paths


    def state_dict(self) -> Dict:
        return {
            'ip': self.ip,
            'port': self.port,
            'address': self.address,
        }


    


    def process_result(self,  result):
        if c.is_generator(result):
            from sse_starlette.sse import EventSourceResponse
            # for sse we want to wrap the generator in an eventsource response
            result = self.generator_wrapper(result)
            return EventSourceResponse(result)
        else:
            # if we are not using sse, then we can do this with json
            result = self.serializer.serialize(result)
            result = self.key.sign(result, return_json=True)
            return result
        
    
    def generator_wrapper(self, generator):

        for item in generator:
 
            # we wrap the item in a json object, just like the serializer does
            item = self.serializer.serialize({'data': item})
            item_size = len(str(item))
            # we need to add a chunk start and end to the item
            if item_size > self.chunk_size:
                # if the item is too big, we need to chunk it
                chunks =[item[i:i+self.chunk_size] for i in range(0, item_size, self.chunk_size)]
                # we need to yield the chunks in a format that the eventsource response can understand
                for chunk in chunks:
                    yield chunk
            else:
                yield item




    @classmethod
    def test(cls):
        module_name = 'storage::test'
        module = c.serve(module_name, wait_for_server=True)
        module = c.connect(module_name)
        module.put("hey",1)
        c.kill(module_name)



    # HISTORY 
    def add_history(self, item:dict):    
        path = self.history_path + '/' + item['address'] + '/'+  str(item['timestamp']) 
        self.put(path, item)

    def set_history_path(self, history_path):
        self.history_path = history_path or f'history/{self.name}'
        return {'history_path': self.history_path}

    @classmethod
    def rm_history(cls, server=None, history_path='history'):
        dirpath  = f'{history_path}/{server}'
        return cls.rm(dirpath)
    
    @classmethod
    def rm_all_history(cls, server=None, history_path='history'):
        dirpath  = f'{history_path}'
        return cls.rm(dirpath)


    @classmethod
    def history(cls, 
                key=None, 
                history_path='history',
                features=[ 'module', 'fn', 'seconds_ago', 'latency', 'address'], 
                to_list=False,
                **kwargs
                ):
        key = c.get_key(key)
        history_path = cls.history_paths(key=key, history_path=history_path)
        df =  c.df([cls.get(path) for path in history_path])
        now = c.timestamp()
        df['seconds_ago'] = df['timestamp'].apply(lambda x: now - x)
        df = df[features]
        
        if to_list:
            return df.to_dict('records')

        return df
    

    def __del__(self):
        c.deregister_server(self.name)
