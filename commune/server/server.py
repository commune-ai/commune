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
        chunk_size: int = 1000,
        max_request_staleness: int = 5, 
        key = None,
        verbose: bool = False,
        access_module: str = 'server.access',
        serializer: str = 'serializer',
        free: bool = False,
        access_token_feature : str = 'access_token',
        save_history:bool= True,
        history_path:str = None , 
        nest_asyncio = True,
        new_loop = True,
        **kwargs
        ) -> 'Server':

        if new_loop:
            c.new_event_loop(nest_asyncio=nest_asyncio)
        self.max_request_staleness = max_request_staleness
        self.network = network
        self.verbose = verbose
        self.chunk_size = chunk_size
        self.free = free
        self.save_history = save_history
        self.access_token_feature = access_token_feature
        self.serializer = c.module(serializer)()
        self.set_history_path(history_path)
        self.set_module(module, key=key,  name=name,  port=port,  access_module=access_module)

    def forward(self, fn:str, input:dict):
        """
        OPTION 1:
        fn (str): the function to call
        input (dict): the input to the function
            data: the data to pass to the function
                kwargs/params (Optional): the keyword arguments to pass to the function
                args (optional): the positional arguments to pass to the function
                timestamp: the timestamp of the request
                address: the address of the caller (ss58_address)
                signature: the signature of the request
        OPTION 2

        input (dict): the input to the function
            **kwargs, # the params
            access_token: {timestamp}::{address}::{signature}
   
        """
        user_info = None

        try:
            # you can verify the input with the server key class
            if 'signature' in input and 'data' in input:
                assert self.key.verify(input), f"Data not signed with correct key"
            elif 'access_token' in input:
                """
                module_tikcet:
                {timestamp}::signature::{signature}::address::{address}
                """
                assert self.key.verify(input['access_token']), f"Data not signed with correct key"
                timestamp = int(input['access_token'].split('::signature::')[0])
                if all([k not in input for k in ['kwargs', 'params', 'args']]):
                    """
                    We assume the data is in the input, and the token
                    """
                    kwargs = input
                    kwargspop('access_token')
                    input['kwargs'] = input

            if 'params' in input:
                # if the params are in the input, we want to move them to the data
                if isinstance(input['params'], dict):
                    input['kwargs'] = input['params']
                elif isinstance(input['params'], list):
                    input['args'] = input['params']
                del input['params']

            if 'args' in input and 'kwargs' in input:
                input['data'] = {'args': input['args'], 
                                 'kwargs': input['kwargs'], 
                                 'timestamp': input['timestamp'], 
                                 'address': input['address']}
                
            # deserialize the data
            input['data'] = self.serializer.deserialize(input['data'])
            
            # here we want to verify the data is signed with the correct key
            request_staleness = c.timestamp() - input['data'].get('timestamp', 0)
            assert request_staleness < self.max_request_staleness, f"Request is too old, {request_staleness} > MAX_STALENESS ({self.max_request_staleness})  seconds old"
            
            # verify the access module
            user_info = self.access_module.verify(fn=fn, address=input['address'])
            if not user_info['success']:
                return user_info
            assert 'args' in input['data'], f"args not in input data"
            data = input['data']
            args = data.get('args',[])
            kwargs = data.get('kwargs', {})
            fn_obj = getattr(self.module, fn)
            result = fn_obj(*args, **kwargs) if callable(fn_obj) else fn_obj
            success = bool(isinstance(result, dict) and 'error' in result) 

            # if the result is a future, we need to wait for it to finish
        except Exception as e:
            result = c.detailed_error(e)
            success = False 

        output = {
            'fn': fn,
            'input': input['data'],
            'output': result,
            'address': input['address'],
            'latency': c.time() - input['data']['timestamp'],
            'datetime': c.time2datetime(input['data']['timestamp']),
            'user_info': user_info,
            'timestamp': c.timestamp(),
            'success': success,

        }
        if not success:
            output['error'] = result
        result = self.process_result(result)

        if self.save_history:
            self.add_history(output)

        return result



    def set_module(self, module, 
                   key=None, 
                   name=None, 
                   port=None, 
                   access_module='server.access'):

        module = module or 'module'
        if isinstance(module, str):
            module = c.module(module)()
        # RESOLVE THE WHITELIST AND BLACKLIST
        whitelist = module.whitelist if hasattr(module, 'whitelist') else module.functions(include_parents=False)
        # Resolve the blacklist
        blacklist = self.blacklist if hasattr(self, 'blacklist') else []
        if name != None:
            module.server_name = name

        self.whitelist = list(set(whitelist + c.whitelist))
        self.blacklist = list(set(blacklist + c.blacklist))
        self.name = module.server_name
        self.module = module 
        self.ip = c.ip()
        port = port or c.free_port()
        while c.port_used(port):
            port =  c.free_port()
        self.port = port
        self.address = f"{self.ip}:{self.port}"
        module.address = self.address
        module.whitelist = whitelist
        module.blacklist = blacklist
        module.ip = self.ip
        module.port = self.port
        module.address  = self.address
        module.network = self.network
        module.subnet = self.subnet
        self.schema = module.schema() 
        self.key = self.module.key = c.get_key(key or self.name)
        self.access_module = c.module(access_module)(module=self.module)  
        self.set_api()
        return {'success': True, 'msg': f'Set module {module}', 'key': self.key.ss58_address}

    def add_fn(self, name:str, fn: str):
        assert callable(fn), 'fn not callable'
        setattr(self.module, name, fn)
           
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
            return self.forward(fn=fn, input=input)
        
        # start the server
        try:
            c.print(f' Served ( {self.name} --> {self.address} ) 🚀\033 ', color='purple')
            c.print(f'🔑 Key: {self.key} 🔑\033', color='yellow')
            c.register_server(name=self.name, address = self.address, network=self.network)
            uvicorn.run(self.app, host='0.0.0.0', port=self.port, loop="asyncio")
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


    def info(self) -> Dict:
        return {
            'name': self.name,
            'address': self.address,
            'key': self.key.ss58_address,
            'network': self.network,
            'port': self.port,
            'whitelist': self.whitelist,
            'blacklist': self.blacklist,
            'free': self.free,
            'save_history': self.save_history,
            
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
        """
        This function wraps a generator in a format that the eventsource response can understand
        """

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


    # HISTORY 
    def add_history(self, item:dict):    
        path = self.history_path + '/' + item['address'] + '/'+  str(item['timestamp']) 
        self.put(path, item)

    def set_history_path(self, history_path):
        self.history_path = self.resolve_path(history_path or f'history/{self.name}')
        return {'history_path': self.history_path}

    @classmethod
    def rm_history(cls, server=None, history_path='history'):
        dirpath  = f'{history_path}/{server}'
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
    
    

    def wait_for_server(self, timeout=10):
        return c.wait_for_server(self.name, timeout=timeout)
    

    
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

Server.run(__name__)