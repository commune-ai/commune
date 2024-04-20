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
        sse: bool = True,
        chunk_size: int = 1000,
        max_request_staleness: int = 2, 
        key = None,
        verbose: bool = False,
        timeout: int = 256,
        access_module: str = 'server.access',
        free: bool = False,
        serializer: str = 'serializer',
        save_history:bool= True,
        history_path:str = None , 
        nest_asyncio = True,
        mnemonic = None,
        new_loop = True,
        subnet = None,
        **kwargs
        ) -> 'Server':

        if new_loop:
            c.new_event_loop(nest_asyncio=nest_asyncio)

        self.ip = c.ip()
        port = port or c.free_port()
        while c.port_used(port):
            port =  c.free_port()
        self.port = port
        self.address = f"http://{self.ip}:{self.port}"
        self.max_request_staleness = max_request_staleness
        self.network = network
        self.verbose = verbose
        self.sse = sse
        self.save_history = save_history
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.free = free
        self.key = key
        self.serializer = c.module(serializer)()
        self.set_module(module)
        self.access_module = c.module(access_module)(module=self.module)  
        self.set_history_path(history_path)
        self.set_api(port=self.port)

    def set_module(self, module, key=None):

        module = module or 'module'
        if isinstance(module, str):
            module = c.module(module)()
        # RESOLVE THE WHITELIST AND BLACKLIST
        whitelist = module.whitelist if hasattr(module, 'whitelist') else module.functions(include_parents=False)
        # Resolve the blacklist
        blacklist = self.blacklist if hasattr(self, 'blacklist') else []

        self.whitelist = list(set(whitelist + c.whitelist))
        self.blacklist = list(set(blacklist + c.blacklist))
        self.name = module.server_name
        self.schema = module.schema() 
        self.module = module 

        module.whitelist = whitelist
        module.blacklist = blacklist
        module.ip = self.ip
        module.port = self.port
        module.address  = self.address
        module.network = self.network
        module.subnet = self.subnet

 

        return {'success': True, 'msg': f'Set module {module}'}

        if self.key == None:
            key = c.get_key(self.name)
        if isinstance(key, str):
            key = c.get_key(key)  
        self.key = self.module.key = key
        c.print(f'🔑 Key: {self.key} 🔑\033')


    def forward(self, fn:str, input:dict):
        """
        fn (str): the function to call
        input (dict): the input to the function
            data: the data to pass to the function
                kwargs: the keyword arguments to pass to the function
                args: the positional arguments to pass to the function
                timestamp: the timestamp of the request
                address: the address of the caller
            hash: the hash of the request (optional)
            signature: the signature of the request
   
        """
        user_info = None
        color = c.random_color()

        try:
            input['fn'] = fn
            # you can verify the input with the server key class
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
            user_info = self.access_module.verify(fn=input['fn'], address=input['address'])
            if not user_info['success']:
                return user_info
            assert 'args' in input['data'], f"args not in input data"

            data = input['data']
            args = data.get('args',[])
            kwargs = data.get('kwargs', {})
            
            fn_obj = getattr(self.module, fn)
            
            if callable(fn_obj):
                result = fn_obj(*args, **kwargs)
            else:
                result = fn_obj

            if isinstance(result, dict) and 'error' in result:
                success = False 
            else:
                success = True

            # if the result is a future, we need to wait for it to finish
        except Exception as e:
            result = c.detailed_error(e)
            success = False 




        print_info = {
            'fn': fn,
            'address': input['address'],
            'latency': c.time() - input['data']['timestamp'],
            'datetime': c.time2datetime(input['data']['timestamp']),
            'success': success,
        }
        if not success:
            print_info['error'] = result

        c.print(print_info, color=color)
        

        result = self.process_result(result)
    
        output = {
        'module': self.name,
        'fn': fn,
        'address': input['address'],
        'args': input['data']['args'],
        'kwargs': input['data']['kwargs'],
        }
        if self.save_history:

            output.update(
                {
                    'success': success,
                    'user': user_info,
                    'timestamp': input['data']['timestamp'],
                    'result': result if c.jsonable(result) else None,
                }
            )

            output.update(output.pop('data', {}))
            output['latency'] = c.time() - output['timestamp']
            self.add_history(output)

        return result

    def set_api(self, port:int = 8888):
        
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
            c.print(f'\033🚀 Registered {self.name} --> {self.address} 🚀\033')
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
    def test_serving(cls):
        module_name = 'storage::test'
        module = c.serve(module_name)
        module = c.connect(module_name)
        module.put("hey",1)
        assert module.get("hey") == 1, f"get failed {module.get('hey')}"
        c.kill(module_name)
        return {'success': True, 'msg': 'server test passed'}


    @classmethod
    def test_serving_with_different_key(cls):
        module_name = 'storage::test'
        module = c.serve(module_name)
        module = c.connect(module_name)
        module.put("hey",1)
        c.kill(module_name)


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
    

    def __del__(self):
        c.deregister_server(self.name)

    @classmethod
    def test_basics(cls) -> dict:
        servers = c.servers()
        c.print(servers)
        tag = 'test'
        module_name = c.serve(module='module', tag=tag)['name']
        c.wait_for_server(module_name)
        assert module_name in c.servers()

        response = c.call(module_name)

        c.kill(module_name)
        assert module_name not in c.servers()
        return {'success': True, 'msg': 'server test passed'}

    
    @classmethod
    def serve(cls, 
              module:Any = None ,
              tag:str=None,
              network = 'local',
              port :int = None, # name of the server if None, it will be the module name
              server_name:str=None, # name of the server if None, it will be the module name
              kwargs:dict = None,  # kwargs for the module
              refresh:bool = True, # refreshes the server's key
              remote:bool = True, # runs the server remotely (pm2, ray)
              tag_seperator:str='::',
              max_workers:int = None,
              mode:str = "http",
              public: bool = False,
              mnemonic = None,
              key = None,
              **extra_kwargs
              ):
        kwargs = kwargs or {}
        kwargs.update(extra_kwargs or {})
        module = module or cls.module_path()

        if tag_seperator in module:
            module, tag = module.split(tag_seperator)

        # resolve the server name ()
        server_name = cls.resolve_server_name(module=module, name=server_name, tag=tag, tag_seperator=tag_seperator)
        
        if tag_seperator in server_name:
            module, tag = server_name.split(tag_seperator)

        # RESOLVE THE PORT FROM THE ADDRESS IF IT ALREADY EXISTS
        if port == None:
            # now if we have the server_name, we can repeat the server
            address = c.get_address(server_name, network=network)
            if address != None :
                port = int(address.split(':')[-1])
            else:
                port = c.free_port()

        # NOTE REMOVE THIS FROM THE KWARGS REMOTE
        if remote:
            # GET THE LOCAL KWARGS FOR SENDING TO THE REMOTE
            remote_kwargs = c.locals2kwargs(locals(), merge_kwargs=False)
            # SET THIS TO FALSE TO AVOID RECURSION
            remote_kwargs['remote'] = False 
            # REMOVE THE LOCALS FROM THE REMOTE KWARGS THAT ARE NOT NEEDED
            for _ in ['extra_kwargs', 'address']:
                remote_kwargs.pop(_, None) # WE INTRODUCED THE ADDRES
            
            cls.remote_fn('serve',name=server_name, kwargs=remote_kwargs)

            return {'success':True, 
                    'name': server_name, 
                    'address':c.ip() + ':' + str(remote_kwargs['port']), 
                    'kwargs':kwargs}
        
        module_class = cls.resolve_module(module)
        
        kwargs.update(extra_kwargs)

        if mnemonic != None:
            c.add_key(server_name, mnemonic)

        if module_class.is_arg_key_valid('tag'):
            kwargs['tag'] = tag
        if module_class.is_arg_key_valid('server_name'):
            kwargs['server_name'] = server_name

        # start the class
        self = module_class(**kwargs)

        self.server_name = server_name

        if tag_seperator in server_name:
            tag = server_name.split(tag_seperator)[-1]
        else:
            tag = None

        self.tag = tag
        self.key = server_name

        address = c.get_address(server_name, network=network)
        if address != None and ':' in address:
            port = address.split(':')[-1]   

        if c.server_exists(server_name, network=network) and not refresh: 
            return {'success':True, 'message':f'Server {server_name} already exists'}

        self(module=self, 
            name=server_name, 
            port=port, 
            network=network, 
            max_workers=max_workers, 
            mode=mode, 
            public=public, 
            key=key)

        return  {'success':True, 
                     'address':  f'{c.default_ip}:{port}' , 
                     'name':server_name, 
                     'kwargs': kwargs,
                     'module':module}

    @staticmethod
    def resolve_function_access(module):
        # RESOLVE THE WHITELIST AND BLACKLIST
        whitelist = module.whitelist if hasattr(module, 'whitelist') else []
        if len(whitelist) == 0 and module != 'module':
            whitelist = module.functions(include_parents=False)
        whitelist = list(set(whitelist + c.helper_functions))
        blacklist = module.blacklist if hasattr(module, 'blacklist') else []
        setattr(module, 'whitelist', whitelist)
        setattr(module, 'blacklist', blacklist)
        return module
    
    

Server.run(__name__)