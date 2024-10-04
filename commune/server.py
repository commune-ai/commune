import commune as c
from typing import *
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from sse_starlette.sse import EventSourceResponse
import uvicorn
import os
import asyncio

class ServerMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_bytes: int):
        super().__init__(app)
        self.max_bytes = max_bytes
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get('content-length')
        if content_length:
            if int(content_length) > self.max_bytes:
                return JSONResponse(status_code=413, content={"error": "Request too large"})
        body = await request.body()
        if len(body) > self.max_bytes:
            return JSONResponse(status_code=413, content={"error": "Request too large"})
        response = await call_next(request)
        return response

class Server(c.Module):
    def __init__(
        self, 
        module: Union[c.Module, object] = None,
        functions:Optional[List[str]] = None, # list of endpoints
        key:str = None, # key for the server (str)
        name: str = None, # the name of the server
        port: Optional[int] = None, # the port the server is running on
        network:str = 'subspace', # the network used for incentives
        helper_functions  = ['info',  'metadata', 'schema', 'server_name', 'functions', 'forward', 'rate_limit'], 
        user_functions = ['user_count', 'user_paths', 'user_info', 'user_data', 'user_info', 'user_functions'],
        functions_attributes = ['helper_functions', 'whitelist','endpoints', 'server_functions'],
        max_bytes:int = 10 * 1024 * 1024,  # max bytes within the request (bytes)
        allow_origins = ["*"], # allowed origins
        allow_credentials=True, # allow credentials
        allow_methods=["*"], # allowed methods
        allow_headers=["*"],  # allowed headers
        period = 60, # the period for 
        fn2cost = None, # the cost of the function
        max_request_staleness = 5, # the time it takes for the request to be too old
        max_network_staleness = 60, # the time it takes for the network to refresh
        data_path = None, # the path to store the data
        **kwargs,
        ) -> 'Server':
        module = module or 'module'
        if isinstance(module, str):
            name = module
            module = c.module(module)()

        module.name = module.server_name = name
        module.fn2cost = fn2cost or {}
        module.port = module.server_port =  port if port not in ['None', None] else c.free_port()
        module.address  = module.server_address =  f"{c.ip()}:{module.port}"
        module.key  = module.server_key = c.get_key(key or module.name, create_if_not_exists=True)
        self.user_functions = user_functions
        for fn in self.user_functions:
            setattr(module, fn, getattr(self, fn))
        module.helper_functions = list(helper_functions + user_functions)
        module.schema =  module.server_schema = self.get_server_schema(module, functions_attributes=functions_attributes)
        module.functions  = module.server_functions = list(set(module.helper_functions + (functions or  list(module.schema.keys()))))
        module.info  =  module.server_info =  self.get_server_info(module)

        self.data_path = self.resolve_path(data_path or f'data/{module.server_name}')
        self.max_request_staleness = max_request_staleness
        self.serializer = c.module('serializer')()
        self.network = network
        self.module = module
        self.period = period
        self.max_network_staleness = max_network_staleness
        self.loop = asyncio.get_event_loop()
        app = FastAPI()    
        app.add_middleware(ServerMiddleware, max_bytes=max_bytes)    
        app.add_middleware(CORSMiddleware, allow_origins=allow_origins, allow_credentials=allow_credentials, allow_methods=allow_methods,allow_headers=allow_headers)
        def api_forward(fn:str, request: Request):
            return self.forward(fn, request)
        app.post("/{fn}")(api_forward)
        c.print(f' Served(name={module.name}, address={module.address}, key=🔑{module.key}🔑 ) 🚀 ', color='purple')
        c.register_server(name=module.name,address = module.address)
        c.thread(self.sync_network_loop) # start the network loop
        uvicorn.run(app, host='0.0.0.0', port=module.port, loop='asyncio')

    def __del__(self):
        c.deregister_server(self.name)

    @classmethod
    def serve(cls, 
              module: Any = None,
              kwargs:Optional[dict] = None,  # kwargs for the module
              params: Optional[dict] = None, # kwargs for the module
              tag:Optional[str]=None,
              network: Optional[str] = 'subspace', # network to run the server
              port :Optional[int] = None, # name of the server if None, it will be the module name
              server_name:str=None, # name of the server if None, it will be the module name
              name = None, # name of the server if None, it will be the module name
              remote:bool = True, # runs the server remotely (pm2, ray)
              tag_seperator:str='::',
              max_workers:int = None,
              public: bool = False,
              mnemonic = None, # mnemonic for the server
              key = None,
              **extra_kwargs
              ):
        module = module or 'module'
        kwargs = {**(params or kwargs or {}), **extra_kwargs}
        name = (name or server_name or module) or c.module_name()
        if tag_seperator in name:
            module, tag = name.split(tag_seperator)
        if tag != None:
            name = f'{module}{tag_seperator}{tag}'
        if port == None:
            namespace = c.get_namespace()
            if name in namespace:
                port = int(namespace.get(name).split(':')[-1])
            else:
                port = c.free_port()
        if c.port_used(port):
            c.kill_port(port)
            c.kill(name)

        response =  { 'module':module, 
                     'name': name, 
                     'address':f'0.0.0.0:{port}',
                       'kwargs':kwargs, 
                       'port': port} 
        if remote:
            remote = False
            remote_kwargs = c.locals2kwargs(locals())  # GET THE LOCAL KWARGS FOR SENDING TO THE REMOTE
            for _ in ['extra_kwargs', 'address', 'response', 'namespace']:
                remote_kwargs.pop(_, None) # WE INTRODUCED THE ADDRES
            c.remote_fn('serve', name=name, kwargs=remote_kwargs)
            print(f'Serving {name} remotely')
            return response
        cls(module=c.module(module)(**kwargs), 
            name=name, 
            port=port, 
            network=network,   
            max_workers=max_workers, 
            mnemonic = mnemonic,
            public=public, 
            key=key)
        return  response
        
    def rate_limit(self, 
                   address, 
                   module=None, 
                   fn = 'info',
                    multipliers = {'network': 1, 'direct': 1,'validator': 1 },
                    rates = {'max': 10, 'stake2rate': 1000, 'admin': 1000}, # the maximum rate
                    ) -> float:
        if not hasattr(self, 'state'):
            self.state = {'sync_time': 0,  'stake': {},'stake_from': {},  'fn_info': {}, 'stake_to': {}}
            c.thread(self.sync_network_loop())
        # stake rate limit
        module = module or self.module
        key2address = c.key2address()
        address = key2address.get(address, address)
        if c.is_admin(address) or address == module.key.ss58_address or address in self.address2key:
            return rates['admin']
        validator_stake = self.state['stake'].get(address, 0) + multipliers['validator']
        network_stake = (sum(self.state['stake_to'].get(address, {}).values())) * multipliers['network']
        direct_stake = self.state['stake_from'].get(module.key.ss58_address, {}).get(address, 0) * multipliers['direct']
        stake = validator_stake + network_stake + direct_stake
        rate_limit = min((stake / rates['stake2rate']), rates['max'])# convert the stake to a rate
        return rate_limit

    def user_folder(self, address):
        return self.resolve_path(f'{self.data_path}/{address}')

    def user_paths(self, address):
        user_folder = self.user_folder(address)
        if not os.path.exists(user_folder):
            return []
        user_paths = os.listdir(user_folder)
        return sorted(user_paths, key=self.extract_timestamp)
    
    def user_data(self, address):
        user_folder = self.user_folder(address)
        user_paths = self.user_paths(address)
        data = []
        for user_path in user_paths:
            data += [c.get(user_folder + '/' + user_path)]
        return data
    
    def user_count(self, address):
        user_path2timestamp = self.user_path2timestamp(address)
        current_timestamp = c.timestamp()
        for p, ts in user_path2timestamp.items():
            if current_timestamp - ts > self.period and os.path.exists(p):
                os.remove(p)
        return len(self.user_paths(address))
    
    def user_info(self, address):
        user_folder = self.user_folder(address)
        user_paths = os.listdir(user_folder)
        user_data = []
        for user_path in user_paths:
            user_data += [c.get(user_folder + '/' + user_path)]
        return user_data

    def extract_timestamp(self, x):
        try:
            x = float(x.split('timestamp=')[-1].split('_')[0])
        except Exception as e:
            print(e)
            x = 0
        return x
    def user_path2timestamp(self, address):
        user_paths = self.user_paths(address)
        user_path2timestamp = {user_path: self.extract_timestamp(user_path) for user_path in user_paths}
        return user_path2timestamp
    
    def clear_user_data(self, address):
        return c.rm(self.user_folder(address))

    def forward(self, fn:str, request: Request, catch_exception:bool=True) -> dict:
        if catch_exception:
            try:
                return self.forward(fn, request, catch_exception=False)
            except Exception as e:
                return c.detailed_error(e)
        color = c.random_color()
        headers = dict(request.headers.items())
        address = headers.get('key', headers.get('address', None))
        assert address, 'No key or address in headers'
        request_staleness = c.timestamp() - int(headers['timestamp'])
        assert  request_staleness < self.max_request_staleness, f"Request is too old ({request_staleness}s > {self.max_request_staleness}s (MAX)" 
        data = self.loop.run_until_complete(request.json())
        data = self.serializer.deserialize(data) 
        request = {'data': data, 'headers': headers}
        headers['hash'] = c.hash(data)
        auth={'data': headers['hash'], 'timestamp': headers['timestamp']}
        signature = headers.get('signature', None)
        assert c.verify(auth=auth,signature=signature, address=address), 'Invalid signature'
        kwargs = dict(data.get('kwargs', {}))
        args = list(data.get('args', []))
        c.print(f'REQUESTING :: 🚀User(address={address[:3]}..) ----> Request(fn={fn} args={args} kwargs={kwargs})🚀', color= color)
        if 'params' in data:
            if isinstance(data['params', dict]):
                kwargs = {**kwargs, **data['params']}
            elif isinstance(data['params'], list):
                args = [*args, *data['params']]
            else:
                raise ValueError('params must be a list or a dictionary')
            
        data = {'args': args, 'kwargs': kwargs}
        is_admin = bool(c.is_admin(address) or address == self.module.key.ss58_address or address in self.address2key)
        if not is_admin:
            assert not bool(fn.startswith('__') or fn.startswith('_')), f'Function {fn} is private'
            assert fn in self.module.server_functions , f"Function {fn} not in endpoints={self.module.server_functions}"
        user_folder = self.user_folder(address)
        count = self.user_count(address)
        rate_limit = self.rate_limit(module=self.module, fn=fn, address=address)
        assert count <= rate_limit, f'rate limit exceeded {count} > {rate_limit}'
        fn_obj = getattr(self.module, fn)
        is_user_fn = fn in self.user_functions
        if callable(fn_obj):
            if is_user_fn:
                data['args'] = [address] + data['args']
            result = fn_obj(*data['args'], **data['kwargs'])
        else:
            result = fn_obj

        timestamp = int(headers['timestamp'])
        latency = c.time() - timestamp
    
        user_data = {
            'module': self.module.server_name,
            'fn': fn,
            'data': data, # the data of the request
            'latency':  c.round(c.time() - int(timestamp), 3), # the latency
            'timestamp': timestamp, # the timestamp of the request
            'user_key': address, # the key of the user
            'server_key': self.module.key.ss58_address, # the key of the server
            'result': [], # the response
            'cost': self.module.fn2cost.get(fn, 1), # the cost of the function
            'is_user_fn': is_user_fn, # is the function a user function
        }

        c.print(f'✅Response(fn={fn} speed={latency}s) --> User(key={address}..)✅', color=color)

        if c.is_generator(result):
            def generator_wrapper(generator):
                for item in generator:
                    output_item = self.serializer.serialize(item)
                    user_data['result'] += [output_item]
                    yield output_item
            result = EventSourceResponse(generator_wrapper(result))
        else:
            result =  self.serializer.serialize(result)
            user_data['result'] = result
        self.save_user_data(user_data)
        return result

    def save_user_data(self, user_data):
        is_user_fn = user_data['is_user_fn']
        if is_user_fn:
            user_data['data'], user_data['result'] = None, None
        user_folder = self.user_folder(user_data['user_key'])
        fn = user_data['fn']
        timestamp = user_data['timestamp']
        user_path = user_folder + f'/timestamp={timestamp}_fn={fn}.json' # get the user info path
        return c.put(user_path, user_data)
    sync_network_loop_running = False
    def sync_network_loop(self):
        self.sync_network_loop_running = True
        while True:
            try:
                r = self.sync_network()
            except Exception as e:
                r = c.detailed_error(e)
            c.print(r)
            c.sleep(self.max_network_staleness)

    def sync_network(self, update=False):
        network_path = self.resolve_path(f'{self.network}/state.json')
        state = self.get(network_path, {}, max_age=self.max_network_staleness)
        network = self.network
        staleness = c.time() - state.get('sync_time', 0)
        self.address2key = c.address2key()
        response = { 'path': network_path,  'max_network_staleness':  self.max_network_staleness,  'network': network,'staleness': int(staleness), }
        if staleness < self.max_network_staleness:
            response['msg'] = f'synced too earlly waiting {self.max_network_staleness - staleness} seconds'
            return response
        else:
            response['msg'] =  'Synced with the network'
            response['staleness'] = 0
        c.get_namespace(max_age=self.max_network_staleness)
        self.subspace = c.module('subspace')(network=network)
        state['stake_from'] = self.subspace.stake_from(fmt='j', update=update, max_age=self.max_network_staleness)
        state['stake_to'] = self.subspace.stake_to(fmt='j', update=update, max_age=self.max_network_staleness)
        state['stake'] =  {k: sum(v.values()) for k,v in state['stake_from'].items()}
        self.state = state
        self.put(network_path, self.state)
        return response

    @classmethod
    def wait_for_server(cls,
                          name: str ,
                          network: str = 'local',
                          timeout:int = 600,
                          sleep_interval: int = 1, 
                          verbose:bool = False) -> bool :
        
        time_waiting = 0
        while time_waiting < timeout:
            namespace = c.get_namespace(network=network)
            if name in namespace:
                c.print(f'{name} is ready', color='green')
                return True
            time_waiting += sleep_interval
            c.print(f'Waiting for {name} for {time_waiting} seconds', color='red')
            c.sleep(sleep_interval)
        raise TimeoutError(f'Waited for {timeout} seconds for {name} to start')

 # the default
    network : str = 'local'

    def is_endpoint(self, fn) -> bool:
        if isinstance(fn, str):
            fn = getattr(self, fn)
        return hasattr(fn, '__metadata__')

    @classmethod
    def endpoint(cls, 
                 cost=1, # cost per call 
                 user2rate : dict = None, 
                 rate_limit : int = 100, # calls per minute
                 timestale : int = 60,
                 public:bool = False,
                 cost_keys = ['cost', 'w', 'weight'],
                 **kwargs):
        
        for k in cost_keys:
            if k in kwargs:
                cost = kwargs[k]
                break

        def decorator_fn(fn):
            metadata = {
                **cls.fn_schema(fn),
                'cost': cost,
                'rate_limit': rate_limit,
                'user2rate': user2rate,   
                'timestale': timestale,
                'public': public,            
            }
            import commune as c
            fn.__dict__['__metadata__'] = metadata

            return fn

        return decorator_fn
    

    def get_server_info(self , 
             module,
             **kwargs
             ) -> Dict[str, Any]:
        '''
        hey, whadup hey how is it going
        '''
        info = {}
        info['schema'] = module.schema
        info['name'] = module.name 
        info['address'] = module.address
        info['key'] = module.key.ss58_address
        return info

    def get_server_schema(self, module,  functions_attributes) -> 'Schema':
        schema = {}
        functions =  []
        for k in functions_attributes:
            if hasattr(module, k):
                fn_obj = getattr(module, k)
                if isinstance(fn_obj, list):
                    functions += fn_obj
        for f in dir(module):
            try:
                if hasattr(getattr(module, f), '__metadata__'):
                    functions.append(f)
            except Exception as e:
                print(f'Error in get_endpoints: {e} for {f}')
        fns =  sorted(list(set(functions)))

        for fn in fns:
            if hasattr(module, fn):
                
                fn_obj = getattr(module, fn )
                if callable(fn_obj):
                    schema[fn] = c.fn_schema(fn_obj)  
                else: 
                    schema[fn] = {'type': str(type(fn_obj)).split("'")[1]}    
        # sort by keys
        schema = dict(sorted(schema.items()))
        return schema

    def add_endpoint(self, name, fn):
        setattr(self, name, fn)
        self.endpoints.append(name)
        assert hasattr(self, name), f'{name} not added to {self.__class__.__name__}'
        return {'success':True, 'message':f'Added {fn} to {self.__class__.__name__}'}

    @classmethod
    def endpoint(cls, 
                 cost=1, # cost per call 
                 user2rate : dict = None, 
                 rate_limit : int = 100, # calls per minute
                 timestale : int = 60,
                 public:bool = False,
                 cost_keys = ['cost', 'w', 'weight'],
                 **kwargs):
        
        for k in cost_keys:
            if k in kwargs:
                cost = kwargs[k]
                break

        def decorator_fn(fn):
            metadata = {
                **cls.fn_schema(fn),
                'cost': cost,
                'rate_limit': rate_limit,
                'user2rate': user2rate,   
                'timestale': timestale,
                'public': public,            
            }
            import commune as c
            fn.__dict__['__metadata__'] = metadata

            return fn

        return decorator_fn
    
    serverfn = endpoint

Server.run(__name__)