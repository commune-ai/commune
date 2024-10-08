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
    network : str = 'local'

    user_functions = [ 'user_count',  
                        'users_path', 
                        'user_paths',
                        'user_data',
                        'user2count', 
                        'user_path2latency',
                        'user_path2timestamp',
                        'remove_user_data',
                        'add_user_rate_limit',
                        'users'
                        ]
    def __init__(
        self, 
        module: Union[c.Module, object] = None,
        functions:Optional[List[str]] = None, # list of endpoints
        key:str = None, # key for the server (str)
        name: str = None, # the name of the server
        port: Optional[int] = None, # the port the server is running on
        network:str = 'subspace', # the network used for incentives
        helper_functions  = ['info',  'metadata',  'schema', 'server_name', 'functions', 'forward', 'rate_limit'], 
        max_bytes:int = 10 * 1024 * 1024,  # max bytes within the request (bytes)
        allow_origins = ["*"], # allowed origins
        allow_credentials=True, # allow credentials
        allow_methods=["*"], # allowed methods
        allow_headers=["*"],  # allowed headers
        period = 60, # the period for 
        max_request_staleness = 4, # the time it takes for the request to be too old
        max_network_staleness = 60, # the time it takes for. the network to refresh
        users_path = None, # the path to store the data
        fn2cost = None, # the cost of the function
        create_key_if_not_exists = True, # create the key if it does not exist
        **kwargs,
        ) -> 'Server':
        module = module or 'module'
        if isinstance(module, str):
            name = name or module
            module = c.module(module)()
        self.max_request_staleness = max_request_staleness
        self.network = network
        self.period = period
        self.users_path = users_path or self.resolve_path(f'{name}/users')
        self.serializer = c.module('serializer')()
        self.max_network_staleness = max_network_staleness

        module.name = module.server_name = name
        module.fn2cost = module.server_fn2cost = fn2cost or {}
        module.port = module.server_port =  port if port not in ['None', None] else c.free_port()
        module.address  = module.server_address =  f"{c.ip()}:{module.port}"
        module.key  = module.server_key = c.get_key(key or module.name, create_if_not_exists=create_key_if_not_exists)
        module.schema =  module.server_schema = self.get_server_schema(module)
        module.functions  = module.server_functions = functions or list(set(helper_functions + list(module.schema.keys())))
        module.info  =  module.server_info =  self.get_server_info(module)
        module.network_path = self.resolve_path(f'{self.network}/state.json')
        module.users_path = self.users_path
        for fn in self.user_functions:
            setattr(module, fn, getattr(self, fn))
        self.module = module

        self.sync(update=False)  
        c.thread(self.sync_loop)
        self.loop = asyncio.get_event_loop()
        app = FastAPI()  
        app.add_middleware(ServerMiddleware, max_bytes=max_bytes)    
        app.add_middleware(CORSMiddleware, 
                           allow_origins=allow_origins, 
                           allow_credentials=allow_credentials,
                           allow_methods=allow_methods,
                           allow_headers=allow_headers)
        
        def api_forward(fn:str, request: Request):
            return self.forward(fn, request)
        app.post("/{fn}")(api_forward)
        c.print(f'Served(name={module.name}, address={module.address}, key=🔑{module.key}🔑 ) 🚀 ', color='purple')
        c.register_server(name=module.name,address=module.address)
        uvicorn.run(app, host='0.0.0.0', port=module.port, loop='asyncio')

    def __del__(self):
        c.deregister_server(self.name)

    @classmethod
    def fleet(cls, module, n:int = 1, **kwargs):
        for _ in range(n):
            c.print(c.serve(module=module, name = module + '::' + str(_),  **kwargs))

        return {'success':True, 'message':f'Served {n} servers'} 

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
    
    def remove_all_history(self):
        return c.rm(self.module.user_path)
    
    user_rate_limit = {}
    def add_user_rate_limit(self, address, rate_limit):
        self.user_rate_limit[address] = rate_limit
        return self.user_rate_limit[address]
        
        
    def rate_limit(self, 
                   address:str, 
                   fn = 'info',
                    multipliers = {'stake': 1, 'stake_to': 1,'stake_from': 1 },
                    module=None, 
                    rates = {'max': 10, 
                             'local': 1000,
                             'stake2rate': 1000, 
                             'admin': 1000}, # the maximum rate
                    ) -> float:
        # stake rate limit
        module = module or self.module
        if c.is_admin(address) or address == module.key.ss58_address:
            return rates['admin']
        if address in self.state['address2key']:
            return rates['local']
        stake_score = self.state['stake'].get(address, 0) + multipliers['stake']
        stake_to_score = (sum(self.state['stake_to'].get(address, {}).values())) * multipliers['stake_to']
        stake_from_score = self.state['stake_from'].get(module.key.ss58_address, {}).get(address, 0) * multipliers['stake_from']
        stake = stake_score + stake_to_score + stake_from_score
        rates['stake2rate'] = rates['stake2rate'] * module.fn2cost.get(fn, 1)
        return min((stake / rates['stake2rate']), rates['max'])

    def user_path(self, address):
        return self.resolve_path(f'{self.module.user_path}/{address}')

    def user2count(self):
        user2count = {}
        for user in self.users():
            user2count[user] = self.user_count(user)
        return user2count

    def user_paths(self, address ):
        user_paths = c.ls(self.user_path(address))
        return sorted(user_paths, key=self.extract_timestamp)
    
    def user_data(self, address):
        for i, user_path in enumerate(self.user_paths(address)):
            yield c.get(user_path)
        
    def user_path(self, address):
        return self.users_path + '/' + address

    def user_count(self, address):
        self.check_user_data(address)
        return len(self.user_paths(address))
    
    def user_path2timestamp(self, address):
        user_paths = self.user_paths(address)
        user_path2timestamp = {user_path: self.extract_timestamp(user_path) for user_path in user_paths}
        return user_path2timestamp
    
    def user_path2latency(self, address):
        user_paths = self.user_paths(address)
        t0 = c.time()
        user_path2timestamp = {user_path: t0 - self.extract_timestamp(user_path) for user_path in user_paths}
        return user_path2timestamp
    
    
    def check_user_data(self, address):
        path2latency = self.user_path2latency(address)
        for path, latency  in path2latency.items():
            if latency > self.period:
                os.remove(path)
        
    
    def check_all_users(self):
        for user in self.users():
            print('Checking', user)
            self.chekcer_user_data()

        

    
    

    def extract_timestamp(self, x):
        try:
            x = float(x.split('timestamp=')[-1].split('_')[0])
        except Exception as e:
            print(e)
            x = 0
        return x

    def remove_user_data(self, address):
        return c.rm(self.user_path(address))

    def users(self):
        return os.listdir(self.module.users_path)

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
        request_staleness = c.timestamp() - float(headers['timestamp'])
        assert  request_staleness < self.max_request_staleness, f"Request is too old ({request_staleness}s > {self.max_request_staleness}s (MAX)" 
        data = self.loop.run_until_complete(request.json())
        data = self.serializer.deserialize(data) 
        request = {'data': data, 'headers': headers}
        auth={'data': c.hash(data), 'timestamp': headers['timestamp']}
        signature = headers.get('signature', None)
        assert c.verify(auth=auth,signature=signature, address=address), 'Invalid signature'
        server_signature = self.module.key.sign(signature)
        kwargs = dict(data.get('kwargs', {}))
        args = list(data.get('args', []))
        if 'params' in data:
            if isinstance(data['params', dict]):
                kwargs = {**kwargs, **data['params']}
            elif isinstance(data['params'], list):
                args = [*args, *data['params']]
            else:
                raise ValueError('params must be a list or a dictionary')
            
        data = {'args': args, 'kwargs': kwargs}
        is_admin = bool(c.is_admin(address) or  address == self.module.key.ss58_address)
        if not is_admin:
            assert not bool(fn.startswith('__') or fn.startswith('_')), f'Function {fn} is private'
            assert fn in self.module.server_functions , f"Function {fn} not in endpoints={self.module.server_functions}"
        count = self.user_count(address)
        rate_limit = self.rate_limit(fn=fn, address=address)
        assert count <= rate_limit, f'rate limit exceeded {count} > {rate_limit}'
        timestamp = float(headers['timestamp'])
        fn_obj = getattr(self.module, fn)
        adress_string = self.state['address2key'].get(address, address)
        user_str = f'User({adress_string[:8]})'
        c.print(f'{user_str} >> Stats(count={count} limit={rate_limit})')
        c.print(f'{user_str} >> Request(fn={fn} args={args} kwargs={kwargs})', color= color)
        result = fn_obj(*data['args'], **data['kwargs']) if callable(fn_obj) else fn_obj
        latency = c.round(c.time() - timestamp, 3)
        c.print(f'{user_str} >> Result(fn={fn} latency={latency})', color=color)
        c.print('-'*16)
        if c.is_generator(result):
            output = []
            def generator_wrapper(generator):
                for item in generator:
                    output_item = self.serializer.serialize(item)
                    output += [output_item]
                    yield output_item
            result = EventSourceResponse(generator_wrapper(result))
        else:
            output =  self.serializer.serialize(result)

        user_data = {
            'module': self.module.server_name,
            'fn': fn,
            'input': data, # the data of the request
            'output': output, # the response
            'latency':  latency, # the latency
            'timestamp': timestamp, # the timestamp of the request
            'user_key': address, # the key of the user
            'server_key': self.module.key.ss58_address, # the key of the server
            'user_signature': signature, # the signature of the user
            'server_signature': server_signature, # the signature of the server
            'cost': self.module.fn2cost.get(fn, 1), # the cost of the function
        }
        
        
        self.save_user_data(user_data)
        
        return result

    def save_user_data(self, user_data):
        user_path = self.user_path(user_data["user_key"]) + f'/timestamp={user_data["timestamp"]}_fn={user_data["fn"]}.json' # get the user info path
        c.put(user_path, user_data)

    def sync_loop(self, sync_loop_initial_sleep=4):
        c.sleep(sync_loop_initial_sleep)
        while True:
            try:
                r = self.sync()
            except Exception as e:
                r = c.detailed_error(e)
                c.print('Error in sync_loop -->', r, color='red')
            c.sleep(self.max_network_staleness)

    def sync(self, 
            update=False,
            state_keys = ['stake_from', 'stake_to', 'address2key', 
                           'stake', 'key2address', 'timestamp', 'latency']):
        t0 = c.time()
        if hasattr(self, 'state'):
            latency = c.time() - self.state['timestamp']
            if latency < self.max_network_staleness:
                return {'msg': 'state is fresh'}
        max_age = self.max_network_staleness
        network_path = self.module.network_path
        state = self.get(network_path, {}, max_age=max_age, updpate=update)
        is_valid_state = lambda x: all([k in x for k in state_keys])
        network = self.network
        state = {}
        state['address2key'] =  c.address2key()
        state['key2address'] = {v:k for k,v in state['address2key'].items()}
        state['stake'] = {}
        state['stake_to'] = {}
        state['stake_from'] = {}
        if update:
            try  : 
                c.get_namespace(max_age=max_age)
                self.subspace = c.module('subspace')(network=network)
                state['stake_from'] = self.subspace.stake_from(fmt='j', update=update, max_age=max_age)
                state['stake_to'] = self.subspace.stake_to(fmt='j', update=update, max_age=max_age)
                state['stake'] =  {k: sum(v.values()) for k,v in state['stake_from'].items()}
            except Exception as e:
                print(f'Error {e} while syncing network--> {network}')
        
        state['timestamp'] = c.time()
        state['latency'] = state['timestamp'] - t0
        assert is_valid_state(state), f'Format for network state is {[k for k in state_keys if k not in state]}'
        self.put(network_path, state)
        self.state = state
        return {'msg': 'state synced successfully'}

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
    
    def get_server_info(self , module,**kwargs ) -> Dict[str, Any]:
        '''
        hey, whadup hey how is it going
        '''
        info = {}
        info['schema'] = module.schema
        info['name'] = module.name 
        info['address'] = module.address
        info['key'] = module.key.ss58_address
        return info

    def get_server_schema(self,
                           module,  
                          functions_attributes=['helper_functions', 'whitelist','endpoints',
                                                'server_functions'],) -> 'Schema':
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