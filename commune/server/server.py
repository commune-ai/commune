import commune as c
from typing import *
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import uvicorn
import os
import json
import asyncio
from starlette.middleware.base import BaseHTTPMiddleware

class Server(c.Module):
    tag_seperator:str='::'
    lifetime = 3600 # the lifetime of the user data
    max_request_staleness : int = 4 # (in seconds) the time it takes for the request to be too old
    max_network_staleness: int = 60 #  (in seconds) the time it takes for. the network to refresh
    multipliers : Dict[str, float] = {'stake': 1, 'stake_to': 1,'stake_from': 1}
    rates : Dict[str, int]= {'max': 10, 'local': 10000, 'stake': 1000, 'owner': 10000, 'admin': 10000} # the maximum rate  ):
    helper_functions  = ['info', 'metadata', 'schema', 'name', 'functions','fns', 'forward'] # the helper functions
    function_attributes =['whitelist', 'endpoints', 'functions',  'fns', "exposed_functions",'server_functions', 'public_functions'] # the attributes for the functions
    net = {'local': c.module('server.network')()}
    manager = c.module("server.manager")()
    
    def __init__(
        self, 
        module: Union[c.Module, object] = None,
        key:str = None, # key for the server (str)
        functions:Optional[List[Union[str, callable]]] = None, # list of endpoints
        name: Optional[str] = None, # the name of the server
        params : dict = None, # the kwargs for the module
        port: Optional[int] = None, # the port the server is running on
        gate:str = None, # the .network used for incentives
        free : bool = False, # if the server is free (checks signature)
        key_type: Union[int, str] = 'sr25519', # the crypto type of the key
        network = 'subspace', # the network the server is running on
        history_path: Optional[str] = None, # the path to the user data
        serializer: str = 'serializer', # the serializer used for the data
        middleware: Optional[callable] = None, # the middleware for the server
        run_api = True, # if the server should be run
        ) -> 'Server':
        self.set_network(network)  
        self.serializer = c.module(serializer)()
        if module == None:
            if isinstance(name, str):
                module = name
        module = module or 'module'
        params =params or {}
        if isinstance(module, str):
            if self.tag_seperator in module:
                name =  module
                module, tag = name.split(self.tag_seperator) 

        name = name or module

        # NOTE: ONLY ENABLE FREEMODE IF YOU ARE ON A CLOSED NETWORK,
        self.module = c.module(module)(**params)
        self.module.name = name 
        self.module.key = c.get_key(key or name,key_type=key_type)
        self.set_port(port)
        self.set_functions(functions) 
        self.loop = asyncio.get_event_loop()
        self.history_path = history_path or self.resolve_path(f'history/{self.module.name}')
        if run_api:
            self.run_api()

    def run_api(self, max_bytes: int = 10**6, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]):
        def forward(fn: str, request: Request):
            return self.forward(fn, request)
        self.api = FastAPI()
        c.thread(self.sync_loop)
        self.api.add_middleware(self.middleware(max_bytes=max_bytes))
        self.api.add_middleware(CORSMiddleware, allow_origins=allow_origins, allow_credentials=allow_credentials, allow_methods=allow_methods, allow_headers=allow_headers)
        self.api.post("/{fn}")(forward)
        c.print(f'Server(name={self.module.name}, address={self.module.address}, key={self.module.key.key_address})', color='purple')
        self.net['local'].add_server(name=self.module.name, address=self.module.address, key=self.module.key.ss58_address)
        print(f'Network: {self.network}')
        uvicorn.run(self.api, host='0.0.0.0', port=self.module.port, loop='asyncio')


    def set_functions(self,  functions:Optional[List[str]] ):

        self.free = any([(hasattr(self.module, k) and self.module.free)  for k in ['free', 'free_mode']])

        if self.free:
            c.print('THE FOUNDING FATHERS WOULD BE PROUD OF YOU SON OF A BITCH', color='red')
        else:
            if hasattr(self.module, 'free'):
                self.free = self.module.free
        self.module.free = self.free
        functions =  functions or []
        if len(functions) > 0:
            for i, fn in enumerate(functions):
                if callable(fn):
                    print('Adding function -->', f)
                    setattr(self, fn.__name__, fn)
                    functions[i] = fn.__name__
        function_attributes = [fa for fa in self.function_attributes if hasattr(self.module, fa) and isinstance(getattr(self.module, fa), list)]
        assert len(function_attributes) == 1 , f'{function_attributes} is too many funcitonal attributes, choose one dog'
        functions = getattr(self.module, function_attributes[0])
        for f in dir(self.module):
            try:
                if hasattr(getattr(self.module, f), '__fn_metadata__'):
                    functions.append(f)
            except Exception as e:
                c.print(f'Error in get_endpoints: {e} for {f}')
        self.module.schema = {fn: c.schema(getattr(self.module, fn )) for fn in functions}
        functions  =  sorted(list(set(functions + self.helper_functions)))
        self.module.functions = sorted(list(set(functions)))
        self.module.fn2cost = self.module.fn2cost  if hasattr(self.module, 'fn2cost') else {}
        c.print(f'Functions({self.module.functions})')
        assert isinstance(self.module.fn2cost, dict), f'fn2cost must be a dict, not {type(self.module.fn2cost)}'

        ### get the info for the module
        self.module.info = {
            "functions": self.module.functions,
            "schema": self.module.schema,
            "name": self.module.name,
            "address": self.module.address,
            "key": self.module.key.ss58_address,
            "key_type": self.module.key.key_type,
            "free": self.module.free,
            "time": c.time()
        }

        return {'success':True, 'message':f'Set functions to {functions}'}
        
    @classmethod
    def get_port(cls, port=None, name=None):
        name = name or 'module'
        
        if port in [None, 'None']:
            namespace = cls.net['local'].namespace()
            if name in namespace:
                cls.manager.kill(name)
                try:
                    port =  int(namespace.get(self.module.name).split(':')[-1])
                except:
                    port = c.free_port()
            else:
                port = c.free_port()
        while c.port_used(port):
            c.kill_port(port)
            c.sleep(1)
            print(f'Waiting for port {port} to be free')
        return port
    def set_port(self, port:Optional[int]=None, port_attributes = ['port', 'server_port']):
        name = self.module.name
        for k in port_attributes:
            if hasattr(self.module, k):
                port = getattr(self.module, k)
                break
        self.module.port = port or self.get_port(port=port, name=name)
        self.module.address = '0.0.0.0:' + str(self.module.port)
        return {'success':True, 'message':f'Set port to {port}'}
    
    def get_data(self, request: Request):
        data = self.loop.run_until_complete(request.json())
        data = self.serializer.deserialize(data) 
        if isinstance(data, str):
            data = json.loads(data)
        if 'kwargs' in data or 'params' in data:
            kwargs = dict(data.get('kwargs', data.get('params', {}))) 
        else:
            kwargs = data
        if 'args' in data:
            args = list(data.get('args', []))
        else:
            args = []
        data = {'args': args, 'kwargs': kwargs}
        return data 
    
    def get_headers(self, request: Request):
        headers = dict(request.headers)
        headers['time'] = float(headers.get('time', c.time()))
        headers['key'] = headers.get('key', headers.get('address', None))
        return headers

    def forward(self, fn:str, request: Request, catch_exception:bool=True) -> dict:
        if catch_exception:
            try:
                return self.forward(fn, request, catch_exception=False)
            except Exception as e:
                result =  c.detailed_error(e)
                return result
        module = self.module
        data = self.get_data(request)
        headers = self.get_headers(request)
        gate_info = self.gate(fn=fn, data=data, headers=headers)  
        c.print(f'Forward(fn={fn} key={headers["key"][:4]}... gate={gate_info})')
        is_admin = bool(c.is_admin(headers['key']))
        is_owner = bool(headers['key'] == self.module.key.ss58_address)    
        if hasattr(module, fn):
            fn_obj = getattr(module, fn)
        elif (is_admin or is_owner) and hasattr(self, fn):
            fn_obj = getattr(self, fn)
        else:
            raise Exception(f"{fn} not found in {self.module.name}")
        result = fn_obj(*data['args'], **data['kwargs']) if callable(fn_obj) else fn_obj
        latency = c.time() - float(headers['time'])
        if c.is_generator(result):
            output = ''
            def generator_wrapper(generator):
                for item in generator:
                    output += str(item)
                    yield item
            result = EventSourceResponse(generator_wrapper(result))
        else:
            output =  self.serializer.serialize(result)
            
        if not self.free:
            user_data = {
                    'url': self.module.address, # the address of the server
                    'fn': fn, # the function you are calling
                    'data': data, # the data of the request
                    'output': output, # the response
                    'time': headers["time"], # the time of the request
                    'latency': latency, # the latency of the request
                    'key': headers['key'], # the key of the user
                    'cost': self.module.fn2cost.get(fn, 1), # the cost of the function
                    "client_signature": headers.get('signature', None), # the signature of the client
                }
            user_data['server_signature'] = self.module.key.sign(user_data, to_str=True)
            user_path = self.user_path(f'{user_data["key"]}/{user_data["fn"]}/{c.time()}.json') 
            c.put(user_path, user_data)
        return result

    def set_network(self, network):
        self.network = network
        self.network_path = self.resolve_path(f'networks/{self.network}/state.json')
        self.address2key =  c.address2key()
        c.thread(self.sync)
        return {'success':True,  'network':network, 'network_path':self.network_path}

    def sync(self, update=True , state_keys = ['stake_from', 'stake_to']):
        self.network_path = self.resolve_path(f'networks/{self.network}/state.json')
        c.print(f'Network(network={self.network} path={self.network_path.replace(c.homepath, "~")})')
        if hasattr(self, 'state'):
            print('state exists')
            latency = c.time() - self.state.get('time', 0)
            if latency < self.max_network_staleness:
                return {'msg': 'state is fresh'}
        max_age = self.max_network_staleness
        network_path = self.network_path
        state = c.get(network_path, None, max_age=max_age, updpate=update)
        state = {}
        if state == None:
            self.net['local'].namespace(max_age=max_age)
            self.net[self.network] = c.module(self.network)()
            state = self.net[self.network].state()
        self.state = state
        return {'msg': 'state synced successfully'}

    def sync_loop(self, sync_loop_initial_sleep=10):
        c.sleep(sync_loop_initial_sleep)
        while True:
            try:
                r = self.sync()
            except Exception as e:
                r = c.detailed_error(e)
                c.print('Error in sync_loop -->', r, color='red')
            c.sleep(self.max_network_staleness)

    @classmethod
    def wait_for_server(cls,
                          name: int ,
                          network: str = 'local',
                          timeout:int = 100, 
                          sleep_interval=1) -> bool :
        
        time_waiting = 0
        # have a waiting counter on one line

        lights = '🟡'
        # have the time print on the same line and delete the previous line without using c. 
        while time_waiting < timeout:
            namespace = c.namespace()
            c.sleep(sleep_interval)
            c.print(c.logs(name, mode='local'), end='\r')
            c.print(f'🟡ServerLoading(name={name} t={int(time_waiting)} timeout={timeout})🟡', end='\r')
            if name in namespace:
                result = c.call(namespace[name])
                if 'functions' in result:
                    result.pop('schema')
                    break
            time_waiting += sleep_interval
            if time_waiting > timeout:
                raise TimeoutError(f'Waited for {timeout} seconds for {name} to start')
        # checkmarket
        c.print(f'🟢ServerLoading(name={name} t={int(time_waiting)} timeout={timeout})🟢')


        return result
    @classmethod
    def endpoint(cls, 
                 cost = 1,
                 user2rate : dict = None, 
                 rate_limit : int = 100, # calls per minute
                 timestale : int = 60,
                 public:bool = False,
                 **kwargs):
        def decorator_fn(fn):
            metadata = {
                'schema':c.schema(fn),
                'cost': cost,
                'rate_limit': rate_limit,
                'user2rate': user2rate,   
                'timestale': timestale,
                'public': public,            
            }
            fn.__dict__['__fn_metadata__'] = metadata
            return fn
        return decorator_fn
    
    serverfn = endpoint
    @classmethod
    def serve(cls, 
              module: Union[str, 'Module', Any] = None, # the module in either a string
              params:Optional[dict] = None,  # kwargs for the module
              port :Optional[int] = None, # name of the server if None, it will be the module name
              name = None, # name of the server if None, it will be the module name
              remote:bool = True, # runs the server remotely (pm2, ray)
              functions = None, # list of functions to serve, if none, it will be the endpoints of the module
              key = None, # the key for the server
              cwd = None,
              **extra_params
              ):
        module = module or 'module'
        name = name or module
        print(f'Serving(module={module}, name={name})')
        params = {**(params or {}), **extra_params}
        if remote and isinstance(module, str):
            rkwargs = {k : v for k, v  in c.locals2kwargs(locals()).items()  if k not in ['extra_params', 'response', 'namespace']}
            rkwargs['remote'] = False
            cls.manager.start( module="server", fn='serve', name=name, kwargs=rkwargs, cwd=cwd)
            return cls.wait_for_server(name)
        print(locals())
        return Server(module=module, name=name, functions=functions, params=params, port=port,  key=key)

    def add_endpoint(self, name, fn):
        setattr(self, name, fn)
        self.endpoints.append(name)
        assert hasattr(self, name), f'{name} not added to {self.__class__.__name__}'
        return {'success':True, 'message':f'Added {fn} to {self.__class__.__name__}'}

    @classmethod
    def test(cls, **kwargs):
        from .test import Test
        return Test().test()

    @classmethod
    def kill_all(cls): 
        return cls.manager.kill_all()

    @classmethod
    def kill(cls, name, **kwargs):
        return cls.manager.kill(name, **kwargs)

    @classmethod
    def server_exists(cls, name):
        return cls.net['local'].server_exists(name)

    @classmethod
    def logs(cls, name, **kwargs):
        return cls.manager.logs(name, **kwargs)
    def is_admin(self, address):
        return c.is_admin(address)

    def gate(self, fn:str, data:dict,  headers:dict ) -> bool:
            if self.free: 
                assert fn in self.module.functions , f"Function {fn} not in endpoints={self.module.functions}"
                return True
            auth = {'data': data, 'time': str(headers['time'])}
            signature = headers['signature']
            assert c.verify(auth=auth,signature=signature, address=headers['key']), 'Invalid signature'
            request_staleness = c.time() - float(headers['time'])
            assert  request_staleness < self.max_request_staleness, f"Request is too old ({request_staleness}s > {self.max_request_staleness}s (MAX)" 
            auth={'data': data, 'time': str(headers['time'])}
            address = headers['key']
            stake = 0
            if c.is_admin(address):
                rate_limit =  self.rates['admin']
            elif address == self.module.key.ss58_address:
                rate_limit =  self.rates['owner']
            elif address in self.address2key:
                rate_limit =  self.rates['local']
            else:
                stake_score = self.state['stake'].get(address, 0) + self.multipliers['stake']
                stake_to_score = (sum(self.state['stake_to'].get(address, {}).values())) * self.multipliers['stake_to']
                stake_from_score = self.state['stake_from'].get(self.module.key.ss58_address, {}).get(address, 0) * self.multipliers['stake_from']
                stake = stake_score + stake_to_score + stake_from_score
                self.rates['stake'] = self.rates['stake'] * self.module.fn2cost.get(fn, 1)
                rate_limit =  min((stake / self.rates['stake']), self.rates['max'])
            count = self.call_count(headers['key'])
            assert count <= rate_limit, f'rate limit exceeded {count} > {rate_limit}'     
            return {'rate_limit': rate_limit, 'count': count, 'stake': stake}

    def users(self):
        try:
            return os.listdir(self.history_path)
        except:
            return []
    
    def user_call_path2latency(self, address):
        user_paths = self.call_paths(address)
        t0 = c.time()
        user_path2time = {user_path: t0 - self.extract_time(user_path) for user_path in user_paths}
        return user_path2time
    
    def check_user_data(self, address):
        path2latency = self.user_call_path2latency(address)
        for path, latency  in path2latency.items():
            if latency > self.lifetime:
                c.print(f'RemovingPath(path={path} age(s)={latency} lifetime={self.lifetime})')
                if os.path.exists(path):
                    os.remove(path)

    def user_path(self, key_address):
        return self.history_path + '/' + key_address

    def call_count(self, user):
        self.check_user_data(user)
        return len(self.call_paths(user))

    def user_data(self, address, stream=False):
        user_paths = self.call_paths(address)
        if stream:
            def stream_fn():
                for user_path in user_paths:
                    yield c.get(user_path)
            return stream_fn()
        
        else:
            return [c.get(user_path) for user_path in user_paths]
        
    def user2fn2count(self):
        user2fn2count = {}
        for user in self.users():
            user2fn2count[user] = {}
            for user_data in self.user_data(user):
                fn = user_data['fn']
                user2fn2count[user][fn] = user2fn2count[user].get(fn, 0) + 1
        return user2fn2count

    def call_paths(self, address ):
        user_paths = c.glob(self.user_path(address))
        return sorted(user_paths, key=self.extract_time)

    def users(self):
        return os.listdir(self.history_path)

    @classmethod
    def rm_all_data(cls):
        c.rm(cls.history_path)
        return {'success':True, 'message':'Removed all user data'}

    def user2count(self):
        user2count = {}
        for user in self.users():
            user2count[user] = self.call_count(user)
        return user2count
    
    def history(self, user):
        return self.user_data(user)



    @classmethod
    def all_history(cls, module=None):
        self = cls(module=module, run_api=False)
        all_history = {}
        for user in self.users():
            all_history[user] = self.history(user)
        return all_history

    def extract_time(self, x):
        try:
            x = float(x.split('/')[-1].split('.')[0])
        except Exception as e:
            x = 0
        return x


    def middleware(self, max_bytes = 10**6):
        class Middleware(BaseHTTPMiddleware):
            def __init__(self, app, max_bytes: int = max_bytes):
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

        return Middleware
if __name__ == '__main__':
    Server.run()

