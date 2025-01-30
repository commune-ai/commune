import commune as c
from typing import *
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import uvicorn
import os
import json
import asyncio

class Server:
    network = 'subspace'
    max_user_history_age = 3600 # the lifetime of the user call data
    max_network_age: int = 60 #  (in seconds) the time it takes for. the network to refresh
    helper_functions  = ['info', 'schema', 'functions', 'forward'] # the helper functions
    function_attributes =['endpoints', 'functions', "exposed_functions",'server_functions', 'public_functions', 'pubfns'] # the attributes for the functions
    server_network =  c.module('server.network')()
    def __init__(
        self, 
        module: Union[c.Module, object] = None,
        key:str = None, # key for the server (str)
        functions:Optional[List[Union[str, callable]]] = None, # list of endpoints
        name: Optional[str] = None, # the name of the server
        params : dict = None, # the kwargs for the module
        port: Optional[int] = None, # the port the server is running on
        network = 'subspace', # the network the server is running on
        # -- ADVANCED PARAMETERS --
        gate:str = None, # the .network used for incentives
        free : bool = False, # if the server is free (checks signature)
        crypto_type: Union[int, str] = 'sr25519', # the crypto type of the key
        serializer: str = 'serializer', # the serializer used for the data
        middleware: Optional[callable] = None, # the middleware for the server
        history_path: Optional[str] = None, # the path to the user data
        run_api = False, # if the server should be run
        ) -> 'Server':
        module = module or 'module'
        if isinstance(module, str):
            if '::' in str(module):
                module, tag = name.split('::') 
        name = name or module
        params = params or {}
        self.module = c.module(module)(**params)
        self.module.name = name 
        key = key or name
        self.module.key = c.get_key(key, crypto_type=crypto_type)
        self.set_port(port)
        self.set_functions(functions) 
        self.history_path = history_path or self.resolve_path(f'history/{self.module.name}')
        self.serializer = c.module(serializer)()
        if run_api:
            self.sync_network(network)  
            self.run_api()

    @classmethod
    def resolve_path(cls, path):
        return  c.storage_path + c.module_name(Server) + path
            
    def run_api(self, 
                 max_bytes = 10**6, 
                 allow_origins=["*"], 
                 allow_credentials=True, 
                 allow_methods=["*"], 
                 allow_headers=["*"]):

        self.loop = asyncio.get_event_loop()
        self.app = FastAPI()
        c.thread(self.sync_loop)
        from starlette.middleware.base import BaseHTTPMiddleware
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
        def forward(fn: str, request: Request):
            return self.forward(fn, request)
        self.app.add_middleware(Middleware)
        self.app.add_middleware(CORSMiddleware, 
                                allow_origins=allow_origins, 
                                allow_credentials=allow_credentials, 
                                allow_methods=allow_methods, 
                                allow_headers=allow_headers)
         # add the endpoints to the app
        self.app.post("/{fn}")(forward)
        c.print(f'Served(name={self.module.name}, url={self.module.url}, key={self.module.key.key_address})', color='purple')
        self.server_network.add_server(name=self.module.name, url=self.module.url, key=self.module.key.ss58_address)
        print(f'Network: {self.network}')
        uvicorn.run(self.app, host='0.0.0.0', port=self.module.port, loop='asyncio')
    
    def set_functions(self,  functions:Optional[List[str]] ):
        self.free = any([(hasattr(self.module, k) and self.module.free)  for k in ['free', 'free_mode']])
        if self.free:
            c.print('THE FOUNDING FATHERS WOULD BE PROUD OF YOU SON OF A BITCH', color='red')
        else:
            if hasattr(self.module, 'free'):
                self.free = self.module.free
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
        self.module.schema = {fn: c.schema(getattr(self.module, fn )) for fn in functions if hasattr(self.module, fn)}
        self.module.free = self.free
        self.module.functions = sorted(list(set(functions + self.helper_functions)))
        self.module.fn2cost = self.module.fn2cost  if hasattr(self.module, 'fn2cost') else {}
        c.print(f'Functions({self.module.functions} fn2cost={self.module.fn2cost} free={self.free})')
        assert isinstance(self.module.fn2cost, dict), f'fn2cost must be a dict, not {type(self.module.fn2cost)}'
        self.module.info = {
            "name": self.module.name,
            "url": self.module.url,
            "key": self.module.key.ss58_address,
            "crypto_type": self.module.key.crypto_type,
            "fn2cost": self.module.fn2cost,
            "time": c.time(),
            "functions": self.module.functions,
            "schema": self.module.schema,
        }
        return {'success':True, 'message':f'Set functions to {functions}'}
        
    def set_port(self, port:Optional[int]=None, port_attributes = ['port', 'server_port']):
        name = self.module.name
        for k in port_attributes:
            if hasattr(self.module, k):
                port = getattr(self.module, k)
                break
        if port in [None, 'None']:
            namespace = self.server_network.namespace()
            if name in namespace:
                m.kill(name)
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
        self.module.port = port
        self.module.url = self.module.url = '0.0.0.0:' + str(self.module.port)
        return {'success':True, 'message':f'Set port to {port}'}
    
    def get_params(self, request: Request) -> dict:
        params = self.loop.run_until_complete(request.json())
        params = self.serializer.deserialize(params) 
        params = json.loads(params) if isinstance(params, str) else params
        assert isinstance(params, dict), f'Params must be a dict, not {type(params)}'
        if len(params) == 2 and 'args' in params and 'kwargs' in params :
            kwargs = dict(params.get('kwargs')) 
            args = list(params.get('args'))
        else:
            args = []
            kwargs = dict(params)
        return {'args': args, 'kwargs': kwargs} 
    

    def get_headers(self, request: Request):
        headers = dict(request.headers)
        headers['time'] = float(headers.get('time', c.time()))
        headers['key'] = headers.get('key', headers.get('url', None))
        return headers

    def forward(self, fn:str, request: Request, catch_exception:bool=True) -> dict:
        if catch_exception:
            try:
                return self.forward(fn, request, catch_exception=False)
            except Exception as e:
                result =  c.detailed_error(e)
                return result
        module = self.module
        params = self.get_params(request)
        headers = self.get_headers(request)
        gate_info = self.gate(fn=fn, params=params, headers=headers)   
        is_admin = bool(c.is_admin(headers['key']))
        is_owner = bool(headers['key'] == self.module.key.ss58_address)    
        if hasattr(module, fn):
            fn_obj = getattr(module, fn)
        else:
            raise Exception(f"{fn} not found in {self.module.name}")
        result = fn_obj(*params['args'], **params['kwargs']) if callable(fn_obj) else fn_obj
        latency = c.time() - float(headers['time'])
        if c.is_generator(result):
            c.print(f"Generator({result})")
            # get a hash for the generator
            output = str(result)
            def generator_wrapper(generator):
                for item in generator:
                    yield item
            result = EventSourceResponse(generator_wrapper(result))       
        else:
            output = result 

        output =  self.serializer.serialize(output)
            
        if not self.free:
            data = {
                    'url': self.module.url, # the url of the server
                    'fn': fn, # the function you are calling
                    'params': params, # the data of the request
                    'output': output, # the response
                    'time': headers["time"], # the time of the request
                    'latency': latency, # the latency of the request
                    'key': headers['key'], # the key of the user
                    'cost': self.module.fn2cost.get(fn, 1), # the cost of the function
                }
            call_data_path = self.get_call_data_path(f'{data["key"]}/{data["fn"]}/{c.time()}.json') 
            c.put(call_data_path, data)
        return result

    def  resolve_path(self, path):
        return  c.storage_path + '/' + self.module_name() + '/' + path
    @classmethod
    def processes(cls):
        return cls.server_network.processes()

    state = {}
    def sync_network(self, network=None):
        self.network = network or self.network
        self.network_path = self.resolve_path(f'networks/{self.network}/state.json')
        self.address2key =  c.address2key()
        c.print(f'Network(network={self.network} path={self.network_path})')
        self.state = c.get(self.network_path, {}, max_age=self.max_network_age)
        if self.state == {}:
            def sync():
                self.network_module = c.module(self.network)()
                self.state = self.network_module.state()
            c.thread(sync)

        return {'network':self.network}

    def sync_loop(self):
        c.sleep(self.max_network_age/2)
        while True:
            try:
                r = self.sync_network()
            except Exception as e:
                r = c.detailed_error(e)
                c.print('Error in sync_loop -->', r, color='red')
            c.sleep(self.max_network_age)

    @classmethod
    def wait_for_server(cls,
                          name: str ,
                          network: str = 'local',
                          timeout:int = 600,
                          max_age = 1,
                          sleep_interval: int = 1) -> bool :
        
        time_waiting = 0
        # rotating status thing
        c.print(f'waiting for {name} to start...', color='cyan')
        future = c.submit(c.logs, [name])

        while time_waiting < timeout:
                namespace = cls.server_network.namespace(network=network, max_age=max_age)
                if name in namespace:
                    try:
                        result = c.call(namespace[name]+'/info')
                        if 'key' in result:
                            c.print(f'{name} is running', color='green')
                        result.pop('schema', None)
                        return result
                    except Exception as e:
                        c.print(f'Error getting info for {name} --> {e}', color='red')
                c.sleep(sleep_interval)
                
                time_waiting += sleep_interval
        future.cancel()
        raise TimeoutError(f'Waited for {timeout} seconds for {name} to start')

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
        params = {**(params or {}), **extra_params}
        if remote and isinstance(module, str):
            rkwargs = {k : v for k, v  in c.locals2kwargs(locals()).items()  if k not in ['extra_params', 'response', 'namespace']}
            rkwargs['remote'] = False
            cls.server_network.start( module="server", fn='serve', name=name, kwargs=rkwargs, cwd=cwd)
            return cls.wait_for_server(name)
        return Server(module=module, name=name, functions=functions, params=params, port=port,  key=key, run_api=1)

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
        return cls.server_network.kill_all()

    @classmethod
    def kill(cls, name, **kwargs):
        return cls.server_network.kill(name, **kwargs)

    @classmethod
    def server_exists(cls, name):
        return cls.server_network.server_exists(name)

    @classmethod
    def servers(cls, **kwargs):
        return cls.server_network.servers(**kwargs)

    @classmethod
    def logs(cls, name, **kwargs):
        return cls.server_network.logs(name, **kwargs)

    def is_admin(self, key_address):
        return c.is_admin(key_address)

    def get_user_role(self, key_address):
        if c.is_admin(key_address):
            return 'admin'
        if key_address == self.module.key.ss58_address:
            return 'owner'
        if key_address in self.address2key:
            return 'local'
        return 'stake'

    def gate(self, 
                fn:str, 
                params:dict,  
                headers:dict, 
                multipliers : Dict[str, float] = {'stake': 1, 'stake_to': 1,'stake_from': 1}, 
                rates : Dict[str, int]= {'local': 10000, 'owner': 10000, 'admin': 10000}, # the maximum rate  ):
                max_request_staleness : int = 4 # (in seconds) the time it takes for the request to be too old
            ) -> bool:
            role = self.get_user_role(headers['key'])
            if role == 'admin':
                return True
            if self.free: 
                return True
            stake = 0
            assert fn in self.module.functions , f"Function {fn} not in endpoints={self.module.functions}"
            request_staleness = c.time() - float(headers['time'])
            assert  request_staleness < max_request_staleness, f"Request is too old ({request_staleness}s > {max_request_staleness}s (MAX)" 
            auth = {'params': params, 'time': str(headers['time'])}
            assert c.verify(auth=auth,signature=headers['signature'], address=headers['key']), 'Invalid signature'
            role = self.get_user_role(headers['key'])
            if role in rates:
                rate_limit = rates[role]
            else:
                stake = self.state['stake'].get(headers['key'], 0) * self.multipliers['stake']
                stake_to = (sum(self.state['stake_to'].get(headers['key'], {}).values())) * multipliers['stake_to']
                stake_from = self.state['stake_from'].get(self.module.key.ss58_address, {}).get(headers['key'], 0) * multipliers['stake_from']
                stake = stake + stake_to + stake_from
                raet_limit = rates['stake'] / self.module.fn2cost.get(fn, 1)
                rate_limit =  min(raet_limit, rates['max'])
            rate = self.call_rate(headers['key'])
            assert rate <= rate_limit, f'RateLimitExceeded({rate}>{rate_limit})'     
            return {'rate': rate, 
                    'rate_limit': rate_limit, 
                    'cost': self.module.fn2cost.get(fn, 1)
                    }

    
    def user_call_path2latency(self, key_address):
        user_paths = self.call_paths(key_address)
        t1 = c.time()
        user_path2time = {p: t1 - self.path2time(p) for p in user_paths}
        return user_path2time

    def get_call_data_path(self, key_address):
        return self.history_path + '/' + key_address

    def call_rate(self, key_address, max_age = 60):
        path2latency = self.user_call_path2latency(key_address)
        for path, latency  in path2latency.items():
            if latency > self.max_user_history_age:
                c.print(f'RemovingUserPath(path={path} latency(s)=({latency}/{self.max_user_history_age})')
                if os.path.exists(path):
                    os.remove(path)
        return len(self.call_paths(key_address))

    def user_history(self, key_address, stream=False):
        call_paths = self.call_paths(key_address)
        if stream:
            def stream_fn():
                for p in call_paths:
                    yield c.get(p)
            return stream_fn()
        else:
            return [c.get(call_path) for call_path in call_paths]
        
    def user2fn2calls(self):
        user2fn2calls = {}
        for user in self.users():
            user2fn2calls[user] = {}
            for user_history in self.user_history(user):
                fn = user_history['fn']
                user2fn2calls[user][fn] = user2fn2calls[user].get(fn, 0) + 1
        return user2fn2calls

    def call_paths(self, key_address ):
        user_paths = c.glob(self.get_call_data_path(key_address))
        return sorted(user_paths, key=self.path2time)

    def users(self):
        return os.listdir(self.history_path)

    def history(self, module=None , simple=True):
        module = module or self.module.name
        all_history = {}
        users = self.users()
        for user in users:
            all_history[user] = self.user_history(user)
            if simple:
                all_history[user].pop('output')
        return all_history
    @classmethod
    def all_history(cls, module=None):
        self = cls(module=module, run_api=False)
        all_history = {}
        return all_history

    def path2time(self, path:str) -> float:
        try:
            x = float(path.split('/')[-1].split('.')[0])
        except Exception as e:
            x = 0
        return x
        return Middleware
if __name__ == '__main__':
    Server.run()

