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
        helper_functions  = ['info', 'metadata','schema', 'server_name', 'functions', 'forward','rate_limit', 'user_info'], 
        helper_function_attributes = ['helper_functions', 'whitelist','endpoints', 'server_functions'],
        max_bytes:int = 10 * 1024 * 1024,  # max bytes within the request (bytes)
        allow_origins = ["*"], # allowed origins
        allow_credentials=True, # allow credentials
        allow_methods=["*"], # allowed methods
        allow_headers=["*"],  # allowed headers
        stake2rate = 1000, # the stake to rate ratio
        max_rate = 100, # the maximum rate
        period = 60, # the period for 
        stake_multiplier = {'network': 1, 'direct': 1,'validator': 1 },
        max_request_staleness = 5, # the time it takes for the request to be too old
        max_network_staleness = 60, # the time it takes for the network to refresh
        **kwargs,
        ) -> 'Server':
        module = module or 'module'
        if isinstance(module, str):
            name = module
            module = c.module(module)()

        module.schema =  module.server_schema = self.get_server_schema(module, helper_function_attributes=helper_function_attributes)
        module.functions  = module.server_functions = list(set(helper_functions + (functions or  list(module.schema.keys()))))
        module.name = module.server_name = name
        module.port = module.server_port =  port if port not in ['None', None] else c.free_port()
        module.address  = module.server_address =  f"{c.ip()}:{module.port}"
        module.key  = module.server_key = c.get_key(key or module.name, create_if_not_exists=True)
        module.info  =  module.server_info =  self.get_server_info(module)
        if hasattr(module, 'rate_limit'):
            self.rate_limit = module.rate_limit
        module.rate_limit = self.rate_limit

        c.print('FUNCTIONS ---->', module.functions)
        

        self.stake2rate = stake2rate
        self.max_rate = max_rate
        self.stake_multiplier = stake_multiplier
        self.max_bytes = max_bytes
        self.max_request_staleness = max_request_staleness
        self.serializer = c.module('serializer')()
        self.network = network
        self.module = module
        self.period = period
        self.max_network_staleness = max_network_staleness
        self.state = {'sync_time': 0,  'stake': {},'stake_from': {},  'fn_info': {}, 'stake_to': {}}
        c.thread(self.sync_network_loop)
        self.loop = asyncio.get_event_loop()
        app = FastAPI()    
        app.add_middleware(ServerMiddleware, max_bytes=self.max_bytes)    
        app.add_middleware(CORSMiddleware,
                           allow_origins=allow_origins, 
                           allow_credentials=allow_credentials,
                           allow_methods=allow_methods,
                           allow_headers=allow_headers)
        def api_forward(fn:str, request: Request):
            return self.forward(fn, request)
        app.post("/{fn}")(api_forward)
        c.print(f' Served(name={module.name}, address={module.address}, key=🔑{module.key}🔑 ) 🚀 ', color='purple')
        c.register_server(name=module.name,address = module.address)
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
        if c.server_exists(name):
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
        
    def rate_limit(self, module, fn, address) -> float:
        # stake rate limit
        key2address = c.key2address()
        address = key2address.get(address, address)
        if c.is_admin(address) or address == self.module.key.ss58_address or address in self.address2key:
            return self.max_rate
        validator_stake = self.state['stake'].get(address, 0) + self.stake_multiplier['validator']
        network_stake = (sum(self.state['stake_to'].get(address, {}).values())) * self.stake_multiplier['network']
        direct_stake = self.state['stake_from'].get(self.module.key.ss58_address, {}).get(address, 0) * self.stake_multiplier['direct']
        stake = validator_stake + network_stake + direct_stake
        fn_info = self.state.get('fn_info', {}).get(fn, {'stake2rate': self.stake2rate, 'max_rate': self.max_rate})
        rate_limit = (stake / fn_info['stake2rate']) # convert the stake to a rate
        return rate_limit

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
        c.print(f'🚀User(address={address[:3]}..) ----> Request(fn={fn} args={args} kwargs={kwargs})🚀', color= color)
        if 'params' in data:
            if isinstance(data['params', dict]):
                kwargs = {**kwargs, **data['params']}
            elif isinstance(data['params'], list):
                args = [*args, *data['params']]
            else:
                raise ValueError('params must be a list or a dictionary')
        data = {'fn': fn, 'args': args, 'kwargs': kwargs}
        assert not bool(fn.startswith('__') or fn.startswith('_')), f'Function {fn} is private'
        assert fn in self.module.server_functions , f"Function {fn} not in endpoints={self.module.server_functions}"
        timestamp = c.timestamp()
        user_folder = self.resolve_path(f'users/{address}/{self.module.server_name}')
        user_path = f'{user_folder}/{fn}_{timestamp}.json' # get the user info path
        user = c.get(user_path, {}, max_age=self.period) # get the user info, refresh if it is too old (> period)
        user['timestamp'] = user.get('timestamp', timestamp)
        user['time_til_reset'] = self.period - (timestamp - user['timestamp'])
        if bool(user['time_til_reset']  < 0):
            c.rmdir(user_folder)
            user['timestamp'] = timestamp

        user['address'] = address
        user['data'] = data
        user['count'] = len(c.ls(user_folder))
        user['count'] = user['count'] + 1 # increment the count
        rate_limit = int(self.rate_limit(module=self.module, fn=fn, address=address)) # get the rate limit for the user
        assert user['count'] <= rate_limit, f'rate limit exceeded for {fn} rate_limit={rate_limit} count={user["count"]}'
        data, headers  = request['data'], request['headers']
        fn_obj = getattr(self.module, fn)
        if callable(fn_obj):
            args, kwargs = data.get('args', []), data.get('kwargs', {})
            response = fn_obj(*args, **kwargs)
        else:
            response = fn_obj
        user['response'] = response
        c.put(user_path, user)
        latency = c.round(c.time() - int(headers['timestamp']), 3)
        c.print(f'✅Response(fn={fn} speed={latency}s) --> User(key={headers["key"][:3]}..)✅', color=color)
        if c.is_generator(response):
            def generator_wrapper(generator):
                for item in generator:
                    yield self.serializer.serialize(item)
            return EventSourceResponse(generator_wrapper(response))
        else:
            return self.serializer.serialize(response)

    def sync_network_loop(self):
        while True:
            try:
                r = self.sync_network()
            except Exception as e:
                r = c.detailed_error(e)
            c.print(r)
            c.sleep(self.max_network_staleness)

    def sync_network(self, update=False):
        path = self.resolve_path(f'{self.network}/state.json')
        state = self.get(path, {}, max_age=self.max_network_staleness)
        network = self.network
        staleness = c.time() - state.get('sync_time', 0)
        self.address2key = c.address2key()
        response = { 'path': path,  'max_network_staleness':  self.max_network_staleness,  'network': network,'staleness': int(staleness), }
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
        self.put(path, self.state)
        return response

    @classmethod
    def kill(cls, 
             module,
             mode:str = 'pm2',
             verbose:bool = False,
             prefix_match = False,
             network = 'local', # local, dev, test, main
             **kwargs):
        kill_fn = getattr(cls, f'{mode}_kill')
        kill_fn(module, verbose=verbose,prefix_match=prefix_match, **kwargs)
        c.deregister_server(module, network=network)
        assert c.server_exists(module, network=network) == False, f'module {module} still exists'
        return {'msg': f'removed {module}'}

    @classmethod
    def kill_many(cls, servers, search:str = None, network='local',  timeout=10, **kwargs):
        servers = c.servers(network=network)
        servers = [s for s in servers if  search in s]
        futures = []
        n = len(servers)
        for i, s in enumerate(servers):
            future = c.submit(c.kill, kwargs={'module':s, **kwargs}, timeout=timeout)
            futures.append(future)
        results = []
        for r in c.as_completed(futures, timeout=timeout):
            c.print(f'Killed {s} ({i+1})/{n})', color='red')
            results += [r.result()]
        c.print(f'Killed {len(results)} servers', color='red')
        return results

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

    @classmethod
    def kill_all(cls, network='local', timeout=20, verbose=True):
        futures = []
        servers = c.servers(network=network)
        n = len(servers)
        progress = c.tqdm(n)
        for s in servers:
            c.print(f'Killing {s}', color='red')
            futures += [c.submit(c.kill, kwargs={'module':s, 'update': False}, return_future=True)]
        results_list = []
        for f in c.as_completed(futures, timeout=timeout):
            result = f.result()
            print(result)
            progress.update(1)
            results_list += [result]
        namespace = c.get_namespace(network=network, update=True)
        print(namespace)
        new_n = len(servers)
        c.print(f'Killed {n - new_n} servers, with {n} remaining {servers}', color='red')
        return {'success':True, 'old_n':n, 'new_n':new_n, 'servers':servers, 'namespace':namespace}

 # the default
    network : str = 'local'

    @classmethod
    def rm_server(self,  name:str, network=network):
        return self.deregister_server(name, network=network)
    
    @classmethod
    def get_address(cls, name:str, network:str=network, external:bool = True) -> dict:
        namespace = cls.namespace(network=network)
        address = namespace.get(name, None)
        if external and address != None:
            address = address.replace(c.default_ip, c.ip()) 
        return address

    @classmethod
    def modules(cls, network:List=network) -> List[str]:
        return list(cls.namespace(network=network).keys())
    
    @classmethod
    def addresses(cls, network:str=network, **kwargs) -> List[str]:
        return list(cls.namespace(network=network, **kwargs).values())
    
    @classmethod
    def check_servers(self, *args, **kwargs):
        servers = c.pm2ls()
        namespace = c.get_namespace(*args, **kwargs)
        c.print('Checking servers', color='blue')
        for server in servers:
            if server in namespace:
                c.print(c.pm2_restart(server))

        return {'success': True, 'msg': 'Servers checked.'}
    
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

    def get_server_schema(self, module,  helper_function_attributes) -> 'Schema':
        schema = {}
        functions =  []
        for k in helper_function_attributes:
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