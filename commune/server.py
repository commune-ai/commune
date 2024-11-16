import commune as c
from typing import *
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from sse_starlette.sse import EventSourceResponse
import uvicorn
import os
import json
import asyncio

class Server(c.Module):
    tag_seperator:str='::'
    pm2_dir = os.path.expanduser('~/.pm2')
    functions_attributes : List[str] =['helper_functions', 'whitelist','endpoints','functions', 'fns', 'server_functions', 'public']
    user_functions = ['user_count', 'user_paths','user_data','user2count', 'user_path2latency','user_path2time', 'remove_user_data', 'users']
    helper_functions : List[str]  = ['info', 'metadata', 'schema', 'name', 'functions','key_address', 'crypto_type','fns', 'forward', 'rate_limit']
    max_bytes:int = 10 * 1024 * 1024  # max bytes within the request (bytes)
    allow_origins: List[str] = ["*"] # allowed origins
    allow_credentials: bool =True # allow credentials
    allow_methods: List[str] = ["*"] # allowed methods
    allow_headers: List[str] = ["*"]  # allowed headers
    period : int = 3600 # the period for 
    max_request_staleness : int = 4 # (in seconds) the time it takes for the request to be too old
    max_network_staleness: int = 60 #  (in seconds) the time it takes for. the network to refresh

    def __init__(
        self, 
        module: Union[c.Module, object] = None,
        functions:Optional[List[str]] = None, # list of endpoints
        key:str = None, # key for the server (str)
        name: str = None, # the name of the server
        port: Optional[int] = None, # the port the server is running on
        network:str = 'subspace', # the network used for incentives
        fn2cost : Dict[str, float] = None, # the cost of the function
        serializer = 'serializer',
        kwargs : dict = None, # the kwargs for the module
        ) -> 'Server':

        functions = functions or []
        module = module or 'module'
        if self.tag_seperator in name:
            # module::fam -> module=module, name=module::fam key=module::fam (default)
            module, tag = name.split(self.tag_seperator) 
        if isinstance(module, str):
            name = name or module
            module_class = c.module(module)
            kwargs = kwargs or {}
            module =  module_class(**kwargs)
        module.name = name
        module.key = c.get_key(key or module.name, create_if_not_exists=True)
        module.key_address = module.key.ss58_address
        module.crypto_type = module.key.crypto_type
        if not hasattr(module, 'fn2cost'):
            module.fn2cost = fn2cost or {}
        functions  =  sorted(list(set(functions + self.helper_functions)))
        schema = {}
        functions =  functions or []
        for k in self.functions_attributes:
            if hasattr(module, k):
                fn_obj = getattr(module, k)
                if isinstance(fn_obj, list):
                    functions += fn_obj
        # get function decorators form c.endpoint()
        for f in dir(module):
            try:
                if hasattr(getattr(module, f), '__metadata__'):
                    functions.append(f)
            except Exception as e:
                c.print(f'Error in get_endpoints: {e} for {f}')
        for fn in functions :
            if hasattr(module, fn):
                fn_obj = getattr(module, fn )
                if callable(fn_obj):
                    schema[fn] = c.fn_schema(fn_obj)['input']
                else: 
                    schema[fn] = {'type': str(type(fn_obj)).split("'")[1]}
        for fn in self.user_functions:
            setattr(module, fn, getattr(self, fn))
        if port in [None, 'None']:
            namespace = c.namespace()
            if name in namespace:
                try:
                    port =  int(namespace.get(module.name).split(':')[-1])
                except:
                    port = c.free_port()
            else:
                port = c.free_port()
        if c.port_used(port):
            port = c.free_port()
        module.ip = c.ip()
        module.port =  port or c.free_port()
        module.address =  f"{module.ip}:{module.port}"
        module.functions = functions
        module.schema = dict(sorted(schema.items()))
        module.info = self.get_info(module)
        self.network = network
        self.network_path = self.resolve_path(f'networks/{self.network}/state.json')
        self.users_path = self.resolve_path(f'users/{name}')
        self.serializer = c.module(serializer)()
        self.sync(update=False)  
        c.thread(self.sync_loop)
        self.loop = asyncio.get_event_loop()
        app = FastAPI()  
        app.add_middleware(self.Middleware, max_bytes=self.max_bytes)    
        app.add_middleware(CORSMiddleware, 
                           allow_origins=self.allow_origins, 
                           allow_credentials=self.allow_credentials,
                           allow_methods=self.allow_methods,
                           allow_headers=self.allow_headers)
        def api_forward(fn:str, request: Request):
            return self.forward(fn, request)
        app.post("/{fn}")(api_forward)
        c.print(f'Served(name={module.name}, address={module.address}, key={module.key}) 🚀 ', color='purple')
        c.register_server(name=module.name,address=module.address, key=module.key.ss58_address)
        self.module = module 

        uvicorn.run(app, host='0.0.0.0', port=module.port, loop='asyncio')


    def rate_limit(self, 
                   address:str,  
                    fn: str= 'info', 
                    multipliers : Dict[str, float] = {'stake': 1, 'stake_to': 1,'stake_from': 1},
                    rates : Dict[str, int]= {'max': 10,  'local': 10000, 'stake2rate': 1000, 'admin': 10000}, # the maximum rate 
                    ) -> float:
        # stake rate limit
        module = self.module
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

    def forward(self, fn:str, request: Request, catch_exception:bool=True) -> dict:
        if catch_exception:
            try:
                return self.forward(fn, request, catch_exception=False)
            except Exception as e:
                return c.detailed_error(e)
        module = self.module
        headers = dict(request.headers.items())
        address = headers.get('key', headers.get('address', None))
        assert address, 'No key or address in headers'
        request_staleness = c.time() - float(headers['time'])
        assert  request_staleness < self.max_request_staleness, f"Request is too old ({request_staleness}s > {self.max_request_staleness}s (MAX)" 
        data = self.loop.run_until_complete(request.json())
        data = self.serializer.deserialize(data) 
        request = {'data': data, 'headers': headers}
        auth={'data': c.hash(data), 'time': headers['time']}
        signature = headers.get('signature', None)
        assert c.verify(auth=auth,signature=signature, address=address), 'Invalid signature'
        server_signature = self.module.key.sign(headers)
        kwargs = dict(data.get('kwargs', {}))
        args = list(data.get('args', []))
        data = {'args': args, 'kwargs': kwargs}
        is_admin = bool(c.is_admin(address) or  address == self.module.key.ss58_address)
        if not is_admin:
            assert not bool(fn.startswith('__') or fn.startswith('_')), f'Function {fn} is private'
            assert fn in self.module.functions , f"Function {fn} not in endpoints={self.module.functions}"
        count = self.user_count(address)
        rate_limit = self.rate_limit(fn=fn, address=address)
        assert count <= rate_limit, f'rate limit exceeded {count} > {rate_limit}'
        fn_obj = getattr(self.module, fn)
        if is_admin:
            RANK = 'ADMIN'
        elif address in self.state['address2key']:
            RANK = 'LOCAL'
        else:
            RANK = 'NA'
        start_time = float(headers['time'])
        result = fn_obj(*data['args'], **data['kwargs']) if callable(fn_obj) else fn_obj
        end_time = c.time()
        latency = c.round(end_time - start_time, 3)
        if c.is_generator(result):
            output = []
            def generator_wrapper(generator):
                try:
                    for item in generator:
                        output_item = self.serializer.serialize(item)
                        yield output_item
                except Exception as e:
                    c.print(e)
                    yield str(c.detailed_error(e))
            result = EventSourceResponse(generator_wrapper(result))
        else:
            output =  self.serializer.serialize(result)

        user_data = {
            'module': module.name,
            'fn': fn,
            'input': data, # the data of the request
            'output': output, # the response
            'latency':  latency, # the latency
            'time': start_time, # the time of the request
            'user_key': address, # the key of the user
            'server_key': self.module.key.ss58_address, # the key of the server
            'user_signature': signature, # the signature of the user
            'server_signature': server_signature, # the signature of the server
            'cost': self.module.fn2cost.get(fn, 1), # the cost of the function
        }
        
        user_path = self.user_path(user_data["user_key"]) + f'/{user_data["fn"]}/{c.time()}.json' # get the user info path
        c.put(user_path, user_data)
        return result
    
    def sync_loop(self, sync_loop_initial_sleep=4):
        c.sleep(sync_loop_initial_sleep)
        while True:
            try:
                r = self.sync()
            except Exception as e:
                r = c.detailed_error(e)
                c.print('Error in sync_loop -->', r, color='red')
            c.sleep(self.max_network_staleness)

    def sync(self, update=True ):
        t0 = c.time()
        if hasattr(self, 'state'):
            latency = c.time() - self.state['time']
            if latency < self.max_network_staleness:
                return {'msg': 'state is fresh'}
        max_age = self.max_network_staleness
        network_path = self.network_path
        state = self.get(network_path, {}, max_age=max_age, updpate=update)
        network = self.network
        state = {}
        state['address2key'] =  c.address2key()
        state['key2address'] = {v:k for k,v in state['address2key'].items()}
        state['stake'] = {}
        state['stake_to'] = {}
        state['stake_from'] = {}
        if update:
            try  : 
                c.namespace(max_age=max_age)
                self.subspace = c.module('subspace')(network=network)
                state['stake_from'] = self.subspace.stake_from(fmt='j', update=update, max_age=max_age)
                state['stake_to'] = self.subspace.stake_to(fmt='j', update=update, max_age=max_age)
                state['stake'] =  {k: sum(v.values()) for k,v in state['stake_from'].items()}
            except Exception as e:
                c.print(f'Error {e} while syncing network--> {network}')
        state['time'] = c.time()
        state['latency'] = state['time'] - t0
        state_keys = ['stake_from', 'stake_to', 'address2key', 'stake', 'key2address', 'time', 'latency']
        is_valid_state = lambda x: all([k in x for k in state_keys])
        assert is_valid_state(state), f'Format for network state is {[k for k in state_keys if k not in state]}'
        self.put(network_path, state)
        self.state = state
        return {'msg': 'state synced successfully'}

    @classmethod
    def wait_for_server(cls,
                          name: str ,
                          network: str = 'local',
                          timeout:int = 600,
                          max_age = 1,
                          sleep_interval: int = 1) -> bool :
        
        time_waiting = 0
        # rotating status thing
        c.print(f'Waiting for {name} to start', color='cyan')
        
        while time_waiting < timeout:
                namespace = c.namespace(network=network, max_age=max_age)
                if name in namespace:
                    try:
                        result = c.call(namespace[name]+'/info')
                        if 'key' in result:
                            c.print(f'{name} is running', color='green')
                            return result
                    except Exception as e:
                        c.print(f'Error getting info for {name} --> {e}', color='red')
                c.sleep(sleep_interval)
                time_waiting += sleep_interval
            # c.logs(name)
            # c.kill(name)
        raise TimeoutError(f'Waited for {timeout} seconds for {name} to start')

    def get_info(self, module):
        info = {}
        info['schema'] = module.schema
        info['name'] = module.name 
        info['address'] = module.address
        info['key'] = module.key.ss58_address
        info['crypto_type'] = module.key.crypto_type
        return info

    
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
                **c.fn_schema(fn),
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


    class Middleware(BaseHTTPMiddleware):
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

    @classmethod
    def kill(cls, name:str, verbose:bool = True, **kwargs):
        try:
            if name == 'all':
                return cls.kill_all(verbose=verbose)
            c.cmd(f"pm2 delete {name}", verbose=False)
            cls.rm_logs(name)
            result =  {'message':f'Killed {name}', 'success':True}
        except Exception as e:
            result =  {'message':f'Error killing {name}', 'success':False, 'error':e}

        c.deregister_server(name)
        return result
    
    @classmethod
    def kill_all_processes(cls, verbose:bool = True, timeout=20):
        servers = cls.processes()
        futures = [c.submit(cls.kill, kwargs={'name':s, 'update': False}, return_future=True) for s in servers]
        return c.wait(futures, timeout=timeout)

    @classmethod
    def kill_all_servers(cls, network='local', timeout=20, verbose=True):
        servers = c.servers(network=network)
        futures = [c.submit(cls.kill, kwargs={'module':s, 'update': False}, return_future=True) for s in servers]
        return c.wait(futures, timeout=timeout)
    
    @classmethod
    def kill_all(cls, mode='process', verbose:bool = True, timeout=20):
        if mode == 'process':
            return cls.kill_all_processes(verbose=verbose, timeout=timeout)
        elif mode == 'server':
            return cls.kill_all_servers(verbose=verbose, timeout=timeout)
        else:
            raise NotImplementedError(f'mode {mode} not implemented')


    @classmethod
    def logs_path_map(cls, name=None):
        logs_path_map = {}
        for l in c.ls(f'{cls.pm2_dir}/logs/'):
            key = '-'.join(l.split('/')[-1].split('-')[:-1]).replace('-',':')
            logs_path_map[key] = logs_path_map.get(key, []) + [l]
        for k in logs_path_map.keys():
            logs_path_map[k] = {l.split('-')[-1].split('.')[0]: l for l in list(logs_path_map[k])}
        if name != None:
            return logs_path_map.get(name, {})

        return logs_path_map
    
   
    @classmethod
    def rm_logs( cls, name):
        logs_map = cls.logs_path_map(name)
        for k in logs_map.keys():
            c.rm(logs_map[k])

    @classmethod
    def logs(cls, 
                module:str, 
                tail: int =100, 
                mode: str ='cmd',
                **kwargs):
        
        if mode == 'local':
            text = ''
            for m in ['out','error']:
                # I know, this is fucked 
                path = f'{cls.pm2_dir}/logs/{module.replace("/", "-")}-{m}.log'.replace(':', '-').replace('_', '-')
                try:
                    text +=  c.get_text(path, tail=tail)
                except Exception as e:
                    c.c.print('ERROR GETTING LOGS -->' , e)
                    continue
            return text
        elif mode == 'cmd':
            return c.cmd(f"pm2 logs {module}", verbose=True)
        else:
            raise NotImplementedError(f'mode {mode} not implemented')
        
    @classmethod
    def kill_many(cls, search=None, verbose:bool = True, timeout=10):
        futures = []
        for name in c.servers(search=search):
            f = c.submit(c.kill, dict(name=name, verbose=verbose), return_future=True, timeout=timeout)
            futures.append(f)
        return c.wait(futures)
    
    @classmethod
    def launch(cls, 
                  fn: str = 'serve',
                   module:str = None,  
                   name:Optional[str]=None, 
                   args : list = None,
                   kwargs: dict = None,
                   interpreter:str='python3', 
                   autorestart: bool = True,
                   verbose: bool = False , 
                   force:bool = True,
                   meta_fn: str = 'module_fn',
                   cwd : str = None,
                   env : Dict[str, str] = None,
                   refresh:bool=True , 
                   **extra_kwargs):
        env = env or {}
        # get the module and fn
        if '/' in fn:
            module, fn = fn.split('/')
        module = module or cls
        if not isinstance(module, str):
            if hasattr(module, 'module_name'):
                module = module.module_name()
            else: 
                module = module.__name__
        name = name or module
        if refresh:
            c.kill(name)
        cmd = f"pm2 start {c.filepath()} --name {name} --interpreter {interpreter}"
        if not autorestart :
            cmd += cmd + ' --no-autorestart'
        if force:
            cmd = cmd + ' -f '
        kwargs =  {'module': module , 'fn': fn, 'args': args if args else [], 'kwargs': kwargs if kwargs else {} }
        kwargs_str = json.dumps(kwargs).replace('"', "'")
        cmd = cmd +  f' -- --fn {meta_fn} --kwargs "{kwargs_str}"'
        stdout = c.cmd(cmd, env=env, verbose=verbose, cwd=cwd)
        return {'success':True, 
                'msg':f'Launched {module}', 
                'cmd': cmd, 
                'stdout':stdout}
    remote_fn = launch

    @classmethod
    def restart(cls, name:str):
        assert name in cls.processes()
        c.c.print(f'Restarting {name}', color='cyan')
        c.cmd(f"pm2 restart {name}", verbose=False)
        cls.rm_logs(name)  
        return {'success':True, 'message':f'Restarted {name}'}
    
    @classmethod
    def processes(cls, search=None,  **kwargs) -> List[str]:
        output_string = c.cmd('pm2 status', verbose=False)
        module_list = []
        for line in output_string.split('\n')[3:]:
            if  line.count('│') > 2:
                name = line.split('│')[2].strip()
                module_list += [name]
        if search != None:
            module_list = [m for m in module_list if search in m]
        module_list = sorted(list(set(module_list)))
        return module_list

    pm2ls = pids = procs = processes 

    @classmethod
    def process_exists(cls, name:str, **kwargs) -> bool:
        return name in cls.processes(**kwargs)

    @classmethod
    def serve(cls, 
              module: Any = None,
              kwargs:Optional[dict] = None,  # kwargs for the module
              port :Optional[int] = None, # name of the server if None, it will be the module name
              name = None, # name of the server if None, it will be the module name
              remote:bool = True, # runs the server remotely (pm2, ray)
              functions = None,
              key = None,
              **extra_kwargs
              ):
        module = module or 'module'
        name = name or module
        kwargs = {**(kwargs or {}), **extra_kwargs}
        if remote:
            rkwargs = {k : v for k, v  in c.locals2kwargs(locals()).items()  if k not in ['extra_kwargs', 'response', 'namespace']}
            rkwargs['remote'] = False
            c.remote_fn('serve', name=name, kwargs=rkwargs)
            return cls.wait_for_server(name)
        return Server(module=module, 
                      name=name, 
                      port=port, 
                      key=key, 
                      functions = functions,
                      kwargs=kwargs)
    

    @classmethod
    def fleet(cls, module, n:int = 1, **kwargs):
        futures = []
        for _ in range(n):

            future = c.submit(c.serve, dict(module=module, name = module + '::' + str(_),  **kwargs))
            futures.append(future)
        for future in c.as_completed(futures):
            c.c.print(future.result())
        return {'success':True, 'message':f'Served {n} servers', 'namespace': c.namespace()} 

    def check_all_users(self):
        for user in self.users():
            c.print('Checking', user)
            self.chekcer_user_data()

    def extract_time(self, x):
        try:
            x = float(x.split('/')[-1].split('.')[0])
        except Exception as e:
            c.print(e)
            x = 0
        return x

    def remove_user_data(self, address):
        return c.rm(self.user_path(address))

    def users(self):
        return os.listdir(self.users_path)

    def user2count(self):
        user2count = {}
        for user in self.users():
            user2count[user] = self.user_count(user)
        return user2count

    def user_paths(self, address ):
        user_paths = c.glob(self.user_path(address))
        return sorted(user_paths, key=self.extract_time)
    
    def user_data(self, address):
        for i, user_path in enumerate(self.user_paths(address)):
            yield c.get(user_path)
        
    def user_path(self, address):
        return self.users_path + '/' + address

    def user_count(self, address):
        self.check_user_data(address)
        return len(self.user_paths(address))
    
    def user_path2time(self, address):
        user_paths = self.user_paths(address)
        user_path2time = {user_path: self.extract_time(user_path) for user_path in user_paths}
        return user_path2time
    
    def user_path2latency(self, address):
        user_paths = self.user_paths(address)
        t0 = c.time()
        user_path2time = {user_path: t0 - self.extract_time(user_path) for user_path in user_paths}
        return user_path2time
    
    def check_user_data(self, address):
        path2latency = self.user_path2latency(address)
        for path, latency  in path2latency.items():
            if latency > self.period:
                c.print(f'Removing stale path {path} ({latency}/{self.period})')
                os.remove(path)


Server.run(__name__)
