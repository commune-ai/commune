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

class Server:
    tag_seperator:str='::'
    user_data_lifetime = 3600
    pm2_dir = os.path.expanduser('~/.pm2')
    period : int = 3600 # the period for 
    max_request_staleness : int = 4 # (in seconds) the time it takes for the request to be too old
    max_network_staleness: int = 60 #  (in seconds) the time it takes for. the network to refresh
    multipliers : Dict[str, float] = {'stake': 1, 'stake_to': 1,'stake_from': 1}
    rates : Dict[str, int]= {'max': 10, 'local': 10000, 'stake': 1000, 'owner': 10000, 'admin': 10000} # the maximum rate  ):
    helper_functions  = ['info', 'metadata', 'schema', 'free', 'name', 'functions','key_address', 'crypto_type','fns', 'forward', 'rate_limit'] # the helper functions
    functions_attributes =['helper_functions', 'whitelist', "whitelist_functions", 'endpoints', 'functions',  'fns', "exposed_functions",'server_functions', 'public_functions'] # the attributes for the functions
    def __init__(
        self, 
        module: Union[c.Module, object] = None,
        key:str = None, # key for the server (str)
        name: str = None, # the name of the server
        functions:Optional[List[Union[str, callable]]] = None, # list of endpoints
        port: Optional[int] = None, # the port the server is running on
        network:str = 'subspace', # the network used for incentives
        fn2cost : Dict[str, float] = None, # the cost of the function
        free : bool = False, # if the server is free (checks signature)
        kwargs : dict = None, # the kwargs for the module
        crypto_type = 'sr25519', # the crypto type of the key
        users_path: Optional[str] = None, # the path to the user data
        serializer: str = 'serializer', # the serializer used for the data
        ) -> 'Server':
        module = module or 'module'
        kwargs = kwargs or {}
        if self.tag_seperator in str(name):
            # module::fam -> module=module, name=module::fam key=module::fam (default)
            module, tag = name.split(self.tag_seperator) 
            module = c.module(module)(**kwargs)
        if isinstance(module, str):
            name = name or module
            module =   c.module(module)(**kwargs)
        print(f'Launching', module, name, functions)
        # NOTE: ONLY ENABLE FREEMODE IF YOU ARE ON A CLOSED NETWORK,
        self.free = free
        self.module = module
        self.module.name = name    
        self.set_key(key=key, crypto_type=crypto_type)
        self.set_port(port)
        self.set_network(network)  
        self.set_functions(functions=functions, fn2cost=fn2cost)
        self.set_user_path(users_path)
        self.serializer = c.module(serializer)()
        self.start_server()

    def set_functions(self, functions:Optional[List[str]] , fn2cost=None):
        if self.free:
            c.print('THE FOUNDING FATHERS WOULD BE PROUD OF YOU SON OF A BITCH', color='red')
        else:
            if hasattr(self.module, 'free'):
                self.free = self.module.free
        self.module.free = self.free
        functions =  functions or []
        for i, fn in enumerate(functions):
            if callable(fn):
                print('Adding function', f)
                setattr(self, fn.__name__, fn)
                functions[i] = fn.__name__
        functions  =  sorted(list(set(functions + self.helper_functions)))
        module = self.module
        for k in self.functions_attributes:
            if hasattr(module, k) and isinstance(getattr(module, k), list):
                print('Found ', k)
                functions = getattr(module, k)
                break
        # get function decorators form c.endpoint()
        for f in dir(module):
            try:
                if hasattr(getattr(module, f), '__metadata__'):
                    functions.append(f)
            except Exception as e:
                c.print(f'Error in get_endpoints: {e} for {f}')
        module.functions = sorted(list(set(functions)))
        ## get the schema for the functions
        schema = {}
        for fn in functions :
            if hasattr(module, fn):
                schema[fn] = c.schema(getattr(module, fn ))
            else:
                print(f'Function {fn} not found in {module.name}')
        module.schema = dict(sorted(schema.items()))
        module.fn2cost = module.fn2cost if hasattr(module, 'fn2cost') else (fn2cost or {})
        assert isinstance(module.fn2cost, dict), f'fn2cost must be a dict, not {type(module.fn2cost)}'

        ### get the info for the module
        module.info = {
            "functions": functions,
            "schema": schema,
            "name": module.name,
            "address": module.address,
            "key": module.key.ss58_address,
            "crypto_type": module.key.crypto_type,
            "fn2cost": module.fn2cost,
            "free": module.free,
            "time": c.time()
        }
        
    def set_key(self, key, crypto_type):
        module = self.module
        module.key = c.get_key(key or module.name, create_if_not_exists=True, crypto_type=crypto_type)
        module.key_address = module.key.key_address
        module.crypto_type = module.key.crypto_type
        return {'success':True, 'message':f'Set key to {module.key.ss58_address}'}
    
    def start_server(self,
            max_bytes = 10 * 1024 * 1024 , # max bytes within the request (bytes)
            allow_origins  = ["*"], # allowed origins
            allow_credentials  =True, # allow credentials
            allow_methods  = ["*"], # allowed methods
            allow_headers = ["*"] , # allowed headers
    ):
        module = self.module
        c.thread(self.sync_loop)
        self.loop = asyncio.get_event_loop()
        app = FastAPI()  
        app.add_middleware(Middleware, max_bytes=max_bytes)    
        app.add_middleware(CORSMiddleware,  
                           allow_origins=allow_origins, 
                           allow_credentials=allow_credentials, 
                           allow_methods=allow_methods,  
                           allow_headers=allow_headers)
        def api_forward(fn:str, request: Request):
            return self.forward(fn, request)
        app.post("/{fn}")(api_forward)
        c.print(f'Served(name={module.name}, address={module.address}, key={module.key.key_address})', color='purple')
        c.print(c.add_server(name=module.name, address=module.address, key=module.key.ss58_address))
        self.module = module 
        uvicorn.run(app, host='0.0.0.0', port=module.port, loop='asyncio')

    def set_port(self, port:Optional[int]=None, port_attributes = ['port', 'server_port'], ip = None):
        module = self.module
        name = module.name
        for k in port_attributes:
            if hasattr(module, k):
                port = getattr(module, k)
                break
        if port in [None, 'None']:
            namespace = c.namespace()
            if name in namespace:
                c.kill(name)
                try:
                    port =  int(namespace.get(module.name).split(':')[-1])
                except:
                    port = c.free_port()
            else:
                port = c.free_port()

        while c.port_used(port):
            c.kill_port(port)
            c.sleep(1)
            print(f'Waiting for port {port} to be free')

        module.port = port
        ip = ip or '0.0.0.0' 
        module.address = ip + ':' + str(module.port)
        self.module = module
        return {'success':True, 'message':f'Set port to {port}'}
    
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
        module = self.module
        address = headers['key']
        if c.is_admin(address):
            rate_limit =  self.rates['admin']
        elif address == module.key.ss58_address:
            rate_limit =  self.rates['owner']
        elif address in self.address2key:
            rate_limit =  self.rates['local']
        else:
            stake_score = self.state['stake'].get(address, 0) + self.multipliers['stake']
            stake_to_score = (sum(self.state['stake_to'].get(address, {}).values())) * self.multipliers['stake_to']
            stake_from_score = self.state['stake_from'].get(module.key.ss58_address, {}).get(address, 0) * self.multipliers['stake_from']
            stake = stake_score + stake_to_score + stake_from_score
            self.rates['stake'] = self.rates['stake'] * module.fn2cost.get(fn, 1)
            rate_limit =  min((stake / self.rates['stake']), self.rates['max'])
        count = self.user_call_count(headers['key'])
        assert count <= rate_limit, f'rate limit exceeded {count} > {rate_limit}'     
        return True
    
    def get_data(self, request: Request):
        data = self.loop.run_until_complete(request.json())
        # data = self.serializer.deserialize(data) 
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
                c.print(result, color='red')
                return result
        module = self.module
        data = self.get_data(request)
        headers = self.get_headers(request)
        self.gate(fn=fn, data=data, headers=headers)   
        is_admin = bool(c.is_admin(headers['key']))
        is_owner = bool(headers['key'] == module.key.ss58_address)    
        if hasattr(module, fn):
            fn_obj = getattr(module, fn)
        elif (is_admin or is_owner) and hasattr(self, fn):
            fn_obj = getattr(self, fn)
        else:
            raise Exception(f"{fn} not found in {module.name}")
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
                    'fn': fn,
                    'data': data, # the data of the request
                    'output': output, # the response
                    'time': headers["time"], # the time of the request
                    'latency': latency, # the latency of the request
                    'key': headers['key'], # the key of the user
                    'cost': module.fn2cost.get(fn, 1), # the cost of the function
                }
            user_path = self.user_path(f'{user_data["key"]}/{user_data["fn"]}/{c.time()}.json') 
            c.put(user_path, user_data)
        return result
    
    def sync_loop(self, sync_loop_initial_sleep=10):
        c.sleep(sync_loop_initial_sleep)
        while True:
            try:
                r = self.sync()
            except Exception as e:
                r = c.detailed_error(e)
                c.print('Error in sync_loop -->', r, color='red')
            c.sleep(self.max_network_staleness)

    def set_network(self, network):
        self.network = network
        self.network_path = self.resolve_path(f'networks/{self.network}/state.json')
        self.address2key =  c.address2key()
        c.thread(self.sync_loop)
        return {'success':True, 'message':f'Set network to {network}', 'network':network, 'network_path':self.network_path}

    def sync(self, update=True , state_keys = ['stake_from', 'stake_to']):
        self.network_path = self.resolve_path(f'networks/{self.network}/state.json')
        print(f'Sync({self.network_path})')
        if hasattr(self, 'state'):
            latency = c.time() - self.state.get('time', 0)
            if latency < self.max_network_staleness:
                return {'msg': 'state is fresh'}
        max_age = self.max_network_staleness
        network_path = self.network_path
        state = c.get(network_path, {}, max_age=max_age, updpate=update)
        state = {}
        state['stake'] = {}
        state['stake_to'] = {}
        state['stake_from'] = {}
        if update:
            try  : 
                c.namespace(max_age=max_age)
                self.subspace = c.module('subspace')(network=self.network)
                state['stake_from'] = self.subspace.stake_from(fmt='j', update=update, max_age=max_age)
                state['stake_to'] = self.subspace.stake_to(fmt='j', update=update, max_age=max_age)
                state['stake'] =  {k: sum(v.values()) for k,v in state['stake_from'].items()}
            except Exception as e:
                c.print(f'Error {e} while syncing network')
        is_valid_state = lambda x: all([k in x for k in state_keys])
        assert is_valid_state(state), f'Format for network state is {[k for k in state_keys if k not in state]}'
        c.put(network_path, state)
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
        c.print(f'waiting for {name} to start...', color='cyan')
    
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
        raise TimeoutError(f'Waited for {timeout} seconds for {name} to start')

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
            fn.__dict__['__metadata__'] = metadata
            return fn
        return decorator_fn
    
    serverfn = endpoint

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

        c.remove_server(name)
        return result
    
    @classmethod
    def kill_all_processes(cls, verbose:bool = True, timeout=20):
        servers = cls.processes()
        futures = [c.submit(cls.kill, kwargs={'name':s, 'update': False}, return_future=True) for s in servers]
        results = c.wait(futures, timeout=timeout)

        return results

    @classmethod
    def kill_all_servers(cls, network='local', timeout=20, verbose=True):
        servers = c.servers(network=network)
        futures = [c.submit(cls.kill, kwargs={'module':s, 'update': False}, return_future=True) for s in servers]
        return c.wait(futures, timeout=timeout)
    
    @classmethod
    def kill_all(cls, mode='process', verbose:bool = True, timeout=20):
        if mode == 'process':
            results =  cls.kill_all_processes(verbose=verbose, timeout=timeout)
        elif mode == 'server':
            results =  cls.kill_all_servers(verbose=verbose, timeout=timeout)
        else:
            raise NotImplementedError(f'mode {mode} not implemented')
        c.namespace(update=True)
        return results
    
    @classmethod
    def killall(cls, **kwargs):
        return cls.kill_all(**kwargs)
    
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
    def logs(cls, module:str,  tail: int =100,   mode: str ='cmd', **kwargs):
        
        if mode == 'local':
            text = ''
            for m in ['out','error']:
                # I know, this is fucked 
                path = f'{cls.pm2_dir}/logs/{module.replace("/", "-")}-{m}.log'.replace(':', '-').replace('_', '-')
                try:
                    text +=  c.get_text(path, tail=tail)
                except Exception as e:
                    c.print('ERROR GETTING LOGS -->' , e)
                    continue
            return text
        elif mode == 'cmd':
            return c.cmd(f"pm2 logs {module}", verbose=True)
        else:
            raise NotImplementedError(f'mode {mode} not implemented')
        
    def get_logs(self, tail=100, mode='local'):
        return self.logs(self.module.name, tail=tail, mode=mode)
        
    @classmethod
    def kill_many(cls, search=None, verbose:bool = True, timeout=10):
        futures = []
        for name in c.servers(search=search):
            f = c.submit(c.kill, dict(name=name, verbose=verbose), return_future=True, timeout=timeout)
            futures.append(f)
        return c.wait(futures)
    
    @classmethod
    def start_process(cls, 
                  fn: str = 'serve',
                   module:str = None,  
                   name:Optional[str]=None, 
                   args : list = None,
                   kwargs: dict = None,
                   interpreter:str='python3', 
                   autorestart: bool = True,
                   verbose: bool = False , 
                   force:bool = True,
                   run_fn: str = 'run_fn',
                   cwd : str = None,
                   env : Dict[str, str] = None,
                   refresh:bool=True , 
                   **extra_kwargs):
        env = env or {}
        if '/' in fn:
            module, fn = fn.split('/')
        module = module or cls.module_name()
        name = name or module
        if refresh:
            cls.kill(name)
        cmd = f"pm2 start {c.filepath()} --name {name} --interpreter {interpreter}"
        cmd = cmd  if autorestart else ' --no-autorestart' 
        cmd = cmd + ' -f ' if force else cmd
        kwargs =  {'module': module ,  'fn': fn, 'args': args or [],  'kwargs': kwargs or {} }
        kwargs_str = json.dumps(kwargs).replace('"', "'")
        cmd = cmd +  f' -- --fn {run_fn} --kwargs "{kwargs_str}"'
        stdout = c.cmd(cmd, env=env, verbose=verbose, cwd=cwd)
        return {'success':True, 'msg':f'Launched {module}',  'cmd': cmd, 'stdout':stdout}
    remote_fn = launch = start_process

    @classmethod
    def restart(cls, name:str):
        assert name in cls.processes()
        c.print(f'Restarting {name}', color='cyan')
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
    
    @classmethod
    def procs(cls, **kwargs):
        return cls.processes(**kwargs)
    

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
              functions = None, # list of functions to serve, if none, it will be the endpoints of the module
              key = None, # the key for the server
              free = False,
              cwd = None,
              **extra_kwargs
              ):
        module = module or 'module'
        name = name or module
        kwargs = {**(kwargs or {}), **extra_kwargs}
        c.print(f'Serving(module={module} params={kwargs} name={name} function={functions})')
        if not isinstance(module, str):
            remote = False
        if remote:
            rkwargs = {k : v for k, v  in c.locals2kwargs(locals()).items()  if k not in ['extra_kwargs', 'response', 'namespace']}
            rkwargs['remote'] = False
            cls.start_process(fn='serve', name=name, kwargs=rkwargs, cwd=cwd)
            return cls.wait_for_server(name)
        return Server(module=module, name=name, functions = functions, kwargs=kwargs, port=port,  key=key, free = free)

    def extract_time(self, x):
        try:
            x = float(x.split('/')[-1].split('.')[0])
        except Exception as e:
            x = 0
        return x

    @classmethod
    def fleet(cls, module, n=10, timeout=10):
        futures = [ c.submit(c.serve, {'module':module + '::' + str(i)}, timeout=timeout) for i in range(n)]
        progress = c.progress(futures)
        results = []
        for f in c.as_completed(futures, timeout=timeout):
            r = f.result()
            results.append(r)
            progress.update()
        return results
    def remove_user_data(self, address):
        return c.rm(self.user_path(address))

    def users(self):
        return os.listdir(self.users_path)

    def user2count(self):
        user2count = {}
        for user in self.users():
            user2count[user] = self.user_call_count(user)
        return user2count
    
    def history(self, user):
        return self.user_data(user)
    
    def user2fn2count(self):
        user2fn2count = {}
        for user in self.users():
            user2fn2count[user] = {}
            for user_data in self.user_data(user):
                fn = user_data['fn']
                user2fn2count[user][fn] = user2fn2count[user].get(fn, 0) + 1
        return user2fn2count

    def user_call_paths(self, address ):
        user_paths = c.glob(self.user_path(address))
        return sorted(user_paths, key=self.extract_time)
    
    def user_data(self, address, stream=False):
        user_paths = self.user_call_paths(address)
        if stream:
            def stream_fn():
                for user_path in user_paths:
                    yield c.get(user_path)
            return stream_fn()
        
        else:
            return [c.get(user_path) for user_path in user_paths]
        
    def user_path(self, key_address):
        return self.users_path + '/' + key_address

    def user_call_count(self, user):
        self.check_user_data(user)
        return len(self.user_call_paths(user))
    
    def users(self):
        return os.listdir(self.users_path)
    
    def user_path2time(self, address):
        user_paths = self.user_call_paths(address)
        user_path2time = {user_path: self.extract_time(user_path) for user_path in user_paths}
        return user_path2time
    
    def user_call_path2latency(self, address):
        user_paths = self.user_call_paths(address)
        t0 = c.time()
        user_path2time = {user_path: t0 - self.extract_time(user_path) for user_path in user_paths}
        return user_path2time
    
    def check_user_data(self, address):
        path2latency = self.user_call_path2latency(address)
        for path, latency  in path2latency.items():
            if latency > self.user_data_lifetime:
                c.print(f'Removing stale path {path} ({latency}/{self.period})')
                if os.path.exists(path):
                    os.remove(path)

    def resolve_path(self, path):
        return c.resolve_path(path, storage_dir=self.storage_dir())

    def set_user_path(self, users_path):
        self.users_path = users_path or self.resolve_path(f'users/{self.module.name}')

    
    def add_endpoint(self, name, fn):
        setattr(self, name, fn)
        self.endpoints.append(name)
        assert hasattr(self, name), f'{name} not added to {self.__class__.__name__}'
        return {'success':True, 'message':f'Added {fn} to {self.__class__.__name__}'}

if __name__ == '__main__':
    Server.run()

