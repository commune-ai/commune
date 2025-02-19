import commune as c
from typing import *
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import uvicorn
import os
import json
import asyncio
import os
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

    pm_dir = c.home_path + '/.pm2'
    helper_functions  = ['info', 'forward'] # the helper functions
    def __init__(
        self, 
        module: Union[c.Module, object] = None,
        key:str = None, # key for the server (str)
        functions:Optional[List[Union[str, callable]]] = None, # list of endpoints
        name: Optional[str] = None, # the name of the server
        params : dict = None, # the kwargs for the module
        port: Optional[int] = None, # the port the server is running on
        network = 'local', # the network the server is running on
        free : bool = False, # if the server is free (checks signature)
        middleware: Optional[callable] = None, # the middleware for the server
        history_path: Optional[str] = None, # the path to the user data
        run_api = False, # if the server should be run as an api
        executor = 'server.executor',  # the executor for the server to run the functions
        serializer = 'server.serializer', # the serializer for the server serializes and deserializes the data for the server if it is not a string
        tempo:int = 60 , # (in seconds) the maximum age of the history
        task_timeout:int = 10, # (in seconds) the maximum time to wait for a response
        ) -> 'Server':
        self.set_network(network)
        module = module or 'module'
        if isinstance(module, str):
            if '::' in str(module):
                name = module
                module, tag = name.split('::') 
        name = name or module
        self.module = c.module(module)(**(params or {}))
        self.module.name = name 
        self.module.key = c.get_key(key or name)
        self.history_path = history_path or self.resolve_path('history')
        if run_api:
            self.address2key =  c.address2key()
            self.serializer = c.module(serializer)()
            self.executor = c.module(executor)()
            self.set_functions(functions) 
            self.set_port(port)
            self.module.info = self.get_info(self.module)
            self.loop = asyncio.get_event_loop()
            self.app = FastAPI()
            self.set_middleware(self.app)
            def forward(fn: str, request: Request):
                try:
                    result =  c.wait(self.executor.submit(self.forward, params={"fn": fn, "request":request}), timeout=task_timeout)
                except Exception as e:
                    result =  c.detailed_error(e)
                return result
            self.app.post("/{fn}")(forward)
            c.print(f'Served(url={self.module.url}, name={self.module.name}, key={self.module.key.key_address})', color='purple')
            uvicorn.run(self.app, host='0.0.0.0', port=self.module.port, loop='asyncio')
            
    def forward(self, fn:str, request: dict) -> dict:
        t0 = c.time()
        request = self.get_request(request) # process the request
        params = request['params']
        headers = request['headers']
        client_key_address = headers['key']
        rate_limit = self.rate_limit(fn=fn, params=params, headers=headers)   
        rate = self.rate(self.module.name+'/'+headers['key'])
        assert rate < rate_limit, f'RateLimitExceeded({rate}>{rate_limit})'     
        c.print(f'Request(fn={fn} params={params} client={client_key_address} size={len(str(params))})', color='green')
        fn_obj = getattr(self.module, fn)
        result = fn_obj(*params['args'], **params['kwargs']) if callable(fn_obj) else fn_obj
        fn_latency = c.time() - float(t0)
        net_latency = t0 - float(headers['time']) # the latency of the request (in seconds) by subtracting the time the request was made from the current time
        if c.is_generator(result):
            output = str(result)
            def generator_wrapper(generator):
                for item in generator:
                    yield item
            result = EventSourceResponse(generator_wrapper(result))       
        else:
            call_data = headers
            call_data['fn'] = fn
            call_data['params'] = params
            call_data['output'] = result
            call_data['fn_latency'] = fn_latency
            call_data['net_latency'] = net_latency
            call_data['rate'] = rate
            call_data['rate_limit'] = rate_limit
            call_data['server'] = self.module.key.ss58_address
            call_data['server_signature'] = c.sign(call_data, key=self.module.key, mode='str')
            self.save_call_data(call_data)
        return result

    @classmethod
    def resolve_path(cls, path):
        return  c.storage_path + '/server/' + path
            
    def set_functions(self,  
                functions:Optional[List[str]], 
                freedom_attributes=['free', 'free_mode', 'freedom', 'freemee', 'freeme'],
                function_attributes =['endpoints', 'functions', 'expose', "exposed_functions",'server_functions', 'public_functions', 'pubfns'] # the attributes for the functions
):
        self.free = any([(hasattr(self.module, k) and self.module.free)  for k in freedom_attributes])
        if self.free:
            c.print('YOUVE ENABLED AMURICA --> FREEDOM', color='red')
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
        for fa in function_attributes:
            if hasattr(self.module, fa) and isinstance(getattr(self.module, fa), list):
                functions = getattr(self.module, function_attributes[0]) 
                break       
        # does not start with _ and is not a private function
        self.module.functions = self.module.fns = [fn for fn in sorted(list(set(functions + self.helper_functions))) if not fn.startswith('_')]
        self.module.fn2cost = self.module.fn2cost  if hasattr(self.module, 'fn2cost') else {}
        c.print(f'SetFunctions(fns={self.module.fns} fn2cost={self.module.fn2cost} free={self.free})')
        return {'functions': self.module.fns, 'fn2cost': self.module.fn2cost, 'free': self.free}
        
    def get_info(self, module):
        info =  {
            "name": module.name,
            "url": module.url,
            "key": module.key.ss58_address,
            "time": c.time(),
            "schema": {fn:c.fn_schema(getattr(module, fn)) for fn in module.fns if hasattr(module, fn)},
        }
        print(info)
        return info

    def set_port(self, port:Optional[int]=None, port_attributes = ['port', 'server_port']):
        name = self.module.name
        if port == None:
            in_port_attribute = any([k for k in port_attributes])
            if in_port_attribute:
                for k in port_attributes:
                    if hasattr(self.module, k):
                        port = getattr(self.module, k)
                        c.kill_port(port)
                        break
            else:
                namespace = self.namespace()
                if name in namespace:
                    port = int(namespace.get(name).split(':')[-1])
                port = port or c.free_port()
        if str(port) == 'None':
            port = c.free_port()
        while c.port_used(port):
            c.kill_port(port)
            c.sleep(1)
            print(f'Waiting for port {port} to be free')
        self.module.port = port
        self.module.url = f'0.0.0.0:{self.module.port}' 
        return {'success':True, 'message':f'Set port to {port}'}
    max_request_staleness = 4
    
    def get_request(self, request: Request) -> dict:
        headers = dict(request.headers)
        headers['time'] = float(headers.get('time', c.time()))
        headers['key'] = headers.get('key', headers.get('url', None))
        max_request_staleness : int = 4, # (in seconds) the time it takes for the request to be too old
        request_staleness = c.time() - float(headers['time'])
        assert  request_staleness < self.max_request_staleness, f"Request is too old ({request_staleness}s > {max_request_staleness}s (MAX)" 
        params = self.loop.run_until_complete(request.json())
        params = self.serializer.deserialize(params) 
        params = json.loads(params) if isinstance(params, str) else params
        if not self.free:
            assert c.verify({'params': params, 'time': str(headers['time'])}, address=headers['key'], signature=headers['signature'])
        assert isinstance(params, dict), f'Params must be a dict, not {type(params)}'
        if len(params) == 2 and 'args' in params and 'kwargs' in params :
            kwargs = dict(params.get('kwargs')) 
            args = list(params.get('args'))
        else:
            args = []
            kwargs = dict(params)
        params = {"args": args, "kwargs": kwargs}
        return {"params": params, "headers": headers}

    def get_byte_size(self, data):
        return len(str(data).encode('utf-8'))

    def wait_for_server(self,
                          name: str ,
                          network: str = 'local',
                          timeout:int = 600,
                          max_age = 1,
                          sleep_interval: int = 1) -> bool :
        
        time_waiting = 0
        c.print(f'waiting for {name} to start...', color='cyan')
        while time_waiting < timeout:
            namespace = self.namespace(network=network, max_age=max_age)
            if name in namespace:
                try:
                    result = c.call(namespace[name]+'/info')
                    return result
                except Exception as e:
                    c.print(f'Error getting info for {name} --> {c.detailed_error(e)}', color='red')
            c.sleep(sleep_interval)
            c.print(c.logs(name, tail=10))
            time_waiting += sleep_interval
        future.cancel()
        raise TimeoutError(f'Waited for {timeout} seconds for {name} to start')

    def serve(self, 
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
            params = {k : v for k, v  in c.locals2kwargs(locals()).items()  if k not in ['extra_params', 'response', 'namespace']}
            self.run_process(module="server", fn='serve', name=name, params=params, cwd=cwd)
            return self.wait_for_server(name)
        return Server(module=module, name=name, functions=functions, params=params, port=port,  key=key, run_api=1)

    def kill(self, name:str, verbose:bool = True, rm_server=True, **kwargs):
        process_name = self.resolve_process_name(name)
        try:
            c.cmd(f"pm2 delete {process_name}", verbose=False)
            self.rm_logs(process_name)
            result =  {'message':f'Killed {process_name}', 'success':True}
        except Exception as e:
            result =  {'message':f'Error killing {process_name}', 'success':False, 'error':e}
        return result
    
    def kill_all(self, verbose:bool = True, timeout=20):
        servers = self.procs()
        futures = [c.submit(self.kill, kwargs={'name':s, 'update': False}) for s in servers]
        results = c.wait(futures, timeout=timeout)
        return results
    
    def killall(self, **kwargs):
        return self.kill_all(**kwargs)

    def get_logs_path(self, name:str, mode='out')->str:
        assert mode in ['out', 'error'], f'Invalid mode {mode}'
        name = self.resolve_process_name(name)
        return f'{self.pm_dir}/logs/{name.replace("/", "-")}-{mode}.log'.replace(':', '-').replace('_', '-')

    def get_logs_path(self, name:str, mode='out')->str:
        assert mode in ['out', 'error'], f'Invalid mode {mode}'
        name = self.resolve_process_name(name)
        return f'{self.pm_dir}/logs/{name.replace("/", "-")}-{mode}.log'.replace(':', '-').replace('_', '-') 
 
    def rm_logs( self, name):
        for m in ['out', 'error']:
            c.rm(self.get_logs_path(name, m))
        return {'success':True, 'message':f'Removed logs for {name}'}

    def logs(self, module:str, top=None, tail: int =None , stream=True, **kwargs):
        module = self.resolve_process_name(module)
        if tail or top:
            stream = False
        if stream:
            return c.cmd(f"pm2 logs {module}", verbose=True)
        else:
            text = ''
            for m in ['out', 'error']:
                # I know, this is fucked 
                path = self.get_logs_path(module, m)
                try:
                    text +=  c.get_text(path)
                except Exception as e:
                    c.print('ERROR GETTING LOGS -->' , e)
            if top != None:
                text = '\n'.join(text.split('\n')[:top])
            if tail != None:
                text = '\n'.join(text.split('\n')[-tail:])
        return text

    def get_port(self, name:str,  tail:int=100, **kwargs):
        logs = self.logs(name, tail=tail, stream=False, **kwargs)
        port = None
        tag = 'Uvicorn running on '
        for i, line in enumerate(logs.split('\n')[::-1]):
            if tag in line:
                return int(line.split(tag)[-1].split(' ')[0].split(':')[-1])
        return port

    def namespace(self,  search=None, **kwargs) -> dict:
        namespace =  {s: u for s, u in zip(self.servers(), self.urls())}
        return {k:v for k, v in namespace.items() if search in k} if search != None else namespace

    def get_url(self, name:str,  tail:int=100, **kwargs):
        return f'0.0.0.0:{self.get_port(name, tail=tail, **kwargs)}'

    def servers(self, search=None,  **kwargs) -> List[str]:
        return [ p[len(self.process_prefix):] for p in self.procs(search=search, **kwargs) if p.startswith(self.process_prefix)]

    def urls(self, search=None,  **kwargs) -> List[str]:
        return [self.get_url(s) for s in self.servers(search=search, **kwargs)]
    
    def run_process(self, 
                  fn: str = 'serve',
                   name:Optional[str] = None, 
                   module:str = 'server',  
                   params: dict = None,
                   interpreter:str='python3', 
                   verbose: bool = False , 
                   run_fn: str = 'run_fn',
                   cwd : str = None,
                   refresh:bool=True ):
        self.ensure_env()
        params['remote'] = False
        name = name or module
        process_name = self.resolve_process_name(name)
        if self.process_exists(process_name):
            self.kill(process_name, rm_server=False)
        params_str = json.dumps({'module': module , 'fn': fn, 'params': params or {}}).replace('"','\\"')
        cmd = f"pm2 start {c.filepath()} --name {process_name} --interpreter {interpreter} -f --no-autorestart "
        cmd += f"-- --fn {run_fn} --kwargs  \"{params_str}\""
        stdout = c.cmd(cmd, verbose=verbose, cwd=cwd)
        return {'success':True, 'msg':f'Launched {module}',  'cmd': cmd, 'stdout':stdout}

    def procs(self, search=None,  **kwargs) -> List[str]:
        output_string = c.cmd('pm2 status')
        procs = []
        tag = ' default '
        for line in output_string.split('\n'):
            if  tag in line:
                name = line.split(tag)[0].strip()
                name = name.split(' ')[-1]
                procs += [name]
        if search != None:
            search = self.resolve_process_name(search)
            procs = [m for m in procs if search in m]
        procs = sorted(list(set(procs)))
        return procs

    def resolve_process_name(self, name:str, **kwargs) -> str:
        if  name != None and not name.startswith(self.process_prefix):
            name = self.process_prefix + name
        return name
        
    def process_exists(self, name:str, **kwargs) -> bool:
        name = self.resolve_process_name(name)
        return name in self.procs(**kwargs)

    def ensure_env(self,**kwargs):
        '''ensure that the environment variables are set for the process'''
        is_pm2_installed = bool( '/bin/pm2' in c.cmd('which pm2', verbose=False))
        if not is_pm2_installed:
            c.cmd('npm install -g pm2')
            c.cmd('pm2 update')
        return {'success':True, 'message':f'Ensured env '}

    def set_network(self, 
                    network:str, 
                    tempo:int=60, 
                    n=100, 
                    path=None,
                    process_prefix='server',
                    **kwargs):
        self.network = network or 'local'
        self.tempo = tempo
        self.network_path = self.resolve_path(self.network)
        self.modules_path =  f'{self.network_path}/modules'
        self.process_prefix = process_prefix + '/' + network + '/'
        return {'params': self.params(),'network_path': self.network_path}

    def params(self,*args,  **kwargs):
        return { 'network': self.network, 'tempo' : self.tempo}

    def most_recent_call_path(self, module='module',search=None,  **kwargs):
        paths = self.call_paths(module)
        path2time = {p: self.get_path_time(p) for p in paths}
        return max(path2time, key=path2time.get)

    def most_recent_call_time(self, module='module',search=None,  **kwargs):
        return self.get_path_time(self.most_recent_call_path(module=module, search=search, **kwargs))
        
    def time2date(self, time:float):
        import datetime
        return datetime.datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')

    def modules(self, 
                search=None, 
                max_age=60, 
                update=False, 
                features=['name', 'url', 'key'], 
                timeout=8, 
                **kwargs):

        modules = c.get(self.modules_path, max_age=max_age, update=update)
        if modules == None:
            futures  = [c.submit(c.call, [s + '/info'], timeout=timeout) for s in self.urls()]
            modules = c.wait(futures, timeout=timeout)
            c.put(self.modules_path, modules)
        if search != None:
            modules = [m for m in modules if search in m['name']]
        return modules

    def module2called(self, search=None, max_age=None, **kwargs):
        modules = self.modules(search=search, max_age=max_age)
        module2called = {m['name']: self.calls(m['name']) for m in modules}
        return module2called
    
    def server_exists(self, name:str, **kwargs) -> bool:
        return bool(name in self.servers(**kwargs))

    def set_middleware(self, app, max_bytes=1000000):
    
        from starlette.middleware.base import BaseHTTPMiddleware
        class Middleware(BaseHTTPMiddleware):
            def __init__(self, app, max_bytes: int = 1000000):
                super().__init__(app)
                self.max_bytes = max_bytes
            async def dispatch(self, request: Request, call_next):
                print(request)
                content_length = request.headers.get('content-length')
                if content_length:
                    if int(content_length) > self.max_bytes:
                        return JSONResponse(status_code=413, content={"error": "Request too large"})
                body = await request.body()
                if len(body) > self.max_bytes:
                    return JSONResponse(status_code=413, content={"error": "Request too large"})
                response = await call_next(request)
                return response
        app.add_middleware(Middleware, max_bytes=max_bytes)
        app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
        return app


    def rate_limit(self, 
                fn:str, 
                params:dict,  
                headers:dict, 
                network:str = 'subspace', # the network to gate on
                role2rate:dict = {'admin': 100000000, 'owner': 10000000, 'local': 1000000}, # the rate limits for each role
                stake_per_call:int = 1000, # the amount of stake required per call
            ) -> dict:
        if self.module.free: 
            return 100000000
        role = self.get_user_role(headers['key'])
        if role != 'admin':
            assert fn in self.module.fns , f"Function {fn} not in endpoints={self.module.fns}"
        if role in role2rate:
            rate_limit = role2rate[role]
        else:
            path = self.resolve_path(f'rate_limiter/{network}/{role}')
            if self.state == None:
                self.state = c.module(network)().state()
            stake = self.state['stake'].get(headers['key'], 0) 
            stake_to = (sum(self.state['stake_to'].get(headers['key'], {}).values())) 
            stake_from = self.state['stake_from'].get(self.module.key.ss58_address, {}).get(headers['key'], 0) 
            stake = stake + stake_to + stake_from
            rate_limit = stake / stake_per_call
        cost = self.module.fn2cost.get(fn, 1)
        rate_limit = rate_limit / cost
        return rate_limit
        
    def get_user_role(self, address, module=None):
        module = module or self.module
        if c.is_admin(address):
            return 'admin'
        if address == module.key.ss58_address:
            return 'owner'
        if address in self.address2key:
            return 'local'
        return 'stake'

    def path2age(self, address='module'):
        user_paths = self.call_paths(address)
        user_path2time = {p: self.get_path_age(p) for p in user_paths}
        return user_path2time

    def get_path_age(self, path:str) -> float:
        return c.time() - self.get_path_time(path)

    def rate(self, address):
        path2age = self.path2age(address)
        for path, age  in path2age.items():
            if age > self.tempo:
                if os.path.exists(path):
                    print(f'Removing({path})')
                    os.remove(path)
        return len(self.call_paths(address))

    def call_paths(self, address = '' ):
        path = self.history_path + '/' + address
        user_paths = c.glob(path)
        return sorted(user_paths, key=self.get_path_time)

    def history(self, address = 'module' ):
        return [c.get_json(p)["data"] for p in self.call_paths(address)]

    def clear_history(self, address = 'module' ):
        paths = self.call_paths(address)
        for p in paths:
            c.rm(p)
        return {'message':f'Cleared {len(paths)} paths for {address}'}

    def calls(self, address = 'module' ):
        return len(self.call_paths(address))

    def users(self, module='module'):
        return [p.split('/')[-1] for p in c.ls(self.history_path + '/' + module)]
        
    def user2calls(self, module='module'):  
        return {u: self.calls(module+'/'+u) for u in self.users(module)}

    def get_path_time(self, path:str) -> float:
        try:
            x = float(path.split('/')[-1].split('.')[0])
        except Exception as e:
            x = 0
        return x

    def save_call_data(self, data):
        """
        Save the call data to the history path.
        """
        address = self.module.name + '/'  + data["key"]
        calls_t0 = self.calls(address)
        call_data_path = self.history_path + '/' + address +  (f'/{data["fn"]}/{c.time()}.json') 
        c.put(call_data_path, data)
        calls_t1 = self.calls(address)
        assert calls_t1 == calls_t0 + 1, f'Expected {calls_t0+1} calls, but got {calls_t1}'
        return call_data_path

    def fleet(self, module, n=1, tag=None, **kwargs):
        futures = []
        if tag != None:
            module = f'{module}::{tag}'
        for i in range(n):
            futures += [c.submit(c.serve, dict(module=module, name=f'{module}{i}'))]
        for f in c.as_completed(futures):
            c.print(f.result())
        return  {'message':f'Launched {n} servers for {module}', 'namespace':self.namespace()}
        