from typing import *
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import uvicorn
import os
import json
import os
import os
import json
import asyncio
import commune as c

shortkey = lambda x: x[:3] + '...' + x[-2:]

class Server:
    def __init__(
        self, 
        module: Union[c.Module, object] = 'module',
        key: Optional[str] = None, # key for the server (str), defaults to being the name of the server
        params : Optional[dict] = None, # the kwargs for the module

        # FUNCTIONS
        functions:Optional[List[Union[str, callable]]] = None, # list of endpoints
        helper_functions  = ['info', 'forward'], # the helper functions

        # NETWORK
        port: Optional[int] = None, # the port the server is running on
        tempo:int = 10000 , # (in seconds) the maximum age of the history
        name: Optional[str] = None, # the name of the server, 
        network: Optional[str] = 'local', # the network the server is running on
        middleware: Optional[callable] = None, # the middleware for the server
        history_path: Optional[str] = None, # the path to the user data
        timeout:int = 10, # (in seconds) the maximum time to wait for a response

        # EXTERNAL MODULES
        executor = 'executor', # the executor for the server to run the functions
        serializer = 'serializer', # the serializer for the server serializes and deserializes the data for the server if it is not a string
        auth = 'server.auth.jwt', # the auth for the server
        
        # PM2 (PROCESS MANAGEMENT)
        process_manager_path = c.home_path + '/.pm2',
        proc_prefix = 'server', # the prefix for the proc name

        # MISC
        verbose:bool = True, # whether to print the output
        info = None, # the info for the server
        run_api : Optional[bool] = False, # if the server should be run as an api

        ) -> 'Server':
        self.network = network or 'local'
        self.tempo = tempo

        self.history_path = history_path or self.get_path('history')
        self.modules_path = f'{self.get_path(self.network)}/modules'
        self.verbose = verbose
        self.timeout = timeout
        # process manager
        self.proc_prefix = 'server/' + network + '/'
        self.process_manager_path = process_manager_path

        if run_api:
            self.serializer = c.module(serializer)()
            self.executor = c.module(executor)()
            self.auth = c.module(auth)() if auth else Auth()
            module = module or 'module'
            if isinstance(module, str):
                if '::' in str(module):
                    name = module
                    tag = name.split('::')[-1]
                    module = '::'.join(name.split('::')[:-1])
            self.module = c.module(module)(**(params or {}))
            self.module.name = name = name or module 
            # settle the module and name
            self.module.key = c.get_key(key or self.module.name)
            self.set_port(port)
            self.helper_functions = helper_functions
            self.set_functions(functions) 
            self.set_info(info)
            self.loop = asyncio.get_event_loop()
            self.app = FastAPI()
            self.set_middleware(self.app)
            def server_function(fn: str, request: Request):
                return self.forward(fn, request)
            self.app.post("/{fn}")(server_function)
            c.print(f'ServedModule({self.module.info})', color='purple')
            uvicorn.run(self.app, host='0.0.0.0', port=self.module.port, loop='asyncio')

    def fleet(self, module='module', n=2, timeout=10):
        if '::' not in module:
            module = module + '::'
        names = [module+str(i) for i in range(n)]
        return c.wait([c.submit(self.serve, [names[i]])  for i in range(n)], timeout=timeout)

    def forward(self, fn:str, request: Request):

        """
        gets and verifies the request
        params:
            fn : str
                the function to call
            request : dict
                the request object
        result:
            data : dict
                fn : str
                params : dict
                client : dict (headers)
        """
        params = self.loop.run_until_complete(request.json())
        params = self.serializer.deserialize(params) 
        params = json.loads(params) if isinstance(params, str) else params
        # GET HE HEADERS
        # verify the data is legit and the headers are correct , the output will be the headers with additional metatada
        headers = dict(request.headers)
        data = {'fn': fn, 'params': params}
        data['client'] = self.auth.verify_headers(headers=dict(request.headers), data=data) # verify the headers
        self.rate_limit(data)   # check the rate limit
        if self.verbose:
            c.print(f'fn(fn={fn} client={shortkey(data["client"]["key"])})', color='green')
        fn_obj = getattr(self.module, fn)
        if len(params) == 2 and 'args' in params and 'kwargs' in params :
            kwargs = dict(params.get('kwargs')) 
            args = list(params.get('args'))
        else:
            args = []
            kwargs = dict(params)
        params = {"args": args, "kwargs": kwargs}
        result = fn_obj(*args, **kwargs) if callable(fn_obj) else fn_obj
        if c.is_generator(result):
            output = str(result)
            def generator_wrapper(generator):
                for item in generator:
                    yield item
            result = EventSourceResponse(generator_wrapper(result))   
        
        data['time'] = data['client']['time']
        data['result'] = 'stream' if isinstance(result, EventSourceResponse) else result
        data['server'] = self.auth.get_headers(data=data, key=self.module.key)
        data['duration'] = c.time() - float(data['client']['time'])
        data['schema'] = self.module.schema.get(data['fn'], {})
        self.save_data(data)
        return data

    def save_data(self, data):
        path = f'{self.history_path}/{self.module.name}/{data["client"]["key"]}/{data["fn"]}/{data["time"]}.json' 
        c.put(path, data) # save the call data
        return {'message':f'Saved data to {path}', 'success':True}

    @classmethod    
    def get_path(cls, path):
        return  c.storage_path + '/server/' + path
        
    def set_info(self, info=None, required_info_features=['name', 'url', 'key', 'time', 'schema']):
            
        if info == None:
            if hasattr(self.module, 'info') and isinstance(self.module.info, dict):
                info = self.module.info 
            else:
                info = {   
                    "name": self.module.name,
                    "url": self.module.url,
                    "key": self.module.key.ss58_address,
                    "time": c.time(),
                    "schema": self.module.schema,
                }
        assert isinstance(info, dict), f'Info must be a dict, not {type(info)}'
        assert all([k in info for k in required_info_features]), f'Info must have keys name, url, key, time, schema'
        c.print('Setting info -->', info)
        info['signature'] = c.sign(info, key=self.module.key, mode='str')
        assert isinstance(info, dict), f'Info must be a dict, not {type(info)}'
        assert all([k in info for k in ['name', 'url', 'key', 'time', 'signature']]), f'Info must have keys name, url, key, time, signature'
        signature= info['signature']
        info = {k: v for k, v in info.items() if k != 'signature'}
        assert c.verify(info, signature=signature, address=info['key']), f'InvalidSignature({info})'
        self.module.info = info
        return info
    
    def set_functions(self, functions:Optional[List[str]]):
        function_attributes =['endpoints', 'functions', 'expose', "exposed_functions",'server_functions', 'public_functions', 'pubfns']  
        functions =  functions or []
        for fa in function_attributes:
            if hasattr(self.module, fa) and isinstance(getattr(self.module, fa), list):
                print(f'Found functions in {fa}')
                functions = getattr(self.module, function_attributes[0]) 
                break       
        # does not start with _ and is not a private function
        self.module.fns = self.module.functions = [fn for fn in sorted(list(set(functions + self.helper_functions))) if not fn.startswith('_')]
        self.module.fn2cost = self.module.fn2cost  if hasattr(self.module, 'fn2cost') else {}
        schema = {}
        for fn in self.module.fns:
            if hasattr(self.module, fn):
                fn_obj = getattr(self.module, fn)
                setattr(self.module, fn, fn_obj)
                schema[fn] = c.fnschema(fn_obj)
            else:
                c.print(f'SEVER_FN_NOT_FOUND({fn}) --> REMOVING FUNCTION FROM FNS', color='red')
                self.module.fns.remove(fn)
        self.module.schema = schema
        c.print(f'ServerFunctions(fns={self.module.fns} fn2cost={self.module.fn2cost} )')
        return {'fns': self.module.fns, 'fn2cost': self.module.fn2cost}
        
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
                port = port or c.free_ports()
        if str(port) == 'None':
            port = c.free_port()
        while c.port_used(port):
            c.kill_port(port)
            c.sleep(1)
            print(f'Waiting for port {port} to be free')
        self.module.port = port
        self.module.url = f'0.0.0.0:{self.module.port}' 
        return {'success':True, 'message':f'Set port to {port}'}

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
            return self.proc("server/serve", name=name, params=params, cwd=cwd)
        return Server(module=module, name=name, functions=functions, params=params, port=port,  key=key, run_api=1)

    def get_port(self, name:str,  tail:int=100, **kwargs):
        """
        get port from the logs
        """

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
        return [ p[len(self.proc_prefix):] for p in self.procs(search=search, **kwargs) if p.startswith(self.proc_prefix)]

    def urls(self, search=None,  **kwargs) -> List[str]:
        return [self.get_url(s) for s in self.servers(search=search, **kwargs)]

    def params(self,*args,  **kwargs):
        return { 'network': self.network, 'tempo' : self.tempo}

    def modules(self, 
                search=None, 
                max_age=60, 
                update=False, 
                features=['name', 'url', 'key'], 
                timeout=8, 
                **kwargs):

        modules = c.get(self.modules_path, max_age=max_age, update=update)
        if modules == None:
            c.print(f'SyncingLocalModules(max_age={max_age})')

            futures  = [c.submit(c.call, [s + '/info'], timeout=timeout) for s in self.urls()]
            modules = c.wait(futures, timeout=timeout)
            c.put(self.modules_path, modules)
        if search != None:
            modules = [m for m in modules if search in m['name']]
        return [m for m in modules if not c.is_error(m)]
    
    def server_exists(self, name:str, **kwargs) -> bool:
        return bool(name in self.servers(**kwargs))

    def set_middleware(self, app, max_bytes=1000000):
    
        from starlette.middleware.base import BaseHTTPMiddleware
        class Middleware(BaseHTTPMiddleware):
            def __init__(self, app, max_bytes: int = 1000000):
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
        app.add_middleware(Middleware, max_bytes=max_bytes)
        app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
        return app

    def rate_limit(self, data:dict, # fn, params and client/headers
                network:str = 'chain', # the network to gate on
                role2rate:dict = {'admin': 100000000, 'owner': 10000000, 'local': 1000000}, # the rate limits for each role
                stake_per_call:int = 1000, # the amount of stake required per call
            ) -> dict:
        fn = data['fn']
        params = data['params']
        client = data['client'] if 'client' in data else data['headers'] # also known as the headers
        self.address2key = c.address2key()
        if not hasattr(self, 'state'):
            self.state = None
        module = self.module
        address = client['key']
        if c.is_admin( client['key']):
            role =  'admin'
        elif address == module.key.key_address:
            role =  'owner'
        elif address in self.address2key:
            role =  'local'
        else:
            role = 'guest'
        if role != 'admin':
            assert fn in module.fns , f"Function {fn} not in endpoints={module.fns}"
        if role in role2rate:
            rate_limit = role2rate[role]
        else:
            path = self.get_path(f'rate_limiter/{network}_state')
            self.state = c.get(path, max_age=self.tempo)
            if self.state == None:
                self.state = c.module(network)().state()
            # the amount of stake the user has as a module only
            stake = self.state['stake'].get(client['key'], 0) 
            stake_to_me = self.state['stake_from'].get(module.key.ss58_address, {}).get(client['key'], 0) 
            stake = stake + stake_to_me
            rate_limit = stake / stake_per_call
        rate_limit = rate_limit / module.fn2cost.get(fn, 1)
        rate = self.rate(self.module.name+'/'+client['key'])
        assert rate < rate_limit, f'RateExceeded(rate={rate} limit={rate_limit}, caller={shortkey(client["key"])})' 
        return rate_limit
        
    def path2age(self, address='module'):
        user_paths = self.call_paths(address)
        user_path2time = {p: self.get_path_age(p) for p in user_paths}
        return user_path2time

    def get_path_age(self, path:str) -> float:
        return c.time() - self.get_path_time(path)

    def rate(self, address:str):
        path2age = self.path2age(address)
        for path, age  in path2age.items():
            if age > self.tempo:
                if os.path.exists(path):
                    print(f'RemovingStalePath(age={age} tempo={self.tempo}) --> {path}')
                    os.remove(path)
        return len(self.call_paths(address))

    def call_paths(self, address = '' ):
        path = self.history_path + '/' + address
        user_paths = c.glob(path)
        return sorted(user_paths, key=self.get_path_time)

    def history(self, address = '' , paths=None,  df=True, features=['time', 'fn', 'cost', 'duration',  'client', 'server']):
        paths = paths or self.call_paths(address)
        history =  [c.get_json(p)["data"] for p in self.call_paths(address)]
        history = [h for h in history if isinstance(h, dict) and  all([f in h for f in features])]
        address2key = c.address2key(max_age=self.tempo)
        print(f'History({address}) --> {len(history)}')
        if df:
            import datetime
            df =  c.df(history)
            if len(df) == 0:
                return df
            features = ['fn', 'cost', 'time', 'duration', 'age', 'client', 'server']
            df['age'] = df['time'].apply(lambda x: c.time() - float(x))
            df['time'] = df['time'].apply(lambda x:  datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') if isinstance(x, float) else x)

            def headers2key(x):
                if isinstance(x, dict):
                    k = x.get('key', None)
                    return address2key.get(k, k)
                return x

            df['client'] = df['client'].apply(lambda x: headers2key(x) )
            df['server'] = df['server'].apply(lambda x: headers2key(x))
            df = df[features]
            return df
        return history

    def all_paths(self):
        
        return c.glob(self.history_path)

    def all_history(self):
        paths = self.all_paths()
    
        return self.history(paths=paths)

    h = history

    def clear_history(self, address = 'module' ):
        paths = self.call_paths(address)
        for p in paths:
            c.rm(p)
        assert len(self.call_paths(address)) == 0, f'Failed to clear paths for {address}'
        return {'message':f'Cleared {len(paths)} paths for {address}'}

    def all_n(self):
        return len(self.all_paths())

    def clear_all_history(self):
        paths = self.all_paths()
        for p in paths:
            c.rm(p)
        assert len(self.all_paths()) == 0, f'Failed to clear all paths'
        return {'message':f'Cleared {len(paths)} paths for all'}

    def calls(self, address = 'module' ):
        return len(self.call_paths(address))

    def num_calls(self, address = 'module' ):
        return len(self.call_paths(address))

    def callers(self, module='module'):
        return [p.split('/')[-1] for p in c.ls(self.history_path + '/' + module)]
        
    def caller2calls(self, module='module'):  
        return {u: self.calls(module+'/'+u) for u in self.callers(module)}

    def clear_module_history(self, module='module'):
        return os.system(f'rm -r {self.history_path}/{module}')

    def get_path_time(self, path:str) -> float:
        try:
            x = float(path.split('/')[-1].split('.')[0])
        except Exception as e:
            x = 0
        return x

    def kill(self, name:str, verbose:bool = True, rm_server=True, **kwargs):
        proc_name = self.get_procname(name)
        try:
            c.cmd(f"pm2 delete {proc_name}", verbose=False)
            for m in ['out', 'error']:
                os.remove(self.get_logs_path(name, m))
            result =  {'message':f'Killed {proc_name}', 'success':True}
        except Exception as e:
            result =  {'message':f'Error killing {proc_name}', 'success':False, 'error':e}
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
        name = self.get_procname(name)
        return f'{self.process_manager_path}/logs/{name.replace("/", "-")}-{mode}.log'.replace(':', '-').replace('_', '-') 

    def logs(self, module:str, top=None, tail: int =None , stream=True, **kwargs):
        module = self.get_procname(module)
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

    def sync_env(self,**kwargs):
        '''ensure that the environment variables are set for the proc'''
        is_pm2_installed = bool( '/bin/pm2' in c.cmd('which pm2', verbose=False))
        if not is_pm2_installed:
            c.cmd('npm install -g pm2')
            c.cmd('pm2 update')
        return {'success':True, 'message':f'Ensured env '}
  
    def proc(self, 
                  fn: str = 'serve',
                   name:str = None, 
                   module:str = 'server',  
                   params: dict = None,
                   network:str = 'local',
                   interpreter:str='python3', 
                   verbose: bool = False , 
                   cwd : str = None,
                    max_age = 10,
                    trials:int=3,
                    trial_backoff:int=1,
                    refresh:bool=True ):
        """
        run a process with pm2

        Args:
            fn (str, optional): The function to run. Defaults to 'serve'.
            name (str, optional): The name of the proc. Defaults to None.
            module (str, optional): The module to run. Defaults to 'server'.
            params (dict, optional): The parameters for the function. Defaults to None.
            interpreter (str, optional): The interpreter to use. Defaults to 'python3'.
            verbose (bool, optional): Whether to print the output. Defaults to False.
            wait_for_server (bool, optional): Whether to wait for the server to start. Defaults to True.
            cwd (str, optional): The current working directory. Defaults to None.
            refresh (bool, optional): Whether to refresh the environment. Defaults to True.
        Returns:
            dict: The result of the command
         
        """
        self.sync_env()
        params['remote'] = False
        name = name or module
        if '/' in fn:
            module, fn = fn.split('/')
        else:
            module = 'server'
            fn = fn
        params_str = json.dumps({'fn': module +'/' + fn, 'params': params or {}}).replace('"','\\"')
        proc_name = self.get_procname(name)
        if self.proc_exists(proc_name):
            self.kill(proc_name, rm_server=False)
        cmd = f"pm2 start {c.filepath()} --name {proc_name} --interpreter {interpreter} -f --no-autorestart -- --fn run_fn --params \"{params_str}\""
        c.cmd(cmd, verbose=verbose, cwd=cwd)

        # wait for the server to start
        for trial in range(trials):
            namespace = self.namespace(network=network, max_age=max_age)
            if name in namespace:
                try:
                    return  c.call(namespace[name]+'/info')
                except Exception as e:
                    if trial > 1:
                        c.print(f'Error getting info for {name} --> {c.detailed_error(e)}', color='red')
                    # c.print(c.logs(name, tail=10))
            c.sleep(trial_backoff)
        raise Exception(f'Failed to start {name} after {trials} trials')
        return info

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
            search = self.get_procname(search)
            procs = [m for m in procs if search in m]
        procs = sorted(list(set(procs)))
        return procs

    def get_procname(self, name:str, **kwargs) -> str:
        if  name != None and not name.startswith(self.proc_prefix):
            name = self.proc_prefix + name
        return name
        
    def proc_exists(self, name:str, **kwargs) -> bool:
        name = self.get_procname(name)
        return name in self.procs(**kwargs)