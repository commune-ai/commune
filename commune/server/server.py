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

class Server:
    def __init__(
        self, 
        module: Union[c.Module, object] = 'module',
        key: Optional[str] = None, # key for the server (str), defaults to being the name of the server
        params : Optional[dict] = None, # the kwargs for the module
        
        # FUNCTIONS
        functions:Optional[List[Union[str, callable]]] = None, # list of endpoints
        # NETWORK
        port: Optional[int] = None, # the port the server is running on
        tempo:int = 10000, # (in seconds) the maximum age of the history
        name: Optional[str] = None, # the name of the server, 
        network: Optional[str] = 'local', # the network the server is running on
        history_path: Optional[str] = None, # the path to the user data
        timeout:int = 10, # (in seconds) the maximum time to wait for a response

        # EXTERNAL MODULES
        executor = 'executor', # the executor for the server to run the functions
        serializer = 'server.serializer', # the serializer for the server serializes and deserializes the data for the server if it is not a string
        auth = 'server.auth.jwt', # the auth for the server,
        middleware = 'server.middleware', # the middleware for the server
        history = 'history', # the history for the server
        pm = 'pm2', # the process manager for the server
        helper_functions  = ['info', 'forward'], # the helper functions

        # MISC
        verbose:bool = True, # whether to print the output
        info = None, # the info for the server
        run_api : Optional[bool] = False, # if the server should be run as an api

        ) -> 'Server':


        self.helper_functions = helper_functions
        self.network = network or 'local'
        self.tempo = tempo
        self.history_manager_path = history_path or self.get_path('history')
        self.modules_path = f'{self.get_path(self.network)}/modules'
        self.verbose = verbose
        self.timeout = timeout
        self.pm = c.module(pm)(proc_prefix= 'server/' + network + '/')
        self.hist = c.module(history)(path=self.history_manager_path)

        if run_api:
            # set modules 
            self.serializer = c.module(serializer)()
            self.executor = c.module(executor)()
            self.auth = c.module(auth)()
            self.middleware = c.module(middleware)
            self.set_module(module=module, name=name, key=key, params=params, functions=functions, port=port)
            self.loop = asyncio.get_event_loop() # get the event loop
            self.app = FastAPI()
            self.app.add_middleware(self.middleware)
            self.app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
            def server_function(fn: str, request: Request):
                return self.forward(fn, request)
            self.app.post("/{fn}")(server_function)
            c.print(f'Served({self.module.info})', color='purple')
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
        headers = dict(request.headers)
        data = {'fn': fn, 'params': params}
        data['client'] = self.auth.verify_headers(headers=dict(request.headers), data=data) # verify the headers
        self.rate_limit(data)   # check the rate limit
        if self.verbose:
            shortkey = lambda x: x[:3] + '...' + x[-3:]
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
        path = f'{self.module.name}/{data["client"]["key"]}/{data["fn"]}/{data["time"]}.json'
        self.hist.save_data(path, data)
        return data
  
    def get_path(self, path):
        return  c.storage_path + '/server/' + path
        
    def set_module(self, module:str, functions: List[str], name: str , key:Union[str, 'Key'], params:dict, port:int):
        module = module or 'module'
        if isinstance(module, str):
            if '::' in str(module):
                name = module
                tag = name.split('::')[-1]
                module = '::'.join(name.split('::')[:-1])
        self.module = c.module(module)(**(params or {}))
        self.module.name = name = name or module 
        self.module.key = c.get_key(key or self.module.name)
        self.set_functions(functions) 
        self.set_port(port)
        self.module.info = {   
            "name": self.module.name,
            "url": self.module.url,
            "key": self.module.key.ss58_address,
            "time": c.time(),
            "schema": self.module.schema,
        }
        self.module.info['signature'] = c.sign(self.module.info, key=self.module.key, mode='str')
        self.verify_info(self.module.info) # verify the info
        return {'success':True, 'message':f'Set module to {self.module.name}'}

    def verify_info(self, info:dict) -> dict:
        """
        verifies the info of the server
        params:
            info : dict
                the info of the server
        """
        assert isinstance(info, dict), f'Info must be a dict, not {type(info)}'
        assert all([k in info for k in ['name', 'url', 'key', 'time', 'signature']]), f'Info must have keys name, url, key, time, signature'
        signature= info['signature']
        payload = {k: v for k, v in info.items() if k != 'signature'}
        assert c.verify(payload, signature=signature, address=info['key']), f'InvalidSignature({info})'
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
        return {'fns': self.module.fns, 'fn2cost': self.module.fn2cost}
        
    def set_port(self, port:Optional[int]=None, port_attributes = ['port', 'server_port']):
        if port == None:
            in_port_attribute = any([k for k in port_attributes])
            if in_port_attribute:
                for k in port_attributes:
                    if hasattr(self.module, k):
                        port = getattr(self.module, k)
                        break
            else:
                namespace = self.namespace()
                if self.module.name in namespace:
                    port = int(namespace.get(self.module.name).split(':')[-1])
                    self.kill(self.module.name)
        if port == None:
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
            self.pm.run("server/serve", name=name, params=params, cwd=cwd)
            return self.wait_for_server(name)
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
        return [ p[len(self.pm.process_prefix):] for p in self.pm.procs(search=search, **kwargs) if p.startswith(self.pm.process_prefix)]

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
            futures  = [c.submit(c.call, [s + '/info'], timeout=timeout) for s in self.urls()]
            modules = c.wait(futures, timeout=timeout)
            c.put(self.modules_path, modules)
        if search != None:
            modules = [m for m in modules if search in m['name']]
        return [m for m in modules if not c.is_error(m)]
    
    def server_exists(self, name:str, **kwargs) -> bool:
        return bool(name in self.servers(**kwargs))

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
        
    def rate(self, address:str):
        path2age = self.hist.path2age(address)
        for path, age  in path2age.items():
            if age > self.tempo:
                if os.path.exists(path):
                    print(f'RemovingStalePath(age={age} tempo={self.tempo}) --> {path}')
                    os.remove(path)
        return len(self.hist.call_paths(address))

    def history(self, address = '' , paths=None,  df=True, features=['time', 'fn', 'cost', 'duration',  'client', 'server']):
        return self.hist.history(address=address, paths=paths, df=df, features=features)

    def clear_history(self, address = 'module' ):
        return self.hist.clear_history(address)

    def kill(self, name):
        return self.pm.kill(name)

    def kill_all(self):
        return self.pm.kill_all()
    
    def logs(self, name, **kwargs):
        return self.pm.logs(name, **kwargs)

    def wait_for_server(self, name:str, trials:int=10, trial_backoff:int=1, network:str='local', max_age:int=60):
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

    
    