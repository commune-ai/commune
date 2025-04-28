from typing import *
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import uvicorn
import os
import sys
import hashlib
import pandas as pd
import json
import asyncio
import commune as c
import argparse
from .utils import *
class Server:

    def __init__(
        self, 
        module: Union[str, object] = 'module',
        params : Optional[dict] = None, # the kwargs for the module
        key: Optional[str] = None, # key for the server (str), defaults to being the name of the server
        functions:Optional[List[Union[str, callable]]] = None, # list of endpoints
        port: Optional[int] = None, # the port the server is running on
        tempo:int = 10000, # (in seconds) the maximum age of the history
        name: Optional[str] = None, # the name of the server, 
        network: Optional[str] = 'local', # the network the server is running on
        timeout:int = 10, # (in seconds) the maximum time to wait for a response

        # EXTERNAL MODULES
        auth = 'server.auth', # the auth for the server,
        middleware = 'server.middleware', # the middleware for the server
        store = 'server.store', # the history for the server
        pm = 'server.pm', # the process manager for the server
        helper_functions  = ['info', 'forward'], # the helper functions

        # MISC
        verbose:bool = True, # whether to print the output
        info = None, # the info for the server
        run_api : Optional[bool] = False, # if the server should be run as an api
        path = '~/.commune/server' # the path to store the server data
        ):

        self.path = abspath(path)
        self.helper_functions = helper_functions
        self.network = network or 'local'
        self.tempo = tempo
        self.verbose = verbose
        self.timeout = timeout
        self.store = c.module(store)(abspath(path))
        self.pm = c.module(pm)(proc_prefix= 'server/' + network + '/')
        self.set_module(module=module, name=name, key=key, params=params, functions=functions, port=port)
        # set modules 
        if run_api:
            self.reg()
            self.auth = c.module(auth)()
            self.loop = asyncio.get_event_loop() # get the event loop
            app = FastAPI()
            app.add_middleware(c.module(middleware))
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],  # or your specific origins
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            def server_function(fn: str, request: Request):
                try:
                    return self.forward(fn, request)
                except Exception as e:
                    err = c.detailed_error(e)
                    c.print(f'Error({fn}) --> {err}', color='red')
                    return err
            app.post("/{fn}")(server_function)
            c.print(f'Served({self.module.info["url"]})', color='purple')
            try:
                uvicorn.run(app, host='0.0.0.0', port=self.module.port, loop='asyncio')
            except:
                self.dereg()
                pass

    def get_params(self, request: Request):
        params = self.loop.run_until_complete(request.json())
        params = json.loads(params) if isinstance(params, str) else params
        if len(params) == 2 and 'args' in params and 'kwargs' in params :
            kwargs = dict(params.get('kwargs')) 
            args = list(params.get('args'))
        else:
            args = []
            kwargs = dict(params)
        params = {"args": args, "kwargs": kwargs}
        return params

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
        headers = dict(request.headers)
        params = self.get_params(request)
        data = {'fn': fn, 'params': params}
        data['client'] = self.auth.verify_headers(headers=headers, data=params) # verify the headers
        self.rate_limit(data)   # check the rate limit
        fn_obj = getattr(self.module, fn)
        result = fn_obj(*args, **kwargs) if callable(fn_obj) else fn_obj
        if c.is_generator(result):
            output = str(result)
            def generator_wrapper(generator):
                for item in generator:
                    c.print(item, end='')
                    yield item
            result = EventSourceResponse(generator_wrapper(result))   
        data['time'] = data['client']['time']
        data[f'result'] = 'stream' if isinstance(result, EventSourceResponse) else result
        data['server'] = self.auth.get_headers(data=data, key=self.module.key)
        data['duration'] = c.time() - float(data['client']['time'])
        data['schema'] = self.module.schema.get(data['fn'], {})
        cid = self.hash(data)
        client_key = data['client']['key']
        path = f'results/{self.module.name}/{data["client"]["key"]}/{cid}.json'
        self.store.put(path, data)
        c.print(f'fn({data["fn"]}) --> {data["duration"]} seconds')
        c.print('Saved data to -->', path)
        return result
    
    def reg(self):
        self.store.put(self.module.info_path, self.module.info)

    def dereg(self):
        self.store.rm(self.module.info_path)

    def hash(self, data:dict) -> str:
        return  hashlib.sha256(json.dumps(data).encode()).hexdigest()

    def get_path(self, path):
        if not path.startswith(self.path):
            path = os.path.join(self.path, path)
        return abspath(path)
        
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
        self.modules_path = self.store.get_path('modules')
        self.module.info_path = f'{self.modules_path}/{self.module.name}.json'
        
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
            c.print(f'Waiting for port {port} to be free')
        self.module.port = port
        self.module.url = f'0.0.0.0:{self.module.port}' 
        return {'success':True, 'message':f'Set port to {port}'}

    def serve(self, 
              module: Union[str, 'Module', Any] = None, # the module in either a string
              params:Optional[dict] = None,  # kwargs for the module
              port :Optional[int] = None, # name of the server if None, it will be the module name
              name = None, # name of the server if None, it will be the module name
              remote:bool = False, # runs the server remotely (pm2, ray)
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
            self.pm.run(name=name, params=params)
            return self.wait_for_server(name)
        return Server(module=module, name=name, functions=functions, params=params, port=port,  key=key, run_api=1)

    def get_port(self, name:str,  tail:int=100, **kwargs):
        return self.namespace().get(name, None).split(':')[-1] if name in self.namespace() else c.free_port()

    def namespace(self,  search=None, **kwargs) -> dict:
        return {m['name']: m['url'] for m in self.modules(search=search, **kwargs)}

    def get_url(self, name:str,  tail:int=100, **kwargs):
        return self.namespace().get(name, None)

    def names(self, search=None,  **kwargs) -> List[str]:
        return [m['name'] for m in self.modules(search=search, **kwargs)]

    servers = names

    def urls(self, search=None,  **kwargs) -> List[str]:
        return list(self.namespace(search=search, **kwargs).values())   

    def params(self,*args,  **kwargs):
        return { 'network': self.network, 'tempo' : self.tempo}

    def modules(self, 
                search=None, 
                max_age=None, 
                update=False, 
                features=['name', 'url', 'key'], 
                timeout=8, 
                **kwargs):
        paths = self.store.paths(self.modules_path)
        modules = []
        for p in paths:
            data = self.store.get(p, max_age=max_age, update=update)
            if search != None:
                if search not in data['name']:
                    continue
            module_port = int(data['url'].split(':')[-1])
            data = {k: data[k] for k in features if k in data}
            if c.port_used(module_port):
                modules.append(data)
            else: 
                self.store.rm(p)
        return modules
    
    def server_exists(self, name:str, **kwargs) -> bool:
        return bool(name in self.servers(**kwargs))

    def rate(self, key:str, # the key to rate
             network:str = 'chain', # the network to gate on
             max_age:int = 60, # the maximum age of the rate
             update:bool = False, # whether to update the rate
             module = None, # the module to rate on
             ) -> float:
        if module == None:
            if '/' in key:
                module, key = key.split('/')
            else:
                module = self.module.name
        return len( self.store.paths(f'results/{module}/{key}')) 

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
        address = client['key']
        if c.is_admin( client['key']):
            role =  'admin'
        elif address == self.module.key.key_address:
            role =  'owner'
        elif address in self.address2key:
            role =  'local'
        else:
            role = 'guest'
        if role != 'admin':
            assert fn in self.module.fns , f"Function {fn} not in endpoints={self.module.fns}"
        if role in role2rate:
            rate_limit = role2rate[role]
        else:
            path = self.get_path(f'rate_limiter/{network}_state')
            self.state = c.get(path, max_age=self.tempo)
            if self.state == None:
                self.state = c.module(network)().state()
            # the amount of stake the user has as a module only
            stake = self.state['stake'].get(client['key'], 0) 
            stake_to_me = self.state['stake_from'].get(self.module.key.key_address, {}).get(client['key'], 0) 
            stake = stake + stake_to_me
            rate_limit = stake / stake_per_call
        rate_limit = rate_limit / self.module.fn2cost.get(fn, 1)
        rate = self.rate(self.module.name+'/'+client['key'])
        assert rate < rate_limit, f'RateExceeded(rate={rate} limit={rate_limit}, caller={shortkey(client["key"])})' 
        return rate_limit

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

    def kill(self, name):
        return self.pm.kill(name)

    def kill_all(self):
        return self.pm.kill_all()
    
    def logs(self, name, **kwargs):
        return self.pm.logs(name, **kwargs)

    def __delete__(self):
        self.dereg()


    def rund(self, module='module', port: int = 8080, verbose: bool = False):
        import subprocess
        """
        Run code-server in the background using nohup.

        Args:
            module (str, optional): The module to serve. Defaults to 'module'.
            port (int, optional): The port to run the server on. Defaults to 8080.
            verbose (bool, optional): Whether to print the output. Defaults to False.
        
        Returns:
            dict: The result of the command
        """
        log_dir = os.path.join(self.path, 'running')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f'{module}.logs')

        # cmd = f"nohup c serve {module} > {log_path} 2>&1 &"
        if verbose:
            print(f"Running command: {cmd}")
        return c.cmd(f'python3 --version',  verbose=verbose)