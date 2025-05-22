from typing import *
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import uvicorn
import os
import hashlib
import os
import pandas as pd
import json
import asyncio
import commune as c
from .utils import shortkey, abspath


print = c.print

class Server:

    helper_functions  = ['info', 'forward'] # the helper functions

    def __init__(
        self, 
        module: Union[str, object] = 'module',
        key: Optional[str] = None, # key for the server (str), defaults to being the name of the server
        params : Optional[dict] = None, # the kwargs for the module
        
        # FUNCTIONS
        functions:Optional[List[Union[str, callable]]] = ["forward", "info"] , # list of endpoints
        # NETWORK
        port: Optional[int] = None, # the port the server is running on
        tempo:int = 10000, # (in seconds) the maximum age of the history
        name: Optional[str] = None, # the name of the server, 
        network: Optional[str] = 'local', # the network the server is running on
        # STORAGE
        store = 'store', # the store for the server
        path = '~/.commune/server', # the path to store the server data

        # AUTH
        auth = 'server.auth', # the auth for the server,
        middleware = 'server.middleware', # the middleware for the server
        # PROCESS MANAGER
        pm = 'server.pm.pm2', # the process manager to use

        # MISC
        verbose:bool = True, # whether to print the output
        timeout = 10, # (in seconds) the maximum time to wait for a response
        run_api:bool = False, # whether to run the api
        ):
        self.store = c.module(store)(path)
        self.network = network or 'local'
        self.tempo = tempo
        self.verbose = verbose
        self.pm = pm # sets the module to the pm
        if run_api:
            self.set_module(module=module, name=name, key=key, params=params, functions=functions, port=port)
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
                    print(f'Error({fn}) --> {err}', color='red')
                    return err
            app.post("/{fn}")(server_function)
            self.register_server(self.module.info['name'], self.module.info['url'])
            uvicorn.run(app, host='0.0.0.0', port=self.module.port, loop='asyncio')

    @property
    def pm(self):
        if not hasattr(self, '_pm'):
            self._pm = c.module('server.pm')()
        return self._pm

    # set the pm
    @pm.setter
    def pm(self, pm='server.pm'):
        self._pm = c.module(pm)()
        return self

    def fleet(self, module='module', n=2, timeout=10):
        if '::' not in module:
            module = module + '::'
        names = [module+str(i) for i in range(n)]
        return c.wait([c.submit(self.serve, [names[i]])  for i in range(n)], timeout=timeout)

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
        params = self.get_params(request)
        headers = self.auth.verify_headers(headers=dict(request.headers), data=params) # verify the headers
        user = headers['key'] # the user key
        rate_limit = self.rate_limit(user=user, fn=fn) # the rate limit 
        rate = self.rate(user) * self.module.fn2cost.get(fn, 1)
        assert rate < rate_limit, f'RateLimitExceeded({rate} > {rate_limit})'
        data = {}


        with c.timer('SERVER_FN'):
            fn_obj = getattr(self.module, fn)

            if callable(fn_obj):
                result = fn_obj(*params['args'], ** params['kwargs'])
            else:
                result = fn_obj
                
            if c.is_generator(result):
                output = str(result)
                def generator_wrapper(generator):
                    for item in generator:
                        print(item, end='')
                        yield item
                result = EventSourceResponse(generator_wrapper(result))   
       
        data['fn'] = fn
        data['params'] = params
        data['result'] = 'stream' if isinstance(result, EventSourceResponse) else result
        data['client'] = headers
        data['server'] = self.auth.get_headers(data=data, key=self.module.key)
        data['time'] = data['client']['time']
        data['duration'] = c.time() - float(data['client']['time'])
        data['schema'] = self.module.schema.get(data['fn'], {})
        self.save_result(data)
        print(f'fn({data["fn"]}) --> {data["duration"]} seconds')
        return result

    def save_result(self, data) -> Union[pd.DataFrame, List[Dict]]:
        path = f'results/{self.module.name}/{data["client"]["key"]}/{self.hash(data)}.json'
        c.print('RESULT({}): {}'.format(path, data), color='green')
        self.store.put(path, [data])

    def results(self,
                    module:str = 'module', 
                    df: bool = True, 
                    features: List = ['time', 'fn', 'duration', 'client', 'server']) -> Union[pd.DataFrame, List[Dict]]:
        """
        Get history data for a specific address
        
        Args:
            address: The address to get history for
            paths: Optional list of paths to load from
            as_df: Whether to return as pandas DataFrame
            features: Features to include
            
        Returns:
            DataFrame or list of history records
        """
        paths = self.store.paths('results/'+module)
        history = [self.store.get(p) for p in paths]
        if df and len(history) > 0:
            history = pd.DataFrame(history)
            if len(history) == 0:
                return history
            history = history[features]
            def _shorten(x):
                if x in self.address2key: 
                    return self.address2key.get(x) + ' (' + shortkey(x) + ')'
                else:
                    return shortkey(x)
                return x
            history['server'] = history['server'].apply(lambda x: _shorten(x['key']))

            history['client'] = history['client'].apply(lambda x: _shorten(x['key']))
            history['age'] = history['time'].apply(lambda x:c.time() - float(x))
            del history['time']
        return history

    def hash(self, data:dict) -> str:
        return  hashlib.sha256(json.dumps(data).encode()).hexdigest()

        
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

    def verify_info(self, info:dict) -> bool:
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
        assert self.module.key.verify(payload, signature=signature, address=info['key']), f'InvalidSignature({info})'
        return True
    
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
                schema[fn] = c.fn_schema(fn_obj)
            else:
                print(f'SEVER_FN_NOT_FOUND({fn}) --> REMOVING FUNCTION FROM FNS', color='red')
                self.module.fns.remove(fn)
        self.module.schema = schema
        self.module.free_mode = self.module.free_mode if hasattr(self.module, 'free_mode') else False
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

    def servers(self, search=None,  **kwargs) -> List[str]:
        return list(self.namespace(search=search, **kwargs).keys())

    def urls(self, search=None,  **kwargs) -> List[str]:
        return list(self.namespace(search=search, **kwargs).values())   

    def params(self,*args,  **kwargs):
        return { 'network': self.network, 'tempo' : self.tempo}

    def modules(self, 
                search=None, 
                max_age=60, 
                update=False, 
                features=['name', 'url', 'key'], 
                timeout=8, 
                **kwargs):
        modules_path = 'modules.json'
        modules = self.store.get(modules_path, max_age=max_age, update=update)
        if modules == None:
            futures  = [c.submit(c.call, [s + '/info'], timeout=timeout) for s in self.urls()]
            modules = c.wait(futures, timeout=timeout)
            self.store.put(modules_path, modules)
        if search != None:
            modules = [m for m in modules if search in m['name']]
        return [m for m in modules if not c.is_error(m) and  m != None]
    
    def server_exists(self, name:str, **kwargs) -> bool:
        return bool(name in self.servers(**kwargs))

    def exists(self, name:str, **kwargs) -> bool:
        """
        check if the server exists
        """
        return bool(name in self.servers(**kwargs))

    def rate(self, user:str, # the key to rate
             max_age:int = 60, # the maximum age of the rate
             update:bool = False, # whether to update the rate
             module = None, # the module to rate on
             ) -> float:
        if module == None:
            module = self.module.name
        if '/' in user:
            module, user = user.split('/')
        path = f'results/{module}/{user}'
        return len( self.store.paths(path)) 

    @property
    def address2key(self):  
        if not hasattr(self, '_address2key'):
            self._address2key = c.address2key()
        return self._address2key

    def role(self, user) -> str:
        """
        get the role of the address ( admin, owner, local, public)
        """
        assert not self.is_blacklisted(user), f"Address {user} is blacklisted"

        if c.is_admin(user):
            # can call any function
            role =  'admin'
        elif user == self.module.key.key_address:
            # can call any function
            role =  'owner'
        else:
            # non admin roles (cant call every function)
            assert fn in self.module.fns , f"Function {fn} not in endpoints={self.module.fns}"
            roles = self.roles(max_age=60, update=False)
            if user in roles:
                role = roles[user]
            elif address in self.address2key:
                role =  'local'
            else:
                # this is a public address that is not in any of the roles
                role = 'public'
        
        return role

    def rate_limit(self, user:str, fn:str,  role2rate = {'admin': 100000000, 'owner': 10000000, 'local': 1000000}):
        role = self.role(user)
        rate = role2rate.get(role, 1000)
        if role in ['admin', 'owner']:
            return rate
        else:
            assert fn in self.module.fns, f"Function {fn} not in endpoints={self.module.fns}"
            network_rate = self.network_rate(user=user, network=self.network)
            rate = min(rate, network_rate)

        return rate

    def roles(self, max_age:int = 60, update:bool = False):
        """
        get the roles of the addresses
        """
        roles = self.store.get(f'roles.json', {}, max_age=max_age, update=update)
        return roles

    def add_role(self, address:str, role:str, max_age:int = 60, update:bool = False):
        """
        add a role to the address
        """
        roles = self.store.get(f'roles.json', {}, max_age=max_age, update=update)
        roles[address] = role
        self.store.put(f'roles.json', roles)
        return {'roles': roles, 'address': address }

    def remove_role(self, address:str, role:str, max_age:int = 60, update:bool = False):
        """
        remove a role from the address
        """
        roles = self.store.get(f'roles.json', {}, max_age=max_age, update=update)
        if address in roles:
            del roles[address]
        self.store.put(f'roles.json', roles)
        return {'roles': roles, 'address': address }

    def get_role(self, address:str, max_age:int = 60, update:bool = False):
        """
        get the role of the address
        """
        roles = self.store.get(f'roles.json', {}, max_age=max_age, update=update)
        if address in roles:
            return roles[address]
        else:
            return 'public'

    def has_role(self, address:str, role:str, max_age:int = 60, update:bool = False):
        """
        check if the address has the role
        """
        roles = self.store.get(f'roles.json', {}, max_age=max_age, update=update)
        if address in roles:
            return roles[address] == role
        else:
            return False


    def blacklist_user(self, user:str, max_age:int = 60, update:bool = False):
        """
        check if the address is blacklisted
        """
        blacklist = self.store.get(f'blacklist.json', [], max_age=max_age, update=update)
        blacklist.append(user)
        blacklist = list(set(blacklist))
        self.store.put(f'blacklist.json', blacklist)
        return {'blacklist': blacklist, 'user': user }

    def unblacklist_user(self, user:str, max_age:int = 60, update:bool = False):
        """
        check if the address is blacklisted
        """
        blacklist = self.store.get(f'blacklist.json', [], max_age=max_age, update=update)
        blacklist.remove(user)
        blacklist = list(set(blacklist))
        self.store.put(f'blacklist.json', blacklist)
        return {'blacklist': blacklist, 'user': user }

    def blacklist(self,  max_age:int = 60, update:bool = False):
        """
        check if the address is blacklisted
        """
        blacklist = self.store.get(f'blacklist.json', [], max_age=max_age, update=update)
        return blacklist

    def is_blacklisted(self, user:str, max_age:int = 60, update:bool = False):
        """
        check if the address is blacklisted
        """
        blacklist = self.blacklist(max_age=max_age, update=update)
        return user in blacklist


    
    
    def network_rate(self, user:str, network:str = 'chain', max_age:int = 60, update:bool = False):  
        state = self.network_state(network=network, max_age=60, update=False)
        server_address = self.module.key.key_address
        stake = state['stake'].get(user, 0) + state['stake_to'].get(user, {}).get(server_address, 0) 
        stake_per_call = state.get('stake_per_call', 1000)
        rate = stake / stake_per_call
        return rate

    def network_state(self, network:str = 'chain', max_age:int = 360, update:bool = False):
        path = self.store.get_path(f'network_state/{network}.json')
        self.state = self.store.get(path, max_age=self.tempo, update=update)
        if self.state == None:
            self.state = c.module(network)().state()
            self.store.put(path, self.state)
        return self.state

    def wait_for_server(self, name:str, trials:int=10, trial_backoff:int=0.5, network:str='local', verbose=False, max_age:int=20):
        # wait for the server to start
        for trial in range(trials):
            namespace = self.namespace(network=network)
            if name in namespace:
                try:
                    return  c.call(namespace[name]+'/info')
                except Exception as e:
                    if verbose:
                        print(f'Error getting info for {name} --> {c.detailed_error(e)}', color='red')
                        if trial > 1:
                            print(f'Error getting info for {name} --> {c.detailed_error(e)}', color='red')
                    
                        # print(c.logs(name, tail=10))
            c.sleep(trial_backoff)
        raise Exception(f'Failed to start {name} after {trials} trials')

    def kill(self, name):
        self.pm.kill(name)
        self.deregister_server(name)

    def kill_all(self):
        for server in self.servers():
            self.kill(server)
            print(f'Killing -> {server}')
        return {'servers': self.servers(update=1)}
    def logs(self, name, **kwargs):
        return self.pm.logs(name, **kwargs)

    def namespace(self,  search=None,  max_age=600, update=False, path='namespace', **kwargs) -> dict:
        t0 = c.time()
        namespace = self.store.get(path , max_age=max_age, update=update)
        if namespace == None:
            names = [p[len(self.pm.process_prefix):] for p in self.pm.ps(search=search, **kwargs) if p.startswith(self.pm.process_prefix)]
            urls = [self.get_url(s) for s in names]
            namespace = {s: urls[i] for i, s in enumerate(names)}
            self.store.put(path, namespace)
        if search != None:
            namespace = {k:v for k, v in namespace.items() if search in k}
        return namespace

    def register_server(self, name, url, path='namespace'):
        namespace = self.store.get(path ,{})
        namespace[name] = url
        return self.store.put(path, namespace)

    def deregister_server(self, name, path='namespace'):
        namespace = self.store.get(path ,{})
        namespace.pop(name, None)
        return self.store.put(path, namespace)

    def get_url(self, name:str,  tail:int=100, **kwargs):
        return f'0.0.0.0:{self.get_port(name, tail=tail, **kwargs)}'

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
   

