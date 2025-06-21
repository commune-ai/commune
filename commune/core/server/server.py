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

print = c.print

class Server:

    helper_fns  = ['info', 'forward'] # the helper fns
    fn_attributes =['endpoints', 
                    'fns',
                    'expose', 
                    "exposed_fns",'server_fns', 
                    'public_fns', 
                    'pubfns',  
                    'public_functions', 
                    'exposed_functions'] # the attributes that can contain the fns

    def __init__(
        self, 
        module: Union[str, object] = 'module',
        key: Optional[str] = None, # key for the server (str), defaults to being the name of the server
        params : Optional[dict] = None, # the kwargs for the module
        
        # FUNCTIONS
        fns:Optional[List[Union[str, callable]]] = ["forward", "info"] , # list of endpoints

        # NETWORK
        port: Optional[int] = None, # the port the server is running on
        tempo:int = 10000, # (in seconds) the maximum age of the txs
        name: Optional[str] = None, # the name of the server, 
        network: Optional[str] = 'local', # the network the server is running on

        # STORAGE
        store = 'store', # the store for the server
        path = '~/.commune/server', # the path to store the server data

        # AUTHENTICATION
        tx = 'tx', # the tx for the server
        tx_path = '~/.commune/server/tx',
        auth = 'auth', # the auth for the server,
        private = True, # whether the store is private or not
        middleware = 'server.middleware', # the middleware for the server
        role2rate = {'admin': 100000000, 'owner': 10000000, 'local': 1000000, 'public': 100}, # the rate for each role,
        admin_roles:List[str] = ['admin', 'owner'], # the roles that can call any fn

        # PROCESS MANAGER
        pm = 'pm', # the process manager to use
        free_mode:bool = False, # whether the server is in free mode or not

        # MISC
        verbose:bool = True, # whether to print the output
        timeout = 10, # (in seconds) the maximum time to wait for a response
        serve:bool = False, # whether to run the api
        ):
        
        self.store = c.mod(store)(path)
        self.network = network or 'local'
        self.tempo = tempo
        self.verbose = verbose
        self.tx = c.mod(tx)(tx_path=tx_path)
        self.role2rate = role2rate
        self.admin_roles = admin_roles
        self.auth = c.mod(auth)()
        self.pm = c.mod(pm)() # sets the module to the pm

    @property
    def info(self):
        info  = {   
            "name": self.name,
            "url": self.url,
            "key": self.key.ss58_address,
            "time": c.time(),
            'free_mode': self.free_mode,
            "schema": self.schema,
        }
        info['signature'] = self.key.sign(info, mode='str')
        self.verify_info(info) # verify the info
        return info

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
        return  {"args": args, "kwargs": kwargs}



    

    def forward(self, fn:str, request: Request):

        params = self.get_params(request)
        headers = dict(request.headers)

        self.check_call(fn, params, headers) # check the user and the fn
        fn_obj = getattr(self.module, fn)
        
        # get the result
        if callable(fn_obj):
            result = fn_obj(*params['args'], **params['kwargs']) # call the fn
        else:
            result = fn_obj # if not callable, just return the object

        # if the result is a generator, wrap it in an EventSourceResponse (SSE) for streaming
        if c.is_generator(result):
            def generator_wrapper(generator):
                output =  ''
                for item in generator:
                    print(item, end='')
                    output += str(item)
                    yield item
        else:

            # save the transaction between the headers and server for future auditing
            
            if not self.free_mode:
                server_headers = self.auth.headers(data={'fn': fn, 'params': params, 'result': result}, key=self.key)
                self.tx.forward(
                    module=self.name,
                    fn=fn, # 
                    params=params, # params of the inputes
                    result=self.serializer.forward(result),
                    schema=self.schema.get(fn, {}), # schema of the fn
                    client=headers,
                    server=server_headers,
                )
        return result


    def txs(self, *args, **kwargs) -> Union[pd.DataFrame, List[Dict]]:
        return  self.tx.txs( *args, **kwargs)

    def hash(self, data:dict) -> str:
        return  hashlib.sha256(json.dumps(data).encode()).hexdigest()


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
        print('Verifying info', payload)
        assert self.key.verify(payload, signature=signature, address=info['key']), f'InvalidSignature({info.keys()})'
        return True
    
    def set_fns(self, fns:Optional[List[str]]):
        """
        set functions
        """
        fns =  fns or []
        if len(fns) == 0:
            for fa in self.fn_attributes:
                if hasattr(self.module, fa) and isinstance(getattr(self.module, fa), list):
                    fns = getattr(self.module, fa) 
                    break
        # does not start with _ and is not a private fn
        fns = [fn for fn in sorted(list(set(fns + self.helper_fns))) if not fn.startswith('_')]
        fn2cost = {} if not hasattr(self.module, 'fn2cost') else self.module.fn2cost
        self.fn2cost = {fn: fn2cost.get(fn, 1) for fn in fns}
        schema = c.schema(self.module, code=False)
        schema = {fn: schema[fn] for fn in fns if fn in schema}
        self.schema = schema
        self.fns = list(schema.keys())


        return {'fns': self.fns, 'fn2cost': self.fn2cost}
        
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
                if self.name in namespace:
                    port = int(namespace.get(self.name).split('::')[-1])
                    self.kill(self.name)
        if port == None:
            port = c.free_port()
        while c.port_used(port):
            c.kill_port(port)
            c.sleep(1)
            print(f'Waiting for port {port} to be free')
        self.module.port = port
        self.url = f'0.0.0.0:{self.module.port}' 
        return {'success':True, 'message':f'Set port to {port}'}



    def servers(self, search=None,  **kwargs) -> List[str]:
        return self.pm.servers(search=search,  **kwargs)

    def urls(self, search=None,  **kwargs) -> List[str]:
        return list(self.pm.namespace(search=search, **kwargs).values())   

    def params(self,*args,  **kwargs):
        return { 'network': self.network, 'tempo' : self.tempo}

    modules_path = 'modules.json'

    def modules(self, 
                search=None, 
                max_age=None, 
                update=False, 
                features=['name', 'url', 'key'], 
                timeout=24, 
                **kwargs):


        def module_filter(m: dict) -> bool:
            """Filter function to check if a module contains the required features."""

            return isinstance(m, dict) and all(feature in m for feature in features )
            
        modules = self.store.get(self.modules_path, None, max_age=max_age, update=update)
        if modules == None :
            urls = self.urls(search=search, **kwargs)
            print(f'Updating modules from {self.modules_path}', color='yellow')
            
            futures  = [c.submit(c.call, [url + '/info'], timeout=timeout, mode='thread') for url in urls]
            modules =  c.wait(futures, timeout=timeout)
            print(f'Found {len(modules)} modules', color='green')
            modules = list(filter(module_filter, modules))
            self.store.put(self.modules_path, modules)
        else:
            modules = list(filter(module_filter, modules))

        if search != None:
            modules = [m for m in modules if search in m['name']]
        return modules

    def n(self, search=None, **kwargs):
        return len(self.modules(search=search, **kwargs))

    def exists(self, name:str, **kwargs) -> bool:
        """check if the server exists"""
        return bool(name in self.servers(**kwargs))

    def rate(self, user:str, # the key to rate
             max_age:int = 60, # the maximum age of the rate
             update:bool = False, # whether to update the rate
             module = None, # the module to rate on
             ) -> float:
        if module == None:
            module = self.name
        if '/' in user:
            module, user = user.split('/')
        path = f'results/{module}/{user}'
        return len( self.store.paths(path)) 


    def role(self, user) -> str:
        """
        get the role of the address ( admin, owner, local, public)
        """
        assert not self.is_blacklisted(user), f"Address {user} is blacklisted"

        if c.is_admin(user):
            # can call any fn
            role =  'admin'
        elif hasattr(self, 'module') and user == self.key.key_address:
            # can call any fn
            role =  'owner'
        else:
            # non admin roles (cant call every fn)
            roles = self.roles(max_age=60, update=False)
            if user in roles:
                role = roles[user]

            if not hasattr(self, 'address2key'):
                self.address2key = c.address2key()
            
            elif user in self.address2key:
                role =  'local'
            else:
                # this is a public address that is not in any of the roles
                role = 'public'
        
        return role

    def check_call(self, fn:str, params:dict, headers:dict) -> float:
        if self.free_mode:
            assert fn in self.fns, f"Function {fn} not in fns={self.fns}"
            rate = 1
        else:
            headers = self.auth.verify_headers(headers=headers, data={'fn': fn , 'params': params}) # verify the headers
            rate = self.rate(headers['key']) * self.fn2cost.get(fn, 1)
            role = self.role(headers['key'])
            rate_limit = self.role2rate.get(role) # admin and owner can call any fn
            if role not in ['admin', 'owner']:
                rate_limit = self.role2rate.get(role, 1000)
                assert fn in self.fns, f"Function {fn} not in endpoints={self.fns}"
                network_rate_limit = self.network_rate(user=headers['key'], network=self.network)
                rate_limit = max(rate , rate_limit)
            assert rate < rate_limit, f'RateLimitExceeded({rate} > {rate_limit}) for user={headers}, fn={fn}, params={params}'

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
        server_address = self.key.key_address
        stake = state.get('stake', {}).get(user, 0) + state.get('stake_to', {}).get(user, {}).get(server_address, 0) 
        stake_per_call = state.get('stake_per_call', 1000)
        rate = stake / stake_per_call
        return rate


    def network_state(self, network:str = 'chain', max_age:int = 360, update:bool = False):
        path = self.store.get_path(f'network_state/{network}.json')
        self.state = self.store.get(path, max_age=self.tempo, update=update)
        if network in ['local', 'local_chain']:
            return {}
        else:
            if self.state == None:
                self.state = c.mod(network)().state()
                self.store.put(path, self.state)
            return self.state

    def wait_for_server(self, name:str, max_time:int=10, trial_backoff:int=0.5, network:str='local', verbose=True, max_age:int=20):
        # wait for the server to start
        t0 = c.time()
        while c.time() - t0 < max_time:
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

    def kill_all(self):
        return self.pm.kill_all()
    killall = kill_all
    def logs(self, name, **kwargs):
        return self.pm.logs(name, **kwargs)
        

    def namespace(self,  search=None,  max_age=None, update=False,**kwargs) -> dict:
        return self.pm.namespace(search=search, max_age=max_age, update=update, **kwargs)


    def serve(self, 
              module: Union[str, 'Module', Any] = None, # the module in either a string
              params:Optional[dict] = None,  # kwargs for the module
              port :Optional[int] = None, # name of the server if None, it will be the module name
              name = None, # name of the server if None, it will be the module name
              fns = None, # list of fns to serve, if none, it will be the endpoints of the module
              key = None, # the key for the server
              free_mode:bool = False, # whether the server is in free mode or not
              cwd = None,
              auth = 'auth',
              serializer = 'serializer',
              **extra_params
              ):
        module = module or 'module'
        name = name or module
        params = {**(params or {}), **extra_params}
        self =  Server(module=module, name=name, fns=fns, params=params, port=port,  key=key, serve=True, free_mode=free_mode)

        module = module or 'module'
        if isinstance(module, str):
            if '::' in module:
                name = module
                module = '::'.join(name.split('::')[:-1])

        self.serializer = c.mod(serializer)() # sets the serializer
        self.module = c.mod(module)(**(params or {}))
        self.name = name = name or module 
        self.key = c.get_key(key or self.name)
        self.set_fns(fns) 
        self.set_port(port)
        self.free_mode = bool(free_mode)
        self.module.info = self.info
        self.auth = c.mod(auth)()
        self.loop = asyncio.get_event_loop() # get the event loop

        app = FastAPI()
        # app.add_middleware(c.mod(middleware), auth=)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # or your specific origins
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        def server_fn(fn: str, request: Request):
            try:
                return self.forward(fn, request)
            except Exception as e:
                return c.detailed_error(e)
        app.post("/{fn}")(server_fn)

        c.print(f'Serving(name={name} port={port} free_mode={free_mode} key={self.key.key_address})', color='green')
        uvicorn.run(app, host='0.0.0.0', port=self.module.port, loop='asyncio')
        return {'success':True, 'message':f'Set module to {self.name}'}


