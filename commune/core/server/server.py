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
import time
import commune as c

print = c.print

class Server:

    helper_fns  = ['info', 'forward'] # the helper fns
    def __init__(
        self, 
        module: Union[str, object] = 'module',
        params : Optional[dict] = None, # the kwargs for the module
        key: Optional[str] = None, # key for the server (str), defaults to being the name of the server
        # FUNCTIONS
        fns:Optional[List[Union[str, callable]]] = ["forward", "info"] , # list of endpoints
        port: Optional[int] = None, # the port the server is running on
        tempo:int = 10000, # (in seconds) the maximum age of the txs
        name: Optional[str] = None, # the name of the server, 
        network: Optional[str] = 'local', # the network the server is running on
        # STORAGE
        store = 'store', # the store for the server
        path = '~/.commune/server', # the path to store the server data
        # AUTHENTICATION
        tx = 'tracker', # the tx for the server
        tx_path = '~/.commune/server/tx',
        auth = 'auth', # the auth for the server,
        private = True, # whether the store is private or not
        middleware = 'server.middleware', # the middleware for the server
        # PROCESS MANAGER
        pm = 'pm', # the process manager to use
        verbose:bool = True, # whether to print the output
        timeout = 10, # (in seconds) the maximum time to wait for a response
        ):
        
        self.store = c.mod(store)(path)
        self.network = network or 'local'
        self.tempo = tempo
        self.verbose = verbose
        self.tracker = c.mod(tx)(tx_path=tx_path)
        self.auth = c.mod(auth)()
        self.pm = c.mod(pm)() # sets the module to the pm
        self.timeout = timeout
    
    def forward(self, fn:str, request: Request):
        """
        forwards the request to the module
        1. get the request
        2. verify the request
        3. run the function
        4. save the transaction
        5. return the result (if generator, return a stream, else return the result)

        params: 
            fn: the function to run
            request: the request object
        returns:
            the result of the function
        """
        request = self.get_request(fn=fn, request=request) # get the request
        fn = request['fn']
        params = request['params']
        print(request, color='blue', verbose=self.verbose)
        # NOW RUN THE FUNCTION
        fn_obj = getattr(self.module, fn) # get the function object from the module
        if callable(fn_obj):
            if len(params) == 2 and 'args' in params and 'kwargs' in params :
                kwargs = dict(params.get('kwargs')) 
                args = list(params.get('args'))
            else:
                args = []
                kwargs = dict(params)
            result = fn_obj(*args, **kwargs) 
        else:
            # if the fn is not callable, return it
            result = fn_obj
        fn = request['fn']
        if c.is_generator(result):
            def generator_wrapper(generator):
                _gen_result =  ''
                for item in generator:
                    print(item, end='')
                    _gen_result += str(item)
                    yield item

                # save the transaction between the headers and server for future auditing

                self.tracker.forward(
                    module=self.module.info['name'],
                    fn=fn, # 
                    params=params, # params of the inputes
                    result=_gen_result,
                    schema=self.module.info['schema'][fn],
                    client=request['client'],
                    server=self.auth.headers(data={'fn': fn, 'params': params, 'result': _gen_result}, key=self.key),
                    )
            # if the result is a generator, return a stream
            return  EventSourceResponse(generator_wrapper(result))
        else:

            # save the transaction between the headers and server for future auditing
            result = self.serializer.forward(result) # serialize the result
            fn = request['fn']
            params = request['params']

            tx = self.tracker.forward(
                module=self.module.info['name'],
                fn=fn, # 
                params=params, # params of the inputes
                result=result,
                schema=self.module.info['schema'][fn],
                client=request['client'], # client auth
                server=self.auth.headers(data={"fn": fn, "params": params, "result": result}, key=self.key),
                )
            return result

    def get_request(self, fn:str, request) -> float:
        if fn == '':
            fn = 'info'
        # params
        # headers
        headers = dict(request.headers)    
        server_cost = float(self.module.info['schema'].get(fn, {}).get('cost', 0))
        client_cost = float(headers.get('cost', 0))
        assert client_cost >= server_cost, f'Insufficient cost {client_cost} for fn {fn} with cost {server_cost}'
        self.auth.verify(headers) # verify the headers
        params = self.loop.run_until_complete(request.json())
        params = json.loads(params) if isinstance(params, str) else params
        assert self.auth.hash({"fn": fn, "params": params}) == headers['data'], f'Invalid data hash for {params}'
        role = self.role(headers['key']) # get the role of the user
        if role not in ['admin', 'owner']:
            assert fn in self.module.info['schema'], f"Function {fn} not in fns={self.fns}"
        return {'fn': fn, 'params': params, 'client': headers}

    def fleet(self, module='module', n=2, timeout=10):
        if '::' not in module:
            module = module + '::'
        names = [module+str(i) for i in range(n)]
        return c.wait([c.submit(self.serve, [names[i]])  for i in range(n)], timeout=timeout)

    def txs(self, *args, **kwargs) -> Union[pd.DataFrame, List[Dict]]:
        return  self.tracker.txs( *args, **kwargs)

    def hash(self, data:dict) -> str:
        return  hashlib.sha256(json.dumps(data).encode()).hexdigest()

    def set_fns(self, 
                fns:Optional[List[str]], 
                fn_attributes =['endpoints', 
                        'fns',
                        'expose', 
                        "exposed_fns",
                        'server_fns', 
                        'public_fns', 
                        'pubfns',  
                        'public_functions', 
                        'exposed_functions'] # the attributes that can contain the fns
        ):
        """
        sets the fns of the server
        """
        fns =  fns or []
        if len(fns) == 0:
            for fa in fn_attributes:
                if hasattr(self.module, fa) and isinstance(getattr(self.module, fa), list):
                    fns = getattr(self.module, fa) 
                    break
        # does not start with _ and is not a private fn
        fns = [fn for fn in sorted(list(set(fns + self.helper_fns))) if not fn.startswith('_')]
        schema = c.schema(self.module, code=False)
        schema = {fn: schema[fn] for fn in fns if fn in schema}
        self.fns = fns
        return {'fns': self.fns}
        
    def set_port(self, port:Optional[int]=None, port_attributes = ['port', 'server_port']):
        if port == None:
            in_port_attribute = any([k for k in port_attributes])
            if in_port_attribute:
                for k in port_attributes:
                    # Check if the attribute exists and is an integer
                    if hasattr(self.module, k) and isinstance(getattr(self.module, k), int):
                        port = getattr(self.module, k)
                        print(f'Setting port from attribute {k} to {port}', color='green')
                        break
            else:
                namespace = self.namespace()
                if self.name in namespace:
                    port = int(namespace.get(self.name).split('::')[-1])
                    self.kill(self.name)
        # if the port is still None, get a free port
        port = port or c.free_port()
        while c.port_used(port):
            c.kill_port(port)
            c.sleep(0.1)
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

    mods_path = 'modules.json'

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
            
        modules = self.store.get(self.mods_path, None, max_age=max_age, update=update)
        if modules == None :
            urls = self.urls(search=search, **kwargs)
            print(f'Updating modules from {self.mods_path}', color='yellow')
            
            futures  = [c.submit(c.call, [url + '/info'], timeout=timeout, mode='thread') for url in urls]
            modules =  c.wait(futures, timeout=timeout)
            print(f'Found {len(modules)} modules', color='green')
            modules = list(filter(module_filter, modules))
            self.store.put(self.mods_path, modules)
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

    def call_count(self, user:str, # the key to rate
            fn = 'info', # the function to rate
             max_age:int = 60, # the maximum age of the rate
             update:bool = False, # whether to update the rate
             module = None, # the module to rate on
             ) -> float:
        if module == None:
            module = self.name
        if '/' in user:
            module, user = user.split('/')
        path = f'results/{module}/{fn}/{user}'
        return len( self.store.paths(path)) 


    def role(self, user) -> str:
        """
        get the role of the address ( admin, owner, local, public)
        """
        assert not self.is_blacklisted(user), f"Address {user} is blacklisted"
        role = 'public'
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
            
            if user in self.address2key:
                role =  'local'
            else:
                role = 'public' # default role is public

        return role

    def roles(self, max_age:int = 60, update:bool = False):
        """
        get the roles of the addresses
        """
        roles = self.store.get(f'roles', {}, max_age=max_age, update=update)
        return roles

    def add_role(self, address:str, role:str, max_age:int = 60, update:bool = False):
        """
        add a role to the address
        """
        roles = self.store.get(f'roles', {}, max_age=max_age, update=update)
        roles[address] = role
        self.store.put(f'roles', roles)
        return {'roles': roles, 'address': address }

    def remove_role(self, address:str, role:str, max_age:int = 60, update:bool = False):
        """
        remove a role from the address
        """
        roles = self.store.get(f'roles', {}, max_age=max_age, update=update)
        if address in roles:
            del roles[address]
        self.store.put(f'roles', roles)
        return {'roles': roles, 'address': address }

    def get_role(self, address:str, max_age:int = 60, update:bool = False):
        """
        get the role of the address
        """
        roles = self.store.get(f'roles', {}, max_age=max_age, update=update)
        if address in roles:
            return roles[address]
        else:
            return 'public'

    def has_role(self, address:str, role:str, max_age:int = 60, update:bool = False):
        """
        check if the address has the role
        """
        roles = self.store.get(f'roles', {}, max_age=max_age, update=update)
        if address in roles:
            return roles[address] == role
        else:
            return False

    def blacklist_user(self, user:str, max_age:int = 60, update:bool = False):
        """
        check if the address is blacklisted
        """
        blacklist = self.store.get(f'blacklist', [], max_age=max_age, update=update)
        blacklist.append(user)
        blacklist = list(set(blacklist))
        self.store.put(f'blacklist', blacklist)
        return {'blacklist': blacklist, 'user': user }

    def unblacklist_user(self, user:str, max_age:int = 60, update:bool = False):
        """
        check if the address is blacklisted
        """
        blacklist = self.store.get(f'blacklist', [], max_age=max_age, update=update)
        blacklist.remove(user)
        blacklist = list(set(blacklist))
        self.store.put(f'blacklist', blacklist)
        return {'blacklist': blacklist, 'user': user }

    def blacklist(self,  max_age:int = 60, update:bool = False):
        """
        check if the address is blacklisted
        """
        return self.store.get(f'blacklist', [], max_age=max_age, update=update)

    def is_blacklisted(self, user:str, max_age:int = 60, update:bool = False):
        """
        check if the address is blacklisted
        """
        blacklist = self.blacklist(max_age=max_age, update=update)
        return user in blacklist

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

    @classmethod
    def serve(cls, 
              module: Union[str, 'Module', Any] = None, # the module in either a string
              params:Optional[dict] = None,  # kwargs for the module
              port :Optional[int] = None, # name of the server if None, it will be the module name
              name = None, # name of the server if None, it will be the module name
              fns = None, # list of fns to serve, if none, it will be the endpoints of the module
              key = None, # the key for the server
              cwd = None,
              remote = False,
              auth = 'auth',
              serializer = 'serializer',
              **extra_params
              ):

        module = module or 'module'
        name = name or module
        params = {**(params or {}), **extra_params}
        if remote:
            return c.fn('pm/serve')(module=module, params=params, name=name, fns=fns, port=port, key=key, auth=auth, serializer=serializer)
        self =  Server(module=module, name=name, fns=fns, params=params, port=port,  key=key)
        if self.exists(name):
            self.kill(name) # kill the server if it exists

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
        self.auth = c.mod(auth)()

        info = c.info(module, key=self.key)
        info['url'] = self.url
        self.module.info = info
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

        c.print(f'Serving(name={name} port={port} key={self.key.address})', color='green')
        uvicorn.run(app, host='0.0.0.0', port=self.module.port, loop='asyncio')
        return {'success':True, 'message':f'Set module to {self.name}'}

    def get_info(self, name:str, timeout:int = 60, interval:int = 1):
        elapsed_seconds = 0
        while elapsed_seconds < timeout:
            try:
                info = c.call(name + '/info')
                if 'key' in info:
                    return info
            except Exception as e:
                print(c.detailed_error(e))
            time.sleep(interval)
            elapsed_seconds += interval
            print(f'waiting for {name} to be available')
        raise TimeoutError(f"Timeout waiting for {name} to be available after {timeout} seconds")

    def test_serializer(self):
        return c.mod('serializer')().test()  

    def test_server(self, 
                    server = 'module::test_serving', 
                    key="test_deployer"):
        c.serve(server, key=key)
        info = self.get_info(server)
        assert info['key'] == c.key(key).ss58_address
        c.kill(server)
        return {'success': True, 'msg': 'server test passed'}

    def test_executor(self):
        return c.mod('executor')().test()

    def test_auth(self, auths=['auth.jwt', 'auth']):
        for auth in auths:
            print(f'testing {auth}')
            c.mod(auth)().test()
        return {'success': True, 'msg': 'server test passed', 'auths': auths}

    def test_role(self, address:str=None, role:str='viber', max_age:int = 60, update:bool = False):
        """
        test the role of the address
        """
        address = address or c.get_key('test').ss58_address
        self.add_role(address, role, max_age=max_age, update=update)
        assert self.get_role(address, max_age=max_age, update=update) == role, f"Failed to add {address} to {role}"
        self.remove_role(address, role, max_age=max_age, update=update)
        assert self.get_role(address, max_age=max_age, update=update) == 'public', f"Failed to remove {address} from {role}"
        return {'roles': True, 'address': address , 'roles': self.get_role(address, max_age=max_age, update=update)}

    def test_blacklist(self,  max_age:int = 60, update:bool = False):  
        """
        check if the address is blacklisted
        """
        key = c.get_key('test').ss58_address
        self.blacklist_user(key, max_age=max_age, update=update)
        assert key in self.blacklist(max_age=max_age, update=update), f"Failed to add {key} to blacklist"
        self.unblacklist_user(key, max_age=max_age, update=update)
        assert key not in self.blacklist(max_age=max_age, update=update), f"Failed to remove {key} from blacklist"
        return {'blacklist': True, 'user': key , 'blacklist': self.blacklist(max_age=max_age, update=update)}
