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

    def __init__(
        self, 
        path = '~/.commune/server', # the path to store the server data
        verbose:bool = True, # whether to print the output
        pm = 'server.pm', # the process manager to use
        middleware = 'server.middleware', # the middleware to use
        serializer = 'server.serializer', # the serializer to use
        tracker = 'server.tracker', # the tracker to use
        auth = 'server.auth', # the auth to use
        **_kwargs):
        
        self.store = c.mod('store')(path)
        self.verbose = verbose
        self.pm = c.mod(pm)() # sets the mod to the pm
        self.middleware = c.mod(middleware)
        self.serializer = c.mod(serializer)() # sets the serializer
        self.tracker = c.mod(tracker)()
        self.auth = c.mod(auth)()
    
    def forward(self, fn:str, request: Request):
        """
        runt the function
        """
        request = self.get_request(fn=fn, request=request) # get the request
        fn = request['fn']
        params = request['params']
        print(request, color='blue', verbose=self.verbose)
        # NOW RUN THE FUNCTION

        fn_obj = getattr(self.mod, fn) # get the function object from the mod
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
        if c.is_generator(result):
            def generator_wrapper(generator):
                gen_result =  {'data': [], 'start_time': time.time(), 'end_time': None, 'cost': 0}
                for item in generator:
                    print(item, end='')
                    gen_result['data'].append(item)
                    yield item
                # save the transaction between the headers and server for future auditing
                server_auth = self.auth.headers(data={'fn': fn, 'params': params, 'result': gen_result})

                self.tracker.forward(
                    mod=self.name,
                    fn=fn, # 
                    params=params, # params of the inputes
                    result=gen_result,
                    schema=self.schema[fn],
                    client=request['client'],
                    server=server_auth, 
                    key=self.key)
            # if the result is a generator, return a stream
            return  EventSourceResponse(generator_wrapper(result))
        else:

            # save the transaction between the headers and server for future auditing
            result = self.serializer.forward(result) # serialize the result
            fn = request['fn']
            params = request['params']
            server_auth = self.auth.generate(data={"fn": fn, "params": params, "result": result})
            tx = self.tracker.forward(
                mod=self.name,
                fn=fn, # 
                params=params, # params of the inputes
                result=result,
                schema=self.schema[fn],
                client=request['client'], # client auth
                server= server_auth , 
                key=self.key)
            return result
        raise Exception('Should not reach here')

    def get_request(self, fn:str, request) -> float:
        if fn == '':
            fn = 'info'
        # params
        # headers
        headers = dict(request.headers)    
        server_cost = float(self.schema.get(fn, {}).get('cost', 0))
        client_cost = float(headers.get('cost', 0))
        assert client_cost >= server_cost, f'Insufficient cost {client_cost} for fn {fn} with cost {server_cost}'
        self.auth.verify(headers) # verify the headers
        loop = asyncio.get_event_loop()
        params = loop.run_until_complete(request.json())
        params = json.loads(params) if isinstance(params, str) else params
        assert self.auth.hash({"fn": fn, "params": params}) == headers['data'], f'Invalid data hash for {params}'
        role = self.role(headers['key']) # get the role of the user
        if role not in ['admin', 'owner']:
            assert fn in self.fns, f"Function {fn} not in fns={self.fns}"
        return {'fn': fn, 'params': params, 'client': headers}

    def txs(self, *args, **kwargs) -> Union[pd.DataFrame, List[Dict]]:
        return  self.tracker.txs( *args, **kwargs)

    def set_fns(self, 
                fns: List[str] = None, # the fns to set
                helper_fns : List[str] = ['info', 'forward'],# the helper fns
                hide_private_fns: bool = True, # whether to include private fns
                fn_attributes : List[str] =['endpoints',  'fns', 'expose',  'expoed', 'functions'],  # the attributes that can contain the fns
        ):
        """
        sets the fns of the server
        """
        fns =  fns or []
        # if no fns are provided, get them from the mod attributes
        if len(fns) == 0:
            for fa in fn_attributes:
                if hasattr(self.mod, fa) and isinstance(getattr(self.mod, fa), list):
                    fns = getattr(self.mod, fa) 
                    break
        # does not start with _ and is not a private fn
        if hide_private_fns:
            fns = [fn for fn in fns if not fn.startswith('_') ]
        fns = list(set(fns + helper_fns))
        schema = c.schema(self.mod, public=self.public)
        self.schema= {fn: schema[fn] for fn in fns if fn in schema}
        self.fns = sorted(list(self.schema.keys()))
        return schema
        
    def get_port(self, port:Optional[int]=None):
        
        port = port or c.free_port()
        return port

    def servers(self, search=None,  **kwargs) -> List[str]:
        return list(self.namespace(search=search, **kwargs).keys())

    def urls(self, search=None,  **kwargs) -> List[str]:
        return list(self.namespace(search=search, **kwargs).values())   

    mods_path = 'mods.json'

    def mods(self, 
                search=None, 
                max_age=None, 
                update=False, 
                features=['name', 'url', 'key'], 
                timeout=24, 
                **kwargs):

        def module_filter(m: dict) -> bool:
            """Filter function to check if a mod contains the required features."""
            return isinstance(m, dict) and all(feature in m for feature in features )    
        mods = self.store.get(self.mods_path, None, max_age=max_age, update=update)
        if mods == None :
            urls = self.urls(search=search, **kwargs)
            print(f'Updating mods from {self.mods_path}', color='yellow')
            futures  = [c.submit(c.call, {"fn":url + '/info'}, timeout=timeout, mode='thread') for url in urls]
            mods =  c.wait(futures, timeout=timeout)
            print(mods)
            print(f'Found {len(mods)} mods', color='green')
            mods = list(filter(module_filter, mods))
            self.store.put(self.mods_path, mods)
        else:
            mods = list(filter(module_filter, mods))

        if search != None:
            mods = [m for m in mods if search in m['name']]
        return mods

    def n(self, search=None, **kwargs):
        return len(self.mods(search=search, **kwargs))

    def exists(self, name:str, **kwargs) -> bool:
        """check if the server exists"""
        return bool(name in self.servers(**kwargs))

    def call_count(self, user:str, # the key to rate
            fn = 'info', # the function to rate
             max_age:int = 60, # the maximum age of the rate
             update:bool = False, # whether to update the rate
             mod = None, # the mod to rate on
             ) -> float:
        if mod == None:
            mod = self.name
        if '/' in user:
            mod, user = user.split('/')
        path = f'results/{mod}/{fn}/{user}'
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
        elif hasattr(self, 'mod') and user == self.key.key_address:
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

    def namespace(self,  search=None,**kwargs) -> dict:
        return self.pm.namespace(search=search, **kwargs)

    def serve(self, 
              mod: Union[str, 'Module', Any] = None, # the mod in either a string
              params:Optional[dict] = None,  # kwargs for the mod
              port :Optional[int] = None, # name of the server if None, it will be the mod name
              name = None, # name of the server if None, it will be the mod name
              fns = None, # list of fns to serve, if none, it will be the endpoints of the mod
              key = None, # the key for the server
              cwd = None, # the cwd to run the server in
              remote = False, # whether to run the server remotely
              host = '0.0.0.0',
              public=False,
              server_mode = 'http',
              **extra_params 
              ):
        port = self.get_port(port)
        mod = mod or 'mod'
        name = name or mod
        params = {**(params or {}), **extra_params}
        if remote:
            return c.fn('pm/serve')(mod=mod, params=params, name=name, fns=fns, port=port, key=key, cwd=cwd)
        if self.exists(name):
            self.kill(name) # kill the server if it exists
        mod = mod or 'mod'
        if isinstance(mod, str):
            if '::' in mod:
                name = mod
                mod = '::'.join(name.split('::')[:-1])
        self.public = public
        self.mod = c.mod(mod)(**(params or {}))
        self.name = name = name or mod 
        self.key = c.get_key(key or self.name)
        # add the endpoints
        self.set_fns(fns) 
        self.info = c.info(mod, key=self.key, public=self.public, schema=False)
        self.info['schema'] = self.schema
        self.info['url'] = f"{server_mode}://{host}:{port}"
        self.mod.info = self.info

        # start the server
        def server_fn(fn: str, request: Request):
            try:
                return self.forward(fn, request)
            except Exception as e:
                return c.detailed_error(e)
        # add CORS middleware
        self.app = FastAPI()
        self.app.add_middleware(CORSMiddleware,allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
        self.app.post("/{fn}")(server_fn)
        c.print(f'Serving({self.info})', color='green')
        self.store.put(f'servers/{self.name}', self.info)
        uvicorn.run(self.app,  host='0.0.0.0', port=port, loop='asyncio')
        return {'success':True, 'message':f'Set mod to {self.name}'}
