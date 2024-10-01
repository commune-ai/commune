import commune as c
from typing import *
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import asyncio
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from sse_starlette.sse import EventSourceResponse
import os


class ServerMiddleware(BaseHTTPMiddleware):
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


class Server(c.Module):
    def __init__(
        self,
        module: Union[c.Module, object] = None,
        functions:Optional[List[str]] = None, # list of endpoints
        key:str = None, # key for the server (str)
        name: str = None,
        port: Optional[int] = None,
        # ------- ADVANCE ------
        network:str = 'subspace',
        period = 60,
        max_request_staleness:int = 5,
        max_bytes:int = 10 * 1024 * 1024,  # 1 MB limit
        process_request:Optional[Union[callable, str]] = None,
        network_staleness = 60,
        path:str = 'state',
        helper_functions  = ['info','metadata','schema', 'server_name','server_functions','forward'], # whitelist of helper functions to load
        helper_function_attributes = ['helper_functions', 'whitelist', 'endpoints', 'server_functions'],
        allow_origins = ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"], 
        loop = None,
        nest_asyncio = True,
        **kwargs,
        ) -> 'Server':
        
        if isinstance(module, str):
            module = c.module(module)()
        self.helper_function_attributes = helper_function_attributes
        module.name = module.server_name = name or module.server_name
        module.helper_functions = helper_functions
        module.port = module.server_port =  port if port not in ['None', None] else c.free_port()
        module.address  = module.server_address =  f"{c.ip()}:{module.port}"
        module.key  = module.server_key = c.get_key(key or module.name, create_if_not_exists=True)
        module.schema =  module.server_schema = self.get_server_schema(module)
        module.functions  = module.server_functions = list(module.schema.keys())
        module.info  =  module.server_info =  self.get_info(module)
        
        if  nest_asyncio:
            c.new_event_loop(nest_asyncio=nest_asyncio)
        self.loop = asyncio.get_event_loop()
        self.app = FastAPI()    
        self.app.add_middleware(ServerMiddleware, max_bytes=max_bytes)    
        self.app.add_middleware(CORSMiddleware,
                                allow_origins=allow_origins, 
                                allow_credentials=allow_credentials,
                                allow_methods=allow_methods,
                                allow_headers=allow_headers)
        @self.app.post("/{fn}")
        def api_function(fn, request: Request):
            try:
                output = self.forward(fn, request)
            except Exception as e:
                output =  c.detailed_error(e)
                c.print(output, color='red')
            return output

        self.max_request_staleness = max_request_staleness
        self.serializer = c.module('serializer')()
        self.network = network
        self.path = path
        self.network_staleness = network_staleness
        self.set_process_request(process_request)
        self.network = network
        self.module = module
        self.period = period
        self.path = self.resolve_path(path)
        self.state = {'sync_time': 0,  'stake': {},'stake_from': {},  'fn_info': {}}
        c.thread(self.sync_loop)
        self.set_api()

    def add_fn(self, name:str, fn: str):
        assert callable(fn), 'fn not callable'
        setattr(self.module, name, fn)
        return {'success':True, 'message':f'Added {name} to {self.name} module'}
    
    def set_process_request(self, process_request:Union[callable, str]):
        if not process_request:
            return {'success':True, 'message':f'using default access function'}
        if hasattr(process_request, 'forward'):
            process_request = process_request.forward
        assert callable(process_request), 'access_fn must be callable'
        self.process_request = process_request
        return {'success':True, 'message':f'Set access_fn for {self.name}'}
    
    def forward(self, fn:str,  request: Request):
        request = self.process_request(fn=fn, request=request )
        data = request['data']
        headers = request['headers']
        args = data.get('args', [])
        kwargs = data.get('kwargs', {})
        fn_obj = getattr(self.module, fn)
        if callable(fn_obj):
            response = fn_obj(*args, **kwargs)
        else:
            response = fn_obj
        latency = c.round(c.time() - int(headers['timestamp']), 3)
        msg = f"<✅Response(fn={fn} from={headers['key'][:4]}... latency={latency}s)✅>"
        c.print(msg, color='green')
        if c.is_generator(response):
            def generator_wrapper(generator):
                for item in generator:
                    yield self.serializer.serialize(item)
            return EventSourceResponse(generator_wrapper(response))
        else:
            return self.serializer.serialize(response)

    def set_api(self ):
        # start the server
        try:
            c.print(f' Served(name={self.module.name}, address={self.module.address}, key=🔑{self.module.key}🔑 ) 🚀 ', color='purple')
            c.register_server(name=self.module.name,address = self.module.address)
            uvicorn.run(self.app, host='0.0.0.0', port=self.module.port, loop='asyncio' )
        except Exception as e:
            c.print(e, color='red')
        finally:
            c.deregister_server(self.name)

    def __del__(self):
        c.deregister_server(self.name)

    @classmethod
    def serve(cls, 
              module: Any = None,
              kwargs:Optional[dict] = None,  # kwargs for the module
              params: Optional[dict] = None, # kwargs for the module
              tag:Optional[str]=None,
              network: Optional[str] = 'subspace', # network to run the server
              port :Optional[int] = None, # name of the server if None, it will be the module name
              server_name:str=None, # name of the server if None, it will be the module name
              name = None, # name of the server if None, it will be the module name
              remote:bool = True, # runs the server remotely (pm2, ray)
              tag_seperator:str='::',
              max_workers:int = None,
              public: bool = False,
              mnemonic = None, # mnemonic for the server
              key = None,
              **extra_kwargs
              ):
        module = module or 'module'
        kwargs = {**(params or kwargs or {}), **extra_kwargs}
        name = (name or server_name or module) or c.module_name()
        if tag_seperator in name:
            module, tag = name.split(tag_seperator)
        if tag != None:
            name = f'{module}{tag_seperator}{tag}'
        if port == None:
            namespace = c.get_namespace()
            if name in namespace:
                port = int(namespace.get(name).split(':')[-1])
            else:
                port = c.free_port()
        if c.server_exists(name):
            c.kill(name)
        response =  { 'module':module, 'name': name, 'address':f'0.0.0.0:{port}', 'kwargs':kwargs} 
        if remote:
            remote = False
            remote_kwargs = c.locals2kwargs(locals())  # GET THE LOCAL KWARGS FOR SENDING TO THE REMOTE
            for _ in ['extra_kwargs', 'address', 'response', 'namespace']:
                remote_kwargs.pop(_, None) # WE INTRODUCED THE ADDRES
            cls.remote_fn('serve', name=name, kwargs=remote_kwargs)
            return response
        cls(module=c.module(module)(**kwargs), 
            name=name, 
            port=port, 
            network=network,   
            max_workers=max_workers, 
            mnemonic = mnemonic,
            public=public, 
            key=key)
        return  response






    sync_time = 0
    timescale_map  = {'sec': 1, 'min': 60, 'hour': 3600, 'day': 86400, 'minute': 60, 'second': 1}
        
    def get_rate_limit(self, fn, address):
        # stake rate limit
        stake = self.state['stake'].get(address, 0)
        stake_from = self.state['stake_from'].get(address, 0)
        stake = (stake_from * self.stake_from_multipler) + stake
        fn_info = self.state.get('fn_info', {}).get(fn, {'stake2rate': self.stake2rate, 'max_rate': self.max_rate})
        rate_limit = (stake / fn_info['stake2rate']) # convert the stake to a rate
        return rate_limit

    def process_request(self, fn:str, request: Request) -> dict:
        """
        input:
            {
                args: list = [] # the arguments to pass to the function
                kwargs: dict = {} # the keyword arguments to pass to the function
                timestamp: int = 0 # the timestamp to use
                address: str = '' # the address to use
            }

        Rules:
        1. Admins have unlimited access to all functions, do not share your admin keys with anyone
            - Admins can add and remove other admins 
            - to check admins use the is_admin function (c.is_admin(address) or c.admins() for all admins)
            - to add an admin use the add_admin function (c.add_admin(address))
        2. Local keys have unlimited access but only to the functions in the whitelist
        returns : dict
        """

        headers = dict(request.headers.items())
        address = headers.get('key', headers.get('address', None))
        assert address, 'No key or address in headers'
        request_staleness = c.timestamp() - int(headers['timestamp'])
        assert  request_staleness < self.max_request_staleness, f"Request is too old ({request_staleness}s > {self.max_request_staleness}s (MAX)" 
        data = self.loop.run_until_complete(request.json())
        data = self.serializer.deserialize(data) 
        auth={'data': headers['hash'], 'timestamp': headers['timestamp']}
        request = {'data': data, 'headers': headers}
        signature = headers.get('signature', None)
        assert c.verify(auth=auth,signature=signature, address=address), 'Invalid signature'
        kwargs = dict(data.get('kwargs', {}))
        args = list(data.get('args', []))
        if 'params' in data:
            if isinstance(data['params', dict]):
                kwargs = {**kwargs, **data['params']}
            elif isinstance(data['params'], list):
                args = [*args, *data['params']]
            else:
                raise ValueError('params must be a list or a dictionary')
        data = {'args': args, 'kwargs': kwargs}

        if c.is_admin(address):
            return request
        assert fn in self.module.server_functions or fn in self.module.helper_functions , f"Function {fn} not in whitelist={self.module.server_functions}"
        assert not bool(fn.startswith('__') or fn.startswith('_')), f'Function {fn} is private'
        is_local_key = address in self.address2key
        is_user = c.is_user(address)
        if is_local_key or is_user:
            return request
        # check if the user has exceeded the rate limit
        user_info_path = self.resolve_path(f'user_info/{address}.json') # get the user info path
        user_info = self.get(user_info_path, {}, max_age=self.period) # get the user info, refresh if it is too old (> period)
        user_fn_info = user_info.get(fn, {"timestamp": c.time(), 'count': 0}) # get the user info for the function
        reset_count = bool((c.timestamp() -  user_fn_info['timestamp'])  > self.period) # reset the count if the period has passed
        user_fn_info['count'] = (user_fn_info.get('count', 0) if reset_count else 0) + 1 # increment the count
        rate_limit = self.get_rate_limit(fn=fn, address=address) # get the rate limit for the user
        assert user_fn_info['count'] <= rate_limit, f'rate limit exceeded for {fn}'
        user_info[fn] = user_fn_info
        return request

    def sync_loop(self):
        while True:
            try:
                r = self.sync()
            except Exception as e:
                r = c.detailed_error(e)
            c.print(r)
            c.sleep(self.network_staleness)

    def sync(self, update=False):
        path = self.resolve_path(self.path + '/network_state.json')
        state = self.get(path, {}, max_age=self.network_staleness)
        network = self.network
        staleness = c.time() - state.get('sync_time', 0)
        self.address2key = c.address2key()
        response = { 'path': path, 
                    'network_staleness':  self.network_staleness,  
                    'network': network,
                    'staleness': int(staleness), 
                    }
        
        if staleness < self.network_staleness:
            response['msg'] = f'synced too earlly waiting {self.network_staleness - staleness} seconds'
            return response
        else:
            response['msg'] =  'Synced with the network'
            response['staleness'] = 0
        c.get_namespace(max_age=self.network_staleness)
        self.subspace = c.module('subspace')(network=network)
        state['stake_from'] = self.subspace.stake_from(fmt='j', update=update, max_age=self.network_staleness)
        state['stake'] =  {k: sum(v.values()) for k,v in state['stake_from'].items()}
        self.state = state
        self.put(path, self.state)
        return response



    @classmethod
    def kill(cls, 
             module,
             mode:str = 'pm2',
             verbose:bool = False,
             update : bool = True,
             prefix_match = False,
             network = 'local', # local, dev, test, main
             **kwargs):

        kill_fn = getattr(cls, f'{mode}_kill')
        delete_modules = []

        try:
            killed_module =kill_fn(module, verbose=verbose,prefix_match=prefix_match, **kwargs)
        except Exception as e:
            return {'error':str(e)}
        if isinstance(killed_module, list):
            delete_modules.extend(killed_module)
        elif isinstance(killed_module, str):
            delete_modules.append(killed_module)
        else:
            delete_modules.append(killed_module)
        # update modules
        c.deregister_server(module, network=network)
        assert c.server_exists(module, network=network) == False, f'module {module} still exists'
        servers = c.servers()
        for m in delete_modules:
            if m in servers:
                c.deregister_server(m, network=network)
        return {'server_killed': delete_modules, 'update': update}

    @classmethod
    def kill_many(cls, servers, search:str = None, network='local',  timeout=10, **kwargs):
        servers = c.servers(network=network)
        servers = [s for s in servers if  search in s]
        futures = []
        for s in servers:
            c.print(f'Killing {s}', color='red')
            future = c.submit(c.kill, kwargs={'module':s, **kwargs}, imeout=timeout)
            futures.append(future)
        results = []
        for r in c.as_completed(futures, timeout=timeout):
            results += [r.result()]
        c.print(f'Killed {len(results)} servers', color='red')
        return results
    

    @classmethod
    def fleet(cls, module, n=5, timeout=10):
        futures = []
        if '::'  not in module:
            module = f'{module}::'

        
        for i in range(n):
            module_name = f'{module}{i}'
            future = c.submit(cls.serve, kwargs=dict(module=module_name), timeout=timeout)
            futures.append(future)
        results = []
        for future in c.as_completed(futures, timeout=timeout):
            result = future.result()
            results.append(result)

        return results


    @classmethod
    def serve_many(cls, modules:list, **kwargs):

        if isinstance(modules[0], list):
            modules = modules[0]
        
        futures = []
        for module in modules:
            future = c.submit(c.serve, kwargs={'module': module, **kwargs})
            futures.append(future)
            
        results = []
        for future in c.as_completed(futures):
            result = future.result()
            results.append(result)
        return results
    serve_batch = serve_many


    @classmethod
    def wait_for_server(cls,
                          name: str ,
                          network: str = 'local',
                          timeout:int = 600,
                          sleep_interval: int = 1, 
                          verbose:bool = False) -> bool :
        
        time_waiting = 0
        while time_waiting < timeout:
            namespace = c.get_namespace(network=network)
            if name in namespace:
                c.print(f'{name} is ready', color='green')
                return True
            time_waiting += sleep_interval
            c.print(f'Waiting for {name} for {time_waiting} seconds', color='red')
            c.sleep(sleep_interval)
        raise TimeoutError(f'Waited for {timeout} seconds for {name} to start')
    

    @staticmethod
    def kill_all_servers( *args, **kwargs):
        '''
        Kill all of the servers
        '''
        for module in c.servers(*args, **kwargs):
            c.kill(module)

        # c.update(network='local')
            
    @classmethod
    def kill_all(cls, network='local', timeout=20, verbose=True):
        futures = []
        servers = c.servers(network=network)
        n = len(servers)
        progress = c.tqdm(n)
        for s in servers:
            c.print(f'Killing {s}', color='red')
            futures += [c.submit(c.kill, kwargs={'module':s, 'update': False}, return_future=True)]
        results_list = []
        for f in c.as_completed(futures, timeout=timeout):
            result = f.result()
            print(result)
            progress.update(1)
            results_list += [result]
        namespace = c.get_namespace(network=network, update=True)
        print(namespace)
        new_n = len(servers)
        c.print(f'Killed {n - new_n} servers, with {n} remaining {servers}', color='red')
        return {'success':True, 'old_n':n, 'new_n':new_n, 'servers':servers, 'namespace':namespace}

 # the default
    network : str = 'local'
    @classmethod
    def resolve_network_path(cls, network:str, netuid:str=None):
        if netuid != None:
            if network not in ['subspace']:
                network = f'subspace'
            network = f'subspace/{netuid}'
        return cls.resolve_path(network + '.json')

    @classmethod
    def namespace(cls, search=None,
                    network:str = 'local',
                    update:bool = False, 
                    netuid=None, 
                    max_age:int = 60,
                    timeout=6,
                    verbose=False) -> dict:
        network = network or 'local'
        path = cls.resolve_network_path(network)
        namespace = cls.get(path, None, max_age=max_age)
        if namespace == None:
            
            namespace = cls.update_namespace(network=network, 
                                            netuid=netuid, 
                                            timeout=timeout, 
                                            verbose=verbose)
            cls.put(path,namespace)
        if search != None:
            namespace = {k:v for k,v in namespace.items() if search in k} 
        namespace = {k:':'.join(v.split(':')[:-1]) + ':'+ str(v.split(':')[-1]) for k,v in namespace.items()}
        namespace = dict(sorted(namespace.items(), key=lambda x: x[0]))
        ip  = c.ip()
        namespace = {k: v.replace(ip, '0.0.0.0') for k,v in namespace.items() }
        namespace = { k.replace('"', ''): v for k,v in namespace.items() }
        return namespace
    
    @classmethod
    def update_namespace(cls, network, netuid=None, timeout=1, search=None, verbose=False):
        c.print(f'UPDATING --> NETWORK(network={network} netuid={netuid})', color='blue')
        if 'subspace' in network:
            if '.' in network:
                network, netuid = network.split('.')
            else: 
                netuid = netuid or 0
            if c.is_int(netuid):
                netuid = int(netuid)
            namespace = c.module(network)().namespace(search=search, max_age=1, netuid=netuid)
            return namespace
        elif 'local' == network: 
            namespace = {}
            addresses = ['0.0.0.0'+':'+str(p) for p in c.used_ports()]
            future2address = {}
            for address in addresses:
                f = c.submit(c.call, [address+'/server_name'], timeout=timeout)
                future2address[f] = address
            futures = list(future2address.keys())
            try:
                for f in c.as_completed(futures, timeout=timeout):
                    address = future2address[f]
                    name = f.result()
                    if isinstance(name, str):
                        namespace[name] = address
                    else:
                        print(f'Error: {name}')
            except Exception as e:
                print(e)

            namespace = {k:v for k,v in namespace.items() if 'Error' not in k} 
            namespace = {k: '0.0.0.0:' + str(v.split(':')[-1]) for k,v in namespace.items() }
        else:
            return {}
        return namespace 
    
    get_namespace = _namespace = namespace

    @classmethod
    def register_server(cls, name:str, address:str, network=network) -> None:
        namespace = cls.namespace(network=network)
        namespace[name] = address
        cls.put_namespace(network, namespace)
        return {'success': True, 'msg': f'Block {name} registered to {network}.'}
    
    @classmethod
    def deregister_server(cls, name:str, network=network) -> Dict:
        namespace = cls.namespace(network=network)
        address2name = {v: k for k, v in namespace.items()}
        if name in address2name:
            name = address2name[name]
        if name in namespace:
            del namespace[name]
            cls.put_namespace(network, namespace)
            return {'status': 'success', 'msg': f'Block {name} deregistered.'}
        else:
            return {'success': False, 'msg': f'Block {name} not found.'}
    
    @classmethod
    def rm_server(self,  name:str, network=network):
        return self.deregister_server(name, network=network)
    
    @classmethod
    def get_address(cls, name:str, network:str=network, external:bool = True) -> dict:
        namespace = cls.namespace(network=network)
        address = namespace.get(name, None)
        if external and address != None:
            address = address.replace(c.default_ip, c.ip()) 
        return address

    @classmethod
    def put_namespace(cls, network:str, namespace:dict) -> None:
        assert isinstance(namespace, dict), 'Namespace must be a dict.'
        return cls.put(network, namespace)        
    
    add_namespace = put_namespace
    
    @classmethod
    def rm_namespace(cls,network:str) -> None:
        if cls.namespace_exists(network):
            cls.rm(network)
            return {'success': True, 'msg': f'Namespace {network} removed.'}
        else:
            return {'success': False, 'msg': f'Namespace {network} not found.'}
        
    @classmethod
    def networks(cls) -> dict:
        return [p.split('/')[-1].split('.')[0] for p in cls.ls()]
    
    @classmethod
    def namespace_exists(cls, network:str) -> bool:
        path = cls.resolve_network_path( network)
        return os.path.exists(path)
    
    @classmethod
    def modules(cls, network:List=network) -> List[str]:
        return list(cls.namespace(network=network).keys())
    
    @classmethod
    def addresses(cls, network:str=network, **kwargs) -> List[str]:
        return list(cls.namespace(network=network, **kwargs).values())
    
    @classmethod
    def check_servers(self, *args, **kwargs):
        servers = c.pm2ls()
        namespace = c.get_namespace(*args, **kwargs)
        c.print('Checking servers', color='blue')
        for server in servers:
            if server in namespace:
                c.print(c.pm2_restart(server))

        return {'success': True, 'msg': 'Servers checked.'}

    @classmethod
    def add_server(cls, address:str, name=None, network:str = 'local',timeout:int=4, **kwargs):
        """
        Add a server to the namespace.
        """
        module = c.connect(address)
        info = module.info(timeout=timeout)
        name = info['name'] if name == None else name
        # check if name exists
        address = info['address']
        module_ip = address.split(':')[0]
        is_remote = bool(module_ip != c.ip())
        namespace = cls.namespace(network=network)
        if is_remote:
            name = name + '_' + str(module_ip)
        addresses = list(namespace.values())
        if address not in addresses:
            return {'success': False, 'msg': f'{address} not in {addresses}'}
        namespace[name] = address
        cls.put_namespace(network, namespace)

        return {'success': True, 'msg': f'Added {address} to {network} modules', 'remote_modules': cls.servers(network=network), 'network': network}
    
    @classmethod
    def rm_server(cls,  name, network:str = 'local', **kwargs):
        namespace = cls.namespace(network=network)
        if name in namespace.values():
            for k, v in c.copy(list(namespace.items())):
                if v == name:
                    name = k
                    break
        if name in namespace:
            # reregister
            address = cls.get_address(name, network=network)
            cls.deregister_server(name, network=network)
            servers = cls.servers(network=network)
            assert cls.server_exists(name, network=network) == False, f'{name} still exists'
            return {'success': True, 'msg': f'removed {address} to remote modules', 'servers': servers, 'network': network}
        else:
            return {'success': False, 'msg': f'{name} does not exist'}

    @classmethod
    def servers(cls, search=None, network:str = 'local',  **kwargs):
        namespace = cls.namespace(search=search, network=network, **kwargs)
        return list(namespace.keys())
    
    @classmethod
    def server_exists(cls, name:str, network:str = None,  prefix_match:bool=False, **kwargs) -> bool:
        servers = cls.servers(network=network, **kwargs)
        if prefix_match:
            server_exists =  any([s for s in servers if s.startswith(name)])
            
        else:
            server_exists =  bool(name in servers)

        return server_exists

    
    @classmethod
    def server_exists(cls, name:str, network:str = None,  prefix_match:bool=False, **kwargs) -> bool:
        servers = cls.servers(network=network, **kwargs)
        if prefix_match:
            server_exists =  any([s for s in servers if s.startswith(name)])
            
        else:
            server_exists =  bool(name in servers)

        return server_exists  
    


  
    def add_endpoint(self, name, fn):
        setattr(self, name, fn)
        self.server_functions.append(name)
        assert hasattr(self, name), f'{name} not added to {self.__class__.__name__}'
        return {'success':True, 'message':f'Added {fn} to {self.__class__.__name__}'}

    def is_endpoint(self, fn) -> bool:
        if isinstance(fn, str):
            fn = getattr(self, fn)
        return hasattr(fn, '__metadata__')


    

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
                **cls.fn_schema(fn),
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
    

    def get_info(self , 
             module,
             **kwargs
             ) -> Dict[str, Any]:
        '''
        hey, whadup hey how is it going
        '''
        info = {}
        info['schema'] = module.schema
        info['name'] = module.name 
        info['address'] = module.address
        info['key'] = module.key.ss58_address
        return info

    def get_server_schema(self, module ) -> 'Schema':
        schema = {}
        endpoints = []
        for k in self.helper_function_attributes:
            if hasattr(module, k):
                fn_obj = getattr(module, k)
                if isinstance(fn_obj, list):
                    endpoints += fn_obj
        for f in dir(module):
            try:
                if hasattr(getattr(module, f), '__metadata__'):
                    endpoints.append(f)
            except Exception as e:
                print(f'Error in get_endpoints: {e} for {f}')
        fns =  sorted(list(set(endpoints)))

        for fn in fns:
            if hasattr(module, fn):
                
                fn_obj = getattr(module, fn )
                if callable(fn_obj):
                    schema[fn] = c.fn_schema(fn_obj)  
                else: 
                    schema[fn] = {'type': str(type(fn_obj)).split("'")[1]}    
        # sort by keys
        schema = dict(sorted(schema.items()))
        return schema


    def add_endpoint(self, name, fn):
        setattr(self, name, fn)
        self.endpoints.append(name)
        assert hasattr(self, name), f'{name} not added to {self.__class__.__name__}'
        return {'success':True, 'message':f'Added {fn} to {self.__class__.__name__}'}

    def is_endpoint(self, fn) -> bool:
        if isinstance(fn, str):
            fn = getattr(self, fn)
        return hasattr(fn, '__metadata__')

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
                **cls.fn_schema(fn),
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
    



if __name__ == '__main__':
    Server.run()