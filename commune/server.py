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
        name: str = None,
        network:str = 'subspace',
        port: Optional[int] = None,
        key:str = None, # key for the server (str)
        endpoints:Optional[List[str]] = None, # list of endpoints
        max_request_staleness:int = 5,
        loop = None,
        timescale:str = 'min', # 'sec', 'min', 'hour', 'day'
        max_bytes:int = 10 * 1024 * 1024,  # 1 MB limit
        nest_asyncio:bool = True, # whether to use nest asyncio
        process_request:Optional[Union[callable, str]] = None,
        network_staleness = 60,
        path:str = 'state',
        **kwargs,
        ) -> 'Server':
        
        self.max_request_staleness = max_request_staleness
        self.serializer = c.module('serializer')()
        self.network = network
        self.timescale = timescale
        self.path = path
        self.network_staleness = network_staleness
        self.set_module(module=module, 
                        name=name, 
                        port=port, 
                        key=key, 
                        endpoints=endpoints, 
                        process_request=process_request,
                        path=path,
                        network=network)
        self.set_api(max_bytes=max_bytes, loop=loop, nest_asyncio=nest_asyncio,  **kwargs)

    def set_module(self, 
                   module:Union[c.Module, object],
                   name:Optional[str]=None,
                   port:Optional[int]=None,
                   key:Optional[str]=None,
                   endpoints:Optional[List[str]] = None,
                   process_request:Optional[str] = None,
                   network:str = 'subspace',
                   path:str = 'state'):

        self.set_process_request(process_request)
        if isinstance(module, str):
            module = c.module(module)()
        if not  hasattr(module, 'get_endpoints'):
            module.get_endpoints = lambda : dir(module)
        endpoints = endpoints or module.get_endpoints()
        module.endpoints = endpoints
        module.name = module.server_name = name or module.server_name
        module.port = port if port not in ['None', None] else c.free_port()
        module.address = f"{c.ip()}:{module.port}"
        self.network = network
        module.key  = c.get_key(key or module.name, create_if_not_exists=True)

        self.module = module
        self.period = self.timescale_map[self.timescale]
        self.path = self.resolve_path(path)
        self.state = {'sync_time': 0, 
                      'stake': {},
                      'stake_from': {}, 
                      'fn_info': {}}
        c.thread(self.sync_loop)
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

    def set_api(self, 
                max_bytes=1024 * 1024,
                allow_origins = ["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
                loop = None,
                nest_asyncio = True,
                **kwargs
                ):

        if  nest_asyncio:
            c.new_event_loop(nest_asyncio=nest_asyncio)
        self.loop = loop or asyncio.get_event_loop()
        self.app = FastAPI()
        # add the middleware
        self.app.add_middleware(ServerMiddleware, max_bytes=max_bytes)    
        self.app.add_middleware(
                CORSMiddleware,
                allow_origins=allow_origins,
                allow_credentials=allow_credentials,
                allow_methods=allow_methods,
                allow_headers=allow_headers,
            )


        def fn_forward(fn, request: Request):
            try:
                output = self.forward(fn, request)
            except Exception as e:
                output =  c.detailed_error(e)
                print(output)
            return output

        self.app.post("/{fn}")(fn_forward)
        # start the server
        try:
            c.print(f' Served(name={self.module.name}, address={self.module.address}, key=🔑{self.module.key}🔑 ) 🚀 ', color='purple')
            c.register_server(name=self.module.name, 
                              address = self.module.address)
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
        if c.port_used(port):
            c.kill_port(port)
        address = f'0.0.0.0:{port}'
        # RESOLVE THE PORT FROM THE ADDRESS IF IT ALREADY EXISTS
        # # NOTE REMOVE is FROM THE KWARGS REMOTE
        response =  { 'module':module, 'name': name, 'address':address, 'kwargs':kwargs} 
        if remote:
            remote = False
            remote_kwargs = c.locals2kwargs(locals())  # GET THE LOCAL KWARGS FOR SENDING TO THE REMOTE
            for _ in ['extra_kwargs', 'address', 'response']:
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

    @classmethod
    def launch(cls, 
                   module:str = None,  
                   fn: str = 'serve',
                   name:Optional[str]=None, 
                   tag : str = None,
                   args : list = None,
                   kwargs: dict = None,
                   device:str=None, 
                   interpreter:str='python3', 
                   autorestart: bool = True,
                   verbose: bool = False , 
                   force:bool = True,
                   meta_fn: str = 'module_fn',
                   tag_seperator:str = '::',
                   cwd = None,
                   refresh:bool=True ):
        import commune as c

        if hasattr(module, 'module_name'):
            module = module.module_name()
            
        # avoid these references fucking shit up
        args = args if args else []
        kwargs = kwargs if kwargs else {}

        # convert args and kwargs to json strings
        kwargs =  {
            'module': module ,
            'fn': fn,
            'args': args,
            'kwargs': kwargs 
        }

        kwargs_str = json.dumps(kwargs).replace('"', "'")

        name = name or module
        if refresh:
            cls.pm2_kill(name)
        module = c.module()
        # build command to run pm2
        filepath = c.filepath()
        cwd = cwd or module.dirpath()
        command = f"pm2 start {filepath} --name {name} --interpreter {interpreter}"

        if not autorestart:
            command += ' --no-autorestart'
        if force:
            command += ' -f '
        command = command +  f' -- --fn {meta_fn} --kwargs "{kwargs_str}"'
        env = {}
        if device != None:
            if isinstance(device, int):
                env['CUDA_VISIBLE_DEVICES']=str(device)
            if isinstance(device, list):
                env['CUDA_VISIBLE_DEVICES']=','.join(list(map(str, device)))
        if refresh:
            cls.pm2_kill(name)  
        
        cwd = cwd or module.dirpath()
        
        stdout = c.cmd(command, env=env, verbose=verbose, cwd=cwd)
        return {'success':True, 'message':f'Launched {module}', 'command': command, 'stdout':stdout}

    @classmethod
    def remote_fn(cls, 
                    fn: str='train', 
                    module: str = None,
                    args : list = None,
                    kwargs : dict = None, 
                    name : str =None,
                    tag: str = None,
                    refresh : bool =True,
                    mode = 'pm2',
                    tag_seperator : str = '::',
                    cwd = None,
                    **extra_launch_kwargs
                    ):
        import commune as c
        
        kwargs = c.locals2kwargs(kwargs)
        if 'remote' in kwargs:
            kwargs['remote'] = False
        if len(fn.split('.'))>1:
            module = '.'.join(fn.split('.')[:-1])
            fn = fn.split('.')[-1]
            
        kwargs = kwargs if kwargs else {}
        args = args if args else []
        if 'remote' in kwargs:
            kwargs['remote'] = False

        cwd = cwd or cls.dirpath()
        kwargs = kwargs or {}
        args = args or []
        module = cls.resolve_object(module)
        # resolve the name
        if name == None:
            # if the module has a module_path function, use that as the name
            if hasattr(module, 'module_path'):
                name = module.module_name()
            else:
                name = module.__name__.lower() 
        
        c.print(f'[bold cyan]Launching --> <<[/bold cyan][bold yellow]class:{module.__name__}[/bold yellow] [bold white]name[/bold white]:{name} [bold white]fn[/bold white]:{fn} [bold white]mode[/bold white]:{mode}>>', color='green')

        launch_kwargs = dict(
                module=module, 
                fn = fn,
                name=name, 
                tag=tag, 
                args = args,
                kwargs = kwargs,
                refresh=refresh,
                **extra_launch_kwargs
        )
        assert fn != None, 'fn must be specified for pm2 launch'
    
        return  cls.launch(**launch_kwargs)
    
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
        assert fn in self.module.endpoints , f"Function {fn} not in whitelist={self.module.endpoints}"
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
    def kill_prefix(cls, prefix:str, **kwargs):
        servers = c.servers(network='local')
        killed_servers = []
        for s in servers:
            if s.startswith(prefix):
                c.kill(s, **kwargs)
                killed_servers.append(s)
        return {'success':True, 'message':f'Killed servers with prefix {prefix}'}
    


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



Server.run(__name__)