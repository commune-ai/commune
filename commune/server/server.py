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
from .gate import Gate

class Server:
    pm2_dir = c.home_path + '/.pm2'
    network = 'subspace'
    max_user_history_age = 3600 # the lifetime of the user call data
    max_network_age: int = 60 #  (in seconds) the time it takes for. the network to refresh
    helper_functions  = ['info', 'schema', 'functions', 'forward'] # the helper functions
    function_attributes =['endpoints', 'functions', "exposed_functions",'server_functions', 'public_functions', 'pubfns'] # the attributes for the functions
    def __init__(
        self, 
        module: Union[c.Module, object] = None,
        key:str = None, # key for the server (str)
        functions:Optional[List[Union[str, callable]]] = None, # list of endpoints
        name: Optional[str] = None, # the name of the server
        params : dict = None, # the kwargs for the module
        port: Optional[int] = None, # the port the server is running on
        network = 'local', # the network the server is running on
        # -- ADVANCED PARAMETERS --
        gate:str = None, # the .network used for incentives
        free : bool = False, # if the server is free (checks signature)
        serializer: str = 'serializer', # the serializer used for the data
        middleware: Optional[callable] = None, # the middleware for the server
        history_path: Optional[str] = None, # the path to the user data
        run_api = False, # if the server should be run
        ) -> 'Server':
        self.set_network(network)

        # default to false but enabled with c serve
        if run_api:
            module = module or 'module'
            if isinstance(module, str):
                if '::' in str(module):
                    module, tag = name.split('::') 
            name = name or module
            self.module = c.module(module)(**(params or {}))
            self.module.name = name 
            self.module.key = c.get_key(key or name)
            self.serializer = c.module(serializer)()
            self.set_port(port)
            self.set_functions(functions) 
            self.ensure_env()
            self.gate = gate or Gate(module=self.module, history_path=history_path or self.resolve_path(f'history/{self.module.name}'))
            self.loop = asyncio.get_event_loop()
            self.app = FastAPI()
            def forward(fn: str, request: Request):
                return self.forward(fn, request)
            self.gate.set_middleware(self.app)
            self.app.post("/{fn}")(forward)
            c.print(f'Served(name={self.module.name}, url={self.module.url}, key={self.module.key.key_address})', color='purple')
            self.add_server(name=self.module.name, url=self.module.url, key=self.module.key.ss58_address)
            print(f'Network: {self.network}')
            uvicorn.run(self.app, host='0.0.0.0', port=self.module.port, loop='asyncio')

    @classmethod
    def resolve_path(cls, path):
        return  c.storage_path + '/server/' + path
            
    def set_functions(self,  functions:Optional[List[str]] ):
        self.free = any([(hasattr(self.module, k) and self.module.free)  for k in ['free', 'free_mode']])
        if self.free:
            c.print('THE FOUNDING FATHERS WOULD BE PROUD OF YOU SON OF A BITCH', color='red')
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
        function_attributes = [fa for fa in self.function_attributes if hasattr(self.module, fa) and isinstance(getattr(self.module, fa), list)]
        assert len(function_attributes) == 1 , f'{function_attributes} is too many funcitonal attributes, choose one dog'
        functions = getattr(self.module, function_attributes[0])
        self.module.schema = {fn: c.schema(getattr(self.module, fn )) for fn in functions if hasattr(self.module, fn)}
        self.module.free = self.free
        self.module.functions = sorted(list(set(functions + self.helper_functions)))
        self.module.fn2cost = self.module.fn2cost  if hasattr(self.module, 'fn2cost') else {}
        c.print(f'Functions({self.module.functions} fn2cost={self.module.fn2cost} free={self.free})')
        assert isinstance(self.module.fn2cost, dict), f'fn2cost must be a dict, not {type(self.module.fn2cost)}'
        self.module.info = {
            "name": self.module.name,
            "url": self.module.url,
            "key": self.module.key.ss58_address,
            "time": c.time(),
            "functions": self.module.functions,
            "schema": self.module.schema,
        }
        return {'success':True, 'message':f'Set functions to {functions}'}
        
    def set_port(self, port:Optional[int]=None, port_attributes = ['port', 'server_port']):
        name = self.module.name
        for k in port_attributes:
            if hasattr(self.module, k):
                port = getattr(self.module, k)
                break
        port = port or c.free_port()
        while c.port_used(port):
            port = c.free_port()
            c.sleep(1)
            print(f'Waiting for port {port} to be free')
        self.module.port = port
        self.module.url = self.module.url = '0.0.0.0:' + str(self.module.port)
        return {'success':True, 'message':f'Set port to {port}'}
    
    def get_params(self, request: Request) -> dict:
        params = self.loop.run_until_complete(request.json())
        params = self.serializer.deserialize(params) 
        params = json.loads(params) if isinstance(params, str) else params
        assert isinstance(params, dict), f'Params must be a dict, not {type(params)}'
        if len(params) == 2 and 'args' in params and 'kwargs' in params :
            kwargs = dict(params.get('kwargs')) 
            args = list(params.get('args'))
        else:
            args = []
            kwargs = dict(params)
        return {'args': args, 'kwargs': kwargs} 
    

    def get_headers(self, request: Request):
        headers = dict(request.headers)
        headers['time'] = float(headers.get('time', c.time()))
        headers['key'] = headers.get('key', headers.get('url', None))
        return headers

    def forward(self, fn:str, request: Request, catch_exception:bool=True) -> dict:
        if catch_exception:
            try:
                return self.forward(fn, request, catch_exception=False)
            except Exception as e:
                result =  c.detailed_error(e)
                return result
        module = self.module
        params = self.get_params(request)
        headers = self.get_headers(request)
        gate_info = self.gate.forward(fn=fn, params=params, headers=headers)   
        is_admin = bool(c.is_admin(headers['key']))
        is_owner = bool(headers['key'] == self.module.key.ss58_address)    
        if hasattr(module, fn):
            fn_obj = getattr(module, fn)
        else:
            raise Exception(f"{fn} not found in {self.module.name}")
        result = fn_obj(*params['args'], **params['kwargs']) if callable(fn_obj) else fn_obj
        latency = c.time() - float(headers['time'])
        if c.is_generator(result):
            c.print(f"Generator({result})")
            # get a hash for the generator
            output = str(result)
            def generator_wrapper(generator):
                for item in generator:
                    yield item
            result = EventSourceResponse(generator_wrapper(result))       
        else:
            output = result 

        output =  self.serializer.serialize(output)
            
        if not self.free:
            data = {
                    'url': self.module.url, # the url of the server
                    'fn': fn, # the function you are calling
                    'params': params, # the data of the request
                    'output': output, # the response
                    'time': headers["time"], # the time of the request
                    'latency': latency, # the latency of the request
                    'key': headers['key'], # the key of the user
                    'cost': self.module.fn2cost.get(fn, 1), # the cost of the function
                }

            self.gate.save_data(data)
        return result

    def  resolve_path(self, path):
        return  c.storage_path + '/' + self.module_name() + '/' + path
    def processes(cls):
        return self.processes()

    def wait_for_server(cls,
                          name: str ,
                          network: str = 'local',
                          timeout:int = 600,
                          max_age = 1,
                          sleep_interval: int = 1) -> bool :
        
        time_waiting = 0
        # rotating status thing
        c.print(f'waiting for {name} to start...', color='cyan')
        while time_waiting < timeout:
            namespace = cls.namespace(network=network, max_age=max_age)
            if name in namespace:
                try:
                    result = c.call(namespace[name]+'/info')
                    if 'key' in result:
                        c.print(f'{name} is running', color='green')
                    result.pop('schema', None)
                    return result
                except Exception as e:
                    c.print(f'Error getting info for {name} --> {e}', color='red')
            c.sleep(sleep_interval)
                
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
            self.serve_background(module="server", fn='serve', name=name, params=params, cwd=cwd)
            return self.wait_for_server(name)
        return Server(module=module, name=name, functions=functions, params=params, port=port,  key=key, run_api=1)

    def add_endpoint(self, name, fn):
        setattr(self, name, fn)
        self.endpoints.append(name)
        assert hasattr(self, name), f'{name} not added to {self.__class__.__name__}'
        return {'success':True, 'message':f'Added {fn} to {self.__class__.__name__}'}

    @classmethod
    def test(cls, **kwargs):
        from .test import Test
        return Test().test()
    
    def kill(self, name:str, verbose:bool = True, **kwargs):
        process_name = self.resolve_process_name(name)
        try:
            c.cmd(f"pm2 delete {process_name}", verbose=False)
            self.rm_logs(process_name)
            result =  {'message':f'Killed {process_name}', 'success':True}
        except Exception as e:
            result =  {'message':f'Error killing {process_name}', 'success':False, 'error':e}
        c.rm_server(name)
        return result
    
    def kill_all(self, verbose:bool = True, timeout=20):
        servers = self.processes()
        futures = [c.submit(self.kill, kwargs={'name':s, 'update': False}) for s in servers]
        results = c.wait(futures, timeout=timeout)
        return results
    
    def killall(self, **kwargs):
        return self.kill_all(**kwargs)

    pm2_dir_logs = c.home_path + '/.pm2/logs'

    
    def logs_path_map(self, name=None):
        logs_path_map = {}
        for l in c.ls(f'{self.pm2_dir}/logs/'):
            key = '-'.join(l.split('/')[-1].split('-')[:-1]).replace('-',':')
            logs_path_map[key] = logs_path_map.get(key, []) + [l]
        for k in logs_path_map.keys():
            logs_path_map[k] = {l.split('-')[-1].split('.')[0]: l for l in list(logs_path_map[k])}
        if name != None:
            return logs_path_map.get(name, {})
        return logs_path_map

    
    def rm_logs( self, name):
        name = self.resolve_process_name(name)
        for m in ['out', 'error']:
            c.rm(self.get_logs_path(name, m))

    def get_logs_path(self, name:str, mode='out')->str:
        name = self.resolve_process_name(name)
        return f'{self.pm2_dir}/logs/{name.replace("/", "-")}-{mode}.log'.replace(':', '-').replace('_', '-') 

    def logs(self, module:str,  tail: int =100, stream=True, **kwargs):
        module = self.resolve_process_name(module)
        if stream:
            return c.cmd(f"pm2 logs {module}", verbose=True)
        else:
            text = ''
            for m in ['out', 'error']:
                # I know, this is fucked 
                path = self.get_logs_path(module, m)
                try:
                    text +=  c.get_text(path, tail=tail)
                except Exception as e:
                    c.print('ERROR GETTING LOGS -->' , e)
                    continue
            return text
    def get_server_port(self, name:str,  tail:int=100, **kwargs):
        logs = self.logs(name, tail=tail, stream=False, **kwargs)
        port = None
        for i, line in enumerate(logs.split('\n')[::-1]):
            if f'Served(' in line and 'url=' in line:
                return int(line.split('url=')[1].split(' ')[0].split(',')[0].split(':')[-1])
        return None

    def get_server_url(self, name:str,  tail:int=100, **kwargs):
        return f'0.0.0.0:{self.get_server_port(name, tail=tail, **kwargs)}'
    
    def kill_many(self, search=None, verbose:bool = True, timeout=10):
        futures = []
        for name in c.servers(search=search):
            f = c.submit(c.kill, dict(name=name, verbose=verbose), timeout=timeout)
            futures.append(f)
        return c.wait(futures)

    def servers(self, search=None,  **kwargs) -> List[str]:
        return [ p[len(self.process_prefix):] for p in self.processes(search=search, **kwargs) if p.startswith(self.process_prefix)]

    def urls(self, search=None,  **kwargs) -> List[str]:
        return [self.get_server_url(s) for s in self.servers(search=search, **kwargs)]
    def namespace(self, search=None,  max_age:int = None, update:bool = False, **kwargs) -> dict:
        servers = self.servers(search=search, **kwargs)
        urls = self.urls(search=search, **kwargs)
        namespace = {s: u for s, u in zip(servers, urls)}
        return namespace

    def serve_background(self, 
                  fn: str = 'serve',
                   name:Optional[str] = None, 
                   module:str = 'server',  
                   params: dict = None,
                   interpreter:str='python3', 
                   autorestart: bool = True,
                   verbose: bool = False , 
                   run_fn: str = 'run_fn',
                   cwd : str = None,
                   env : Dict[str, str] = None,
                   refresh:bool=True ):
        params['remote'] = False
        env = env or {}
        if '/' in fn:
            module, fn = fn.split('/')
        if self.server_exists(module):
            self.kill(name)

        process_name = self.resolve_process_name(name)
        cmd = f"pm2 start {c.filepath()} --name {process_name} --interpreter {interpreter} -f"
        cmd = cmd  if autorestart else ' --no-autorestart' 
        params_str = json.dumps({'module': module ,  'fn': fn, 'params': params or {}}).replace('"', "'")
        cmd = cmd +  f' -- --fn {run_fn} --params "{params_str}"'
        stdout = c.cmd(cmd, env=env, verbose=verbose, cwd=cwd)
        return {'success':True, 'msg':f'Launched {module}',  'cmd': cmd, 'stdout':stdout}

    def reserve_background(self, name:str):
        assert name in self.processes()
        c.print(f'Restarting {name}', color='cyan')
        c.cmd(f"pm2 restart {name}", verbose=False)
        self.rm_logs(name)  
        return {'success':True, 'message':f'Restarted {name}'}
    
    def processes(self, search=None,  **kwargs) -> List[str]:
        output_string = c.cmd('pm2 status')
        processes = []
        tag = ' default '
        for line in output_string.split('\n'):
            if  tag in line:
                name = line.split(tag)[0].strip()
                name = name.split(' ')[-1]
                processes += [name]
        if search != None:
            search = self.resolve_process_name(search)
            processes = [m for m in processes if search in m]
        processes = sorted(list(set(processes)))
        return processes

    def resolve_process_name(self, name:str, **kwargs) -> str:
        if  name != None and not name.startswith(self.process_prefix):
            name = self.process_prefix + name
        return name
        
    def process_exists(self, name:str, **kwargs) -> bool:
        name = self.resolve_process_name(name)
        return name in self.processes(**kwargs)

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
        self.n = n 
        self.network_path = self.resolve_path(self.network)
        self.modules_path =  f'{self.network_path}/modules'
        self.process_prefix = process_prefix + '/' + network + '/'


        return {'network': self.network, 
                'tempo': self.tempo, 
                'n': self.n,
                'network_path': self.network_path}
    
    def params(self,*args,  **kwargs):
        return { 'network': self.network, 'tempo' : self.tempo,'n': self.n}

    def modules(self, 
                search=None, 
                max_age=None, 
                update=False, 
                features=['name', 'url', 'key'], 
                timeout=8, 
                **kwargs):
        modules = c.get(self.modules_path, max_age=max_age or self.tempo, update=update)
        if modules == None:
            modules = []
            urls = ['0.0.0.0'+':'+str(p) for p in c.used_ports()]
            futures  = [c.submit(c.call, [s + '/info'], timeout=timeout) for s in urls]
            try:
                for f in c.as_completed(futures, timeout=timeout):
                    data = f.result()
                    if all([k in data for k in features]):
                        modules.append({k: data[k] for k in features})
            except Exception as e:
                c.print('Error getting modules', e)
                modules = []
            c.put(self.modules_path, modules)
        if search != None:
            modules = [m for m in modules if search in m['name']]
        return modules

    def namespace(self, search=None,  max_age:int = None, update:bool = False, **kwargs) -> dict:
        processes = self.processes(search=search, **kwargs)
        modules = self.modules(search=search, max_age=max_age, update=update, **kwargs)
        processes = [ p.replace(self.process_prefix, '') for p in processes if p.startswith(self.process_prefix)]
        namespace = {m['name']: m['url'] for m in modules if m['name'] in processes}
        return namespace

    def add_server(self, name:str, url:str, key:str) -> None:
        modules = self.modules()
        modules.append( {'name': name, 'url': url, 'key': key})
        c.put(self.modules_path, modules)
        return {'success': True, 'msg': f'Block {name}.'}
    
    def rm_server(self, name:str, features=['name', 'key', 'url']) -> Dict:
        modules = self.modules()
        modules = [m for m in modules if not any([m[f] == name for f in features])]
        c.put(self.modules_path, modules)

    def resolve_network(self, network:str) -> str:
        return network or self.network
    
    def server_exists(self, name:str, **kwargs) -> bool:
        return bool(name in self.servers(**kwargs))

if __name__ == '__main__':
    Server.run()

