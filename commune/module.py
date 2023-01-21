import inspect
import os
from copy import deepcopy
from typing import Optional, Union, Dict, List, Any, Tuple
from tuwang.utils import *
from munch import Munch
import json
from glob import glob
import ray
import sys

class Module:
    port_range = [50050, 50150] # the range of ports the moddule can be a server for
    default_ip = '0.0.0.0'
    root_path  = root = '/'.join(os.path.dirname(__file__).split('/')[:-1])
    pwd = os.getenv('PWD')
    root_dir = root_path.split('/')[-1]
    def __init__(self, config:dict=None, *args,  **kwargs):
        self.set_config(config=config, kwargs =  kwargs)        

    def getattr(self, k)-> Any:
        return getattr(self,  k)

    @classmethod
    def get_module_path(cls, obj: Any =None,  simple:bool=True) -> str:
        if obj == None:
            obj = cls
        module_path =  inspect.getmodule(obj).__file__
        # convert into simple
        if simple:
            module_path = cls.path2simple(path=module_path)

        return module_path


    @classmethod
    def __file__(cls, simple:bool=False) -> str:
        '''
        
        Gets the absolute file path of the Module.
        
        '''
        
        return cls.get_module_path(simple=simple)
    
    
    @classmethod
    def __local_file__(cls) -> str:
        '''
        removes the PWD with respect to where module.py is located
        '''
        return cls.__file__().replace(cls.pwd+'/', '')
    
    @classmethod
    def __simple_file__(cls) -> str:
        '''
        The simple representation of a module path with respect to the module.py
        home/commune/module.py would assume the module_path would be home/commune/
        
        Using this we convert the full path of the module into a simple path for more
        human readable strings. We do the following
        
        1. Remove the MODULE_PATH and assume the module represents the directory
        2. replace the "/" with "."
        
    
        Examples:
            commune/dataset/text/dataset.py -> dataset.text
            commune/model/transformer/dataset.py -> model.transformer
        
        '''
        return cls.__file__(simple=True)
    
    
    @classmethod
    def module_path(cls):
        if not hasattr(cls, '_module_path'):
            cls._module_path = cls.__simple_file__()
        return cls._module_path

        
    @classmethod
    def module_name(cls):
        '''
        Another name for the module path
        '''
        return cls.module_path()

    @property
    def module_tag(self):
        '''
        The tag of the module for many flavors of the module to avoid name conflicts
        (TODO: Should we call this flavor?)
        
        '''
        if not hasattr(self, '_module_tag'):
            self._module_tag = None
        return self._module_tag
    
    
    @module_tag.setter
    def module_tag(self, value):
        self._module_tag = value
        return self._module_tag

    @classmethod
    def __config_file__(cls) -> str:
        
        __config_file__ =  cls.__file__().replace('.py', '.yaml')
        
        if not os.path.exists(__config_file__):
            cls.save_config(config=cls.minimal_config(), path=__config_file__)
            
        return __config_file__

    @classmethod
    def minimal_config(cls) -> Dict:
        '''
        The miminal config a module can be
        
        '''
        minimal_config = {
            'module': cls.__name__
        }
        return minimal_config
        

    @classmethod
    def get_module_config_path(cls) -> str:
        return cls.get_module_path(simple=False).replace('.py', '.yaml')


    @classmethod
    def default_config(cls, *args, **kwargs):
        '''
        
        Loads a default config
        '''
        cls.load_config( *args, **kwargs)

    @classmethod
    def load_config(cls, path:str=None, to_munch:bool = True) -> Union[Munch, Dict]:
        path = path if path else cls.__config_file__()
        
        if not os.path.exists(path):
            cls.save_config({'module': cls.__name__})
        config = load_yaml(path)
        
        if to_munch:
            config =  cls.dict2munch(config)
        
        return config

    @property
    def class_name(self):
        return self.__class__.__name__

    @classmethod
    def save_config(cls, config:Union[Munch, Dict]= None, path:str=None) -> Munch:
        path = path if path else cls.__config_file__()
        
        if isinstance(config, Munch):
            config = cls.munch2dict(deepcopy(config))
        config = save_yaml(data=config , path=path)

        return config
    
    
    @classmethod
    def dict2munch(cls,x:dict, recursive:bool=True)-> Munch:
        '''
        Turn dictionary into Munch
        '''
        if isinstance(x, dict):
            for k,v in x.items():
                if isinstance(v, dict) and recursive:
                    x[k] = cls.dict2munch(v)
            x = Munch(x)
        return x 
    @classmethod
    def munch2dict(cls,x:Munch, recursive:bool=True)-> dict:
        '''
        Turn munch object  into dictionary
        '''
        if isinstance(x, Munch):
            x = dict(x)
            for k,v in x.items():
                if isinstance(v, Munch) and recursive:
                    x[k] = cls.munch2dict(v)

        return x 

    def set_config(self, config:Optional[Union[str, dict]]=None, kwargs:dict={}):
        '''
        Set the config as well as its local params
        '''
        if config == False:
            self.config =  {}
            return self.config
        
        # ensure to include the inner kwargs if that is provided (Which isnt great practice lol)
        kwargs  = {**kwargs, **kwargs.pop('kwargs', {}) }
        
        # ensure there are no inner_args to avoid ambiguous args 
        inner_args = kwargs.pop('args', [])
        assert len(inner_args) == 0, f'Please specify your keywords for this to act nicely, args: {inner_args}'
    
        if type(config) in [dict, Munch]:
            config = config
        elif type(config) in [str, type(None)]:
            config = self.load_config(path=config)
        
        config.update(kwargs)
        

        self.config = self.dict2munch(config)
        
        return self.config

    @classmethod
    def add_args( cls, config: dict , prefix: str = None  ):
        import argparse
        parser =  argparse.ArgumentParser()
        """ Accept specific arguments from parser
        """
        
        prefix_str = '' if prefix == None else prefix + '.'
        flat_config = deep2flat(config)
        for k,v in flat_config.items():

            if type(v) in [str, int, float, int, bool]:
                parser.add_argument('--' + prefix_str + k, type=type(v),  help=f'''The value for {k}''', default = v)
            elif type(v) in [list]:
                parser.add_argument('--' + prefix_str + k, nargs='+', help=f'''The value for {k}''', default = v)

        args = parser.parse_args()
        flat_config.update(args.__dict__)
        config = flat2deep(flat_config)
        return config

    @staticmethod
    def run_command(command:str, background=False, env:dict = {}):
        '''
        Runs  a command in the shell.
        
        '''
        import subprocess
        import shlex
        if background:
            
            process = subprocess.Popen(shlex.split(command), env={**os.environ, **env})
            process = process.__dict__
        else:
            
            process = subprocess.run(shlex.split(command), 
                                stdout=subprocess.PIPE, 
                                universal_newlines=True,
                                env={**os.environ, **env})
            
        return process


    # @classmethod
    # def launch(cls, module:str, fn:str=None ,kwargs:dict={}, args=[]):
    #     '''
        
    #     Args:
    #         module: path of the module {module_path}.{object_path}
    #         fn: the function of the module
    #         kwargs: the kwargs of the function
    #         args: the args of the function
        
    #     '''
        
        
    #     module_class = cls.import_object(module)
    #     if fn == None:
    #         module_object =  module_class(*args,**kwargs)
    #     else:
    #         fn = getattr(module_class,fn)
    #         module_object =  fn(*args, **kwargs)
    #     return module_object

    @classmethod
    def import_module(cls, import_path:str) -> 'Object':
        from importlib import import_module

        return import_module(import_path)


    @classmethod
    def import_object(cls, key:str)-> 'Object':
        
        '''
        
        Import an object from a string with the format of 
            {module_path}.{object}
        
        Examples:
            import_object("torch.nn"): imports nn from torch
        
        '''
        from importlib import import_module

        module = '.'.join(key.split('.')[:-1])
        object_name = key.split('.')[-1]
        obj =  getattr(import_module(module), object_name)
        return obj

    
    @classmethod
    def module_list(cls)-> List[str]:
        '''
        List of module paths with respect to module.py file
        
        Assumes the module root directory is the directory containing module.py
        '''
        return list(cls.module_tree().keys())
    
    
    @staticmethod
    def port_available(port:int, ip:str ='0.0.0.0'):
        '''
        Check if port is available
        '''
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0

    @classmethod
    def get_available_ports(cls, ports:List[int] = None, ip:str = '0.0.0.0'):
        '''
        Get availabel ports out of port range
        
        Args:
            ports: list of ports
            ip: ip address
        
        '''
        if ports:
            ports = list(range(*cls.port_range))
        
        available_ports = []
        for port in ports: 
            if cls.port_available(port=port, ip=ip):
                available_ports.append(port)
        
        return port
   
    @classmethod
    def resolve_path(cls, path:str, extension:Optional[str]=None):
        '''
        Resolves path for saving items that relate to the module
        
        The path is determined by the module path 
        
        '''
        tmp_dir = cls.tmp_dir()
        if tmp_dir not in path:
            path = os.path.join(tmp_dir, path)
        if extension and extension != path.split('.')[-1]:
            path = path + '.' + extension

        return path
    @classmethod
    def resolve_port(cls, port:int=None, find_available:bool = False):
        
        '''
        
        Resolves the port and finds one that is available
        '''
        port = port if port else cls.get_available_port()
        port_available = cls.port_available(port)
        if port_available:
            if find_available:
                port = cls.get_available_port()
            else:
                raise Exception(f"Port: {port} is already in use, try , {cls.get_available_ports()}")
        return port
   
    
    @classmethod
    def get_available_port(cls, port_range: List[int] = None, ip:str='0.0.0.0' ) -> int:
        port_range = port_range if port_range else cls.port_range
        for port in range(*port_range): 
            if cls.port_available(port=port, ip=ip):
                return port
    
        raise Exception(f'ports {port_range[0]} to {port_range[1]} are occupied, change the port_range to encompase more ports')

    def kwargs2attributes(self, kwargs:dict, ignore_error:bool = False):
        for k,v in kwargs.items():
            if k != 'self': # skip the self
                # we dont want to overwrite existing variables from 
                if not ignore_error: 
                    assert not hasattr(self, k)
                setattr(self, k)

    @staticmethod
    def kill_port(port:int)-> str:
        from psutil import process_iter
        '''
        Kills the port {port} on the localhost
        '''
        for proc in process_iter():
            for conns in proc.connections(kind='inet'):
                if conns.laddr.port == port:
                    proc.send_signal(signal.SIGKILL) # or SIGKILL
        return port

    @classmethod
    def kill_server(cls, module:str):
        '''
        Kill the server by the name
        '''
        port = cls.server_registry()[module]
        return cls.kill_port(port)


    @classmethod
    def get_module_python_paths(cls) -> List[str]:
        
        '''
        Search for all of the modules with yaml files. Format of the file
        
        
        - MODULE_PATH/dataset_module.py
        - MODULE_PATH/dataset_module.yaml
        
        
        '''
        modules = []
        failed_modules = []

        for f in glob(Module.root_path + '/**/*.py', recursive=True):
            if os.path.isdir(f):
                continue
            file_path, file_ext =  os.path.splitext(f)
            if file_ext == '.py':
                if os.path.exists(file_path+'.yaml'):
                    modules.append(f)
        return modules

    @classmethod
    def path2simple(cls, path:str) -> str:
        simple_path = os.path.dirname(path)[len(os.path.join(cls.pwd, cls.root_dir))+1:].replace('/', '.')
        if simple_path == '':
            simple_path = 'module'
        return simple_path
    @classmethod
    def path2localpath(cls, path:str) -> str:
        local_path = path.replace(cls.pwd, cls.root_dir)
        return local_path
    @classmethod
    def path2config(cls, path:str, to_munch=False)-> dict:
        path = cls.path2configpath(path=path)
        return cls.load_config(path, to_munch=to_munch)
    
    @classmethod
    def path2configpath(cls, path:str):
        return path.replace('.py', '.yaml')
    @classmethod
    def simple2configpath(cls,  path:str):
        return cls.path2configpath(cls.simple2path(path))
    @classmethod
    def simple2config(cls, path:str, to_munch=False)-> dict:
        return cls.load_config(cls.simple2configpath(path), to_munch=to_munch)
    @classmethod
    def path2objectpath(cls, path:str) -> str:
        config = cls.path2config(path=path, to_munch=False)
        object_name = config['module']
        if cls.pwd in path:
            # get the path
            path = path[len(cls.pwd)+1:]
        path = path.replace(cls.pwd, '').replace('.py','.').replace('/', '.') + object_name
        return path

    @classmethod
    def path2object(cls, path:str) -> str:
        path = cls.path2objectpath(path)
        return cls.import_object(path)
    @classmethod
    def simple2object(cls, path:str) -> str:
        path = cls.simple2path(path)
        path = cls.path2objectpath(path)
        return cls.import_object(path)


    @classmethod
    def module_tree(cls, mode='path') -> List[str]:
        assert mode in ['path', 'object']
        if mode == 'path':
            return {cls.path2simple(f):f for f in cls.get_module_python_paths()}

        elif mode == 'object':
            return {cls.path2object(f):f for f in cls.get_module_python_paths()}

    @staticmethod
    def module_config_tree() -> List[str]:
        return [f.replace('.py', '.yaml')for f in  Module.get_module_python_paths()]

    @staticmethod
    def is_imported(package:str) :
        return  bool(package in sys.modules)

    @classmethod
    def get_module_path(cls, obj=None,  simple=True):
        if obj == None:
            obj = cls
        module_path =  inspect.getmodule(obj).__file__
        # convert into simple
        if simple:
            module_path = cls.path2simple(path=module_path)

        return module_path


    @classmethod
    def simple2path_map(cls) -> Dict[str, str]:
        return {cls.path2simple(f):f for f in cls.get_module_python_paths()}
    @classmethod
    def simple2path(cls, path) -> Dict[str, str]:
        simple2path_map = cls.simple2path_map()
        return simple2path_map[path]

    @classmethod
    def path2simple_map(cls) -> Dict[str, str]:
        return {v:k for k,v in cls.simple2path_map().items()}
    
    @classmethod
    def simple2config_map(cls) -> Dict[str, str]:
        return {cls.path2simple(f):f for f in cls.get_module_config_paths()}


    module_python_paths = None
    @classmethod
    def get_module_python_paths(cls) -> List[str]:
        '''
        Search for all of the modules with yaml files. Format of the file
        '''
        if isinstance(cls.module_python_paths, list): 
            return cls.module_python_paths
        modules = []
        failed_modules = []

        for f in glob(Module.root_path + '/**/*.py', recursive=True):
            if os.path.isdir(f):
                continue
            file_path, file_ext =  os.path.splitext(f)
            if file_ext == '.py':
                if os.path.exists(file_path+'.yaml'):
                    modules.append(f)
        cls.module_python_paths = modules
        return modules

    @staticmethod
    def get_module_config_paths() -> List[str]:
        return [f.replace('.py', '.yaml')for f in  Module.get_module_python_paths()]


    ##############
    #   RAY LAND
    ##############
    @classmethod
    def get_default_actor_name(cls):
        return cls.get_module_path(simple=True)


    @classmethod
    def ray_stop(cls):
        cls.run_command('ray stop')

    @classmethod
    def ray_import(cls):
        import ray
        return ray
    @classmethod
    def ray_start(cls):
        '''
        Start the ray cluster 
        (TODO: currently supports head)
        '''
        return cls.run_command('ray start --head')

    @classmethod
    def ray_restart(cls, stop:dict={}, start:dict={}):
        '''
        
        Restart  ray cluster
        
        '''
        command_out_dict = {}
        command_out_dict['stop'] = cls.ray_stop(**stop)
        command_out_dict['start'] = cls.ray_start(**start)
        return command_out_dict

    @classmethod
    def ray_status(cls):
        return cls.run_command('ray status')

    @staticmethod
    def ray_initialized():
        
        return ray.is_initialized()


    default_ray_env = {'address':'auto', 
                     'namespace': 'default',
                      'ignore_reinit_error': False,
                      'dashboard_host': '0.0.0.0',
                      '_system_config': {
                                "object_spilling_config": json.dumps(
                                    {"type": "filesystem", "params": {"directory_path": "/tmp/spill"}},
                                )
                            }
                      
                      }
    
    
    @classmethod
    def ray_init(cls,init_kwargs={}):


        init_kwargs =  {**cls.default_ray_env, **init_kwargs}
        if cls.ray_initialized():
            # shutdown if namespace is different
            if cls.ray_namespace() == cls.default_ray_env['namespace']:
                return cls.ray_runtime_context()
            else:
                ray.shutdown()
  
        ray_context = ray.init(**init_kwargs)
        return ray_context


    @staticmethod
    def ray_runtime_context():
        return ray.get_runtime_context()

    @property
    def actor_id(self):
        return self.get_id()
     
    @classmethod
    def module_config_tree(cls):         
        return {m: cls.simple2config(m) for m in cls.module_list()}
    
   
    @classmethod
    def tmp_dir(cls):
        return f'/tmp/{cls.__local_file__().replace(".py", "")}'

    ############ JSON LAND ###############

    @classmethod
    def get_json(cls,path, default=None, **kwargs):
        path = cls.resolve_path(path=path)
        data = load_json(path, **kwargs)
        return data
    load_json = get_json

    @classmethod
    def put_json(cls, path:str, data:Dict, **kwargs) -> str:
        path = cls.resolve_path(path=path)
        put_json(path=path, data=data, **kwargs)
        return path
    save_json = put_json
    

    
    
    @classmethod
    def exists(cls, path:str)-> bool:
        path = cls.resolve_path(path=path)
        return os.path.exists(path)

    @classmethod
    def rm(cls, path=None):
        if path == 'all':
            return [cls.rm(f) for f in cls.glob()]
        path = cls.resolve_path(path)
        return rm_json(path )

    @classmethod
    def glob(cls,  path ='**'):
        path = cls.resolve_path(path, extension=None)
        paths = glob(path, recursive=True)
        
        return list(filter(lambda f:os.path.isfile(f), paths))
    
    @classmethod
    def test_json(cls):
        self = cls()
        self.rm_json('all')
        assert len(self.glob('**')) == 0
        print(self.put_json('bro/fam', data={'bro': 2200}))
        print(self.put_json('bro/dawg', data={'bro': 2200}))
        assert len(self.glob('**')) == 2
        self.rm_json('bro/fam')
        assert len(self.glob('**')) == 1, len(self.glob('**'))
        self.rm_json('bro/dawg')
        assert len(self.glob('**')) == 0
        print(self.put_json('bro/fam/fam', data={'bro': 2200}))
        print(self.put_json('bro/fam/dawg', data={'bro': 2200}))
        assert len(self.glob('bro/**')) == 2

    @classmethod
    def __str__(cls):
        return cls.__name__


    @classmethod
    def connect(cls,name:str=None, port:int=None , ip:str=None,*args, **kwargs ):
        
        from tuwang.server import Client
        server_registry =  Module.server_registry()
        if name:
            assert name in server_registry, f'{name} is not deployed'
            client_kwargs = server_registry[name]
        else:
            client_kwargs = dict(ip=ip, port=port)
        client = Client( *args, **kwargs,**client_kwargs)
        return client
   
    @classmethod
    def server_registry(cls)-> dict:
        '''
        
        The module port is where modules can connect with each othe.
        
        When a module is served "module.serve())"
        it will register itself with the server_registry dictionary.
        
        
        
        '''
        from copy import deepcopy
        
        # get the module port if its saved.
        # if it doesnt exist, then return default ({})
        server_registry = Module.get_json('server_registry', handle_error=True, default={})
        for k in deepcopy(list(server_registry.keys())):
            if not Module.port_available(**server_registry[k]):
                del server_registry[k]
        Module.put_json('server_registry',server_registry)
        return server_registry
  
    @classmethod
    def servers(cls, search:str = None) -> List[str]:
        servers =  list(cls.server_registry().keys())
        
        # filter based on the search
        if search:
            servers = [s for s in servers if search in s]
            
        return servers
    list_servers = servers
    
    @classmethod
    def register_server(cls, name: str, server: 'tuwang.Server')-> dict:
        server_registry = cls.server_registry()
        server_registry[name] = dict(ip=server.ip, port=server.port)
        server_registry = Module.put_json(path='server_registry', data=server_registry) 
        return server_registry
  
    @classmethod
    def set_event_loop(cls, loop=None, new_loop:bool = False):
        if new_loop:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        else:
            loop = loop if loop else asyncio.get_event_loop()
        
        return loop
    @classmethod
    def resolve_module_id(cls, name:str=None, tag:str=None):
        module_id = name if name else cls.module_name()
        if tag:
            module_id = f'{module_id}::{tag}'
        return module_id
    
    def server_exists(cls, name:str) -> bool:
        server_registry = cls.server_registry()
        return bool(name in server_registry)
        
    @classmethod
    def serve(cls, port:int=None , ip:str=None, name:str=None, tag:str=None, replace:bool = True, *args, **kwargs ):
        
        from tuwang.server import Server
        
        self = cls(*args, **kwargs)
    
        module_id = self.resolve_module_id(name=name, tag=tag if tag else self.module_tag)
           
           
        '''check if the server exists'''
        if self.server_exists(module_id): 
            existing_server_port = self.server_registry()[module_id]
            if replace:
                self.kill_port(existing_server_port)
            else: 
                raise Exception(f'The server {module_id} already exists on port {existing_server_port}')
    
        self.server = Server(ip=ip, port=port, module = self)

        self.module_id = module_id
        cls.register_server(name=module_id, server=self.server)
        self.server.serve()
         
    # def __call__(self, data:dict, metadata:dict={}):

    #     try:
    #         if 'fn' in data:
    #             fn_kwargs = data.get('kwargs', {})
    #             fn_args = data.get('args', [])
            
    #             data = {'result': getattr(self, data['fn'])(*fn_args,**fn_kwargs)}
    #         else:
    #             # print(f'[green]{data}')
    #             data = self.forward(**data)
    #     except RuntimeError as ex:
    #         if "There is no current event loop in thread" in str(ex):
    #             self.loop = asyncio.new_event_loop()
    #             asyncio.set_event_loop(self.loop)
    #             return self.__call__(data=data, metadata=metadata)
        
    #     self.server.stats['call_count'] += 1
    #     torch.cuda.empty_cache()

    #     return {'data': data, 'metadata': metadata}
    
    @classmethod
    def functions(cls, obj:Any=None, exclude_module_functions:bool = True, **kwargs) -> List[str]:
        '''
        List of functions
        '''
        obj = obj if obj else cls
        
        functions = get_functions(obj=obj, **kwargs)
        if exclude_module_functions and (not cls.is_module(obj)) :
            module_functions = Module.functions()
            
            functions = [f for f in functions if f not in module_functions]
            
        return functions
    
    @classmethod
    def is_module(cls, obj=None):
        obj = obj if obj else cls
        return (Module != obj and type(obj) != Module)
        
    @classmethod
    def function_schema_map(cls):
        function_schema_map = {}
        for f in cls.functions():
            if f.startswith('__') and f.endswith('__'):
                continue
            if callable(getattr(cls, f )):
                function_schema_map[f] = {k:str(v) for k,v in getattr(cls, f ).__annotations__.items()}
                
        return function_schema_map
    
    @classmethod
    def function_schema(cls, f:str)->dict:
        '''
        Get function schema of function in cls
        '''
        fn = getattr(cls, f)
        fn_schema = {k:str(v) for k,v in fn.__annotations__.items()}
        return fn_schema

    @staticmethod
    def get_annotations(fn:callable) -> dict:
        return fn.__annotations__


    @classmethod
    def start(cls, module:str, tag:str=None, mode:str = 'pm2'):
        if mode == 'pm2':
            return cls.pm2_launch(module=module, tag=tag)
        else:
            raise NotImplemented(mode)


    @classmethod
    def launch(cls, *args, mode:str='pm2', **kwargs ):
        return getattr(cls, f'{mode}_launch')(*args, **kwargs)
       
    @classmethod
    def pm2_launch(cls, 
                   module:str = None,  
                   fn: str = 'serve',
                   name:Optional[str]=None, 
                   tag:str=None, 
                   args : list = None,
                   kwargs: dict = None,
                   device:str='0', 
                   interpreter:str='python', 
                   refresh:bool=True, ):
        
        # avoid these references fucking shit up
        args = args if args else []
        kwargs = kwargs if kwargs else {}
        
        module = module if module else cls.module_path()
        module_path = cls.simple2path(module)
        assert module in cls.module_tree(), f'{module} is not in the module tree, your options are {cls.module_list()}'
        pm2_name = cls.resolve_module_id(name=name, tag=tag) 

        command = f" pm2 start {module_path} --name {pm2_name} --interpreter python"
        kwargs_str = json.dumps(kwargs).replace('"', "'")
        args_str = json.dumps(args).replace('"', "'")

        command = command + ' -- ' + f'--fn {fn} --kwargs "{kwargs_str}" --args "{args_str}"'
    
        print(command)
        env = dict(CUDA_VISIBLE_DEVICES=device)
        if refresh:
            cls.pm2_kill(pm2_name)    
        return cls.run_command(command, env=env)

    @classmethod
    def pm2_kill(cls, name:str):
        return cls.run_command(f"pm2 delete {name}")
    @classmethod
    def pm2_restart(cls, name:str):
        return cls.run_command(f"pm2 restart {name}")

    @classmethod
    def pm2_status(cls):
        return cls.run_command(f"pm2 status")


    @classmethod
    def pm2_logs(cls, module:str,):
        return cls.run_command(f"pm2 logs {module}")


    @classmethod
    def argparse(cls):
        import argparse
        parser = argparse.ArgumentParser(description='Gradio API and Functions')
        parser.add_argument('-fn', '--fn', dest='function', help='run a function from the module', type=str, default="__init__")
        parser.add_argument('-kwargs', '--kwargs', dest='kwargs', help='key word arguments to the function', type=str, default="{}")  
        parser.add_argument('-args', '--args', dest='args', help='arguments to the function', type=str, default="[]")  
        args = parser.parse_args()
        args.kwargs = json.loads(args.kwargs.replace("'",'"'))
        args.args = json.loads(args.args.replace("'",'"'))
        return args

    @classmethod
    def run(cls): 
        args = cls.argparse()
        self = cls()
        return getattr(self, args.function)(*args.args, **args.kwargs)     
       
    @classmethod
    def sandbox(cls, **kwargs):
        print(kwargs)
     
        import tuwang
        commune = tuwang
        batch_count = 100
        print(cls.server_registry())
        
        # t = tuwang.timer()
        dataset =  cls.connect('dataset.bittensor')
        model =  cls.connect('model.transformer::gptj')
        t = tuwang.timer()

        sample = dataset(fn='sample', kwargs=dict(batch_size=32, sequence_length=256))
        sample['output_hidden_states'] =  False
        sample['output_logits'] =  False
        sample['topk'] =  10
        sample['output_length'] = 10
        # sample['topk'] = True
        print(model(fn='forward', kwargs=sample)['hidden_states'].shape)
        print(t.seconds)
        
    

# def module(obj_class:Any)-> Module:
#     class WrapperModule(Module, obj_class):
#         def __init__(self,config=None, *args, **kwargs):
#             Module.__init__(self, config=config, *args, **kwargs)
#             obj_class.__init__(self, *args, **kwargs)
            
#     return WrapperModule 


module = Module

if __name__ == "__main__":
    print(module.run())
