import inspect
import os
from copy import deepcopy
from typing import Optional, Union, Dict, List, Any, Tuple, Callable
from munch import Munch
import json
from glob import glob
import sys
import argparse


class Module:
    
    # port range for servers
    port_range = [50050, 50150] 
    
    # default ip
    default_ip = '0.0.0.0'
    
    # the root path of the module (assumes the module.py is in ./module/module.py)
    root_path  = root = os.path.dirname(os.path.dirname(__file__))
    
    # get the current working directory  (doesnt have /)
    pwd = os.getenv('PWD')
    
    # get the root directory (default commune)
    # Please note that this assumes that {root_dir}/module.py is where your module root is
    root_dir = root_path.split('/')[-1]
    
    def __init__(self, config:Dict=None, save_config_if_not_exists:bool=False, *args,  **kwargs):
        # set the config of the module (avoid it by setting config=False)
        self.set_config(config=config, save_if_not_exists=save_config_if_not_exists)        

    def getattr(self, k:str)-> Any:
        return getattr(self,  k)
    
    
    
    
    
    @classmethod
    def __module_file__(cls) -> str:
        # get the file of the module
        return inspect.getfile(cls)

    @classmethod
    def __module_dir__(cls) -> str :
        # get the directory of the module
        return os.path.dirname(cls.__module_file__())
    
    @classmethod
    def get_module_path(cls, obj=None,  simple:bool=False) -> str:
        
        # odd case where the module is a module in streamlit
        if obj == None:
            obj = cls
        module_path =  inspect.getfile(obj)
        # convert into simple
        if simple:
            return cls.path2simple(path=module_path)
        return module_path

    
    @classmethod
    def __local_file__(cls) -> str:
        '''
        removes the PWD with respect to where module.py is located
        '''
        return cls.get_module_path(simple=False).replace(cls.pwd+'/', '')
    
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
        file =  cls.get_module_path(simple=True)

        return file
    
    
    @classmethod
    def module_path(cls) -> str:
        # get the module path
        if not hasattr(cls, '_module_path'):
            cls._module_path = cls.get_module_path(simple=True)
        return cls._module_path

        
    @classmethod
    def module_name(cls) -> str:
        '''
        Another name for the module path
        '''
        if hasattr(cls, 'module_name_class'):
            return cls.module_name_class
        
        return cls.__name__

    
    @property
    def module_tag(self) -> str:
        '''
        The tag of the module for many flavors of the module to avoid name conflicts
        (TODO: Should we call this flavor?)
        
        '''
        if not hasattr(self, '_module_tag'):
            self.__dict__['_module_tag'] = None
        return self._module_tag
    
    
    @module_tag.setter
    def module_tag(self, value):
        # set the module tag
        self._module_tag = value
        return self._module_tag

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
    def __config_file__(cls) -> str:
        
        __config_file__ =  cls.__module_file__().replace('.py', '.yaml')
        
        # if the config file does not exist, then create one where the python path is

        return __config_file__


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
    def dict2munch(cls, x:Dict) -> Munch:
        '''
        Converts a dict to a munch
        '''
        from commune.utils.dict import dict2munch
        return dict2munch(x)
    
    @classmethod
    def munch2dict(cls, x:'Munch') -> Dict:
        '''
        Converts a munch to a dict
        '''
        from commune.utils.dict import munch2dict
        return munch2dict(x)
    
    @classmethod
    def load_yaml(cls, path:str=None) -> Dict:
        '''
        Loads a yaml file
        '''
        from commune.utils.dict import load_yaml
        return load_yaml(path)

    @classmethod
    def save_yaml(cls, path:str,  data:Union[Dict, Munch]) -> Dict:
        '''
        Loads a yaml file
        '''
        from commune.utils.dict import save_yaml
        if isinstance(config, Munch):
            config = munch2dict(deepcopy(config))
        config = save_yaml(data=config , path=path)
        return save_yaml(path)

    @classmethod
    def load_config(cls, path:str=None, to_munch:bool = True, save_if_not_exists:bool = False) -> Union[Munch, Dict]:
        '''
        Args:
            path: The path to the config file
            to_munch: If true, then convert the config to a munch
        '''
        
        path = path if path else cls.__config_file__()
            
        if save_if_not_exists:    
            if not os.path.exists(__config_file__):
                cls.save_config(config=cls.minimal_config(), path=__config_file__)
                
        config = cls.load_yaml(path)
        
        if to_munch:
            config =  cls.dict2munch(config)
            
        
        return config

    @classmethod
    def save_config(cls, config:Union[Munch, Dict]= None, path:str=None) -> Munch:

        '''
        Saves the config to a yaml file
        '''
        from commune.utils.dict import save_yaml,munch2dict
        
        
        path = path if path else cls.__config_file__()
        
        if isinstance(config, Munch):
            config = munch2dict(deepcopy(config))
        config = save_yaml(data=config , path=path)

        return config
    


    def set_config(self, config:Optional[Union[str, dict]]=None, kwargs:dict={}, save_if_not_exists:bool = False):
        '''
        Set the config as well as its local params
        '''
        
        from commune.utils.dict import munch2dict, dict2munch
        
        if config == False:
            config =  self.minimal_config()
        
        # ensure to include the inner kwargs if that is provided (Which isnt great practice lol)
        kwargs  = {**kwargs, **kwargs.pop('kwargs', {}) }
        
        # ensure there are no inner_args to avoid ambiguous args 
        inner_args = kwargs.pop('args', [])
        assert len(inner_args) == 0, f'Please specify your keywords for this to act nicely, args: {inner_args}'
    
        if type(config) in [dict]:
            config = config
        elif type(config) in[Munch]:
            config = munch2dict(config)
        elif type(config) in [str, type(None)]:
            config = self.load_config(path=config, save_if_not_exists=False)
        
        config.update(kwargs)
        

        self.__dict__['config'] = dict2munch(config)
        
        return self.config

    @classmethod
    def add_args( cls, config: dict , prefix: str = None , parser: argparse.ArgumentParser = None ):

        '''
        Adds arguments to the parser based on the config. This invol
        '''
        from commune.utils.dict import flat2deep, deep2flat
        
        
        parser = parser if parser else argparse.ArgumentParser()
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

    @classmethod
    def run_command(cls, 
                    command:str,
                    verbose:bool = True, 
                    env:Dict[str, str] = {}, 
                    output_text:bool = True,
                    **kwargs) -> 'subprocess.Popen':
        '''
        Runs  a command in the shell.
        
        '''
        import subprocess
        import shlex
        import time
            
        process = subprocess.Popen(shlex.split(command),
                                    stdout=subprocess.PIPE, 
                                #    universal_newlines=True,
                                    env={**os.environ, **env}, **kwargs)
        new_line = b''
        stdout_text = ''
        line_count_idx = 0
        line_delay_period = 0
        last_time_line_printed = time.time()
 
        try:
            for c in iter(lambda: process.stdout.read(1), b""):
                

                if c == b'\n':
                    line_count_idx += 1
                    stdout_text += (new_line+c).decode()
                    if verbose:
                        log_color = verbose if isinstance(verbose, str) else 'green'
                        cls.log(new_line.decode())
                    new_line = b''
                    continue
                
                new_line += c

        except KeyboardInterrupt:
            import signal
            process.send_signal(signal.SIGINT)
            process.wait()

            
        process.stdout = stdout_text

        if output_text:
            return process.stdout

            
        return process

    shell = cmd = run_command
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
    
    get_object = import_object

    
    @classmethod
    def module_list(cls)-> List[str]:
        '''
        List of module paths with respect to module.py file
        
        Assumes the module root directory is the directory containing module.py
        '''
        return list(cls.module_tree().keys())
    
    
    @staticmethod
    def port_used(port:int, ip:str ='0.0.0.0'):
        '''
        Check if port is available
        '''
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0

    @classmethod
    def get_used_ports(cls, ports:List[int] = None, ip:str = '0.0.0.0'):
        '''
        Get availabel ports out of port range
        
        Args:
            ports: list of ports
            ip: ip address
        
        '''
        if ports == None:
            ports = list(range(*cls.port_range))
        
        used_ports = []
        for port in ports: 
            if cls.port_used(port=port, ip=ip):
                used_ports.append(port)
        
        return used_ports
   
    @classmethod
    def resolve_path(cls, path:str, extension:Optional[str]= None):
        '''
        Resolves path for saving items that relate to the module
        
        The path is determined by the module path 
        
        '''
        tmp_dir = cls.tmp_dir()
        if tmp_dir not in path:
            path = os.path.join(tmp_dir, path)
        if not os.path.isdir(path):
            if extension and extension != path.split('.')[-1]:
                path = path + '.' + extension

        return path
    @classmethod
    def resolve_port(cls, port:int=None, find_available:bool = False):
        
        '''
        
        Resolves the port and finds one that is available
        '''
        port = port if port else cls.get_available_port()
        port_used = cls.port_used(port)
        if port_used:
            if find_available:
                port = cls.get_available_port()
            else:
                raise Exception(f"Port: {port} is already in use, try , {cls.get_available_ports()}")
        return port
    @classmethod
    def get_available_port(cls, port_range: List[int] = None, ip:str='0.0.0.0' ) -> int:
        port_range = port_range if port_range else cls.port_range
        for port in range(*port_range): 
            if cls.port_used(port=port, ip=ip):
                return port
    
        raise Exception(f'ports {port_range[0]} to {port_range[1]} are occupied, change the port_range to encompase more ports')

    def kwargs2attributes(self, kwargs:dict, ignore_error:bool = False):
        for k,v in kwargs.items():
            if k != 'self': # skip the self
                # we dont want to overwrite existing variables from 
                if not ignore_error: 
                    assert not hasattr(self, k)
                setattr(self, k)

    @classmethod
    def kill_port(cls, port:int, mode='python')-> str:
        
        if mode == 'python':
            import signal
            from psutil import process_iter
            '''
            Kills the port {port} on the localhost
            '''
            for proc in process_iter():
                for conns in proc.connections(kind='inet'):
                    if conns.laddr.port == port:
                        proc.send_signal(signal.SIGKILL) # or SIGKILL
                        print('KILLED')
            return port
        elif mode == 'bash':
            return cls.run_command('kill -9 $(lsof -t -i:{port})')

    @classmethod
    def kill_server(cls, module:str, mode:str = 'pm2'):
        '''
        Kill the server by the name
        '''
        server_info = cls.get_server_info(module)
        import streamlit as st
        if 'external_ip' in server_info:
            assert server_info.get('external_ip') == cls.external_ip()
        if isinstance(module, int) or mode == 'local':
            return cls.kill_port(server_info['port'])
        if mode == 'pm2':
            return cls.pm2_kill(module)
        else:
            raise NotImplementedError(f"Mode: {mode} is not implemented")

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

        simple_path =  path.split(deepcopy(cls.root_dir))[-1]
        simple_path = os.path.dirname(simple_path)
        simple_path = simple_path.replace('.py', '')
        simple_path = simple_path.replace('/', '.')[1:]

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
    def import_path(cls):
        return cls.path2objectpath(cls.__module_file__())
    
    
    @classmethod
    def path2objectpath(cls, path:str) -> str:
        
        import streamlit as st
        
        module_file_basename = os.path.basename(path).split('.')[0]
        if module_file_basename[0].isupper():
            object_name = module_file_basename
        else:
            config = cls.path2config(path=path, to_munch=False)
            object_name = config.get('module', config.get('name')) 
        path = path.replace(cls.pwd+'/', '').replace('.py','.').replace('/', '.') 
        if path[-1] != '.':
            path = path + '.'
        path = path + object_name
        
        st.write(path, cls.pwd)
        return path

    @classmethod
    def path2object(cls, path:str) -> str:
        path = cls.path2objectpath(path)
        return cls.import_object(path)
    @classmethod
    def simple2object(cls, path:str) -> str:
        path = cls.simple2path(path)
        object_path = cls.path2objectpath(path)
        return cls.import_object(object_path)

    @classmethod
    def get_module(cls, path:str, verbose:bool = True) -> str:
        
        try:
            
            path = cls.simple2path(path)
            path = cls.path2objectpath(path)
            
        except KeyError as e:
            cls.print(f'{e}', verbose=verbose)
            
        
            
        return cls.import_object(path)

    @classmethod
    def module_tree(cls, mode='path') -> List[str]:
        assert mode in ['path', 'object']
        if mode == 'path':
            return {cls.path2simple(f):f for f in cls.get_module_python_paths()}

        elif mode == 'object':
            return {cls.path2object(f):f for f in cls.get_module_python_paths()}
    @classmethod
    def list_modules(cls):
        return cls.module_tree()
    @staticmethod
    def module_config_tree() -> List[str]:
        return [f.replace('.py', '.yaml')for f in  Module.get_module_python_paths()]

    @staticmethod
    def is_imported(package:str) :
        return  bool(package in sys.modules)

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


    @classmethod
    def is_parent(cls, parent=None):
        parent = Module if parrent == None else parent
        return bool(parent in cls.get_parents(child))

    @classmethod
    def run_python(cls, path:str, interpreter:str='python'):
        cls.run_command(f'{interpreter} {path}')

    @classmethod
    def timer(cls, *args, **kwargs):
        from commune.utils.time import Timer
        return Timer(*args, **kwargs)
    
    
    @classmethod
    def get_parents(cls, obj=None):
        
        if obj == None:
            obj = cls

        return list(obj.__mro__[1:-1])

    @classmethod
    def module_config_tree(cls):         
        return {m: cls.simple2config(m) for m in cls.module_list()}
    
   
    @classmethod
    def tmp_dir(cls):
        return f'/tmp/{cls.__local_file__().replace(".py", "")}'

    ############ JSON LAND ###############



        
    @classmethod
    def get_json(cls,path:str, default=None, resolve_path: bool = True, **kwargs):
        from commune.utils.dict import load_json
        path = cls.resolve_path(path=path, extension='json') if resolve_path else path
        data = load_json(path, **kwargs)
        assert isinstance(data, dict)
        if 'data' in data and 'timestamp' in data:
            data = data['data']
        
        return data

    load_json = get_json

    @classmethod
    def put_json(cls, path:str, data:Dict, resolve_path:bool = True, **kwargs) -> str:
        
        from commune.utils.dict import put_json
        path = cls.resolve_path(path=path, extension='json') if resolve_path else path
        
        put_json(path=path, data=data, **kwargs)
        return path
    
    save_json = put_json
    
    @classmethod
    def exists(cls, path:str, resolve_path:bool = True, extension = 'json')-> bool:
        path = cls.resolve_path(path=path, extension=extension) if resolve_path else path
        return os.path.exists(path)

    @classmethod
    def rm_json(cls, path=None, resolve_path:bool = True):
        from commune.utils.dict import rm_json

        if path in ['all', '**']:
            return [cls.rm_json(f) for f in cls.glob(files_only=False)]
        
        if resolve_path:
            path = cls.resolve_path(path=path, extension='json')

        return rm_json(path )

    @classmethod
    def glob(cls,  path ='**', resolve_path:bool = True, files_only:bool = True):
        
        path = cls.resolve_path(path, extension=None) if resolve_path else path
        
        # if os.path.isdir(path):
        #     path = os.path.join(path, '**')
            
        paths = glob(path, recursive=True)
        
        if files_only:
            paths =  list(filter(lambda f:os.path.isfile(f), paths))
        return paths
            
    @classmethod
    def __str__(cls):
        return cls.__name__

    @classmethod
    def get_server_info(cls,name:str) -> Dict:
        return cls.server_registry().get(name, {})
    @classmethod
    def connect(cls,name:str=None, port:int=None , ip:str=None,virtual:bool = True, **kwargs ):
        
        
        

        server_registry =  Module.server_registry()
        if isinstance(name, str) and len(name.split(':')) == 2:
            port = int(name.split(':')[1])
            ip = name.split(':')[0]
            
        if ip == None and port == None:
            client_kwargs = server_registry[name]
        else:
            client_kwargs = dict(ip=ip, port=port)
        Client = cls.import_object('commune.server.client.Client')
        client_module = Client( **kwargs,**client_kwargs)
        ip = client_kwargs['ip']
        port = client_kwargs['port']
        cls.print(f'Connecting to {name} on {ip}:{port}', 'yellow')

        if virtual:
            return client_module.virtual()
        
        return client_module
   
    @classmethod
    def nest_asyncio(cls):
        import nest_asyncio
        nest_asyncio.apply()
        
    @classmethod
    def peer_registry(cls) -> Dict:
        peer_registry = {}
        for peer in cls.pm2_list():
            peer_stub = cls.connect(peer)
            peer_registry[peer] = peer_stub.server_stats
        return peer_registry
    @classmethod
    def server_registry(cls)-> dict:
        '''
        
        The module port is where modules can connect with each othe.
        
        When a module is served "module.serve())"
        it will register itself with the server_registry dictionary.
        
        
        
        '''
        # from copy import deepcopy
        
        # get the module port if its saved.
        # if it doesnt exist, then return default ({})
        server_registry = Module.get_json('server_registry', handle_error=True, default={})
        
        for k in deepcopy(list(server_registry.keys())):
            if not Module.port_used(**server_registry[k]):
                del server_registry[k]
        Module.put_json('server_registry',server_registry)
        return server_registry
    
    
    
    def server_info(self): 
        self.server_registry(self.module_id)
  
    @classmethod
    def servers(cls, search:str = None, ) -> List[str]:
        servers =  list(cls.server_registry().keys())
        
        # filter based on the search
        if search:
            servers = [s for s in servers if search in s]
            
        return servers
    list_servers = servers
    
    
    
    @classmethod
    def register_server(cls, name: str, server: 'commune.Server')-> dict:
        server_registry = cls.server_registry()
        server_registry[name] = dict(ip=server.ip, port=server.port)
        Module.put_json(path='server_registry', data=server_registry) 
        
        return server_registry
  
    @classmethod
    def is_module(cls, obj=None) -> bool:
        return hasattr(cls, 'module_name')

    @classmethod
    def new_event_loop(cls) -> 'asyncio.AbstractEventLoop':
        import asyncio
        
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
  
    @classmethod
    def set_event_loop(cls, loop=None, new_loop:bool = False) -> 'asyncio.AbstractEventLoop':
        import asyncio
        try:
            if new_loop:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            else:
                loop = loop if loop else asyncio.get_event_loop()
        except RuntimeError as e:
            cls.new_event_loop()
        return loop

    @classmethod
    def get_event_loop(cls) -> 'asyncio.AbstractEventLoop':
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = cls.new_event_loop()
        return loop

    @classmethod
    def get_module_id(cls, name:str=None, tag:str=None) -> str:
        module_id = name if name else cls.module_name()
            
        if tag:
            module_id = f'{module_id}::{tag}'
        return module_id
    
    @classmethod
    def server_exists(cls, name:str) -> bool:
        server_registry = cls.servers()
        return bool(name in cls.servers())
    def server_running(self):
        return hasattr(self, 'server_stats')
    def serve(self, name=None , *args, **kwargs):
        if not self.server_running():
            module_serve_output = self.serve_module( *args, module = self, name=name, **kwargs)
        
    def stop_server(self):
        self.server.stop()
        del self.server
        del self.server_stats
        
        
    @classmethod
    def get_streamlit(cls):
        import streamlit
        return streamlit 
    
    
    @classmethod
    def serve_module(cls, 
              module:Any = None ,
              port:int=None ,
              ip:str=None, 
              name:str=None, 
              tag:str=None, 
              replace:bool = True, 
              wait_for_termination:bool = True,
              *args, 
              **kwargs ):
        '''
        Servers the module on a specified port
        '''
        if module == None:
            self = cls(*args, **kwargs)
        else:
            self = module
            
    
    
        # resolve the module id
        
        # if the module is a class, then use the module_tag 
        # Make sure you have the module tag set
        
        name = name if name != None else self.module_name()
        if hasattr(self, 'module_id'):
            module_id = self.module_id
        else:
            module_id = self.get_module_id(name=name, tag=tag)
           
        '''check if the server exists'''
        if self.server_exists(module_id): 
            if replace:
                self.kill_server(module_id)
            else: 
                raise Exception(f'The server {module_id} already exists on port {existing_server_port}')
    
        
        self.module_id = module_id

    
        Server = cls.import_object('commune.server.server.Server')
        server = Server(ip=ip, port=port, module = self )
        
        self.server_stats = dict(ip=server.ip, port=server.port, external_ip = server.external_ip)
        
        
        cls.register_server(name=module_id, server=server)
    
        
        server.serve(wait_for_termination=wait_for_termination)
        
        
    def functions(self, include_module=False):
        if isinstance(self, Module):
            include_module = True
        functions = self.get_functions(obj=self,include_module=include_module)  
        return functions

        
    @classmethod
    def get_functions(cls, obj:Any=None, include_module:bool = False,) -> List[str]:
        '''
        List of functions
        '''
        from commune.utils.function import get_functions
        
        obj = obj if obj != None else cls
        

        functions = get_functions(obj=obj)
        
        if not include_module:
            module_functions = get_functions(obj=Module)
            new_functions = []
            for f in functions:
                if f == '__init__':
                    new_functions.append(f)
                if f not in module_functions:
                    new_functions.append(f)
            functions = new_functions
        
        cls.print(functions, 'blue')
        return functions

    @classmethod
    def get_function_signature_map(cls, obj=None, include_module:bool = False):
        from commune.utils.function import get_function_signature
        function_signature_map = {}
        obj = obj if obj else cls
        for f in cls.get_functions(obj = obj, include_module=include_module):
            if f.startswith('__') and f.endswith('__'):
                if f in ['__init__']:
                    pass
                else:
                    continue
            if not hasattr(cls, f):
                continue
            if callable(getattr(cls, f )):
                function_signature_map[f] = {k:str(v) for k,v in get_function_signature(getattr(cls, f )).items()}        
        
    
        return function_signature_map
    @property
    def function_signature_map(self, include_module:bool = False):
        return self.get_function_signature_map(obj=self, include_module=include_module)
    
    @property
    def function_default_map(self):
        return self.get_function_default_map(obj=self, include_module=False)
        
    @classmethod
    def get_function_default_map(cls, obj:Any= None, include_module:bool=True) -> Dict[str, Dict[str, Any]]:
        obj = obj if obj else cls
        default_value_map = {}
        function_signature = cls.get_function_signature_map(obj=obj,include_module=include_module)
        for fn_name, fn in function_signature.items():
            default_value_map[fn_name] = {}

            for var_name, var in fn.items():
                if len(var.split('=')) == 1:
                    var_type = var
                    default_value_map[fn_name][var_name] = 'NA'

                elif len(var.split('=')) == 2:
                    var_value = var.split('=')[-1].strip()                    
                    default_value_map[fn_name][var_name] = eval(var_value)
        
        return default_value_map   
    
    @property
    def function_info_map(self):
        return self.get_function_info_map(obj=self, include_module=False)
    
    @classmethod
    def get_function_info_map(cls, obj:Any= None, include_module:bool=True) -> Dict[str, Dict[str, Any]]:
        obj = obj if obj else cls
        function_schema_map = cls.get_function_schema_map(obj=obj,include_module=include_module)
        function_default_map = cls.get_function_default_map(obj=obj,include_module=include_module)
        function_info_map = {}
        for fn in function_schema_map:
            function_info_map[fn] = {
                'default':function_default_map.get(fn, 'NA'),
                **function_schema_map.get(fn, {}),
            }
            
            if 'self' in function_info_map[fn]['schema']:
                function_info_map[fn]['method_type'] = 'self'
                function_info_map[fn]['schema'].pop('self')
            elif 'cls' in function_info_map[fn]['schema']:
                function_info_map[fn]['method_type'] = 'cls'
                function_info_map[fn]['schema'].pop('cls')
            else:
                function_info_map[fn]['method_type'] = 'static'
                

        return function_info_map    
    
    @classmethod
    def get_peer_info(cls, peer: Union[str, 'Module']) -> Dict[str, Any]:
        if isinstance(peer, str):
            peer = cls.connect(peer)
        
        function_schema_map = peer.function_schema_map()
        server_stats = peer.server_stats
        info  = dict(
            module_id = peer.module_id,
            server_stats = peer.server_stats,
            function_schema = function_schema_map,
            intro =function_schema_map.get('__init__', 'No Intro Available'),
            examples =function_schema_map.get('examples', 'No Examples Available'),
            public_ip =  server_stats if not isinstance(server_stats, dict) else server_stats['external_ip'] + ':' + str(server_stats['port']) ,

        )
        
        return info
    
    def peer_info(self) -> Dict[str, Any]:
        function_schema_map = self.function_schema_map()
        info  = dict(
            module_id = self.module_id,
            server_stats = self.server_stats,
            function_schema = function_schema_map,
            intro =function_schema_map.get('__init__', 'No Intro Available'),
            examples =function_schema_map.get('examples', 'No Examples Available'),


        )
        return info


    @classmethod
    def get_function_schema_map(cls, obj = None, include_hidden:bool = False, include_module:bool = False):
        
        obj = obj if obj else cls
        
        function_schema_map = {}
        print(cls.get_functions(obj))
        for fn in cls.get_functions(obj, include_module=include_module):
            # if not include_hidden:
            #     if (fn.startswith('__') and fn.endswith('__')) or fn.startswith('_'):
            #         if fn != '__init__':
            #             continue
            
            if callable(getattr(obj, fn )):
                function_schema_map[fn] = {}
                fn_schema = {}
                for fn_k, fn_v in getattr(obj, fn ).__annotations__.items():
                    
                    fn_v = str(fn_v)  
                    if fn_v == inspect._empty:
                        fn_schema[fn_k]= 'Any'
                    elif fn_v.startswith('<class'):
                        fn_schema[fn_k] = fn_v.split("'")[1]
                    else:
                        fn_schema[fn_k] = fn_v
                                        
                function_schema_map[fn] = {
                    'schema': fn_schema,
                    'docs': getattr(obj, fn ).__doc__
                }
        return function_schema_map
    
    def function_schema_map(self, include_hidden:bool = False, include_module:bool = False):
        function_schema_map = {}
        for fn in self.functions(include_module=include_module):
            if not include_hidden:
                if (fn.startswith('__') and fn.endswith('__')) or fn.startswith('_'):
                    continue
            if callable(getattr(self, fn )):
                function_schema_map[fn] = {}
                fn_schema = {}
                for fn_k, fn_v in getattr(self, fn ).__annotations__.items():
                    
                    fn_v = str(fn_v)
                    print(fn_v, fn_v.startswith('<class'))
                    
                    if fn_v == inspect._empty:
                        fn_schema[fn_k]= 'Any'
                    elif fn_v.startswith('<class'):
                        fn_schema[fn_k] = fn_v.split("'")[1]
                    else:
                        fn_schema[fn_k] = fn_v
                                        
                function_schema_map[fn] = {
                    'schema': fn_schema,
                    'docs': getattr(self, fn ).__doc__
                }
        return function_schema_map
    
    @classmethod
    def get_function_schema(cls, fn:str)->dict:
        '''
        Get function schema of function in cls
        '''
        if not callable(fn):
            fn = getattr(cls, fn)
        fn_schema = {k:str(v) for k,v in fn.__annotations__.items()}
        return fn_schema
    def function_schema(self, fn:str)->dict:
        '''
        Get function schema of function in cls
        '''
        fn = getattr(self, fn)
        fn_schema = {k:str(v) for k,v in fn.__annotations__.items()}
        return fn_schema

    @staticmethod
    def get_annotations(fn:callable) -> dict:
        return fn.__annotations__

    @classmethod
    def start_server(cls,
                module:str = None,  
                name:Optional[str]=None, 
                tag:str=None, 
                device:str='0', 
                interpreter:str='python3', 
                refresh:bool=True, 
                args = None, 
                kwargs = None ):
        
        args = args if args else []
        kwargs = kwargs if kwargs else {}
        kwargs['tag'] = tag
        return cls.launch( 
                   module = module,  
                   fn = 'serve_module',
                   name=name, 
                   tag=tag, 
                   args = args,
                   kwargs = kwargs,
                   device=device, 
                   interpreter=interpreter, 
                   refresh=refresh )
      
      
    @classmethod
    def stop(cls, path, mode:str = 'pm2'):
        cls.pm2_stop(path)
        
        return path
        
    
    ## PM2 LAND
    @classmethod
    def launch(cls, 
               module:str = None, 
               fn: str = None,
               args : list = None,
               kwargs: dict = None,
               refresh:bool=True,
               mode:str = 'pm2',
               name:Optional[str]=None, 
               tag:str=None, 
               serve: bool = True,
               **extra_kwargs):
        '''
        Launch a module as pm2 or ray 
        '''
            

        kwargs = kwargs if kwargs else {}
        args = args if args else []
        if module == None:
            module = cls  
  
        if mode == 'local':

            if isinstance(module, str):
                module_class = cls.get_module(module)
            else:
                module_class = cls
            if fn == None:
                return module_class(*args, **kwargs)
            else:
                return getattr(module_class, fn)(*args, **kwargs)

        elif mode == 'pm2':
            
            if serve:
                fn = 'serve_module'
                kwargs['tag'] = kwargs.get('tag', tag)
                kwargs['name'] = kwargs.get('name', name)
            launch_kwargs = dict(
                    module=module, 
                    fn = fn,
                    name=name, 
                    tag=tag, 
                    args = args,
                    kwargs = kwargs,
                    refresh=refresh,
                    **extra_kwargs
            )
            assert fn != None, 'fn must be specified for pm2 launch'

            launch_fn = getattr(cls, f'pm2_launch')
            launch_fn(**launch_kwargs)
        elif mode == 'ray':
            launch_kwargs = dict(
                    module=module, 
                    name=name, 
                    tag=tag, 
                    args = args,
                    kwargs = kwargs,
                    refresh=refresh,
                    serve = serve,
                    **extra_kwargs
            )
            launch_fn = getattr(cls, f'{mode}_launch')
            launch_fn(**launch_kwargs)
        else: 
            raise Exception(f'launch mode {mode} not supported')
         
    @classmethod
    def pm2_list(cls, verbose:bool = False) -> List[str]:
        output_string = cls.run_command('pm2 status', verbose=False)
        module_list = []
        for line in output_string.split('\n'):
            if '??? default     ??? ' in line:
                module_name = line.split('???')[2].strip()
                module_list += [module_name]
                
        return module_list
            
    # commune.run_command('pm2 status').stdout.split('\n')[5].split('    ???')[0].split('  ??? ')[-1]commune.run_command('pm2 status').stdout.split('\n')[5].split('    ???')[0].split('  ??? ')[-1] 
    @classmethod
    def pm2_launch(cls, 
                   module:str = None,  
                   fn: str = 'serve_module',
                   name:Optional[str]=None, 
                   tag:str=None, 
                   args : list = None,
                   kwargs: dict = None,
                   device:str=None, 
                   interpreter:str='python3', 
                   no_autorestart: bool = False,
                   refresh:bool=True, ):
        
        # avoid these references fucking shit up
        args = args if args else []
        kwargs = kwargs if kwargs else {}
        
        import streamlit as st
        st.write(module, 'bro')
        if isinstance(module, str):
            assert isinstance(module, str), f'module must be a string, not {type(module)}'
            module = cls.get_module(module)
        else:
            module = cls
            
        
        

        name = module.module_name() if name == None else name
            
    
        module_path = module.__module_file__()
        module_id = cls.get_module_id(name=name, tag=tag) 
        
        # build command to run pm2
        command = f" pm2 start {module_path} --name {module_id} --interpreter {interpreter}"
        
        # convert args and kwargs to json strings
        kwargs_str = json.dumps(kwargs).replace('"', "'")
        args_str = json.dumps(args).replace('"', "'")


        if refresh:
            cls.pm2_kill(module_id)   
            

            
        command = command + ' -- ' + f'--fn {fn} --kwargs "{kwargs_str}" --args "{args_str}"'
        cls.print(command,'purple')
        env = {}
        if device != None:
            if isinstance(device, list):
                device = ','.join(device)
            env['CUDA_VISIBLE_DEVICES']=device
        return cls.run_command(command, env=env)

    @classmethod
    def pm2_kill(cls, name:str):
        output_str = cls.run_command(f"pm2 delete {name}")
    @classmethod
    def pm2_restart(cls, name:str):
        return cls.run_command(f"pm2 restart {name}")
        stdout = cls.run_command(f"pm2 status")
        if verbose:
            cls.print(stdout, 'orange')

    @classmethod
    def pm2_status(cls, verbose=True):
        stdout = cls.run_command(f"pm2 status")
        if verbose:
            cls.print(stdout, 'green')
        return stdout


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
        cls.print(args, 'cyan')
        args.kwargs = json.loads(args.kwargs.replace("'",'"'))
        args.args = json.loads(args.args.replace("'",'"'))
        return args

    @classmethod
    def run(cls): 
        args = cls.argparse()
        if args.function == '__init__':
            return cls(*args.args, **args.kwargs) 
        else:
            return getattr(cls, args.function)(*args.args, **args.kwargs)     
       
    @classmethod
    def api(cls, *args, **kwargs):
        from commune.api import API
        return API(*args, **kwargs)
    @classmethod
    def sandbox(cls, **kwargs):
     
        import commune
        commune = commune
        batch_count = 100
        print(cls.server_registry())
        
        # t = commune.timer()
        dataset =  cls.connect('dataset.bittensor')
        model =  cls.connect('model.transformer::gptj')
        t = commune.timer()

        sample = dataset(fn='sample', kwargs=dict(batch_size=32, sequence_length=256))
        sample['output_hidden_states'] =  False
        sample['output_logits'] =  False
        sample['topk'] =  10
        sample['output_length'] = 10
        # sample['topk'] = True
        print(model(fn='forward', kwargs=sample)['hidden_states'].shape)
        print(t.seconds)
        
    
    
    @classmethod
    def get_methods(cls, obj:type= None, modes:Union[str, List[str]] = 'all',  ) -> List[str]:
        '''
        
        Get methods of the obj, which defaults to the class object if None
        
        Args:
            obj (object): object to get methods from
            modes:
        
        '''
        methods = []
        obj = obj if obj else cls
        
        if modes == 'all':
            modes = ['class', 'self']
        
        default_modes = ['class', 'self']
        
        for mode in modes:
            assert mode in default_modes, f'{mode} not in {default_modes}'
            methods.extend(getattr(cls, f'get_{mode}_methods')(obj))
            
    @classmethod
    def get_class_methods(cls, obj=None) -> List[str]:
        from commune.utils.function import get_class_methods
        return get_class_methods(obj if obj else cls)
        
    @classmethod
    def get_self_methods(cls, obj=None) -> List[str]:
        from commune.utils.function import get_self_methods
        return get_self_methods(obj if obj else cls)
        
        
    ## RAY LAND
    
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
        import ray

        init_kwargs =  {**cls.default_ray_env, **init_kwargs}
        if cls.ray_initialized():
            # shutdown if namespace is different
            if cls.ray_namespace() == cls.default_ray_env['namespace']:
                return cls.ray_runtime_context()
            else:
                ray.shutdown()
  
        ray_context = ray.init(**init_kwargs)
        return ray_context

    @classmethod
    def ray_runtime_context(cls):
        return ray.get_runtime_context()


    @classmethod
    def ray_stop(cls):
        return cls.run_command('ray stop')

    @classmethod
    def ray_start(cls):
        return cls.run_command('ray start --head')


    @classmethod
    def ray_status(cls, *args, **kwargs):
        return cls.run_command('ray status',  *args, **kwargs)

    @classmethod
    def ray_initialized(cls):
        import ray
        return ray.is_initialized()

    # def resource_usage(self):
    #     resource_dict =  self.config.get('actor', {}).get('resources', None)
    #     resource_dict = {k.replace('num_', ''):v for k,v in resource_dict.items()}
    #     resource_dict['memory'] = self.memory_usage(mode='ratio')
    #     return  resource_dict
    
    @classmethod
    def ensure_ray_context(cls, ray_config:dict = None):
        ray_config = ray_config if ray_config != None else {}
        
        if cls.ray_initialized():
            ray_context = cls.get_ray_context()
        else:
            ray_context =  cls.ray_init(init_kwargs=ray_config)
        
        return ray_context
    @classmethod
    def ray_env(cls):
        import ray
        if not cls.ray_initialized():
            cls.ray_init()
        return ray
    
    @classmethod 
    def ray_launch(cls, 
                   module= None, 
                   name:Optional[str]=None, 
                   tag:str=None, 
                   args:List = None, 
                   refresh:bool = False,
                   kwargs:Dict = None,
                   serve: bool = False, 
                   **actor_kwargs):
        
        launch_kwargs = dict(locals())
        launch_kwargs.update(launch_kwargs.pop('actor_kwargs'))
        launch_kwargs = deepcopy(launch_kwargs)
        ray = cls.ray_env()
        """
        deploys process as an actor or as a class given the config (config)
        """
        args = args if args != None else []
        kwargs = kwargs if kwargs != None else {}
        module_class = None
        if isinstance(module, str):
            module_class = cls.get_module(module)
        elif module == None :
            module_class = cls

        else:
            module_class = cls.module(module)
            
        if name == None:
            if cls.is_module(module_class):
                name = module_class.module_name()
            else:
                name = module_class.__name__
            
        assert isinstance(name, str)
        
        name = cls.get_module_id(name=name, tag=tag) 
        
        actor_kwargs['name'] = name
        actor_kwargs['refresh'] = refresh

        actor = cls.create_actor(module=module_class,  args=args, kwargs=kwargs, **actor_kwargs) 
        if serve:
            actor = actor.serve(ray_get=False)
        
        return actor
            

    default_ray_env = {'address':'auto', 
                     'namespace': 'default',
                      'ignore_reinit_error': False,
                      'dashboard_host': '0.0.0.0'}
    @classmethod
    def ray_init(cls,init_kwargs={}):
        import ray
        init_kwargs =  {**cls.default_ray_env, **init_kwargs}
        ray_context = {}
        if cls.ray_initialized():
             ray_context =  cls.ray_runtime_context()
        else: 
            ray_context = ray.init(**init_kwargs)
            
        return ray_context
    
    @classmethod
    def create_actor(cls,
                 module : str = None,
                 name:str = None,
                 tag:str = None,
                 kwargs: dict = None,
                 args:list =None,
                 cpus:int = 1.0,
                 gpus:int = 0,
                 detached:bool=True, 
                 max_concurrency:int=50,
                 refresh:bool=True,
                 verbose:bool= True,
                 virtual:bool = True):
        
        # self.ray_init()
        import ray, torch
        module = module if module != None else cls 
        
        cls_kwargs = kwargs if kwargs else {}
        cls_args = args if args else []
        name = name if name != None else module.__name__
        resources = {}
        resources['num_cpus'] = cpus
        resources['num_gpus'] = gpus

        if not torch.cuda.is_available() and 'num_gpus' in resources:
            del resources['num_gpus']

        # configure the option_kwargs
        options_kwargs = {'name': name,
                          'max_concurrency': max_concurrency,
                           **resources}
        
        # detatch the actor from the process when it finishes
        if detached:
            options_kwargs['lifetime'] = 'detached'
            
        # setup class init config
        # refresh the actor by killing it and starting it (assuming they have the same name)
        if refresh:
            if cls.actor_exists(name):
                cls.kill_actor(actor=name,verbose=verbose)
                # assert not Module.actor_exists(name)

        options_kwargs['namespace'] = 'default'

        # create the actor if it doesnt exisst
        # if the actor is refreshed, it should not exist lol (TODO: add a check)
        

        if not hasattr(module, 'set_module_id'):
            def set_module_id(self, name):
                self.module_id = name
                return True
            
            module.set_module_id = set_module_id
        
        if not cls.actor_exists(name):
            
            actor_class = ray.remote(module)
            actor_handle = actor_class.options(**options_kwargs).remote(*cls_args, **cls_kwargs)
            ray.get(actor_handle.set_module_id.remote(name))
        actor = cls.get_actor(name, virtual=virtual)

        
        return actor

    @staticmethod
    def get_actor_id( actor):
        assert isinstance(actor, ray.actor.ActorHandle)
        return actor.__dict__['_ray_actor_id'].hex()

    @classmethod
    def create_pool(cls, replicas=3, actor_kwargs_list=[], **kwargs):
        if actor_list == None:
            actor_kwargs_list = [kwargs]*replicas

        actors = []
        for actor_kwargs in actor_kwargs_list:
            actors.append(cls.deploy(**a_kwargs))

        return ActorPool(actors=actors)

    @classmethod
    def virtual_actor(cls, actor):
        from commune.block.ray.client.ray_client import ClientModule
        return ClientModule(actor=actor)

    @classmethod
    def kill_actor(cls, actor, verbose=True):
        import ray
        killed_actors = None
        if isinstance(actor, list):
            killed_actors = []
            for a in actor:
                killed_actors.append(cls.kill_actor(a))
                
        elif isinstance(actor, str):
            if cls.actor_exists(actor):
                actor = ray.get_actor(actor)
            else:
                if verbose:
                    print(f'{actor} does not exist for it to be removed')
                return None
            ray.kill(actor)
            killed_actors = actor
        
            return killed_actors
        elif hasattr(actor, 'module_id'):
            return self.kill_actor(actor.module_id, verbose=verbose)
            
        
       
    @classmethod
    def actor_exists(cls, actor):
        ray = cls.ray_env()
        if isinstance(actor, str):
            try:
                ray.get_actor(actor)
                actor_exists = True
            except ValueError as e:
                actor_exists = False
            
            return actor_exists
        else:
            raise NotImplementedError

    @classmethod
    def ray_actor(cls ,actor_name:str, virtual:bool=True):
        '''
        Gets the ray actor
        '''
        ray  = cls.ray_env()
        actor =  ray.get_actor(actor_name)
        # actor = Module.add_actor_metadata(actor)
        if virtual:
            actor = cls.virtual_actor(actor=actor)
        return actor
    
    get_actor = ray_actor

    @classmethod
    def ray_runtime_context(cls):
        import ray
        return ray.get_runtime_context()

    @classmethod
    def ray_namespace(cls):
        import ray
        return ray.get_runtime_context().namespace

    @classmethod
    def ray_context(cls):
        import ray
        return ray.runtime_context.get_runtime_context()

    @staticmethod
    def ray_objects( *args, **kwargs):

        return ray.experimental.state.api.list_objects(*args, **kwargs)
    
    @classmethod
    def ray_actors(cls, state='ALIVE', names_only:bool = True,detail:bool=True, *args, **kwargs):
        
        ray = cls.ray_env()
        from ray.experimental.state.api import list_actors
              
        kwargs['filters'] = kwargs.get('filters', [("state", "=", state)])
        kwargs['detail'] = detail

        actor_info_list =  list_actors(*args, **kwargs)
        ray_actors = []
        for i, actor_info in enumerate(actor_info_list):
            # resource_map = {'memory':  Module.get_memory_info(pid=actor_info['pid'])}
            resource_list = actor_info_list[i].pop('resource_mapping', [])
            resource_map = {}
            for resource in resource_list:
                resource_map[resource['name'].lower()] = resource['resource_ids']
            actor_info_list[i]['resources'] = resource_map
            if names_only:
                ray_actors.append(actor_info_list[i]['name'])
            else:
                ray_actors.append(actor_info_list[i])
            
        return ray_actors
    actors = ray_actors
    
    @classmethod
    def actor_resources(cls, actor:str):
        resource_map = cls.ray_actor_map()[actor]['required_resources']
        k_map = {
            'GPU': 'gpus',
            'CPU': 'cpus'
        }
        return {k_map[k]:float(v) for k,v in resource_map.items() }
    @classmethod
    def ray_actor_map(cls, ):
        ray = cls.ray_env()
        actor_list = cls.ray_actors(names_only=False, detail=True)
        actor_map  = {}
        for actor in actor_list:
            actor_name = actor.pop('name')
            actor_map[actor_name] = actor
        return actor_map
    actor_map = ray_actor_map
  
    @classmethod
    def ray_tasks(cls, running=False, name=None, *args, **kwargs):
        ray = cls.ray_env()
        filters = []
        if running == True:
            filters.append([("scheduling_state", "=", "RUNNING")])
        if isinstance(name, str):
            filters.append([("name", "=", name)])
        
        if len(filters)>0:
            kwargs['filters'] = filters

        ray_tasks = ray.experimental.state.api.list_tasks(*args, **kwargs)
        return ray_tasks
   
    @staticmethod
    def ray_nodes( *args, **kwargs):
        from ray.experimental.state.api import list_nodes
        return list_nodes(*args, **kwargs)
    @classmethod
    def ray_get(cls,*jobs):
        cls.ray_env()
        return ray.get(jobs)
    @classmethod
    def ray_wait(cls, *jobs):
        cls.ray_env()
        finished_jobs, running_jobs = ray.wait(jobs)
        return finished_jobs, running_jobs
    
    
    @classmethod
    def ray_put(cls, *items):
        ray = cls.ray_env()
        import ray
        return [ray.put(i) for i in items]

    @staticmethod
    def get_ray_context():
        import ray
        return ray.runtime_context.get_runtime_context()
    
    @classmethod
    def module(cls, module: 'python::class' ,init_module:bool=False , serve:bool=False):
        '''
        Wraps a python class as a module
        '''
        
        if isinstance(module, str):
            module = cls.get_module(module)

        
        # serve the module if the bool is True
        is_class = cls.is_class(module)
        module_class = module if is_class else module.__class__
        class ModuleWrapper(Module):
            def __init__(self, *args,**kwargs): 
                if init_module:
                    Module.__init__(self,**kwargs)
                if is_class:
                    self.module = module_class(*args, **kwargs)
                else:
                    self.module = module
                
                # merge the inner module into the wrappers
                self.merge(self.module)
            @classmethod
            def module_name(cls): 
                return module_class.__name__
            @classmethod
            def __module_file__(cls): 
                return cls.get_module_path(simple=False)
            
            def __call__(self, *args, **kwargs):
                return self.module.__call__(self, *args, **kwargs)

            def __str__(self):
                return self.module.__str__()
            
            def __repr__(self):
                return self.module.__repr__()  
        if is_class:
            return ModuleWrapper
        else:
            return ModuleWrapper()
            
        # return module

    # UNDER CONSTRUCTION (USE WITH CAUTION)
    
    def setattr(self, k, v):
        setattr(self, k, v)
        
    @classmethod
    def default_module_id(cls):
        return cls.module_name()
    
    def set_module_id(self, module_id:str) -> str:
        '''
        Sets the module_id when a module is deployed 
        '''
        self.module_id = module_id
        return module_id
    def setattributes(self, new_attributes:Dict[str, Any]) -> None:
        '''
        Set a dictionary to the slf attributes 
        '''
        assert isinstance(new_attributes, dict), f'locals must be a dictionary but is a {type(locals)}'
        self.__dict__.update(new_attributes)

    @staticmethod
    def get_template_args( template:str) -> List[str]:
        '''
        get the template arguments from a string such that
        template = 'hello {name} {age}' returns ['name', 'age']
        
        Args:
            template (str): template string
        Returns:
            List[str]: list of template arguments
            
            
        '''
        from string import Formatter
        template_args =  [i[1] for i in Formatter().parse(template)  if i[1] is not None] 
        
        return template_args
         
    def merge_dict(self, python_obj: Any, include_hidden:bool=False):
        '''
        Merge the dictionaries of a python object into the current object
        '''
        for k,v in python_obj.__dict__.items():
            if include_hidden == False:
                #i`f the function name starts with __ then it is hidden
                if k.startswith('__'):
                    continue
            self.__dict__[k] = v
      
              
    def merge(self, *args, include_hidden:bool = False) -> 'self':
        '''
        Merge the attributes of a python object into the current object
        '''
        merge = self.import_object('commune.utils.class.merge')
        if len(args) == 1:
            args = [self, *args]
            
            
            
            
        assert len(args) == 2, f'args must be a list of length 2 but is {len(args)}'
    
    
        module = merge(*args, include_hidden=include_hidden)
        

        return module
        
    @classmethod
    def print(cls, text:str, color:str='white', return_text:bool=False, verbose:bool = True):
        if verbose:
            logger = cls.import_object('commune.logger.Logger')
            return logger.print(text=text, color=color, return_text=return_text)

    @classmethod
    def nest_asyncio(cls):
        import nest_asyncio
        nest_asyncio.apply()
        
        
    # JUPYTER NOTEBOOKS
    @classmethod
    def jupyter(cls):
        cls.nest_asyncio()
        
    enable_jupyter = jupyter
        
        
    @classmethod
    def int_to_ip(cls, *args, **kwargs):
        return cls.import_object('commune.utils.network.int_to_ip')(*args, **kwargs)
        
    @classmethod
    def ip_to_int(cls, *args, **kwargs):
        return cls.import_object('commune.utils.network.ip_to_int')(*args, **kwargs)

    @classmethod
    def ip_version(cls, *args, **kwargs):
        return cls.import_object('commune.utils.network.ip_version')(*args, **kwargs)
    
    @classmethod
    def get_external_ip(cls, *args, **kwargs) ->str:
        return cls.import_object('commune.utils.network.get_external_ip')(*args, **kwargs)

    @classmethod
    def external_ip(cls, *args, **kwargs) -> str:
        return cls.get_external_ip(*args, **kwargs)
    
    @classmethod
    def get_external_ip(cls, *args, **kwargs) ->str:
        return cls.import_object('commune.utils.network.get_external_ip')(*args, **kwargs)

    @classmethod
    def public_ip(cls):
        return cls.get_public_ip(*args, **kwargs)
    
    @staticmethod
    def is_class(module: Any) -> bool:
        return type(module).__name__ == 'type' 
    
    external_ip = get_external_ip
    
    @classmethod
    def upnpc_create_port_map(cls, port:int):
        return cls.import_object('commune.utils.network.upnpc_create_port_map')(port=port)

    @classmethod
    def set_env(cls, key:str, value:str)-> None:
        '''
        Pay attention to this function. It sets the environment variable
        '''
        import os
        os.environ[key] = value
        return value 

    @classmethod
    def get_env(cls, key:str)-> None:
        '''
        Pay attention to this function. It sets the environment variable
        '''
        import os
        return  os.environ[key] 


    
    ### GPU LAND
    
    @classmethod
    def gpus(cls) -> List[int]:
        import torch
        available_gpus = [i for i in range(torch.cuda.device_count())]
        return available_gpus
    
    @classmethod
    def gpu_map(cls) -> Dict[int, Dict[str, float]]:
        import torch
        gpu_info = {}
        for gpu_id in cls.gpus():
            mem_info = torch.cuda.mem_get_info(gpu_id)
            gpu_info[gpu_id] = {
                'name': torch.cuda.get_device_name(gpu_id),
                'free': mem_info[0]/1e9,
                'total': mem_info[1]/1e9,
                'used': (mem_info[1]- mem_info[0])/1e9,
            }
        return gpu_info
    
    @classmethod
    def free_gpu_memory(cls) -> int:
        free_gpu_memory = 0
        for gpu_id, gpu_info in cls.gpu_map().items():
            free_gpu_memory += gpu_info['free']
        return free_gpu_memory
    
    @classmethod
    def total_gpu_memory(cls) -> int:
        total_gpu_memory = 0
        for gpu_id, gpu_info in cls.gpu_map().items():
            total_gpu_memory += gpu_info['total']
        return total_gpu_memory

    @classmethod
    def used_gpu_memory(cls) -> int:
        used_gpu_memory = 0
        for gpu_id, gpu_info in cls.gpu_map().items():
            used_gpu_memory += gpu_info['used'] 
        return used_gpu_memory

    @classmethod
    def least_used_gpu(cls) -> int:
        """ Returns a dictionary of gpu_id to max memory for each gpu.
        Args:
            total_memory (int, optional): Total memory to allocate. Defaults to None.
            buffer_memory (int, optional): Buffer memory to leave on each gpu. Defaults to 10.
        
        Returns 
            Dict[int, str]: Dictionary of gpu_id to max memory for each gpu.
        """
        gpu_info_map = cls.gpu_map()
        most_available_gpu_tuples = sorted(gpu_info_map.items(), key=lambda x: x[1]['free'] , reverse=True)
        return most_available_gpu_tuples[0][0]
    
    @classmethod
    def gpu_info(cls, device:int = None) -> Dict[str, Union[int, float]]:
        '''
        Get the gpu info for a given device
        '''
        if device is None:
            device = 0
        gpu_map = cls.gpu_map()
        return gpu_map[device]

    def resolve_tag(self, tag:str = None) -> str:
        if tag is None:
            tag = self.tag
        return tag

    @classmethod
    def resolve_device(cls, device:str = None, verbose:bool=True, find_least_used:bool = True) -> str:
        
        '''
        Resolves the device that is used the least to avoid memory overflow.
        '''
        import torch
        if device == None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            assert torch.cuda.is_available(), 'Cuda is not available'
            gpu_id = 0
            if find_least_used:
                gpu_id = cls.least_used_gpu()
                
            device = f'cuda:{gpu_id}'
        
            if verbose:
                device_info = cls.gpu_info(gpu_id)
                cls.print(f'Using device: {device} with {device_info["free"]} GB free memory', 'yellow')
        
        return device  
    
    
    @classmethod
    def peer_info_list(cls, module = None):
        if module == None:
            module = cls
        if isinstance(module, str):
            module = cls.connect(module)
            return module.peer_info_list()
                 
        external_ip = cls.get_external_ip()
        peers = module.servers()

        peer_info_list = []
        for p in peers:
            peer = cls.connect(p)
            peer_stats = peer.server_stats
            
            peer_info = {}
            peer_info['name'] = p
            peer_info['endpoint'] = peer_stats['external_ip']+':' + str(peer_stats['port'])
            peer_info['is_local'] = external_ip == peer_stats['external_ip']
            peer_info_list.append(peer_info)
        
        return peer_info_list
      

    @classmethod
    def peer_registry(cls, module = None):
        if module == None:
            module = cls
        if isinstance(module, str):
            module = cls.connect(module)
            return module.peer_registry()
                 
        external_ip = cls.get_external_ip()
        peers = module.servers()
        peer_info_list = []
         
        peer_registry = cls.get_json('peer_registry')
        peer_registry = {}
        
        peer_map = {}
        for p in peers:
            peer = cls.connect(p)
            peer_stats = peer.server_stats
            peer_info = {}
            peer_info['endpoint'] = peer_stats['external_ip']+':' + str(peer_stats['port'])
            peer_info['is_local'] = external_ip == peer_stats['external_ip']
            peer_map[p] = peer_info
        peer_registry[external_ip] = peer_map
        
        cls.put_json('peer_registry', peer_registry)
        
        return peer_registry

    @classmethod
    def add_peer_registry(cls, endpoint:str):
        peer_registry = cls.get_json('peer_registry')
        peer_registry = {}
        peer_list = peer_registry.get(endpoint, [])
        remote_peer_list = cls.peer_registry(module=endpoint)
        
        peer_list = remote_peer_list + peer_list
        
        peer_registry[endpoint] = remote_peer_list
        
        cls.put_json('peer_registry', peer_registry)
        
        return peer_registry
        
  
    
    def num_params(self, model:'nn.Module')->int:
        import np
        from torch import nn
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        num_params = sum([np.prod(p.size()) for p in model_parameters])
        return num_params

    def to_dict(self)-> Dict:
        return self.__dict__
    
    @classmethod
    def from_dict(cls, input_dict:Dict[str, Any]) -> 'Module':
        return cls(**input_dict)
        
    def to_json(self) -> str:
        import json
        state_dict = self.to_dict()
        assert isinstance(state_dict, dict), 'State dict must be a dictionary'
        return json.dumps(state_dict)
    
    logger = None
    @classmethod
    def log(cls, text, mode='info'):
        if cls.logger is None:
            from loguru import logger
            cls.logger = logger.opt(colors=True)
        
        specific_logger = getattr(cls.logger, mode)
        return specific_logger(text)

    @classmethod
    def from_json(cls, json_str:str) -> 'Module':
        import json
        return cls.from_dict(json.loads(json_str))
        
    @classmethod
    def test(cls):
        for f in dir(cls):
            if f.startswith('test_'):
                getattr(cls, f)()
               
               
    @classmethod
    def import_bittensor(cls):
        try:
            import bittensor
        except RuntimeError:
            cls.new_event_loop()
            import bittensor
        return bittensor
         
    @classmethod  
    def time( cls) -> float:
        import time
        return time.time()
    @classmethod
    def sleep(cls, seconds:float) -> None:
        import time
        time.sleep(seconds)
        return None
    
    
    # DICT LAND
    
    
    @classmethod
    def dict_put(cls, *args, **kwargs):
        dict_put = cls.import_object('commune.utils.dict.dict_put')
        return dict_put(*args, **kwargs)
    @classmethod
    def dict_get(cls, *args, **kwargs):
        dict_get = cls.import_object('commune.utils.dict.dict_get')
        return dict_get(*args, **kwargs)
    
    @classmethod
    def dict_has(cls, *args, **kwargs):
        dict_has = cls.import_object('commune.utils.dict.dict_has')
        return dict_has(*args, **kwargs)
    
Block = Lego = Module
if __name__ == "__main__":
    Module.run()

