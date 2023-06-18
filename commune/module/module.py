

import inspect
import numpy as np
import os
from copy import deepcopy
from typing import Optional, Union, Dict, List, Any, Tuple, Annotated
from munch import Munch
from rich.console import Console
import json
from glob import glob
import sys
import argparse
import asyncio
import gradio as gr

class c:
    root_module_class = 'c' # WE REPLACE THIS THIS Module at the end, kindof odd, i know, ill fix it fam, chill out dawg, i didnt sleep with your girl
    default_port_range = [50050, 50150] 
    user = None
    default_ip = '0.0.0.0'
    address = None
    root_path  = root = os.path.dirname(os.path.dirname(__file__))
    repo_path  = os.path.dirname(root_path)
    library_name = root_dir = root_path.split('/')[-1]
    default_network = 'commune'
    pwd = os.getenv('PWD')
    console = Console()
    helper_functions = ['getattr', 'functions', 'namespace', 'server_info', 
                'info', 'ip', 'address','ip_address', 'info', 'schema',
                'module_name', 'modules', 'help']

    
        
    def __init__(self, 
                 config:Dict=None, 
                 add_attributes: bool = False,
                 key: str = None,
                 save_config:bool = False,
                 *args, 
                 **kwargs):
        # set the config of the module (avoid it by setting config=False)
        self.set_config(config=config, add_attributes=add_attributes, save_config=save_config)  
        # self.set_key(key)
    
    
    
    def init(self, *args, **kwargs):
        c.__init__(self, *args, **kwargs)
    
    @classmethod
    def boot_peers(cls) -> List[str]: 
        config = c.get_config()
        boot_peers = config.get('boot_peers', [])
        return boot_peers
        
        
    @classmethod
    def add_root_path(cls, root_path:str):
        root_paths = c.getc('root_paths', [])
        if root_path not in root_paths:
            root_paths.append(root_path)
        else: 
            return {'msg': 'root_path already exists'}
        c.putc('root_paths', root_paths)
        return {'msg': 'success'}
    
    
    @classmethod
    def get_root_paths(cls):
        root_paths = c.getc('root_paths', [cls.root_path])
        if cls.root_path not in root_paths:
            cls.add_root_path(cls.root_path)

        return rot_paths
    root_paths = get_root_paths
        

    @classmethod
    def start_node(cls, *args, **kwargs):
        c.module('subspace').start_node(*args, **kwargs)
    def getattr(self, k:str)-> Any:
        return getattr(self,  k)
    @classmethod
    def getclassattr(cls, k:str)-> Any:
        return getattr(cls,  k)
    
    
    
    @classmethod
    def module_file(cls) -> str:
        # get the file of the module
        return inspect.getfile(cls)

    @classmethod
    def __module_dir__(cls) -> str :
        # get the directory of the module
        return os.path.dirname(cls.module_file())
    

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
    def filepath(cls) -> str:
        '''
        removes the PWD with respect to where module.py is located
        '''
        return cls.get_module_path(simple=False)
    
    @classmethod
    def dirpath(cls) -> str:
        '''
        removes the PWD with respect to where module.py is located
        '''
        return os.path.dirname(cls.filepath())
    
    
    @classmethod
    def __local_file__(cls) -> str:
        '''
        removes the PWD with respect to where module.py is located
        '''
        return cls.get_module_path(simple=False).replace(cls.repo_path+'/', '')
    
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
    def module_path(cls, simple:bool=True) -> str:
        # get the module path
        path = cls.get_module_path(simple=simple)
        path = path.replace('modules.', '')
        return path
        
    @classmethod
    def module_class(cls) -> str:
        return cls.__name__

    
    @classmethod
    def class_name(cls) -> str:
        return cls.__name__

    def get_class_name(cls, obj = None) -> str:
        obj = obj if obj != None else cls
        if not cls.is_class(obj):
            obj = type(obj)
        
        return obj.__name__
        
    
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
        
        __config_file__ =  cls.module_file().replace('.py', '.yaml')
        
        # if the config file does not exist, then create one where the python path is

        return __config_file__


    @classmethod
    def get_module_config_path(cls) -> str:
        return cls.get_module_path(simple=False).replace('.py', '.yaml')
    
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
    def munch(cls, x:Dict) -> Munch:
        '''
        Converts a dict to a munch
        '''
        return cls.dict2munch(x)
    
    @classmethod
    def load_yaml(cls, path:str=None, root:bool = False) -> Dict:
        '''f
        Loads a yaml file
        '''
        path = cls.resolve_path(path, root=root)
        
        from commune.utils.dict import load_yaml
        return load_yaml(path)



    @classmethod
    def fn_code_map(cls, module=None)-> Dict[str, str]:
        module = module if module else cls
        functions = cls.get_functions(module)
        fn_code_map = {}
        for fn in functions:
            fn_code_map[fn] = cls.get_function_code(fn=fn, module=module)
        return fn_code_map
    
    code_map = fn_code_map
            
    @classmethod
    def get_function_code(cls, 
                    fn:str, 
                    module:str = None, # defaults to the current module
                    fn_seperator:str="::" ) -> str:
        '''
        Returns the code of a function
        '''
        
        
        if isinstance(fn, str):
            if fn.split(fn_seperator)==2:
                module, fn = fn.split(fn_seperator)
                module = commune.module(module)

            if module is None:
                module = cls 
            
            fn = getattr(module, fn)
        assert callable(fn), f'fn must be callable, got {fn}'       
        fn_code = inspect.getsource(fn)
        return fn_code

    @classmethod
    def function_code(cls, fn ) -> str:
        '''
        Returns the code of a function
        '''
        return cls.get_fn_code(fn)
    
    
    fn_code = function_code
    get_fn_code = get_function_code

    @classmethod
    def sandbox(cls):
        return cls.cmd('python3 sandbox.py')
    sand = sandbox
    @classmethod
    def save_yaml(cls, path:str,  data:Union[Dict, Munch], root:bool = False) -> Dict:
        '''
        Loads a yaml file
        '''
        path = cls.resolve_path(path, root=root)
            
        from commune.utils.dict import save_yaml
        if isinstance(data, Munch):
            data = cls.munch2dict(deepcopy(data))
            
        print('saving yaml', path, data)
        return save_yaml(data=data , path=path)

    def merge_config(self, config:Dict, overrite_keys:bool = False) -> Dict:
        '''
        Merges the config with the current config
        '''
        if hasattr(config, 'to_dict'):
            config = config.to_dict()
        
        elif isinstance(config, Munch):
            config = self.munch2dict(config)
                
        # merge the model config with the config
        
        default_config = self.munch2dict(self.config)
        for k,v in config.items():
            if not overrite_keys:
                assert k not in default_config, f'config key {k} not found in config'
            default_config[k] = config[k]        
        self.config = self.munch(default_config)
        return self.config
    
    
    @classmethod
    def resolve_config_path(cls, path= None, root:bool=False) -> str:
        
        module_tree = cls.module_tree()
        if path in module_tree: 
            path = module_tree[path].replace('.py', '.yaml')
            
        if path is None:
            if root:
                path = c.__config_file__()
            else:
                path = cls.__config_file__()
        assert isinstance(path, str)
        return path
    @classmethod
    def load_config(cls, path:str=None, to_munch:bool = False, root:bool = False) -> Union[Munch, Dict]:
        '''
        Args:
            path: The path to the config file
            to_munch: If true, then convert the config to a munch
        '''
        
        path = cls.resolve_config_path(path, root=root)
        config = cls.load_yaml(path)

        if to_munch:
            config =  cls.dict2munch(config)
        
        return config
    
    
    default_config = load_config
    
    @classmethod
    def put(cls, 
            k, 
            v, 
            password: bool = None,
            include_timestamp : bool = True,
            mode: bool = 'json',
            **kwargs):
        '''
        Puts a value in the config
        '''
        encrypt =  password != None
        v = cls.copy(v)
        if encrypt:
            data = cls.encrypt(v, password=password, return_dict=True)
        else:
            data = {'data': v,
                'encrypted': encrypt}

        if include_timestamp:
            data['timestamp'] = c.timestamp()
            

        
        # default json 
        getattr(cls,f'put_{mode}')(k, data, **kwargs)
        
        
        return data
    
    

        
    @classmethod
    def get(cls,
            key, 
            default=None, 
            password=None, 
            mode:str = 'json',
            **kwargs):
        
        '''
        Puts a value in sthe config
        '''
        kwargs['default'] = default
        data = getattr(cls, f'get_{mode}')(key, **kwargs)
        if data == None: 
            data = {}
        encrypted = c.is_encrypted(data)
        if encrypted:
            data = cls.decrypt(data, password=password)
        if isinstance(data, dict):
            data = data.get('data', data)
            
            
        return data
    
    @classmethod
    def config_keys(self, config:Dict = None) -> List[str]:
        '''
        Returns the keys of the config
        '''
        config = config or self.config
        return list(config.keys())
    
    
    @classmethod
    def mutc(cls, k, v, password:str=None, new_password:str=None):
        old_v = cls.getc(k, password=password)
        password = password if new_password == None else new_password
        v = cls.put_v(old_v, password=password)
    @classmethod
    def putc(cls, k, v, password=None) -> Munch:
        '''
        Saves the config to a yaml file
        '''
        config = cls.config()
        if password:
            v = cls.decrypt(v, password=password)

        cls.dict_put(config, k, v)
        cls.save_config(config=config)
   
    setc = putc
      
    @classmethod
    def popc(cls, key:str):
        config = cls.config()
        config.pop(key, None)
        cls.save_config(config=config)
        
    @classmethod  
    def getc(cls, key, password=None) -> Any:
        '''
        Saves the config to a yaml file
        '''
        
        data = cls.dict_get(cls.config(), key)
        if c.is_encrypted(data):
            assert password != None, 'password must be provided to decrypt the data'
            data = c.decrypt(data, password=password)
            
        return data

    
    @classmethod
    def save_config(cls, config:Union[Munch, Dict]= None, path:str=None) -> Munch:

        '''
        Saves the config to a yaml file
        '''
        if config == None:
            config = cls.get_config()
        
        path = path if path else cls.__config_file__()
        
        if isinstance(config, Munch):
            config = cls.munch2dict(deepcopy(config))
        elif isinstance(config, dict):
            config = deepcopy(config)
        else:
            raise ValueError(f'config must be a dict or munch, not {type(config)}')
        
        config = cls.save_yaml(data=config , path=path)

        return config
    
    
    def config_exists(self, path:str=None) -> bool:
        '''
        Returns true if the config exists
        '''
        path = path if path else self.__config_file__()
        return self.path_exists(path)
    @classmethod
    def get_config(cls, 
                   config = None,
                   kwargs=None, 
                   to_munch:bool = True,
                   root:bool = False) -> Munch:
        '''
        Set the config as well as its local params
        '''

                
        if isinstance(config, str) or config == None:
            try:
                config = cls.load_config(path=config)
            except FileNotFoundError as e:
                cls.print(f'config not found at {config}, loading default config')
                config = {}
            assert isinstance(config, dict), f'config must be a dict, not {type(config)}'
        elif isinstance(config, dict):
            default_config = cls.load_config()
            default_config.update(config)
            config = default_config
            
        assert isinstance(config, dict), f'config must be a dict, not {type(config)}'
        
        kwargs = kwargs if kwargs != None else {}
        kwargs.update(kwargs.pop('kwargs', {}))
        
        for k,v in kwargs.items():
            cls.dict_put(config,k,v )
        # ensure there are no inner_args to avoid ambiguous args 
    
        if isinstance(config, Munch) and to_munch:
            config = cls.munch2dict(config)
        
            
        #  add the config after in case the config has a config attribute lol
        if to_munch:
            config = cls.dict2munch(config)
        
        return config

    config = get_config

    @classmethod
    def cfg(cls, *args, **kwargs):
        return cls.get_config(*args, **kwargs)





    def set_config(self, 
                   config:Optional[Union[str, dict]]=None, 
                   kwargs:dict={},
                   to_munch: bool = True,
                   add_attributes: bool = False,
                   save_config:bool = False) -> Munch:
        '''
        Set the config as well as its local params
        '''
        
        from commune.utils.dict import munch2dict, dict2munch
        

        config =  self.get_config(config=config,kwargs=kwargs, to_munch=to_munch)

        if add_attributes:
            self.__dict__.update(self.munch2dict(config))
        self.config = config 
        
        if save_config:
            self.save_config(config=config)
        
        
        return self.config

    @classmethod
    def flatten_dict(cls, x):
        from commune.utils.dict import deep2flat
        return deep2flat(x)

    # KEY LAND
    @classmethod
    def add_key(cls, *args, **kwargs):
        return cls.module('key').add_key(*args, **kwargs)
    @classmethod
    def ls_keys(cls, *args, **kwargs):
        return cls.module('key').ls_keys(*args, **kwargs)
    @classmethod
    def rm_key(cls, *args, **kwargs):
        return cls.module('key').rm_key(*args, **kwargs)
    @classmethod
    def key_encrypted(cls, *args, **kwargs):
        return cls.module('key').key_encrypted(*args, **kwargs)

    
    @classmethod
    def encrypt_key(cls, *args, **kwargs):
        return cls.module('key').key_encrypted(*args, **kwargs)
        

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
    def st(cls, module = None, fn='dashboard'):
        module = cls.get_module(module)
        module_filepath = module.filepath()
        cls.run_command(f'streamlit run {module_filepath} -- --fn {fn}', verbose=True)

    @staticmethod
    def st_sidebar(fn):
        import streamlit as st
        
        def wrapper(*args, **kwargs):
            with st.sidebar:
                return fn(*args, **kwargs)
        
        return wrapper


    @classmethod
    def run_command(cls, 
                    command:str,
                    verbose:bool = False, 
                    env:Dict[str, str] = {}, 
                    output_text:bool = True,
                    sudo:bool = False,
                    password: bool = None,
                    color: str = 'green',
                    **kwargs) -> 'subprocess.Popen':
        '''
        Runs  a command in the shell.
        
        '''
        import subprocess
        import shlex
        import time
        import signal
        
        def kill_process(process):
            import signal
            process.stdout.close()
            process.send_signal(signal.SIGINT)
            process.wait()
            # sys.exit(0)
            
        if password != None:
            sudo = True
            
        if sudo:
            assert isinstance(password, str)
            command = f'sudo {command}'
            
            
        process = subprocess.Popen(shlex.split(command),
                                    stdout=subprocess.PIPE, 
                                    # stderr=subprocess.PIPE, 
                                    env={**os.environ, **env}, **kwargs)

        if password:
            raise NotImplementedError
            
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
                        
                        cls.print(new_line.decode(), color=color)
                    new_line = b''
                    continue
                
                new_line += c
  
        except KeyboardInterrupt:
            pass
            
            
             
        kill_process(process)
        return stdout_text


    shell = cmd = run_command
    @classmethod
    def import_module(cls, import_path:str) -> 'Object':
        from importlib import import_module

        return import_module(import_path)


    @classmethod
    def import_object(cls, key:str, verbose: bool = False)-> 'Object':
        
        '''
        
        Import an object from a string with the format of 
            {module_path}.{object}
        
        Examples:
            import_object("torch.nn"): imports nn from torch
        
        '''
        from importlib import import_module
        module = '.'.join(key.split('.')[:-1])
        object_name = key.split('.')[-1]
        if verbose:
            cls.print(f'Importing {object_name} from {module}')
        obj =  getattr(import_module(module), object_name)
        return obj
    
    get_object = importobj = import_object
    

    
    @classmethod
    def module_list(cls, search=None)-> List[str]:
        '''
        List of module paths with respect to module.py file
        
        Assumes the module root directory is the directory containing module.py
        '''
        module_list = list(cls.module_tree().keys())
        if search:
            module_list = [m for m in module_list if search in m]
    
        return module_list

    @classmethod
    def port_used(cls, port: int, ip: str = '0.0.0.0', timeout: int = 1):
        import socket
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Set the socket timeout
            sock.settimeout(timeout)

            # Try to connect to the specified IP and port
            try:
                sock.connect((ip, port))
                return True
            except socket.error:
                return False
    
    @classmethod
    def port_free(cls, *args, **kwargs) -> bool:
        return not cls.port_used(*args, **kwargs)

    @classmethod
    def port_available(cls, port:int, ip:str ='0.0.0.0'):
        return not cls.port_used(port=port, ip=ip)
        

    @classmethod
    def used_ports(cls, ports:List[int] = None, ip:str = '0.0.0.0', port_range:Tuple[int, int] = None):
        '''
        Get availabel ports out of port range
        
        Args:
            ports: list of ports
            ip: ip address
        
        '''
        port_range = cls.resolve_port_range(port_range=port_range)
        if ports == None:
            ports = list(range(*port_range))
        
        async def check_port(port, ip):
            return cls.port_used(port=port, ip=ip)
        
        used_ports = []
        jobs = []
        for port in ports: 
            jobs += [check_port(port=port, ip=ip)]
                
        results = cls.gather(jobs)
        for port, result in zip(ports, results):
            if isinstance(result, bool) and result:
                used_ports += [port]
            
        return used_ports
    

    get_used_ports = used_ports
   
    @classmethod
    def resolve_path(cls, path:str, extension:Optional[str]= None, root:bool = False):
        '''
        Resolves path for saving items that relate to the module
        
        The path is determined by the module path 
        
        '''
        
        tmp_dir = c.tmp_dir() if root else cls.tmp_dir()
        
        
        if path.startswith('/'):
            return path
        elif path.startswith('~/'):
            return os.path.expanduser(path)
        elif path.startswith('./'):
            return os.path.abspath(path)
        else:
            if tmp_dir not in path:
                path = os.path.join(tmp_dir, path)
            if not os.path.isdir(path):
                if extension != None and extension != path.split('.')[-1]:
                    path = path + '.' + extension

            return path
    
    @classmethod
    def get_address(cls, module, **kwargs):
        return cls.namespace(**kwargs).get(module, None)
    
    @classmethod
    def get_available_ports(cls, port_range: List[int] = None , ip:str =None) -> int:
        port_range = cls.resolve_port_range(port_range)
        ip = ip if ip else cls.default_ip
        
        available_ports = []
        # return only when the port is available
        for port in range(*port_range): 
            if not cls.port_used(port=port, ip=ip):
                available_ports.append(port)
                
                
        return available_ports
    available_ports = get_available_ports
    
    
    @staticmethod
    def scan_ports(host=None, start_port=1, end_port=50000):
        if host == None:
            host = c.external_ip()
        import socket
        open_ports = []
        for port in range(start_port, end_port + 1):  # ports from start_port to end_port
            if c.port_used(port=port, ip=host):
                open_ports.append(port)
        return open_ports

    @classmethod
    def resolve_port(cls, port:int=None, **kwargs):
        
        '''
        
        Resolves the port and finds one that is available
        '''
        if port == None or port == 0:
            port = cls.free_port(port, **kwargs)
            
        if cls.port_used(port):
            port = cls.free_port(port, **kwargs)
            
        return port
    
    @classmethod
    def free_ports(cls, n=10 ) -> List[int]:
        free_ports = []
        for i in range(n):
            free_ports += [cls.free_port(reserve=True, random_selection=False)]
            
        cls.unreserve_ports(free_ports)    
        return free_ports
    
    @classmethod
    def random_port(cls):
        return cls.choice(cls.free_ports())
    
    @classmethod
    def ports(cls, ip='0.0.0.0') -> List[int]:
        ports = []
        for port in range(*cls.port_range()): 
            ports += [port]
                
        return ports
    
    @classmethod
    def used_ports(cls, ip='0.0.0.0') -> List[int]:
        used_ports = []
        for port in range(*cls.port_range()): 
            if not cls.port_available(port=port, ip=ip):
                used_ports += [port]
                
        return used_ports
    
    @classmethod
    def free_port(cls, 
                  ports = None,
                  port_range: List[int] = None , 
                  ip:str =None, 
                  avoid_ports = None,
                  reserve:bool = False, 
                  random_selection:bool = True) -> int:
        
        '''
        
        Get an availabldefe port within the {port_range} [start_port, end_poort] and {ip}
        '''
        avoid_ports = avoid_ports if avoid_ports else []
        
        if ports == None:
            port_range = cls.resolve_port_range(port_range)
            ports = list(range(*port_range))
            
            
            
        ip = ip if ip else cls.default_ip

        if random_selection:
            ports = cls.shuffle(ports)
            
        reserved_ports = cls.reserved_ports()
        # return only when the port is available
        
        port = None
        for port in ports: 
            if port in reserved_ports:
                continue
            if port in avoid_ports:
                continue
            
            if cls.port_available(port=port, ip=ip):
                if reserve:
                    cls.reserve_port(port)
                return port
        
    
    

        raise Exception(f'ports {port_range[0]} to {port_range[1]} are occupied, change the port_range to encompase more ports')

    get_available_port = free_port

    
    def kwargs2attributes(self, kwargs:dict, ignore_error:bool = False):
        for k,v in kwargs.items():
            if k != 'self': # skip the self
                # we dont want to overwrite existing variables from 
                if not ignore_error: 
                    assert not hasattr(self, k)
                setattr(self, k)

    @classmethod
    def kill_port(cls, port:int, mode='bash')-> str:
        
        port2module = cls.port2module()
        if port in port2module:

            cls.kill(port2module[port])
        
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
                        print(f'killed {port}')
            return port
        elif mode == 'bash':
            return cls.run_command('kill -9 $(lsof -ti:{port})')

    @classmethod
    def kill_server(cls, module:str,mode:str = 'pm2'):
        '''
        Kill the server by the name
        '''
        server_info = cls.get_server_info(module)
        if 'external_ip' in server_info:
            server_info.get('external_ip') == cls.external_ip()
        if isinstance(module, int) or mode == 'local':
            cls.kill_port(server_info['port'])
        if mode == 'pm2':
            cls.pm2_kill(module)
        else:
            raise NotImplementedError(f"Mode: {mode} is not implemented")
        
        cls.update_local_namespace()

    @classmethod
    def restart_server(cls, module:str, mode:str = 'pm2'):
        '''
        Kill the server by the name
        '''
        server_info = cls.get_server_info(module)
        if 'external_ip' in server_info:
            assert server_info.get('external_ip') == cls.external_ip()
        if mode == 'pm2':
            return cls.pm2_restart(module)
        else:
            raise NotImplementedError(f"Mode: {mode} is not implemented")

    @staticmethod
    def kill_all_servers( verbose: bool = True):
        '''
        Kill all of the servers
        '''
        for module in c.servers():
            if verbose:
                c.print(f'Killing {module}', color='red')
            c.kill_server(module)
            
    
    @classmethod
    def kill_all(cls, search):
        for module in cls.modules():
            if search in module:
                cls.kill(module)


    @classmethod
    def restart_all_servers(cls, verbose: bool = True):
        '''
        Kill all of the servers
        '''
        for module in cls.servers():
            if verbose:
                cls.print(f'Restarting {module}', color='red')
            cls.restart_server(module)
    @classmethod
    def restart_all(cls):
        cls.restart_all_servers()

    @classmethod
    def path_config_exists(cls, path:str) -> bool:
        '''
        Checks if the path exists
        '''
        for ext in ['.yaml', '.yml']:
            if os.path.exists(path.replace('.py', ext)):
                return True
        return False
    @classmethod
    def path2simple(cls, path:str, compress:bool = True) -> str:

        # does the config exist

        simple_path =  path.split(deepcopy(cls.root_dir))[-1]

        if cls.path_config_exists(path):
            simple_path = os.path.dirname(simple_path)

        simple_path = simple_path.replace('.py', '')
        
        
        simple_path = simple_path.replace('/', '.')[1:]
        if compress:
            simple_path = cls.compress_name(simple_path, seperator='.')
        simple_path.replace('modules.', '')
        return simple_path
    
    @classmethod
    def get_module_python_object_paths(cls):
        module_python_paths = cls.get_module_python_paths()
        module_python_object_paths = []
        for path in module_python_paths:
            module_python_object_paths  += [cls.root_dir+'.'+path.split(cls.root_dir)[-1][1:].replace('/', '.').replace('.py', '')]
            
        return module_python_object_paths
            
    @staticmethod
    def compress_name( name, seperator='.'):
        '''
        
        '''
        chunks = name.split(seperator)
        new_chunks = []
        for i, chunk in enumerate(chunks):
            if len(new_chunks)>0:
                if new_chunks[-1] == chunks[i]:
                    continue
            new_chunks.append(chunk)
            
        return seperator.join(new_chunks)
    
    
    @classmethod
    def path2localpath(cls, path:str) -> str:
        local_path = path.replace(cls.repo_path, cls.root_dir)
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
        return cls.path2objectpath(cls.module_file())
    
    @classmethod
    def object_path(cls):
        return cls.path2objectpath(cls.module_path(simple=False))
    
    @classmethod
    def object_module_path(cls):
        return '.'.join(cls.object_path().split('.')[:-1])
    
    
    @classmethod
    def __object_name__(cls):
        return '.'.join(cls.object_path().split('.')[:-1])


    @classmethod
    def find_python_classes(cls, path:str = None, class_index=0, search = None):
        import re
        
        if path is None:
            path = cls.filepath()
        # read the contents of the Python script file
        python_script = cls.get_text(path)
        class_names  = []
        lines = python_script.split('\n')
        
        for line in lines:

            key_elements = ['class ', '(', '):']
            self_ref_condition = 'key_elements' not in line

            has_class_bool = all([key_element in line for key_element in key_elements])
            other_exceptions = ['ModuleWrapper' in line, 'key_elements' in line]
            has_exception = any([exception for exception in other_exceptions])
            if has_class_bool and (not has_exception):
                if  search != None:
                    if isinstance(search, str):
                        search = [search]
                    if not any([s in line for s in search]):
                        continue
                        
                class_name = line.split('class ')[-1].split('(')[0].strip()
                class_names.append(class_name)
                
        # return the class names
        return class_names
    
    

    @classmethod
    def path2objectpath(cls, path:str) -> str:
        if path.endswith('module/module.py'):
            return 'commune.Module'
            
        object_name = cls.find_python_classes(path)
        if len(object_name) == 0:
            return None
        object_name = object_name[-1]
        path = path.replace(cls.repo_path+'/', '').replace('.py','.').replace('/', '.') 
        path = path + object_name
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
    def simple2objectpath(cls, path:str) -> str:
        path = cls.simple2path(path)
        object_path = cls.path2objectpath(path)
        return object_path
    @classmethod
    def get_module(cls, path:str, verbose:bool = False, handle_error:bool=True) -> str:
        
        og_path = path
        path = cls.simple2path(path)
        path = cls.path2objectpath(path)
        return cls.import_object(path)

    @classmethod
    def module_tree(cls, search=None, mode='path') -> List[str]:
        assert mode in ['path', 'object']
        module_tree = {}
        if mode == 'path':
            module_tree = {cls.path2simple(f):f for f in cls.get_module_python_paths()}

        elif mode == 'object':
            module_tree = {cls.path2simple(f):cls.path2objectpath(f) for f in cls.get_module_python_paths()}
            
        module_tree = {k:v for k,v in module_tree.items() if search is None or search in k}
        
        if cls.root_module_class in module_tree:
            module_tree[cls.module_path()] = module_tree.pop(cls.root_module_class)

        for k in c.copy(list(module_tree.keys())):
            if k.startswith('modules.'):
                module_tree[k.replace('modules.', '')] = module_tree.pop(k)
        return module_tree
    
    tree = module_tree
    leaves = lsm = module_list
    @classmethod
    def list_modules(cls, search=None):
        modules = list(cls.module_tree(search).keys())
        return modules
    
    @classmethod
    def modules(cls, *args, **kwargs) -> List[str]:
        modules = list(cls.namespace(*args, **kwargs).keys())
        # sorted(modules)
        return modules
    
    @classmethod
    def has_modules(cls, *args, **kwargs):
        return bool(len(cls.modules(*args, **kwargs)) > 0)
        
    
    @classmethod
    def valid_module(cls,module,**kwargs ):
        modules = cls.modules(module, **kwargs)
        return bool(len(modules) > 0)
    
    @classmethod
    def tasks(cls, task = None, mode='pm2',**kwargs) -> List[str]:
        kwargs['network'] = 'local'
        kwargs['update'] = False
        modules = cls.modules( **kwargs)
        tasks = getattr(cls, f'{mode}_list')(task)
        tasks = list(filter(lambda x: x not in modules, tasks))
        return tasks
    @classmethod
    def models(cls, *args, **kwargs) -> List[str]:
        models = cls.modules(*args, **kwargs)
        models = [k for k in models if k.startswith('model')]
        return models
    @classmethod
    def datasets(cls, *args, **kwargs) -> List[str]:
        return [k for k in list(cls.namespace(*args, **kwargs).keys()) if k.startswith('dataset')]
    
    @classmethod
    def datasets(cls, *args, **kwargs) -> List[str]:
        return [k for k in list(cls.namespace(*args, **kwargs).keys()) if k.startswith('dataset')]
    @staticmethod
    def module_config_tree() -> List[str]:
        return [f.replace('.py', '.yaml')for f in  c.get_module_python_paths()]

    
    @staticmethod
    def is_imported(package:str) :
        return  bool(package in sys.modules)

    @classmethod
    def simple2path_map(cls) -> Dict[str, str]:
        return {cls.path2simple(f):f for f in cls.get_module_python_paths()}
    @classmethod
    def simple2path(cls, path) -> Dict[str, str]:
        module_tree = cls.module_tree()
        return module_tree[path]

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

        for f in glob(c.root_path + '/**/*.py', recursive=True):
            if os.path.isdir(f):
                continue
            file_path, file_ext =  os.path.splitext(f)
            if file_ext == '.py':
                has_config = any([os.path.exists(file_path+'.'+ext) for ext in ['yaml', 'yml']])
                if has_config:
                    modules.append(f)
                else:
                    # FIX ME
                    f_classes = cls.find_python_classes(f, search=['commune.Module', 'c.Module'])
                    
                    if len(f_classes) > 0:
                        modules.append(f)
        cls.module_python_paths = modules
        
        return modules

    @classmethod
    def dashboard(cls, *args, **kwargs):
        return cls.get_module('dashboard')(*args, **kwargs)
    @staticmethod
    def get_module_config_paths() -> List[str]:
        return [f.replace('.py', '.yaml')for f in  c.get_module_python_paths()]


    @classmethod
    def is_parent(cls, parent=None):
        parent = c if parent == None else parent
        return bool(parent in cls.get_parents(cls))

    @classmethod
    def run_python(cls, path:str, interpreter:str='python3'):
        cls.run_command(f'{interpreter} {path}')
    @classmethod
    def python(cls, *cmd, interpreter:str='python3'):
        cmd = ' '.join(cmd)
        cls.run_command(f'{interpreter} {cmd}')

    @classmethod
    def timer(cls, *args, **kwargs):
        from commune.utils.time import Timer
        return Timer(*args, **kwargs)
    
    @classmethod
    def locals2kwargs(cls,
                      locals_dict:dict,
                      include_args:bool=True) -> dict:
        kwargs = {}
        locals_dict = locals_dict if locals_dict != None else {}
        assert isinstance(locals_dict, dict)
        kwargs.update(locals_dict)
        kwargs.update(locals_dict.get('kwargs', {}))
        kwargs.pop('cls', None)
        kwargs.pop('self', None)

        if include_args == False:
            args = locals_dict.pop('args', [])
            return dict(kwargs=kwargs, args=args)
        
        return kwargs
    

    get_kwargs = get_params = locals2kwargs 
        
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
        return os.path.expanduser(f'~/.{cls.library_name}/{cls.module_path()}')

    ############ JSON LAND ###############



        
    @classmethod
    def get_json(cls, *args, **kwargs):
        loop = cls.get_event_loop()
        return loop.run_until_complete(cls.async_get_json(*args, **kwargs))
    @classmethod
    async def async_get_json(cls,
                             path:str,
                             default=None,
                             root: bool = False,
                             **kwargs):

        from commune.utils.dict import async_get_json
        path = cls.resolve_path(path=path, extension='json', root=root)
        try:
            data = await async_get_json(path, **kwargs)
        except Exception as e:
            cls.print(f'Failed to load json from {path} with error {e}')
            return default
        if data == None:
            data = {}
        if 'data' in data and 'meta' in data:
            data = data['data']
        
        return data

    load_json = get_json

    @classmethod
    def put_torch(cls, path:str, data:Dict, root:bool = False,  **kwargs):
        import torch
        path = cls.resolve_path(path=path, extension='pt', root=root)
        torch.save(data, path)
        return path
    
    @classmethod
    def get_torch(cls,path:str, root:bool = False, **kwargs):
        import torch
        path = cls.resolve_path(path=path, extension='pt', root=root)
        return torch.load(path)
    
    @classmethod
    def put_json(cls,*args,**kwargs) -> str:
        loop = cls.get_event_loop()
        return loop.run_until_complete(cls.async_put_json(*args, **kwargs))
    
    
    
    @classmethod
    async def async_put_json(cls, 
                 path:str, 
                 data:Dict, 
                 meta = None,
                 root: bool = False,
                 **kwargs) -> str:
        
        from commune.utils.dict import async_put_json
        if meta != None:
            data = {'data':data, 'meta':meta}
        path = cls.resolve_path(path=path, extension='json', root=root)
        # cls.lock_file(path)
        await async_put_json(path=path, data=data, **kwargs)
        # cls.unlock_file(path)
        return path
    
    save_json = put_json
    
    @classmethod
    def file_exists(cls, path:str, extension = 'json', root:bool = False)-> bool:
        path = cls.resolve_path(path=path, extension=extension, root=root)
        return os.path.exists(path)

        

    
    
    exists_json = file_exists

    @classmethod
    def rm_json(cls, path=None, root:bool = False):
        from commune.utils.dict import rm_json

        if path in ['all', '**']:
            return [cls.rm_json(f) for f in cls.glob(files_only=False)]
        
        path = cls.resolve_path(path=path, extension='json', root=root)

        return rm_json(path )
    
    @classmethod
    def rmdir(cls, path, root:bool = False):
        import shutil
        return shutil.rmtree(path)

    @classmethod
    def isdir(cls, path, root:bool = False):
        return os.path.isdir(path)
    
    @classmethod
    def rm(cls, path, root:bool = False):
        if not os.path.exists(path):
            path = cls.resolve_path(path=path, extension=None, root=root)
        assert os.path.exists(path)
        if os.path.isdir(path):
            return cls.rmdir(path)
        return os.remove(path)
    @classmethod
    def glob(cls,  path ='~/', files_only:bool = True, root:bool = False):
        
        path = cls.resolve_path(path, extension=None, root=root)
        
        if os.path.isdir(path):
            path = os.path.join(path, '**')
            
        paths = glob(path, recursive=True)
        
        if files_only:
            paths =  list(filter(lambda f:os.path.isfile(f), paths))
        return paths
         
    @classmethod
    def ls_json(cls, path:str = '', recursive:bool = True):
        return [os.path.basename(p).replace('.json', '')for p in cls.ls(path, recursive=recursive)]
    

    @classmethod
    def ls(cls, path:str = '', 
           recursive:bool = False,
           root:bool = False,
           return_full_path:bool = True):
        path = cls.resolve_path(path, extension=None, root=root)
        try:
            ls_files = cls.lsdir(path) if not recursive else cls.walk(path)
        except FileNotFoundError:
            return []
        if return_full_path:
            ls_files = [os.path.expanduser(os.path.join(path,f)) for f in ls_files]
        return ls_files
    
    @classmethod
    def lsdir(cls, path:str) -> List[str]:
        if path.startswith('~'):
            path = os.path.expanduser(path)
        return os.listdir(path)

    @classmethod
    def walk(cls, path:str) -> List[str]:
        import os
        path_map = {}
        for root, dirs, files in os.walk(path):
            for f in files:
                path = os.path.join(root, f)
                path_map[path] = f
        return list(path_map.keys())
    
       
    @classmethod
    def bt(cls, *args, **kwargs):
        return cls.get_module('bittensor')(*args, **kwargs)
    @classmethod
    def __str__(cls):
        return cls.__name__

    @classmethod
    def get_server_info(cls,name:str) -> Dict:
        return cls.local_namespace().get(name, {})

    @classmethod
    def connect(cls, *args, **kwargs):
        
        return_future = kwargs.pop('return_future', False)
        loop = kwargs.get('loop', cls.get_event_loop())
        future = cls.async_connect(*args, **kwargs)
        if return_future:
            return future
        else:
            
            return loop.run_until_complete(future)
       
    @classmethod
    async def async_connect(cls, 
                name:str=None, 
                ip:str=None, 
                port:int=None , 
                network : str = 'global',
                namespace = None,
                virtual:bool = True, 
                wait_for_server:bool = False,
                trials = 3, 
                verbose: bool = False, 
                ignore_error:bool = False,
                **kwargs ):
    
        if (name == None and ip == None and port == None):
            return cls.root_module()
            
        if wait_for_server:
            cls.wait_for_server(name)
        
        if namespace == None :
            namespace = cls.namespace(network, update=False)
        namespace = cls.copy(namespace)

        # local namespace  



        if isinstance(name, str):
      
            found_modules = []

            if cls.is_address(name):
                found_modules = [name]
            
            else:
                modules = list(namespace.keys())
                module_addresses = list(namespace.values())
                for n in modules + module_addresses:
                    if name == n:
                        # we found the module
                        found_modules = [n]
                        break
                    elif name in n:
                        # get all the modules lol
                        found_modules += [n]
                      
            if len(found_modules)>0:
                name = cls.choice(found_modules)
                name = namespace.get(name, name)
                
            else:
                if ignore_error:
                    return None
                raise ValueError(f'Could not find module {name} in namespace {list(namespace.keys())}')
            

            port = int(name.split(':')[-1])

                
            ip = name.split(':')[0]

        assert isinstance(port, int) , f'Port must be specified as an int inputs({name}, {ip}, {port})'
        assert isinstance(ip, str) , 'IP must be specified as a string,inputs({name}, {ip}, {port})'
        if verbose:
            cls.print(f'Connecting to {name} on {ip}:{port}', color='yellow')
        client= cls.get_client(ip=ip, port=int(port), virtual=virtual)
        
        return client
     
    @classmethod
    def root_module(cls, name:str='module',
                    timeout:int = 100, 
                    sleep_interval:int = 1,
                    return_info = False,
                    refresh:bool = False,
                    **kwargs):
        # if not cls.server_exists(name) or refresh:
        #     cls.launch(name=name, **kwargs)
        #     cls.wait_for_server(name, timeout=timeout, sleep_interval=sleep_interval)
        module = cls.connect(name)
        if return_info:
            return module.server_info
        return module
    

    @staticmethod
    def round(x:Union[float, int], sig: int=6, small_value: float=1.0e-9):
        import math
        """
        Rounds x to the number of {sig} digits
        :param x:
        :param sig: signifant digit
        :param small_value: smallest possible value
        :return:
        """
        x = float(x)
        return round(x, sig - int(math.floor(math.log10(max(abs(x), abs(small_value))))) - 1)

    @classmethod
    def root_address(cls, name:str='module',
                    timeout:int = 100, 
                    sleep_interval:int = 1,
                    return_info = False,
                    refresh:bool = False,
                    **kwargs):
        if not cls.server_exists(name) or refresh:
            cls.launch(name=name, **kwargs)
            cls.wait_for_server(name, timeout=timeout, sleep_interval=sleep_interval)
       
        address =  cls.namespace('local')[name]
        address = address.replace(cls.default_ip,cls.external_ip())
        return address
    
    
    addy = root_address
    anchor = root_module
    anchor_address = root_address

 
    
    @classmethod
    def connect_pool(cls, modules=None, *args, return_dict:bool=False, **kwargs):
        if modules == None:
            modules = cls.modules(modules)
        
        module_clients =  cls.gather([cls.async_connect(m, ignore_error=True,**kwargs) for m in modules])
        if return_dict:
            return dict(zip(modules, module_clients))
        return module_clients

    
    @classmethod
    def get_client(cls, *args, virtual:bool = True, **kwargs):
        client_class = cls.import_object('commune.server.client.Client')
        client = client_class(*args, **kwargs)
        if virtual:
            return client.virtual()
        else:
            return client
    
   
    nest_asyncio_enabled : bool = False
    @classmethod
    def nest_asyncio(cls):
        assert not cls.nest_asyncio_enabled, 'Nest Asyncio already enabled'
        import nest_asyncio
        nest_asyncio.apply()
        nest_asyncio_enabled = True
        
        
    @classmethod
    def get_peer_addresses(cls, ip:str = None  ) -> List[str]:
        used_local_ports = cls.get_used_ports() 
        if ip == None:
            ip = cls.default_ip
        peer_addresses = []
        for port in used_local_ports:
            peer_addresses.append(f'{ip}:{port}')
            
        return peer_addresses
            
    

    @classmethod
    def update_local_namespace(cls) -> None:
        local_namespace = cls.local_namespace(update=True)

    @classmethod
    def port2module(cls, *args, **kwargs):
        namespace = cls.namespace(*args, **kwargs)
        port2module =  {}
        for name, address in namespace.items():
            port = int(address.split(':')[1])
            port2module[port] = name
        return port2module
    port2name = port2module
    
    @classmethod
    def module2port(cls, *args, **kwargs):
        port2module = cls.port2module(*args, **kwargs)
        return {v:k for k,v in port2module.items()}
    name2port = m2p = module2port
    

    @classmethod
    def address2module(cls, *args, **kwargs):
        namespace = cls.namespace(*args, **kwargs)
        port2module =  {}
        for name, address in namespace.items():
            port2module[address] = name
        return port2module
    address2name = address2module
        
        
    @classmethod
    def remote_namespace(cls,  
                         seperator = '::', 
                         verbose: bool = False, 
                         update:bool = False,
                         prefix:bool = 'R')-> dict:
    
        if update:
            remote_namespace = {}
        else:
            remote_namespace = c.get('remote_namespace', {})   
        
        remote_modules = c.get('remote_modules', {})
        remote_namespace.update(remote_modules)

        peer_registry = cls.peer_registry(update=update)  
        
        for peer_id, (peer_address, peer_info) in enumerate(peer_registry.items()):
            
            if isinstance(peer_info, dict):
                peer_name = f'{prefix}{peer_id}'
                peer_namespace = peer_info.get('namespace', None)
                if isinstance(peer_namespace, dict):
                    for name, address in peer_namespace.items():
                        remote_namespace[name+seperator+peer_name] = address
                else:
                    cls.print(f'Peer {peer_name} has no namespace', color='red')
        
        c.put('remote_namespace', remote_namespace)
        
        

        return remote_namespace
        
        
    @staticmethod
    def check_response(x) -> bool:
        if isinstance(x, dict) and 'error' in x:
            return False
        else:
            return True
        
    @classmethod
    def local_namespace(cls, update:bool = False, verbose:bool = False)-> dict:
        '''
        The module port is where modules can connect with each othe.
        When a module is served "module.serve())"
        it will register itself with the local_namespace dictionary.
        '''
        # from copy import deepcopy
        

        address2module = {}

        if update:

            peer_registry = {}
            peer_addresses = cls.get_peer_addresses()  
            async def async_get_peer_name(peer_address):
                peer = await cls.async_connect(peer_address, namespace={}, timeout=5, virtual=False, ignore_error=True)
                if peer == None: 
                    return peer
                module_name =  await peer(fn='getattr', args=['module_name'], return_future=True)
                if verbose:
                    cls.print('Connecting: ',module_name, color='cyan')

                if cls.check_response(module_name):
                    return module_name
                else:
                    return None
                
            # print(namespace)
            
            peer_names = [async_get_peer_name(p) for p in peer_addresses]
            peer_names = cls.gather(peer_names)
            local_namespace = dict(zip(peer_names, peer_addresses))
            
            local_namespace = {p_n:p_a for p_n, p_a in local_namespace.items() if p_n != None}
            
            cls.save_json('local_namespace', local_namespace, root=True)
            
        else:
            local_namespace = cls.__dict__.get('_local_namespace', None)
            if local_namespace == None:
                local_namespace = cls.get_json('local_namespace', {}, root=True)
                
        external_ip = cls.external_ip()
        local_namespace = {k:cls.default_ip + f":{v.split(':')[-1]}" for k,v in local_namespace.items()}
        return local_namespace

        

    @classmethod
    def servers(cls, search:str = None, update=False, **kwargs) -> List[str]:

        servers = list(cls.local_namespace(update=update,**kwargs ).keys())
        if search: 
            servers = [s for s in servers if search in s]
        return servers


    list_servers = servers
    
    
    @classmethod
    def rename_server(cls, name:str, new_name:str) -> Dict:
        local_namespace = cls.local_namespace()
        local_namespace[new_name] = local_namespace.pop(name)
        cls.put_json(path='local_namespace', data=local_namespace, root=True) 
        return {new_name:local_namespace[new_name]}
    
    rename = rename_module = rename_server
    
    
    
    @classmethod
    def lock_file(cls, f):
        import fcntl
        fcntl.flock(f, fcntl.LOCK_EX)
        return f
    @classmethod
    def unlock_file(cls, f):
        import fcntl
        fcntl.flock(f, fcntl.LOCK_UN)
        return f
    
    
    @classmethod
    def register_server(cls, name: str, ip: str,port: int, **kwargs)-> dict:
        local_namespace = cls.local_namespace()    
        
        local_namespace[name] = f'{ip}:{port}'
        cls.put_json('local_namespace', local_namespace, root=True) 
        return local_namespace
    
    @classmethod
    def deregister_server(cls, name: str)-> dict:
        local_namespace = cls.local_namespace()    
        
        local_namespace.pop(name, None)
        cls.put_json('local_namespace', local_namespace, root=True) 
        return local_namespace
  
  
    @classmethod
    def is_address(cls, address:str) -> bool:
        conds = []
        
        conds.append(isinstance(address, str))
        conds.append(':' in address)
        conds.append(cls.is_number(address.split(':')[-1]))
    
        return all(conds)
    
    @classmethod
    def is_module(cls, obj=None) -> bool:
        
        if obj is None:
            obj = cls
        if all([hasattr(obj, k) for k in ['module_class', 'root_module_class']]):
            module_class = obj.module_class()
            return True
            
        return False
    @classmethod
    def is_root_module(cls, obj=None) -> bool:
        
        if obj is None:
            obj = cls
        if hasattr(obj, 'module_class'):
            module_class = obj.module_class()
            if module_class == cls.root_module_class:
                return True
            
        return False
    is_root = is_module_root = is_root_module
    @classmethod
    def new_event_loop(cls, nest_asyncio:bool = True) -> 'asyncio.AbstractEventLoop':
        import asyncio
        if nest_asyncio:
            cls.nest_asyncio()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


        return loop
  

    def set_event_loop(self, loop=None, new_loop:bool = False) -> 'asyncio.AbstractEventLoop':
        import asyncio
        try:
            if new_loop:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            else:
                loop = loop if loop else asyncio.get_event_loop()
        except RuntimeError as e:
            self.new_event_loop()
            
        self.loop = loop
        return self.loop

    @classmethod
    def get_event_loop(cls, nest_asyncio:bool = True) -> 'asyncio.AbstractEventLoop':
        import asyncio
        if nest_asyncio:
            cls.nest_asyncio()
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = cls.new_event_loop()

        return loop

    @classmethod
    def server_exists(cls, name:str) -> bool:
        return bool(name in cls.servers())
    
    @classmethod
    def get_port(cls, port:int = None, **kwargs)->int:
        port = port if port is not None and port != 0 else cls.free_port(**kwargs)
        while cls.port_used(port):
            port += 1   
        return port 
    
    resolve_port = get_port
    
    @classmethod
    def exists(cls, name:str, **kwargs) -> bool:
        namespace = cls.namespace(**kwargs)
        return bool(name in namespace)
    
    @classmethod
    def module_exists(cls, name:str) -> bool:
        return bool(name in cls.module_list())
    
    
    
    @classmethod
    def wait_for_server(cls,
                          name: str ,
                          timeout:int = 600,
                          sleep_interval: int = 4) -> bool :
        
        start_time = cls.time()
        time_waiting = 0
        while not cls.server_exists(name):
            cls.sleep(sleep_interval)
            time_waiting += sleep_interval
            cls.print(f'Waiting for server {name} to start... {time_waiting} seconds', end='\r')

            if time_waiting > timeout:
                raise TimeoutError(f'Timeout waiting for server to start')
        return True
    def server_running(self):
        return hasattr(self, 'server_info')

    def stop_server(self):
        self.server.stop()
        del self.server
        del self.server_info
        
        
        
    @classmethod
    def get_streamlit(cls):
        import streamlit as st
        return st 
    
    
    
    def attributes(self):
        return list(self.__dict__.keys())
    @classmethod
    def get_attributes(cls, obj=None):
        if obj is None:
            obj = cls
        if isinstance(obj, str):
            obj = cls.module(obj)
        # assert hasattr(obj, '__dict__'), f'{obj} has no __dict__'
        return list(obj.__dict__.keys())


    @classmethod
    def global_namespace(cls, update=False) -> Dict:
        
        global_namespace = {
            **cls.local_namespace(update=update),
            **cls.remote_namespace(update=update)
        }
        
        return global_namespace
        
    
        
    @classmethod
    def namespace(cls,
                  search = None,
                  network:str='global',
                  verbose: bool = False,
                  update: bool = False,
                  max_staleness:int = 30,
                  **kwargs):
        
        if isinstance(search, str) :
            if hasattr(cls, f'{search}_namespace'):
                network = search
                search = None
        else:
            search = None

        namespace_fn = getattr(cls, f'{network}_namespace')
        namespace = namespace_fn(update=update, **kwargs)
        
        # namespace.update(c.get('remote_modules', {}))
        
        if search:
            namespace = {k:v for k,v in namespace.items() if str(search) in k}
        return namespace
    
    
    
    @classmethod
    def subspace_namespace(cls, search = None, update=False):
        return {}
    

    @classmethod
    def namespace_options(cls,search=None) -> List[str]:
        namespace  = cls.namespace()
        namespace_names = list(namespace.keys())
        namespace_addresses = list(namespace.values())
        namespace_options =  namespace_names + namespace_addresses
        if search:
            namespace_options = [o for o in namespace_options if search in o]
        return namespace_options
    
    
    kwargs_store = {}
    def save_kwargs(self, fn:str, kwargs:dict):
        kwargs.pop('self', None)
        self.kwargs_store[fn] = kwargs
        return self.kwargs_store
    
    
    
    @classmethod
    def serve(cls, 
              module:Any = None ,
              name:str=None, 
              ip:str=None, 
              port:int=None ,
              network:str = None,
              context= '',
              key = None,
              tag:str=None, 
              replace:bool = True, 
              whitelist:List[str] = None,
              blacklist:List[str] = None,
              wait_for_termination:bool = True,
              wait_for_server:bool = False,
              wait_for_server_timeout:int = 30,
              wait_for_server_sleep_interval: int = 1,
              verbose = False,
              reserve_port = False,
              *args, 
              **kwargs ):
        '''
        Servers the module on a specified port
        '''
        # we want to make sure that the module is loco
        cls.update(network='local')
    
        if port == None:
            port = cls.free_port(reserve=reserve_port)
        if module == None:
            self = cls(*args, **kwargs)
        elif isinstance(module, str):
            self = cls.get_module(module)(*args, **kwargs)
        else:
            self = module
             
        whitelist = whitelist if whitelist else self.whitelist()
        blacklist = blacklist if blacklist else self.blacklist()
    
        # resolve the module id
        
        # if the module is a class, then use the module_tag 
        # Make sure you have the module tag set
        if name == None:
            if hasattr(self, 'module_path'):
                name = self.module_path()
            elif hasattr(self, 'default_module_name'):
                name = self.default_module_name()
            else:
                name = self.__class__.__name__
                
        module_name = name

        '''check if the server exists'''
        cls.print(f'Checking if server {module_name} exists {self}')
        if self.server_exists(module_name): 
            if replace:
                if verbose:
                    cls.print(f'Stopping server {module_name}')
                self.kill_server(module_name)
            else: 
                raise Exception(f'The server {module_name} already exists on port {existing_server_port}')

        for k in ['module_name', 'module_id', 'my_name', 'el_namo', 'name']:
            if k not in self.__dict__:
                self.__dict__[k] = module_name

        Server = cls.import_object('commune.server.Server')
        
        self.save_kwargs('serve', locals())

        server = Server(ip=ip, 
                        port=port,
                        module = self,
                        name= module_name,
                        whitelist=whitelist,
                        blacklist=blacklist)
        
        # register the server
        self.server_info = server.info
        self.ip = server.ip
        self.port = server.port
        self.address = self.ip_address = self.ip_addy =  server.address
        
        if (not hasattr(self, 'config')) or callable(self.config):
            self.config = cls.munch({})
            
        self.config['info'] = self.info()
        

        # self.set_key(key)
            
        # serve the server
        server.serve(wait_for_termination=wait_for_termination,register=True)
        if wait_for_server:
            cls.wait_for_server(name=module_name, timeout=wait_for_server_timeout, sleep_interval=wait_for_server_sleep_interval)
        
    serve_module = serve
    @classmethod
    def functions(cls, search = None, include_module=False):
        functions = cls.get_functions(include_module=include_module)  
        
        if isinstance(search, str):
            functions = [f for f in functions if search in f]
        return functions

    fns = functions
        
    @classmethod
    def get_functions(cls, obj:Any=None, include_module:bool = False,) -> List[str]:
        '''
        List of functions
        '''
        if isinstance(obj, str):
            obj = cls.get_module(obj)
        from commune.utils.function import get_functions
        obj = obj if obj != None else cls

        
        if cls.is_root_module(obj):
            include_module = True
            
    
        functions = get_functions(obj=obj)
        
        if not include_module:
            module_functions = get_functions(obj=c)
            new_functions = []
            for f in functions:
                if f == '__init__':
                    new_functions.append(f)
                if f not in module_functions:
                    new_functions.append(f)
            functions = new_functions
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
            if fn_name in ['self', 'cls']:
                continue
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
        obj = obj if obj != None else cls
        function_schema_map = cls.get_function_schema_map(obj=obj,include_module=include_module)
        function_default_map = cls.get_function_default_map(obj=obj,include_module=include_module)
        function_info_map = {}
        for fn in function_schema_map:
            function_info_map[fn] = {
                'default':function_default_map.get(fn, 'NA'),
                **function_schema_map.get(fn, {}),
            }
            # check if the function is a class method or a static method
            
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
            
        info = peer.info()
        return info
    
    
    def is_fn_allowed(self, fn_name:str) -> bool:
        whitelist = self.whitelist()
        blacklist = self.blacklist()
        if fn_name in whitelist and fn_name not in blacklist:
            return True
        else:
            return False
        
    def info(self , 
             include_schema: bool = False,
             include_namespace:bool = True,
             include_peers: bool = True) -> Dict[str, Any]:
        whitelist = self.whitelist()
        blacklist = self.blacklist()
        fns = [ fn for fn in self.fns() if self.is_fn_allowed(fn)]
        attributes =[ attr for attr in self.attributes() if self.is_fn_allowed(attr)]
        

        info  = dict(
            address = self.address,
            functions =  fns,
            attributes = attributes,
            name = self.module_name,
        )
        if include_namespace:
            info['namespace'] = c.namespace()
        if include_schema:
            info['schema'] = self.schema()
        return info
    
    help = info



    def peer_info(self) -> Dict[str, Any]:
        self.info()
    @classmethod
    def schema(cls, search = None, *args,  **kwargs):
        return {k: v for k,v in cls.get_function_schema_map(*args,search=search,**kwargs).items()}
    @classmethod
    def get_function_schema_map(cls,
                                obj = None,
                                search = None,
                                include_code : bool = False,
                                include_hidden:bool = False, 
                                include_module:bool = False,
                                include_default:bool = False,
                                include_docs: bool = False):
        
        obj = obj if obj else cls
        if isinstance(obj, str):
            obj = cls.module(obj)
        function_schema_map = {}
        
        for fn in cls.get_functions(obj, include_module=include_module):
                    
            if search != None :
                if search not in fn:
                    continue
            if include_default:
                fn_default_map = cls.get_function_default_map(obj=obj, include_module=include_module)
            if callable(getattr(obj, fn )):
                function_schema_map[fn] = {}
                fn_schema = {}
                obj_fn = getattr(obj, fn )
                if not hasattr(obj_fn, '__annotations__'):
                    obj_fn.__annotations__ = {}
                
                for fn_k, fn_v in obj_fn.__annotations__.items():
                    

                    fn_v = str(fn_v)  

                    if fn_v == inspect._empty:
                        fn_schema[fn_k]= 'Any'
                    elif fn_v.startswith('<class'):
                        fn_schema[fn_k] = fn_v.split("'")[1]
                    else:
                        fn_schema[fn_k] = fn_v
                              
                function_schema_map[fn]['schema'] = fn_schema
                if include_docs:         
                    function_schema_map[fn]['docs']: getattr(obj, fn ).__doc__ 
                if include_code:
                    function_schema_map[fn]['code'] = inspect.getsource(getattr(obj, fn ))
                    
                if include_default:
                    function_schema_map[fn]['default'] = fn_default_map.get(fn, 'NA')
                if len(function_schema_map[fn]) == 1:
                    function_schema_map[fn] = fn_schema
                    
                    
        for k,v in function_schema_map.items():
            if search != None:
                if search == k:
                    return v
        return function_schema_map
    
    function_schema_map = get_function_schema_map
    

    @classmethod
    def get_function_schema(cls, fn:str)->dict:
        '''
        Get function schema of function in cls
        '''
        if not callable(fn):
            fn = getattr(cls, fn)
        fn_schema = {k:str(v) for k,v in fn.__annotations__.items()}
        return fn_schema
    
    def module_schema(self, 
                      
                      include_hidden:bool = False, 
                      include_module:bool = False):
        module_schema = {
            'module_name':self.module_name,
            'server':self.server_info,
            'function_schema':self.function_schema_map(include_hidden=include_hidden, include_module=include_module),
        }
        return module_schema
    @classmethod
    def function_schema(cls, fn:str)->dict:
        '''
        Get function schema of function in cls
        '''
        fn = getattr(cls, fn)
        fn_schema = {k:str(v) for k,v in fn.__annotations__.items()}
        return fn_schema

    @classmethod
    def function_info(cls, fn:str)->dict:
        '''
        Get function schema of function in cls
        '''
        fn_info  =  cls.function_info_map().get(fn , {})
        return fn_info

    @staticmethod
    def get_annotations(fn:callable) -> dict:
        return fn.__annotations__



    @classmethod
    def kill(cls, module,
             mode:str = 'pm2',
             verbose:bool = False,
             **kwargs):


        delete_modules = []
        delete_modules =  getattr(cls, f'{mode}_kill')(module, verbose=verbose, **kwargs)
       
        cls.update(network='local')

        return delete_modules

    delete = kill
    def destroy(self):
        self.kill(self.module_name)
        return path
    
    def self_destruct(self):
        self.kill(self.module_name)    
        
    def self_restart(self):
        self.restart(self.module_name)
        
    @classmethod
    def set_shortcut(cls, shortcut: str, kwargs: dict) -> dict:
        self.shortcuts = self.get_shortcuts()
        # remove shortcut if it exists
        kwargs.pop('shortcut', None)
        cls.shortcuts[shortcut] = kwargs
        self.put_json('shortcuts', cls.shortcuts)
        
        return kwargs
    
    @classmethod
    def get_shortcut(cls, shortcut:str) -> dict:
        self.shortcuts = cls.get_shortcuts()
        kwargs =  cls.shortcuts.get(shortcut, None)
        return kwargs
    
    def get_shortcuts(cls) -> dict:
        return cls.get_json('shortcuts')

    @classmethod
    def has_shortcut(cls, shortcut:str):
        return cls.get_shortcut(shortcut) != None
    
    @classmethod
    def rm_shortcut(cls, shortcut) -> str:
        shortcuts = cls.get_shortcuts()
        if shortcut in shortcuts:
            cls.shortcuts.pop(shortcut)
            cls.put_json('shortcuts', cls.shortcuts)
        return shortcut
    ## PM2 LAND
    @classmethod
    def deploy(cls, 
               module:str = None, 
               fn: str = 'serve',
               args : list = None,
               kwargs: dict = None,
               refresh:bool=True,
               mode:str = 'pm2',
               name:Optional[str]=None, 
               tag:str=None, 
               tag_seperator: str = ':',
               user: str = None,
               key : str = None,
               verbose : bool = True, 
               shortcut = None,
               wait_for_server=False,
               device = None,
               update: bool = True,
               **extra_kwargs):
        '''
        Launch a module as pm2 or ray 
        '''
        if update:
            cls.update()
        kwargs = kwargs if kwargs else {}
        args = args if args else []
        if module == None:
            module = cls 
        elif isinstance(module, str):

            module = cls.get_module(module) 
            
        if name == None:
            if hasattr(module, 'module_path'):
                name = module.module_path()
            else:
                name = module.__name__.lower()
                
        if tag != None:
            name = f'{name}{tag_seperator}{tag}'
                
                
        if verbose:
            cls.print(f'[bold cyan]Launching[/bold cyan] [bold yellow]class:{module.__name__}[/bold yellow] [bold white]name[/bold white]:{name} [bold white]fn[/bold white]:{fn} [bold white]mode[/bold white]:{mode}', color='green')
            
        if fn == 'serve':
            kwargs['tag'] = kwargs.get('tag', tag)
            kwargs['name'] = kwargs.get('name', name)
        else:
            wait_for_server = False # invalid command

        if mode == 'local':
            return getattr(module, fn)(*args, **kwargs)

        elif mode == 'pm2':
            
            launch_kwargs = dict(
                    module=module, 
                    fn = fn,
                    name=name, 
                    tag=tag, 
                    args = args,
                    kwargs = kwargs,
                    refresh=refresh,
                    device= device,
                    **extra_kwargs
            )
            

            assert fn != None, 'fn must be specified for pm2 launch'
            stdout = getattr(cls, f'{mode}_launch')(**launch_kwargs)
            
            
            if wait_for_server:
                cls.wait_for_server(name)
            
            
        elif mode == 'ray':
            launch_kwargs = dict(
                    module=module, 
                    name=name, 
                    tag=tag, 
                    args = args,
                    kwargs = kwargs,
                    refresh=refresh,
                    **extra_kwargs
            )
            if wait_for_server:
                self.wait_for_server(name)

        
            getattr(cls, f'{mode}_launch')(**launch_kwargs)
        else: 
            raise Exception(f'launch mode {mode} not supported')

        return name

    launch = deploy
    
    @classmethod
    def pm2_kill_all(cls, verbose:bool = True):
        for module in cls.pm2_list():
            cls.pm2_kill(module)
            if verbose:
                cls.print(f'[red] Killed {module}[/red]')      
                
    @classmethod
    def pm2_list(cls, search=None,  verbose:bool = False) -> List[str]:
        output_string = cls.run_command('pm2 status', verbose=False)
        module_list = []
        for line in output_string.split('\n'):
            if '│ default     │ ' in line:
                module_name = line.split('│')[2].strip()
                # fixes odd issue where there is a space between the name and the front 
                module_name = module_name.split(' ')[-1]
                module_list += [module_name]
                
        
        if search:
            if isinstance(search, str):
                search = [search]
            elif isinstance(search, list):
                pass
                assert all([isinstance(s, str) for s in search]), 'search must be a list of strings'
                
            search_true = lambda x: any([s in x for s in search])
            module_list = [m for m in module_list if search_true(m)]
                
        return module_list
    lspm2 = ls_pm2 = pm2ls = pm2_ls = pm2list = pm2_list
    # commune.run_command('pm2 status').stdout.split('\n')[5].split('    │')[0].split('  │ ')[-1]commune.run_command('pm2 status').stdout.split('\n')[5].split('    │')[0].split('  │ ')[-1] 
    
    
    
    @classmethod
    def pm2_launch(cls, 
                   module:str = None,  
                   fn: str = 'serve',
                   name:Optional[str]=None, 
                   tag:str=None, 
                   args : list = None,
                   kwargs: dict = None,
                   device:str=None, 
                   interpreter:str='python3', 
                   no_autorestart: bool = False,
                   verbose: bool = False , 
                   refresh:bool=True ):
        
        # avoid these references fucking shit up
        args = args if args else []
        kwargs = kwargs if kwargs else {}
        
        if isinstance(module, str):
            assert isinstance(module, str), f'module must be a string, not {type(module)}'
            module = cls.get_module(module)
        elif module == None:
            module = cls
            
        module_name =module.default_module_name() if name == None else name
            
        
        module_path = module.module_file()
        
        # build command to run pm2
        command = f" pm2 start {module_path} --name {module_name} --interpreter {interpreter}"
        if no_autorestart:
            command = command + ' ' + '--no-autorestart'

        # convert args and kwargs to json strings
        kwargs_str = json.dumps(kwargs).replace('"', "'")
        args_str = json.dumps(args).replace('"', "'")

        if refresh:
            cls.pm2_kill(module_name)   
        
        command = command + ' -- ' + f'--fn {fn} --kwargs "{kwargs_str}" --args "{args_str}"'
        env = {}

        if device != None:
            if isinstance(device, int):
                env['CUDA_VISIBLE_DEVICES']=str(device)
            if isinstance(device, list):
                env['CUDA_VISIBLE_DEVICES']=','.join(list(map(str, device)))
                
        if verbose:

            cls.print(f'Launching {module_name} with command: {command}', color='green')
        
        
        stdout = cls.run_command(command, env=env, verbose=verbose)
        # cls.print(f'STDOUT: \n {stdout}', color='green')
        return stdout
    
    @classmethod
    def pm2_kill(cls, name:str, verbose:bool = True, pool_match:bool = False):
        output_list = []
        pm2_list = cls.pm2_list()
        kill_list = []
        
        # check if exact match, if so kill it, if not kill all that start with name

        exact_match = any([name == module for module in pm2_list])
        
        if exact_match and not pool_match:
            # kill exact match
            if verbose:
                cls.print(f'Killing {name}', color='red')
            cls.run_command(f"pm2 delete {name}", verbose=False)
            kill_list.append(name)
            return kill_list
        else:
            # kill all modules that start with name
            for module in pm2_list:
                if module.startswith(name):
                    if verbose:
                        cls.print(f'Killing {module}', color='red')
                    cls.run_command(f"pm2 delete {module}", verbose=False)
                    kill_list.append(module)
            return kill_list
    @classmethod
    def pm2_restart(cls, name:str = None, verbose:bool=False):
        pm2_list = cls.pm2_list()
            
        restarted_modules = []
        for module in pm2_list:
            if module.startswith(name) or name in ['all']:
                if verbose:
                    cls.print(f'Restarting {module}', color='cyan')
                cls.run_command(f"pm2 restart {module}")
                restarted_modules.append(module)

            
        return restarted_modules
            
        
            
    def restart_self(self, mode:str='pm2'):
        assert hasattr(self, 'module_name'), 'self.module_name must be defined to restart'
        return self.restart(self.module_name)
    
    
    
    @classmethod
    def restart(cls, name:str, mode:str='pm2', verbose:bool = True):
        refreshed_modules = getattr(cls, f'{mode}_restart')(name, verbose=verbose)
        return refreshed_modules
    refresh = reset = restart
    @classmethod
    def pm2_status(cls, verbose=True):
        stdout = cls.run_command(f"pm2 status")
        if verbose:
            cls.print(stdout,color='green')
        return stdout

    pm2_dir = os.path.expanduser('~/.pm2')
    @classmethod
    def pm2_logs(cls, module:str,verbose=True, mode='cmd'):
        if mode == 'local':
            path = f'{cls.pm2_dir}/logs/{module}-out.log'.replace(':', '-')
            return cls.get_text(path, last_bytes=1000 )
        elif mode == 'cmd':
            return cls.run_command(f"pm2 logs {module}", verbose=verbose)
        else:
            raise NotImplementedError(f'mode {mode} not implemented')

    @classmethod
    def argparse(cls, verbose: bool = False):
        import argparse
        parser = argparse.ArgumentParser(description='Gradio API and Functions')
        parser.add_argument('-fn', '--fn', dest='function', help='run a function from the module', type=str, default="__init__")
        parser.add_argument('-kwargs', '--kwargs', dest='kwargs', help='key word arguments to the function', type=str, default="{}")  
        parser.add_argument('-args', '--args', dest='args', help='arguments to the function', type=str, default="[]")  
        args = parser.parse_args()
        if verbose:
            cls.print('Argparse Args: ',args, color='cyan')
        args.kwargs = json.loads(args.kwargs.replace("'",'"'))
        args.args = json.loads(args.args.replace("'",'"'))
        return args

    @classmethod
    def run(cls, name:str = None, verbose:bool = False) -> Any: 
        if name == '__main__' or name == None or name == cls.__name__:
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
    def learn(cls, *args, **kwargs):
        return cls.module('model.transformer').learn(*args, **kwargs)
        
    
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
    
    # @classmethod
    # def namespace(cls, data: Dict=None) -> 'Munch':
    #     data = data if data else {}
    #     assert isinstance(data, dict), f'data must be a dict, got {type(data)}'
    #     return cls.dict2munch( data)

    
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
    def get_module_name(cls, name:str=None, tag:str=None, seperator:str='.'):
        name = name if name else cls.__name__.lower()
        if tag != None:
            name = tag + seperator + name
        return name
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
            
        name = self.get_module_name(name=name, tag=tag) 
        assert isinstance(name, str)
        
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

        if cls.actor_exists(actor):
            actor = ray.get_actor(actor)
        else:
            if verbose:
                print(f'{actor} does not exist for it to be removed')
            return None
        ray.kill(actor)
    
        return True
    ray_kill = kill_actor
        
       
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
        import ray
        return ray.runtime_context.get_runtime_context()

    @staticmethod
    def ray_objects( *args, **kwargs):
        import ray
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
    def module(cls,module: Any = None ,*args, **kwargs):
        '''
        Wraps a python class as a module
        '''
        
        if module is None:
            return c.root_module()
        if isinstance(module, str):
            modules = c.module_list()
            if module in modules:
                return c.get_module(module,**kwargs)
            # elif module in cls.servers():
            #     return c.connect(module,**kwargs)
    

        # serve the module if the bool is True
        is_class = cls.is_class(module)
        module_class = module if is_class else module.__class__
        
        
        
        class ModuleWrapper(c):
            def __init__(self, module): 
                c.__init__(self, *args, **kwargs) 
                self.merge(self.module)
                
            @classmethod
            def module_file(cls): 
                return cls.get_module_path(simple=False)
            
            
            def __call__(self, *args, **kwargs):
                return self.module.__call__(self, *args, **kwargs)

            def __str__(self):
                return self.module.__str__()
            
            def __repr__(self):
                return self.module.__repr__() 
            
            @classmethod
            def default_module_name(cls) -> str:
                return module_class.__name__.lower()
 
            @classmethod
            def functions(cls):
                return cls.get_functions(module)


        if is_class:
            return ModuleWrapper
        else:
            return ModuleWrapper()
        
        
            
        # return module

    # UNDER CONSTRUCTION (USE WITH CAUTION)
    
    def setattr(self, k, v):
        setattr(self, k, v)
        
    @classmethod
    def default_module_name(cls) -> str:
        return cls.module_class().lower()

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
      

    @classmethod
    def merge(cls, *args, 
                        include_hidden:bool=True, 
                        allow_conflicts:bool=True, 
                        verbose: bool = False):
        
        '''
        Merge the functions of a python object into the current object (a)
        '''
        if len(args) == 1:
            a = cls
            b = args[0]
        elif len(args) == 2:
            a = args[0]
            b = args[1]
        else:
            raise ValueError('must have 1 or 2 arguments')
        if isinstance(a, str):
            a = cls.get_module(a)
        elif isinstance(b, str):
            b = cls.get_module(b)
        
        for b_fn_name in dir(b):
            if include_hidden == False:
                #i`f the function name starts with __ then it is hidden
                if b_fn_name.startswith('__'):
                    continue
                
            # if the function already exists in the object, raise an error
            if  allow_conflicts:
                if hasattr(a, b_fn_name):
                    if verbose:
                        cls.print(f'Warning: overriding function {b_fn_name} already exists in {a}', color='yellow')
            else:
                assert not hasattr(a, b_fn_name), f'function {b_fn_name} already exists in {a}'
                
            # get the function from the python object
            try: 
                b_fn = getattr(b, b_fn_name)
            except NotImplementedError as e:
                print(e)
            error_fn_list = []
            if callable(b_fn):
                try:
                    setattr(a, b_fn_name, b_fn)  
                except TypeError:
                    error_fn_list.append(b_fn)
                if len(error_fn_list)>0:
                    if verbose:
                        cls.print(error_fn_list, 'DEBUG')        
        return a
   
    @classmethod
    def nest_asyncio(cls):
        import nest_asyncio
        try:
            nest_asyncio.apply()
        except RuntimeError as e:
            cls.print('Broooo, nest-asyncio doesnt work fam')
            pass
        
        
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
    def pip_list(cls, lib=None):
        pip_list =  cls.cmd(f'pip list').split('\n')
        if lib != None:
            pip_list = [l for l in pip_list if l.startswith(lib)]
        return pip_list
    
    @classmethod
    def version(cls, lib:str):
        lines = [l for l in cls.cmd(f'pip list').split('\n') if l.startswith(lib)]
        if len(lines)>0:
            return lines[0].split('         ')[1].strip()
        else:
            return f'No Library Found {lib}'
    
    @classmethod
    def get_external_ip(cls, *args, **kwargs) ->str:
        return cls.import_object('commune.utils.network.get_external_ip')(*args, **kwargs)

    @classmethod
    def external_ip(cls, *args, **kwargs) -> str:
        if not hasattr(cls, '__external_ip__'):
            cls.__external_ip__ =  cls.get_external_ip(*args, **kwargs)
        return cls.__external_ip__
    
    @classmethod
    def resolve_ip(cls, ip=None) -> str:
        if ip == None:
            ip = cls.external_ip()   
        assert isinstance(ip, str)
        return ip
        
    @classmethod
    def get_external_ip(cls, *args, **kwargs) ->str:
        try:
            return cls.import_object('commune.utils.network.get_external_ip')(*args, **kwargs)
        except Exception as e:
            c.print(e, color='red')
            return cls.default_ip
            
    @classmethod
    def public_ip(cls, *args, **kwargs):
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

    env = get_env
    
    
    ### GPU LAND
    
    @classmethod
    def gpus(cls) -> List[int]:
        import torch
        available_gpus = [int(i) for i in range(torch.cuda.device_count())]
        return available_gpus
    
    @classmethod
    def num_gpus(cls):
        return len(cls.gpus())
    
    @classmethod
    def cuda_available(cls) -> bool:
        import torch
        return torch.cuda.is_available()
    @classmethod
    def gpu_memory(cls) -> Dict[int, Dict[str, float]]:
        import torch
        gpu_info = {}
        for gpu_id in cls.gpus():
            mem_info = torch.cuda.mem_get_info(gpu_id)
            gpu_info[int(gpu_id)] = {
                'name': torch.cuda.get_device_name(gpu_id),
                'free': mem_info[0],
                'used': (mem_info[1]- mem_info[0]),
                'total': mem_info[1]
            }
        return gpu_info
    
    gpu_info = gpu_memory_map = gpu_map = gpu_memory
 
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
    def least_used_gpu(cls, free_gpu_memory:dict = None, return_tuple:bool = False, **kwargs) -> int:
        """ Returns a dictionary of gpu_id to max memory for each gpu.
        Args:
            total_memory (int, optional): Total memory to allocate. Defaults to None.
            buffer_memory (int, optional): Buffer memory to leave on each gpu. Defaults to 10.
        
        Returns 
            Dict[int, str]: Dictionary of gpu_id to max memory for each gpu.
        """
        if free_gpu_memory is None:
            free_gpu_memory = cls.free_gpu_memory(**kwargs)
        assert isinstance(free_gpu_memory, dict), f'free_gpu_memory must be a dict, not {type(free_gpu_memory)}'
        most_available_gpu_tuples = sorted(free_gpu_memory.items(), key=lambda x: x[1] , reverse=True)
        if return_tuple:
            return most_available_gpu_tuples[0]
        return most_available_gpu_tuples[0][0]
    
    
    most_free_gpu = least_used_gpu
    
    @classmethod
    def most_free_gpu_memory(cls, *args, **kwargs) -> int:
        gpu_id = cls.most_free_gpu()
        return cls.free_gpu_memory(*args, **kwargs)[gpu_id]
    
    @classmethod
    def gpu_info(cls, device:int = None) -> Dict[str, Union[int, float]]:
        '''
        Get the gpu info for a given device
        '''
        if device is None:
            device = 0
        gpu_map = cls.gpu_map()
        return gpu_map[device]

    # CPU LAND
    
    @classmethod
    def cpu_count(cls):
        try:
            return len(os.sched_getaffinity(0))
        except AttributeError:
            # OSX does not have sched_getaffinity
            return os.cpu_count()

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
                cls.print(f'Using device: {device} with {device_info["free"]} GB free memory', color='yellow')
        return device  
    
    
    @classmethod
    def params_map(cls, model):
        params_map = {}
        state_dict = model.state_dict()
        for k,v in state_dict.items():
            params_map[k] = {'shape': v.shape ,
                             'size': cls.get_tensor_size(v)}
            
        return params_map
    
    @classmethod
    def params_size_map(cls, model):
        return {k: v['size'] for k,v in cls.params_map(model).items()}
    

    @classmethod
    def get_num_params(cls, model:'nn.Module' = None)->int:
        import numpy as np
        from torch import nn
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        num_params = sum([np.prod(p.size()) for p in model_parameters])
        return num_params

    get_model_params = get_num_params
    @classmethod
    def get_tensor_size(cls, tensor:'torch.Tensor'):
        return tensor.nelement() * tensor.element_size()
    @classmethod 
    def get_model_device(cls, model, fast_and_lazy:bool = True) -> 'torch.device':
        if fast_and_lazy:
            return next(model.parameters()).device
        else:
            unique_devices = set()
            for p in model.parameters():
                unique_devices.add(p.device)
            return list(unique_devices)[0]
        return next(model.parameters()).device
    
    
    @classmethod
    def update_loop(cls, period=20, remote=True):
        if remote:
            return cls.remote_fn('update_loop', kwargs=dict(period=period, remote=False), name='update_loop')
        while True:
            cls.print('Updating...', color='yellow')
            modules = cls.modules()
            cls.print(f'Modules (n): {modules}', color='cyan')
            cls.print(modules, color='purple')
            cls.update()
            cls.sleep(period)
    def model_size(self, **kwargs ):
        return self.get_model_size(model=self, **kwargs)
    
    
    @classmethod
    def get_empty_model(cls, model, verbose: bool = False, trust_remote_code:bool=True, **kwargs):
        from transformers import  AutoModelForCausalLM, AutoModel, AutoConfig
        from accelerate import init_empty_weights
        
        kwargs['trust_remote_code'] = trust_remote_code
        model = cls.module('model.transformer').shortcuts.get(model, model)

        if isinstance(model, str):
            if verbose:
                cls.print(f'loading config model from {model}...')

            model_config = AutoConfig.from_pretrained(model, **kwargs)
            model_config_dict = model_config.to_dict()
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(model_config,  **kwargs)
                
        return model
    
    @classmethod
    def init_empty_weights(cls, *args, **kwargs):
        from accelerate import init_empty_weights

        return init_empty_weights(*args, **kwargs)
        
    @classmethod
    def get_model_size(cls, 
                       model: 'nn.Module',
                       model_inflation_ratio: float = 1.0, 
                       keys:List[str]=None):
        
        # get the size of the model by initializing an empty model
        if isinstance(model, str):
            model = cls.get_empty_model(model)
            
            model = cls.module('model').shortcuts().get(model, model)
            
        params = {}
        size_in_bytes = 0 
        for name, param in model.named_parameters():
            if keys != None and name not in keys:
                continue
            
            size_in_bytes += cls.get_tensor_size(param)
          
        return size_in_bytes * model_inflation_ratio


    def num_params(self)->int:
        return self.get_num_params(self)

    def to_dict(self)-> Dict:
        return self.__dict__
    
    @classmethod
    def from_dict(cls, input_dict:Dict[str, Any]) -> 'Module':
        return cls(**input_dict)
        
    def to_json(self) -> str:
        import json
        state_dict = self.to_dict()
        assert isinstance(state_dict, dict), 'State dict must be a dictionary'
        assert self.jsonable(state_dict), 'State dict must be jsonable'
        return json.dumps(state_dict)
    
    @classmethod
    def resolve_logger(cls, logger = None):
        if not hasattr(cls,'logger'):
            from loguru import logger
            cls.logger = logger.opt(colors=True)
        if logger is not None:
            cls.logger = logger
        return cls.logger

    @classmethod
    def resolve_console(cls, console = None):
        if not hasattr(cls,'console'):
            from rich.console import Console
            cls.console = Console()
        if console is not None:
            cls.console = console
        return cls.console
    
    @classmethod
    def critical(cls, *args, **kwargs):
        console = cls.resolve_console()
        return console.critical(*args, **kwargs)
    
    @classmethod
    def log(cls, *args, **kwargs):
        console = cls.resolve_console()
        return console.log(*args, **kwargs)
    
    @classmethod
    def logs(cls, *args, **kwargs):

        return cls.pm2_logs(*args, **kwargs)

    @classmethod
    def print(cls, *text:str, 
              color:str=None, 
              return_text:bool=False, 
              verbose:bool = True,
              console: Console = None,
              **kwargs):
        if verbose:
            if color == 'random':
                color = cls.random_color()
            if color:
                kwargs['style'] = color
            console = cls.resolve_console(console)
            return console.print(*text, **kwargs)

    @classmethod
    def success(cls, *args, **kwargs):
        logger = cls.resolve_logger()
        return logger.success(*args, **kwargs)

    @classmethod
    def warning(cls, *args, **kwargs):
        logger = cls.resolve_logger()
        return logger.warning(*args, **kwargs)
    

    
    def whitelist(self, mode='sudo') -> List[str]:
        if self.is_root():
            return self.helper_functions
        else:
            return self.fns() + self.attributes() 
        
    def role2whitelist(self, role='bro') -> Dict:
        return {self.default_role: self.whitelist()}
        
    def role2blacklist(self, role='foe') -> Dict:
        return {'foe': []}
    
    def blacklist(self) -> List[str]:
        return ['module_name']
    black_fns = blacklist

    @classmethod
    def error(cls, *args, **kwargs):
        logger = cls.resolve_logger()
        return logger.error(*args, **kwargs)
    
    @classmethod
    def debug(cls, *args, **kwargs):
        console = cls.resolve_console()
        return console.debug(*args, **kwargs)

    @classmethod
    def from_json(cls, json_str:str) -> 'Module':
        import json
        return cls.from_dict(json.loads(json_str))
    
    
     
    @classmethod
    def status(cls, *args, **kwargs):
        console = cls.resolve_console()
        return cls.console.status(*args, **kwargs)
    @classmethod
    def log(cls, *args, **kwargs):
        console = cls.resolve_console()
        return cls.console.log(*args, **kwargs)
       
    @classmethod
    def test(cls):
        test_responses = {}
        for fn in cls.fns():
            test_response = {
                'passed':False,
                'response': None
            }
            
            if fn.startswith('test_'):
                try:
            
                    getattr(cls, fn)()
                    test_response['passed'] = True
                except Exception as e:
                   test_response['passed'] = False
                   test_response['response'] = str(e)
                test_responses[fn] =test_response
        
        return test_responses
       
               
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
    def timestamp(cls) -> float:
        return int(cls.time())
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
    def dict_delete(cls, *args, **kwargs):
        dict_delete = cls.import_object('commune.utils.dict.dict_delete')
        return dict_delete(*args, **kwargs)
    @classmethod
    def dict_has(cls, *args, **kwargs):
        dict_has = cls.import_object('commune.utils.dict.dict_has')
        return dict_has(*args, **kwargs)
    
    @classmethod
    def argv(cls, include_script:bool = False):
        import sys
        args = sys.argv
        if include_script:
            return args
        else:
            return args[1:]

    @classmethod
    def parse_args(cls, argv = None):
        if argv is None:
            argv = cls.argv()
        args = []
        kwargs = {}
        parsing_kwargs = False
        for arg in argv:
            # TODO fix exception with  "="
            # if any([arg.startswith(_) for _ in ['"', "'"]]):
            #     assert parsing_kwargs is False, 'Cannot mix positional and keyword arguments'
            #     args.append(cls.determine_type(arg))
            if '=' in arg:
                parsing_kwargs = True
                key, value = arg.split('=', 1)
                # use determine_type to convert the value to its actual type
                kwargs[key] = cls.determine_type(value)
            else:
                assert parsing_kwargs is False, 'Cannot mix positional and keyword arguments'
                args.append(cls.determine_type(arg))
        return args, kwargs

    # BYTES LAND
    
    # STRING2BYTES
    @classmethod
    def str2bytes(cls, data: str, mode: str = 'utf-8') -> bytes:
        return bytes(data, mode)
    
    @classmethod
    def bytes2str(cls, data: bytes, mode: str = 'utf-8') -> str:
        try:
            return bytes.decode(data, mode)
        except UnicodeDecodeError:
            return data.hex()
    
    # JSON2BYTES
    @classmethod
    def dict2str(cls, data: str) -> str:
        return json.dumps(data)
    
    
    @classmethod
    def dict2bytes(cls, data: str) -> bytes:
        return cls.str2bytes(cls.json2str(data))
    
    @classmethod
    def bytes2dict(cls, data: bytes) -> str:
        data = cls.bytes2str(data)
        return json.loads(data)
    
    
    @classmethod
    def python2str(cls, input):
        input = deepcopy(input)
        input_type = type(input)
        if input_type == str:
            return input
        
        if input_type in [dict]:
            input = json.dumps(input)
        elif input_type in [bytes]:
            input = cls.bytes2str(input)
        elif input_type in [list, tuple, set]:
            input = json.dumps(list(input))
        elif input_type in [int, float, bool]:
            input = str(input)
        return input

    @classmethod
    def str2python(cls, input)-> dict:
        assert isinstance(input, str), 'input must be a string, got {}'.format(input)
        try:
            output_dict = json.loads(input)
        except json.JSONDecodeError as e:
            return input

        return output_dict
    
    @staticmethod
    def jsonable( value):
        import json
        try:
            json.dumps(value)
            return True
        except:
            return False
            

    def restart_module(self, module:str) -> None:
        module = self.get_module(module)
        module.restart()
        return None
    
    
    # KEY LAND

    # MODULE IDENTITY LAND
    
    @classmethod
    def get_wallet( cls, *args, mode = 'bittensor', **kwargs) -> 'bittensor.Wallet':
        if mode == 'bittensor':
            return cls.get_module('bittensor').get_wallet(*args, **kwargs)
        elif mode == 'subspace':
            kwargs['mode'] = mode
            raise c.get_key(*args, **kwargs)
               
    
    @classmethod
    def get_key(cls, *args,mode='commune', **kwargs) -> None:


        if mode == 'commune':
            module = cls.module('key')
        elif mode == 'subspace':
            module  = cls.module('subspace.key')
        elif mode == 'substrate':
            module =  cls.module(f'web3.account.substrate')
        elif mode == 'evm':
            module = cls.module('web3.account.evm')
        elif  mode == 'aes':
            module =  cls.module('key.aes')
        else:
            raise ValueError('Invalid mode for key')
        
        # run get_key if the function exists
        if hasattr(module, 'get_key'):
            key = module.get_key(*args, **kwargs)
        else:
            key = module(*args, **kwargs)
            
        return key
    
    key = get_key
        
            
    @classmethod
    def hash(cls, 
             data: Union[str, bytes], 
             mode: str = 'sha256', 
             **kwargs) -> bytes:
        if not hasattr(cls, 'hash_module'):
            cls.hash_module = cls.get_module('crypto.hash')()
        return cls.hash_module(data, mode=mode, **kwargs)
    
    default_password = 'bitconnect'
    @classmethod
    def resolve_password(cls, password: str) -> str:
        if password == None:
            password = cls.default_password
            
            
        password = cls.python2str(password)
        assert isinstance(password, str), f'Password must be a string , not {type(password)}'
        return password


    encrypted_prefix = 'AESKEY'
    @classmethod
    def encrypt(cls, 
                data: Union[str, bytes],
                password: str = 'bitconnect', 
                prefix = encrypted_prefix) -> bytes:
        password = cls.resolve_password(password)
        data = cls.python2str(data)
        

        assert isinstance(password, str),  f'{password}'
        key = c.module('key.aes')(key=password)
        encrypted_data = key.encrypt(data)
        if prefix != None:
            encrypted_data = f'{prefix}::{encrypted_data}'
        return encrypted_data
    
    
    @classmethod
    def decrypt(cls, 
                data: str,
                password= None,
                ignore_error: bool = True,
                prefix = encrypted_prefix,
                verbose:bool = False) -> Any:
        password = cls.resolve_password(password)
        key = c.module('key.aes')(password)
        
        if isinstance(data, str):
            if data.startswith(prefix):
                data = data[len(prefix):]
            else:
                return {'error': 'data does not start with prefix'}
        if isinstance(data, Munch):
            data = c.munch2dict(data)
        if isinstance(data, dict):
            data = data['data']
        try:
            data = key.decrypt(data)
        except Exception as e:
            return None 
        if isinstance(data, str):
            data = cls.str2python(data)
            
        if isinstance(data, str) and len(data) == 0:
    
            if ignore_error:
                data = None
                if verbose:
                    cls.print(f'Exception: Wrong Password, try another',color='red')
            else:
                raise Exception(f'could not decrypt data, try another pasword')
        
        return data

    
    module_cache = {}
    @classmethod
    def put_cache(cls,k,v ):
        cls.module_cache[k] = v
    
    @classmethod
    def get_cache(cls,k, default=None, **kwargs):
        v = cls.module_cache.get(k, default)
        return v

    @classmethod
    def call(cls,  *args, loop=None, **kwargs) -> None:
        loop = cls.get_event_loop()
        return loop.run_until_complete(cls.async_call(*args, **kwargs))
    
    @classmethod
    async def async_call(cls,
                         module,
                         fn,
                         *args,
                         **kwargs) -> None:
        
        if isinstance(module, str) and fn == None:
            
            module, fn = '.'.join(module.split('.')[:-1]),  module.split('.')[-1],
            pool_mode = False
            
            while module.endswith('.'):
                pool_mode = True
                module = module[:-1]
            if pool_mode:
                module = cls.modules(module)
            
        if fn == None:
            fn = 'forward'
            
        if isinstance(module, list):
            modules = module
            jobs = []
            for m in modules:
                print(m, fn, args, kwargs)
                job = cls.async_call(m, fn, *args, **kwargs)
                jobs.append(job)
            results = await asyncio.gather(*jobs)
            return dict(zip(modules, results))
    
        if isinstance(module, str):
            module = await cls.async_connect(module)
        result = getattr(module, fn)
        if inspect.iscoroutinefunction(result):
            return await result(*args, **kwargs)
        elif callable(result):
            return result(*args, **kwargs)
        else:
            return result


    @classmethod
    def live_modules(cls, **kwargs):
        return cls.call_pool(fn='address', **kwargs)
    @classmethod
    def call_pool(cls, *args, **kwargs):
        loop = cls.get_event_loop()
        return loop.run_until_complete(cls.async_call_pool(*args, **kwargs))
    cpool = call_pool
    @classmethod
    async def async_call_pool(cls,
                              modules, 
                              fn = 'address',
                              *args, n=3, **kwargs):
        
        args = args or []
        kwargs = kwargs or {}
        
        if isinstance(modules, str) or modules == None:
            modules = cls.modules(modules)
            
        modules = cls.shuffle(modules)[:n]
        assert isinstance(modules, list), 'modules must be a list'
        cls.print(f'Calling {fn} on {len(modules)} modules', color='green')
        jobs = []
        for m in modules:
            job = cls.async_call(m, fn, *args, **kwargs)
            jobs.append(job)
        
        responses = await asyncio.gather(*jobs)
        
        is_error = lambda r: isinstance(r, dict) and 'error' in r
        successes  = [r for r in responses if not is_error(r)]
        errors = [r for r in responses if is_error(r)]
        
        if len(successes) == 0:
            cls.print(f'ERRORS {errors}', color='red')
        return successes[0]
    

    @classmethod
    def resolve_fn_module(cls, fn, module=None ) -> str:
    
        if module == None and len(fn.split('.')) > 1:
            module = '.'.join(fn.split('.')[:-1])
            module = cls.connect(module)
        
        return  fn, module

    
    def resolve_key(self, key: str) -> str:
        if key == None:
            key = self.key
        assert isinstance(key, str)
        key = self.get_key(key)
            
        return key  
                
    @classmethod  
    def keys(cls, *args, **kwargs):
        return c.module('key').keys(*args, **kwargs)
    
    @staticmethod
    def is_key(key):
        return hasattr(args[0], 'public_key') and hasattr(args[0], 'address')
    
    def set_key(self, *args, **kwargs) -> None:
        # set the key
        if hasattr(args[0], 'public_key') and hasattr(args[0], 'address'):
            # key is already a key object
            self.key = args[0]
            self.public_key = self.key.public_key
            self.address = self.key.address
        else:
            # key is a string
            self.key = self.get_key(*args, **kwargs)
            self.public_key = self.key.public_key
      
    def set_network(self, network: str) -> None:
        self.network = network
        
        
    def sign(self, data:dict  = None, key: str = None) -> bool:
        key = self.resolve_key(key)
        return key.sign(data) 
    
       
    
    @classmethod
    def verify(cls, data:dict ) -> bool:        
        return c.module('key').verify(data)
        
    
    @classmethod
    def get_signer(cls, data:dict ) -> bool:        
        return c.module('key').get_signer(data)
    
    def get_auth(self, 
                 data:dict  = None, 
                 key: str = None,
                 return_dict:bool = True,
                 encrypt: bool = False,
                 ) -> dict:
        
        key = self.resolve_key(key)
        if data == None:
            data = {'utc_timestamp': self.time()}

        sig_dict = key.sign(data, return_dict=return_dict)

        if encrypt:
            sig_dict['data'] = key.encrypt(sig_dict['data'])

        sig_dict['encrypted'] = encrypt
            
        
        
        return sig_dict
    
    
    @classmethod
    def start(cls, *args, **kwargs):
        return cls(*args, **kwargs)
    
    @classmethod
    def auth_data(cls, network = None, key = None) -> dict:
        network = network if network else cls.default_network
        
        data = {
            'timestamp': cls.timestamp(),
            'network': network
        }
        
        if key:
            key = cls.get_key(key)
            data = key.sign(data)
        return data


    @classmethod
    def call_auth_data(cls, 
                       module:str, 
                       fn:str,
                       args:list=None,
                       kwargs:dict = None,
                       network:str = None,
                       key = None) -> dict:
        network = network if network else cls.default_network
        data = {
            'network': network,
            'module':module,
            'fn': fn,
            'args': args or [],
            'kwargs': kwargs or {},
            'timestamp': cls.timestamp(),
        }
        
        if key:
            key = cls.get_key(key)
            data = key.sign(data)
            
        return data


    @classmethod
    def verify_call_auth(cls, auth) -> dict:
        signer_address = cls.get_signer(auth)        
        
        if key:
            key = cls.get_key(key)
            data = key.sign(data)
        return data

      
    def authenticate(self, data, staleness: int = 60, ) -> bool:
        
        '''
        Args:
            auth {
                'signature': str,
                'data': str (json) with ['timestamp'],
                'public_key': str
            }
            
            statleness: int (seconds) - how old the request can be
        return bool
        '''
        if not isinstance(data, dict):
            return False
        
        fn = data.get('fn', None)
        assert fn != None, 'Must provide a function name'
        
        assert fn in self.whitelist(), f'AuthFail: Function {fn} not in whitelist'
        assert fn not in self.blacklist(), f'AuthFail: Function {fn} in blacklist'
        
        # # check if user is in the list of users
        # is_user = self.is_user(auth)
        
        # # check the data
        # data = auth['data']
        
        # expiration  = self.time() - staleness
        # is_user = bool(data['timestamp'] > expiration)
            
        return True
        
    @classmethod
    def is_encrypted(cls, data, prefix='AESKEY'):
        if isinstance(data, str):
            if data.startswith(prefix):
                return True
        elif isinstance(data, dict):
            return bool(data.get('encrypted', False) == True)
        else:
            return False
        
    def is_user(self, auth: dict = None) -> bool:
        assert isinstance(auth, dict), 'Auth must be provided'
        for k in ['signature', 'data', 'public_key']:
            assert k in auth, f'Auth must have key {k}'
            
        user_address = self.verify(user, auth)
        if not hasattr(self, 'users'):
            self.users = {}
        return bool(user_address in self.users)
        
        
    @classmethod
    def add_user(cls, 
                 name: str = None,
                 signature: str = None,
                 role='sudo', **info):
        if not hasattr(self, 'users'):
            self.users = {}
        info.update(dict(timestamp=self.time(), 
                         role=role, 
                         user=user,
                         address=address))
        self.put(f'users/{user}/{role}', info)
    
    @classmethod
    def get_user(cls, user: str = None) -> dict:
        return cls.ls(f'users/{user}')
    
    @classmethod
    def rm_user(cls, user: str = None):
        self.users.pop(user, None)  
        
    @classmethod
    def users(self):
        return self._users
    
    
    
    
    
    @classmethod
    def network(cls,  *args, mode='subspace', **kwargs) -> str:
        if mode == 'subspace':
            return self.get_module('subspace')(*args, **kwargs)
        else:
            raise ValueError('Invalid mode for network')
        
    def remove_user(self, key: str) -> None:
        if not hasattr(self, 'users'):
            self.users = []
        self.users.pop(key, None)
        
    @classmethod
    def reserve_port(cls,port:int = None, var_path='reserved_ports' , root=True):
        if port == None:
            port = cls.free_port()
        reserved_ports =  cls.get(var_path, {}, root=root)
        reserved_ports[str(port)] = {'time': cls.time()}
        cls.put(var_path, reserved_ports, root=root)
        cls.print(f'reserving {port}')
        return {'success':f'reserved port {port}', 'reserved': cls.reserved_ports()}
    
    
    resport = reserve_port
    
    @classmethod
    def reserved_ports(cls,  var_path='reserved_ports'):
        return list(map(int, cls.get(var_path, {}, root=True).keys()))
    resports = reserved_ports

    
    @classmethod
    def unreserve_port(cls,port:int, 
                       var_path='reserved_ports' ,
                       verbose:bool = True, 
                       root:bool = True):
        reserved_ports =  cls.get(var_path, {}, root=True)
        
        port_info = reserved_ports.pop(port,None)
        if port_info == None:
            port_info = reserved_ports.pop(str(port),None)
        
        output = {}
        if port_info != None:
            cls.put(var_path, reserved_ports, root=True)
            output['msg'] = 'port removed'
        else:
            output['msg'] =  f'port {port} doesnt exist, so your good'

        output['reserved'] =  cls.reserved_ports()
        return output
    
    
    
    unresport = unreserve_port
    
    @classmethod
    def unreserve_ports(cls,*ports, 
                       var_path='reserved_ports' ,
                       verbose:bool = True, 
                       root:bool = True):
        output ={}
        reserved_ports =  cls.get(var_path, {}, root=root)
        if len(ports) == 0:
            ports = list(reserved_ports.keys())
        elif len(ports) == 1 and isinstance(ports[0],list):
            ports = ports[0]
        ports = list(map(str, ports))
        reserved_ports = {rp:v for rp,v in reserved_ports.items() if not any([p in ports for p in [str(rp), int(rp)]] )}
        cls.print(reserved_ports)
        cls.put(var_path, reserved_ports, root=root)
        return cls.reserved_ports()
    
    
    unresports = unreserve_ports
    @classmethod
    def fleet(cls, *tags, n=1, **kwargs):
        if len(tags) == 0:
            tags = list(range(n))
            
        for tag in tags: 
            cls.deploy(tag=tag, **kwargs)
        
        
        
    # # ARRAY2BYTES
    # @classmethod
    # def array2bytes(self, data: np.array) -> bytes:
    #     if isinstance(data, np.array):
    #         data = data.astype(np.float64)
    #     return data.tobytes()
    
    

    @classmethod
    def network(cls, network='subtensor', *args, **kwargs) -> str:
        if network == 'subspace':
            return cls.subspace(*args, **kwargs)
        elif network == 'subtensor':
            return cls.subtensor(*args, **kwargs)
        else:
            raise ValueError('Invalid mode for network')
    

    @classmethod
    def resolve_network(cls, subspace: str) -> str:
        if subspace == None:
            subspace = cls.subspace()
        elif isinstance(subspace, str):
            subspace = cls.subspace(subspace)
            
        return subspace
    
    @classmethod
    def client(cls, *args, **kwargs) -> 'Client':
        return cls.import_object('commune.server.Client')(*args, **kwargs)
    
    # @classmethod
    # def server(cls, *args, **kwargs) -> 'Server':
    #     return cls.import_object('commune.server.Server')(*args, **kwargs)
    
    @classmethod
    def serializer(cls, *args, **kwargs) -> 'Serializer':
        return cls.import_object('commune.server.Serializer')(*args, **kwargs)

    
    @classmethod
    def serialize(cls, data, metadata=None,to_json = False, **kwargs):
        metadata = metadata or {}
        if not isinstance(data, dict):
            data = dict(value=data)
        proto_data =  cls.serializer().serialize(data=data, metadata=metadata ,**kwargs)
        if to_json:
            proto_data = cls.proto2json(proto_data)
            
        return proto_data

    @classmethod
    def proto2json(cls, data):
        from google.protobuf.json_format import MessageToJson
        return MessageToJson(data)


    @classmethod
    def json2prot(cls, data):
        from google.protobuf.json_format import JsonToMessage
        return JsonToMessage(data)

    @classmethod
    def copy(cls, data: Any) -> Any:
        import copy
        return copy.deepcopy(data)
    
    @classmethod
    def launchpad(cls):
        return cls.import_object('commune.launchpad.Launchpad')()
    @classmethod
    def determine_type(cls, x):
        if x.lower() == 'null' or x == 'None':
            return None
        elif x.lower() in ['true', 'false']:
            return bool(x.lower() == 'true')
        elif x.startswith('[') and x.endswith(']'):
            # this is a list
            try:
                
                list_items = x[1:-1].split(',')
                # try to convert each item to its actual type
                return [cls.determine_type(item.strip()) for item in list_items]
            except:
                # if conversion fails, return as string
                return x
        elif x.startswith('{') and x.endswith('}'):
            # this is a dictionary
            if len(x) == 2:
                return {}
            try:
                dict_items = x[1:-1].split(',')
                # try to convert each item to a key-value pair
                return {key.strip(): cls.determine_type(value.strip()) for key, value in [item.split(':', 1) for item in dict_items]}
            except:
                # if conversion fails, return as string
                return x
        else:
            # try to convert to int or float, otherwise return as string
            try:
                return int(x)
            except ValueError:
                try:
                    return float(x)
                except ValueError:
                    return x

    @classmethod
    def set_port_range(cls, *port_range: list):
        if len(port_range) ==0 :
            port_range = cls.default_port_range
        elif len(port_range) == 1:
            if port_range[0] == None:
                port_range = cls.default_port_range

        assert len(port_range) == 2, 'Port range must be a list of two integers'        
        for port in port_range:
            assert isinstance(port, int), f'Port {port} range must be a list of integers'
        assert port_range[0] < port_range[1], 'Port range must be a list of integers'
                
        c.put('port_range', port_range)
        return port_range
    
    
    
    
    @classmethod
    def get_port_range(cls, port_range: list = None) -> list:

            
        if port_range == None:
            port_range = c.get('port_range', default=cls.default_port_range)
            
        if len(port_range) == 0:
            port_range = cls.default_port_range
        port_range = list(port_range)
        assert isinstance(port_range, list), 'Port range must be a list'
        assert isinstance(port_range[0], int), 'Port range must be a list of integers'
        assert isinstance(port_range[1], int), 'Port range must be a list of integers'
        return port_range
    
    @classmethod
    def port_range(cls):
        return cls.get_port_range()
    
    # ports = port_range
    @classmethod
    def resolve_port_range(cls, port_range: list = None) -> list:
        return cls.get_port_range(port_range)
        return port_range
    
    # @classmethod 
    # def ansible(cls, *args, fn='shell', **kwargs):
    #     ansible_module = cls.get_module('ansible')()
    #     return getattr(ansible_module, fn)(*args, **kwargs)
    
    
    
    @classmethod
    def add_peer(cls, *args, **kwargs)-> List:
        loop = cls.get_event_loop()
        peer = loop.run_until_complete(cls.async_add_peer(*args, **kwargs))
        
        return peer
    
    
    @classmethod
    async def async_add_peer(cls, 
                             peer_address,
                             network = 'local',
                             timeout:int=1,
                             verbose:bool = True):
        
        peer_registry = await cls.async_get_json('peer_registry', default={}, root=True)


        peer_info = await cls.async_call(module=peer_address, 
                                              fn='info',
                                              include_namespace=True, 
                                              timeout=timeout)
        
        #  add each peer to the registry
        
        c.print(peer_info)

        if 'error' in peer_info:
            if verbose:
                cls.print(f'Error adding peer {peer_address} due to {peer_info["error"]}',color='red')
            return None    
        else:
            if verbose:
                cls.print(f'Successfully added peer {peer_address}', color='green')
        
            
        assert isinstance(peer_info, dict)
        assert 'address' in peer_info
        assert 'namespace' in peer_info
        
        peer_ip = ':'.join(peer_info['address'].split(':')[:-1])
        peer_port = int(peer_info['address'].split(':')[-1])
        
        # relace default local ip with external_ip
        peer_info['namespace'] = {k:v.replace(cls.default_ip,peer_ip) for k,v in peer_info['namespace'].items()}

        peer_registry[peer_address] = peer_info
            
        await cls.async_put_json('peer_registry', peer_registry, root=True)
        
        return peer_registry
    
    @classmethod
    def add_module(cls, module, cache_path='remote_modules', refresh = True):
        module = c.connect(module)
        module_info = module.info(include_namespace=False)
        assert isinstance(module_info, dict), 'Module info must be a dictionary'
        remote_modules = {} if refresh else c.get(cache_path, {})
        remote_modules[ module_info['name']] = module_info['address']
        c.put(cache_path, remote_modules)
        return {'msg': module,
                'address': module_info['address'], 
                'module': module_info}
    
    @classmethod
    def rm_module(cls, module, cache_path='remote_modules'):
        remote_modules = c.get(cache_path, {})
        if module in remote_modules:
            remote_modules.pop(module)
            c.put(cache_path, remote_modules)
            return {'msg': 'Module removed', 'module': module}
        
        return {'msg': 'Module not found', 'module': module}

    @classmethod
    def remote_modules(cls, cache_path='remote_modules'):
       
        
        return c.get(cache_path, {}) 

    
    
    @classmethod
    def is_success(cls, x):
        # assume that if the result is a dictionary, and it has an error key, then it is an error
        if isinstance(x, dict):
            if 'error' in x:
                return False
            
        return True
    
    @classmethod
    def reset_peers(cls, *args, **kwargs):
        cls.rm_peers()
        return cls.add_peers(*args, **kwargs)
    
    
    @classmethod
    def add_peers(cls, *peer_addresses, **kwargs): 
        if len(peer_addresses) == 0:
            peer_addresses = cls.boot_peers()
            
        if len(peer_addresses) == 1 and isinstance(peer_addresses[0], list):
            peer_addresses = peer_addresses[0]
        jobs = []
        for peer_address in peer_addresses:
            job = cls.async_add_peer(peer_address, **kwargs)
            jobs += [job]
            
        loop = cls.get_event_loop()
        peers = loop.run_until_complete(asyncio.gather(*jobs))
        peers = [peer for peer in peers if peer != None]
        return {'added_peers': peers, 'msg': f'Added {len(peers)} peers'}


    @staticmethod
    def is_number(value):
        try:
            int(value)
        except ValueError:
            return False
        return True

        

    
    @classmethod
    def rm_peer(cls, peer_address: str):
        peer_registry = c.get_json('peer_registry', default={})
        result = peer_registry.pop(peer_address, None) 
        if result != None:
            result = peer_address      
            cls.put_json('peer_registry', peer_registry, root=True)
        return result
       
    @classmethod
    def rm_peers(cls, peer_addresses: list = None):
        rm_peers = []
        if peer_addresses == None:
            peer_addresses = cls.peers()
        if isinstance(peer_addresses, str):
            peer_addresses = [peer_addresses]
        for peer_address in peer_addresses:
            
            rm_peers.append(cls.rm_peer(peer_address))
        return rm_peers
            
      

        
        
    def store_value(self, key, value, *args, **kwargs):
        value = {'data': value}
        self.put_json(key, value, *args, **kwargs)
        return key
    def get_value(self, key, *args, **kwargs):
        value = self.get_json(key, *args, **kwargs)
        value = value.get('data', None)
        return value
    @classmethod
    def update(cls, 
               network = 'global',
               verbose:bool = True,
               min_staleness = 30,
               
               ):

            cls.namespace(network=network,verbose=True, update=True)
            
            # cls.root_module()

        
        
    @classmethod
    def peer_registry(cls, peers=None, update: bool = False):
        if update:
            if peers == None:
                peers = cls.peers()
            cls.add_peers(peers)
        return c.get_json('peer_registry', default={})
    
    

    @classmethod
    def run_jobs(cls, jobs: List, mode ='asyncio',**kwargs):
        if mode == 'asyncio':
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(asyncio.gather(*jobs))
            return results
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
    
    @classmethod
    def ls_peers(cls, update=False):
        peer_registry = cls.get_json('peer_registry', default={})
        return list(peer_registry.keys())
      
    @classmethod
    def peers(cls, update=False):
        peer_registry = cls.peer_registry(update=update)
        return list(peer_registry.keys())

    @classmethod
    def filter(cls, text_list: List[str], filter_text: str) -> List[str]:
        return [text for text in text_list if filter_text in text]

    @classmethod
    def get_file_contents(cls, class_name = None):
        if class_name is None:
            class_name = cls
        # Get the module that contains the class
        module = inspect.getmodule(class_name)
        if module is None:
            raise ValueError(f"Could not find module for class {class_name}")

        # Get the file path of the module
        module_file_path = os.path.abspath(module.__file__)

        # Read the contents of the file
        with open(module_file_path, 'r') as file:
            file_contents = file.read()

        return file_contents

    @classmethod
    def put_text(cls, path:str, text:str, root=False) -> None:
        # Get the absolute path of the file
        path = cls.resolve_path(path, root=root)

        # Write the text to the file
        with open(path, 'w') as file:
            file.write(text)
            
            
    @classmethod
    def add_text(cls, path:str, text:str, root=False) -> None:
        # Get the absolute path of the file
        path = cls.resolve_path(path, root=root)
        # Write the text to the file
        with open(path, 'w') as file:
            file.write(text)
            
    @classmethod
    def get_text(cls, path: str, root=False, last_bytes=-1, first_bytes=-1) -> str:
        # Get the absolute path of the file
        path = cls.resolve_path(path, root=root)
        # Read the contents of the file
        with open(path, 'rb') as file:
            # Move to the beginning of the file

            content = ''
            if last_bytes > 0:
                file.seek(0, 2)
                file_size = file.tell()  # Get the file size
                # Move to the position to read the last bytes
                file.seek(max(file_size - last_bytes, 0), 0)
                content = file.read(last_bytes).decode()
            elif first_bytes > 0:
                # Move back to the beginning of the file
                file.seek(0, 0)
                content = file.read(first_bytes).decode()
                
            else:
                content = file.read().decode()

        return content


    @classmethod
    def free_gpu_memory(cls, 
                     max_gpu_ratio: float = 1.0 ,
                     reserved_gpus: bool = False,
                     fmt = 'b') -> Dict[int, float]:
        import torch
        assert max_gpu_ratio <= 1.0 and max_gpu_ratio > 0, 'max_gpu_ratio must be less than 1.0 and greter than 0'
        free_gpu_memory = {}
        
        if fmt == 'gb' or fmt == 'GB':
            scale = 1e9
            
        elif fmt == 'mb':
            scale = 1e6
        elif fmt == 'kb':
            scale = 1e3
        elif fmt == 'b':
            scale = 1
        elif fmt in ['%', 'ratio']:
            scale = 1
        else:
            raise ValueError(f'Invalid format: {fmt}, options are gb, mb, kb, b')

        
        gpu_info_map = cls.gpu_map()
        gpus = [int(gpu) for gpu in gpu_info_map.keys()] 
        
        if  reserved_gpus != False:
            reserved_gpus = reserved_gpus if isinstance(reserved_gpus, dict) else cls.copy(cls.reserved_gpus())
            assert isinstance(reserved_gpus, dict), 'reserved_gpus must be a dict'
            
            for r_gpu, r_gpu_memory in reserved_gpus.items():
                gpu_info_map[r_gpu]['total'] -= r_gpu_memory
               
        for gpu_id, gpu_info in gpu_info_map.items():
            if int(gpu_id) in gpus:
                gpu_memory = max(gpu_info['total']*max_gpu_ratio - gpu_info['used'], 0)
                if gpu_memory <= 0:
                    continue
                free_gpu_memory[gpu_id] = int(cls.copy(gpu_memory /scale))
                if fmt == '%':
                    free_gpu_memory[gpu_id] =int((free_gpu_memory[gpu_id]/gpu_info['total']) * 100)
                    free_gpu_memory[gpu_id] = f'{free_gpu_memory[gpu_id]:.2f}%'
                elif fmt == 'ratio':
                    free_gpu_memory[gpu_id] = free_gpu_memory[gpu_id]/(gpu_info['total']+1e-10)
        if fmt == 'GB':
            free_gpu_memory = {k:f'{int(v)}GB' for k,v in free_gpu_memory.items()}
            
        
        try:
            total_free_memory = sum(free_gpu_memory.values())
        except TypeError as e:
            suffix_length = len(fmt) 
            total_free_memory = sum(list(map(lambda x: float(x[:-suffix_length]), free_gpu_memory.values())))
        assert total_free_memory > 0, 'No free memory on any GPU, please reduce the buffer ratio'

                
        return cls.copy(free_gpu_memory)
    
    
    free_gpus = free_gpu_memory

    @classmethod
    def make_dirs( cls, path ):
        """ Makes directories for path.
        """
        
        directory = os.path.dirname( path )
        if not os.path.exists( directory ):
            os.makedirs( directory ) 

    @classmethod
    def max_gpu_memory(cls, memory:Union[str,int] = None,
                       mode:str = 'most_free', 
                       min_memory_ratio = 0.0,
                       reserve:bool = False, 
                       free_gpu_memory: dict = None,
                       saturate:bool = False,
                       **kwargs):
        
        
        memory = cls.resolve_memory(memory)
        min_memory = min_memory_ratio * memory
        
        assert memory > 0, f'memory must be greater than 0, got {memory}'
        free_gpu_memory = free_gpu_memory if free_gpu_memory else cls.free_gpu_memory(**kwargs)
        
        # free_gpu_memory = {k:v for k,v in free_gpu_memory.items() if v > min_memory}
        gpus = list(free_gpu_memory.keys()) 
        total_gpu_memory = sum(free_gpu_memory.values())
        
        
        assert memory < total_gpu_memory, f'model size {memory} is larger than total gpu memory {total_gpu_memory}, over gpus {gpus}'
        unallocated_memory = memory
        # max_memory = {}
        max_memory = {}
        
        
        

        
        
        selected_gpus = []
        while unallocated_memory > 0:
            if mode =='random':
                gpu = np.random.choice(gpus)
                gpu_memory = free_gpu_memory[gpu]
            elif mode == 'most_free':
                gpu, gpu_memory = cls.most_free_gpu(free_gpu_memory=free_gpu_memory, return_tuple=True)
            else:
                raise ValueError(f'Invalid mode: {mode}, options are random, most_free')
            
            
            if gpu in max_memory:
                continue
            
            if gpu_memory < min_memory:
                continue
                
  
            allocated_memory = min(gpu_memory, unallocated_memory)
            if allocated_memory>0:
                max_memory[gpu] = allocated_memory
                free_gpu_memory[gpu] -= allocated_memory
            unallocated_memory -= allocated_memory
            max_memory[gpu] = allocated_memory
            free_gpu_memory[gpu] -= allocated_memory
            
        max_memory = {k:int(v) for k,v in max_memory.items() if v > 0}
        
        if reserve:
            
            cls.reserve_gpu_memory(max_memory)
            
            
        if saturate:
            free_gpu_memory = cls.free_gpu_memory()
            max_memory = {gpu:free_gpu_memory[gpu] for gpu in max_memory.keys()}
        return max_memory
            
            
    scale_map = {
        'kb': 1e3,
        'mb': 1e6,
        'gb': 1e9,
        'b': 1,
    }
    @classmethod
    def resolve_module(cls, module=None):
        if module == None:
            module = cls
        if isinstance(module, str):
            module = self.get_module(module)
        
        return module
            
            
    @classmethod
    def resolve_memory(cls, memory) -> str:
        
        if isinstance(memory, str):
            scale_found = False
            for scale_key, scale_value in cls.scale_map.items():
                
                if isinstance(memory, str) and memory.lower().endswith(scale_key):
                    memory = int(int(memory[:-len(scale_key)])*scale_value)
    
                if type(memory) in [float, int]:
                    scale_found = True
                    break
                    
        assert type(memory) in [float, int], f'memory must be a float or int, got {type(memory)}'
        return memory
            

    @classmethod
    def reserve_gpus(cls,gpu_memory: Union[Dict, str, int, float], refresh:bool = False, root=True, **kwargs):
        reserved_gpu_memory = {} if refresh else cls.reserved_gpus()
        if type(gpu_memory) in [int, float, str]:
            gpu_memory = cls.max_gpu_memory(gpu_memory, **kwargs)
        for  gpu, memory in gpu_memory.items():
            memory = cls.resolve_memory(memory) 
            gpu = int(gpu)
            if gpu in reserved_gpu_memory:
                reserved_gpu_memory[gpu] += memory
            else:
                reserved_gpu_memory[gpu] = memory
        cls.put('reserved_gpu_memory', reserved_gpu_memory, root=root)
        return reserved_gpu_memory
    
    @classmethod
    def reserved_gpus(cls,*args, **kwargs) -> Dict[str, int]:
        reserved_gpus = cls.get('reserved_gpu_memory', {}, root=True)
        reserved_gpus = {k:int(v) for k,v in reserved_gpus.items() if v > 0} 
        reserved_gpus = {int(k):int(v) for k,v in reserved_gpus.items()}
        return reserved_gpus  
    
    @classmethod
    def unreserve_gpus(cls,gpu_memory: Union[dict] = None,*args,  **kwargs):
        if gpu_memory is None:
            reserved_gpu_memory = {}
        else:
            reserved_gpu_memory =cls.reserved_gpus()
            for  gpu, memory in gpu_memory.items():
                memory = cls.resolve_memory(memory)
    
                if gpu in reserved_gpu_memory:
                    if memory == -1:
                        memory = reserved_gpu_memory[gpu]
                    reserved_gpu_memory[gpu] -= memory
                
        cls.print(f'unreserving {gpu_memory}')
        reserved_gpu_memory = {k:v for k,v in reserved_gpu_memory.items() if v > 0}
        cls.put('reserved_gpu_memory', reserved_gpu_memory, root=True)
        return cls.reserved_gpus()

    release_gpus = unleash_gpus =  unreserve_gpus
    reserve_gpu_memory = reserve_gpus
    unreserve_gpu_memory = unreserve_gpus

    def link_cmd(cls, old, new):
        
        link_cmd = cls.get('link_cmd', {})
        assert isinstance(old, str), old
        assert isinstance(new, str), new
        link_cmd[new] = old 
        
        cls.put('link_cmd', link_cmd)
        

    @classmethod
    def remote_fn(cls, 
                    fn='train', 
                    module = None,
                    args= None,
                    kwargs = None, 
                    tag = None,
                    tag_seperator= '::',
                    name=None):
        
        if len(fn.split('.'))>1:
            module = '.'.join(fn.split('.')[:-1])
            fn = fn.split('.')[-1]
            
        kwargs = kwargs if kwargs else {}
        args = args if args else []
        
        

        if name == None:
            prefix = cls.resolve_module(module).module_path()
            name = f'{prefix}{tag_seperator}{fn}'
    
        if tag != None:
            name = f'{name}{tag_seperator}{tag}'
            
            
        cls.launch(fn=fn, 
                   module = module,
                    kwargs=kwargs,
                    name=name)

    rfn = remote_fn
    @classmethod
    def choice(cls, options:list)->list:
        import random
        assert isinstance(options, list)
        return random.choice(options)
    
    @classmethod
    def colors(cls):
        return ['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white', 'bright_black', 'bright_red', 'bright_green', 'bright_yellow', 'bright_blue', 'bright_magenta', 'bright_cyan', 'bright_white']
    colours = colors
    @classmethod
    def random_color(cls):
        import random
        return random.choice(cls.colors())

    random_colour = random_color

    @classmethod
    def random_ratio_selection(cls, x:list, ratio:float = 0.5)->list:
        import random
        assert len(x)>0
        if ratio == 1:
            return x
        assert ratio > 0 and ratio <= 1
        random.shuffle(x)
        k = max(int(len(x) * ratio),1)
        return x[:k]
    
    @classmethod
    def tags(cls):
        return ['alice', 'bob', 'chris', 'dan', 'fam', 'greg', 'elon', 'huck']
    
    @classmethod
    def rand_tag(cls):
        return cls.choice(cls.tags())
    
    @classmethod
    def gather(cls,jobs:list, mode='asyncio', loop=None)-> list:
        if not isinstance(jobs, list):
            jobs = [jobs]
        assert isinstance(jobs, list)
        
        
        
        if mode == 'asyncio':
            loop = loop if loop != None else cls.get_event_loop()
            results = loop.run_until_complete(asyncio.gather(*jobs))
            
        else:
            raise NotImplementedError
        
        return results
    @classmethod
    def addresses(cls, *args, **kwargs) -> List[str]:
        return list(cls.namespace(*args,**kwargs).values())

    @classmethod
    def address_exists(cls, address:str) -> List[str]:
        addresses = cls.addresses()
        return address in addresses
        
    @classmethod
    def task(cls, fn, timeout=1, mode='asyncio'):
        
        if mode == 'asyncio':
            assert callable(fn)
            future = asyncio.wait_for(fn, timeout=timeout)
            return future
        else:
            raise NotImplemented
        
    @staticmethod
    def is_ss58(value: str) -> bool:
        try:
            decoded = base58.b58decode_check(value)
            # Check if the decoded value starts with a specific prefix byte
            prefix = decoded[0]
            return prefix in (0x00, 0x01) # Or any other prefix values you want to allow
        except ValueError:
            return False
        
    @staticmethod
    def is_mnemonic(s: str) -> bool:
        import re
        # Match 12 or 24 words separated by spaces
        pattern = r'^(\w+\s){11}\w+(\s\w+){11}$|^(\w+\s){23}\w+$'
        return bool(re.match(pattern, s))

        
    @staticmethod   
    def is_private_key(s: str) -> bool:
        import re
        # Match a 64-character hexadecimal string
        pattern = r'^[0-9a-fA-F]{64}$'
        return bool(re.match(pattern, s))

    def is_ss58(value: str) -> bool:
        try:
            decoded = base58.b58decode_check(value)
            # Check if the decoded value starts with a specific prefix byte
            prefix = decoded[0]
            return prefix in (0x00, 0x01) # Or any other prefix values you want to allow
        except ValueError:
            return False
        
        
    @classmethod
    def mv(cls, path1, path2):
        import shutil
        shutil.move(path1, path2)
        return path2

        
        
    @classmethod
    def cp(cls, path1, path2):
        import shutil
        shutil.copy(path1, path2)
        return path2
    
    
    @classmethod
    def get_sample_schema(cls, x:dict) -> dict:
        import torch
        '''
        
        '''
        sample_schema = {}
        for k,v in x.items():
            if isinstance(v, torch.Tensor):
                sample_schema = dict(
                    shape=list(v.shape),
                    dtype= str(v.dtype)
                )
        return sample_schema    
    

    
    @classmethod
    def learn(cls, *args, **kwargs):
        return cls.module('model.transformer').learn(*args, **kwargs)
        
    @classmethod
    def mine(cls,*args, **kwargs):
        kwargs['remote'] = kwargs.get('remote', True)
        return cls.module('bittensor').mine(*args, **kwargs)
    
    @classmethod
    def train_fleet(cls, *args, **kwargs):
        kwargs['remote'] = kwargs.get('remote', True)
        return cls.module('model.transformer').train_fleet(*args, **kwargs)
    
    @classmethod
    def miners(cls, prefix='miner'):
        return cls.pm2_list(prefix)
    
    @classmethod
    def check_miners(cls, *args, module='bittensor', **kwargs):
        return cls.module(module).check_miners( *args, **kwargs)
    
    
    @classmethod
    def shuffle(cls, x:list)->list:
        import random
        random.shuffle(x)
        return x
    

    @classmethod
    def pull(cls):
        return cls.cmd('git pull')
    
    @classmethod
    def commit(cls, msg='update'):
        return cls.cmd(f'git add; git commit -m "{msg}"; git push;')
    
    @classmethod
    def make_pull(cls):
        return cls.cmd('make pull')
    
    
    @staticmethod
    def encode_topk( forward_response_tensor: 'torch.Tensor' , topk:int=4096) -> 'torch.Tensor':
        import torch
        """ Returns topk tokens/probabilities given unnormalized logits as input. """

        #import ipdb; ipdb.set_trace()

        logits = forward_response_tensor  # unnormalized logit scores: [batch_size, sequence_len, vocab_size]
        probs = torch.softmax(logits, dim=-1).to(torch.float32)  # normalized probabilities: [batch_size, sequence_len, vocab_size]

        topk_indices = torch.argsort(probs, dim=-1, descending=True)[...,:topk]
        # topk_values, topk_indices = torch.topk(probs, topk) # topk probs and indices: [batch_size, sequence_len, topk]

        topk_values = probs.gather( index=topk_indices, dim=-1)
        encoded_probs = torch.cat([topk_values, topk_indices], dim=-1)  # [batch_size, sequence_len, topk + topk]
        return encoded_probs  # [batch_size, sequence_len, topk + topk]

    # @staticmethod
    # def private_key_to_mnemonic(private_key):
    #     # Convert the public key to a hex string
    #     public_key_hex = substrate.keccak_256(private_key).hex()

    #     # Convert the public key hex to a mnemonic
    #     mnemonic = bip39.mnemonic_from_entropy(public_key_hex)

    #     return mnemonic
    
    @classmethod
    def docker_ps(cls):
        return cls.cmd('docker ps')
    dps = docker_ps
    
    '''
    SSH LAND
    '''
    @classmethod
    def add_ssh_key(cls,public_key:str, authorized_keys_file:str='~/authorized_keys'):
        authorized_keys_file = os.path.expanduser(authorized_keys_file)
        with open(authorized_keys_file, 'a') as auth_keys_file:
            auth_keys_file.write(public_key)
            auth_keys_file.write('\n')
            
        cls.print('Added the key fam')
        
    @classmethod
    def ssh_authorized_keys(cls, authorized_keys_file:str='~/authorized_keys'):
        authorized_keys_file = os.path.expanduser(authorized_keys_file)
        return cls.get_text(authorized_keys_file)

    @staticmethod
    def get_public_key_from_file(public_key_file='~/.ssh/id_rsa.pub'):
        public_key_file = os.path.expanduser(public_key_file)
        with open(public_key_file, 'r') as key_file:
            public_key_data = key_file.read().strip()

        # Extract the public key from the data
        public_key = None
        if public_key_data.startswith("ssh-rsa"):
            public_key = public_key_data

        return public_key

    @classmethod
    def generate_ssh_key_pair(cls, key_file=os.path.expanduser('~/.ssh/id_rsa'),
                            passphrase=None):
        import paramiko
        key = paramiko.RSAKey.generate(bits=2048)

        # Save the private key to a file
        key.write_private_key_file(key_file, password=passphrase)

        # Save the public key to a file
        with open(f"{key_file}.pub", "w") as pub_key_file:
            pub_key_file.write(f"{key.get_name()} {key.get_base64()} Generated by Python")
        
        cls.print(f"SSH key pair generated and saved to {ssh_key_path}")

    @classmethod
    def miner(cls, 
              api_key = None, 
              wallet = 'ensemble.vali',
              miner = '~/commune/bittensor/neurons/text/prompting/miners/openai/neuron.py',
              port=2012,
              network = 'finney',
              netuid = 1,
              *args, **kwargs):
        miner = os.path.expanduser(miner)
        api_key = api_key or os.environ.get('OPENAI_API_KEY')
        wallet_name, wallet_hotkey = wallet.split('.')
        name = f'miner::{wallet}::{network}::{netuid}'
        command = f"pm2 start {miner} --name {name} --interpreter python3 -- --wallet.name {wallet_name} --wallet.hotkey {wallet_hotkey} --axon.port {port} --openai.api_key {api_key} --neuron.no_set_weights --subtensor.network {network} --netuid {netuid} --logging.debug"
        cls.cmd(command)
        cls.print({'msg': f"Started miner {name} on port {port}"})
        
        
    @staticmethod
    def reverse_map(x):
        return {v:k for k,v in x.items()}

    @classmethod
    def pd(cls):
        return cls.import_module('pandas')

    @classmethod
    def df(cls, *args, **kwargs):
        df =  cls.import_object('pandas.DataFrame')
        if len(args) > 0 or len(kwargs) > 0:
            df = df(*args, **kwargs)
        return df

    @classmethod
    def st(cls):
        return cls.import_module('streamlit')
    @classmethod
    def torch(cls):
        return cls.import_module('torch')

    @staticmethod
    def json2df(json_data):
        """
        Convert JSON data to a pandas DataFrame.
        
        Args:
            json_data (str): JSON data representing a DataFrame.
            
        Returns:
            pandas.DataFrame: DataFrame created from the JSON data.
        """
                
        import pandas as pd
        import json
        dataframe = pd.read_json(json_data)
        return dataframe
    
    @staticmethod
    def ss58_encode(*args, **kwargs):
        from scalecodec.utils.ss58 import ss58_encode, ss58_decode
        return ss58_encode(*args, **kwargs)
    @staticmethod
    def ss58_decode(*args, **kwargs):
        from scalecodec.utils.ss58 import  ss58_decode
        return ss58_decode(*args, **kwargs)

     
    def interpeter(obj, fn, **kwargs):
        if fn["return"] == "Blocks":
            return getattr(obj, fn)() # function rendering is a block so things are prebuilt 
        else:
            # handel output annotations given there is
            if "return" in list(fn.keys()):     
                # converts the return in the function annotation to gradio str components
                if fn["return"].__origin__ in (list, tuple, Tuple):
                     outputs = []
                     for output in fn["return"].__args__:
                        _output = output.__metadata__[-1]
                        if isinstance(_output, gr.component.IOComponent):
                            _output._id += 1
                            outputs.append(_output.get_config())
                        elif isinstance(_output, str):
                            outputs.append(_output)
                        else:
                            raise ValueError("Must provide an annotation that is either and IOComponent or string")
                else:
                    outputs = fn.__annotations__["return"].__metadata__[-1]
                            
                del fn["return"]

                        # handel input annotation 
                if fn: # check if there if there are no more inputs 
                    inputs = []
                    for input in fn.values():
                        _input = input.__metadata__[-1]
                        if isinstance(_input, gr.component.IOComponent):
                            _input._id += 1
                            inputs.append(_input.get_config())
                        elif isinstance(_input, str):
                                inputs.append(_input)
                        else:
                            raise ValueError("Must provide an annotation that is either and IOComponent or string")
            return gr.Interface(fn=getattr(obj, fn), inputs=inputs, outputs=outputs, **kwargs)


    def _annotate(module, fn : str, /, **kwargs) -> Union[gr.Interface, gr.Blocks]:
        try:
            fn = getattr(module, fn)

            if fn.__annotations__ and \
            "return" in list(fn.__annotations__.keys()) and \
            fn.__annotations__["return"].__metadata__[-1] == "Block":
                return fn()
                
            
            else:
                inputs, outputs = [], []
                if "return" in fn.__annotations__.keys():
                    if fn.__annotations__["return"].__origin__ in (list, tuple, Tuple):
                            for output in fn.__annotations__["return"].__args__:
                                _output = output.__metadata__[-1]
                                if isinstance(_output, gr.component.IOComponent):
                                    _output._id += 1
                                    outputs.append(_output.get_config())
                                elif isinstance(_output, str):
                                    outputs.append(_output)
                                else:
                                    raise ValueError("Must provide an annotation that is either and IOComponent or string")
                    else:
                        print(fn.__annotations__["return"].__metadata__[-1])
                        outputs = [fn.__annotations__["return"].__metadata__[-1]]


                    del fn.__annotations__["return"]
                
                if fn.__annotations__: # check if there if there are no more inputs 
                    for input in fn.__annotations__.values():
                        _input = input.__metadata__[-1]
                        if isinstance(_input, gr.components.IOComponent):
                            _input._id += 1
                            inputs.append(gr.components.get_component_instance(input.get_config()))
                        elif isinstance(_input, str):
                            inputs.append(_input)
                        else:
                            raise ValueError("Must provide an annotation that is either and IOComponent or string")
                print(inputs, outputs)
                return gr.Interface(fn=fn, inputs=inputs, outputs=outputs)
        except (AttributeError, ValueError) as e:
            return False

    @classmethod
    def gradioify(cls, module : str=None, fn : Union[str, list[str]]=None, **kwargs) -> None:
        """
            
            serve all possible functions within the module that return blocks or are annotated using
            typing within directory gradio.tpying there are prebuilt annotation that one can uses to annotate
            functions or you can create your own custom annotation that are gradio compiable
            
            
            ### Commandline Serve All Function(s)
            ```python
                >>> c gradioify gradio.example
            ```

            ### Serve All Function(s)
            ```python
                >>> import commune
                >>> commune.gradioify("gradio.example").launch()
            ```

            ### Serve Singualr Function 
            ```python
                >>> import commune
                >>> commune.gradioify("gradio.example", fn="example_").launch()
            ```


        """
        obj = cls.module(module)

        if fn == None:
            schema = cls.schema(obj=obj)
            del schema["__init__"]
            fn = list(schema.keys())
            

        if not isinstance(fn, list):
            fn = [fn]

        try:  
            if len(fn) == 1:
                return cls._annotate(obj, fn[-1], **kwargs)
            elif len(fn) > 1:  
                interfaces, names = list(), list()
                for idx, function in enumerate(fn):
                    print(function)
                    interface = cls._annotate(obj, function)
                    
                    if isinstance(interface, Union[gr.Interface, gr.Blocks]):
                        interfaces.append(interface)
                        names.append(function)
                print(interfaces)
                return gr.TabbedInterface(interface_list=interfaces, tab_names=names)
        except (AttributeError, ValueError, KeyboardInterrupt) as e:
            c.console.print_exception(show_locals=True)
 

Module = c
Module.run(__name__)
    

# pm2 start commune/module/module.py --name module --interpreter python3