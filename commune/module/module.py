

import inspect
import numpy as np
import os
from copy import deepcopy
from typing import Optional, Union, Dict, List, Any, Tuple, Callable
from munch import Munch
from rich.console import Console
import json
from glob import glob
import sys
import argparse
import asyncio
from typing import Union, Dict, Optional, Any, List, Tuple


@classmethod
def cache_result(cls, func):
    import functools
    
    def wrapper(*args, **kwargs):
        fn_name = func.__name__
        cache = kwargs.pop('cache', True)
        update = kwargs.pop('update', False)
        max_age = kwargs.pop('max_age', 60)

        if cache and not update:
            cls.get(fn_name, max_age=max_age, cache=cache)

        result = func(*args, **kwargs)
        
        if cache:
            cls.put(fn_name, result, cache=cache)

        return result

    return wrapper


class c:
    root_module_class = 'c' # WE REPLACE THIS THIS Module at the end, kindof odd, i know, ill fix it fam, chill out dawg, i didnt sleep with your girl
    default_port_range = [50050, 50150] 
    user = None
    default_ip = '0.0.0.0'
    address = None
    root_path  = root = libpath = os.path.dirname(os.path.dirname(__file__))
    modules_path = os.path.join(root_path, 'modules')
    repo_path  = os.path.dirname(root_path)
    library_name = root_dir = root_path.split('/')[-1]
    default_network = 'subspace'
    pwd = os.getenv('PWD')
    console = Console()
    default_key = 'alice'
    helper_whitelist = ['info', 'schema','module_name']
    whitelist = []
    blacklist = []
    def __init__(self, 
                 config:Dict=None,
                 **kwargs):
        
        self.set_config(config=config, 
                        kwargs=kwargs)  
    
    
    @classmethod
    def init(cls, *args, **kwargs):
        cls.__init__(*args, **kwargs)
    
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

    @classmethod
    def start_chain(cls, *args, **kwargs):
        c.module('subspace').start_chain(*args, **kwargs)
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
    def module_dirpath(self, simple:bool=False) -> str:
        return  os.path.dirname(self.module_file(simple=simple))

    @classmethod
    def __module_dir__(cls) -> str :
        # get the directory of the module
        return os.path.dirname(cls.module_file())
    

    @classmethod
    def get_module_path(cls, obj=None,  simple:bool=False) -> str:
        
        # odd case where the module is a module in streamlit
        obj = cls.resolve_module(obj)
        module_path =  inspect.getfile(obj)
        # convert into simple
        if simple:
            return cls.path2simple(path=module_path)
        return module_path
    
    @classmethod
    def get_module_dirpath(cls, obj=None,  simple:bool=False) -> str:
       
        return  os.path.dirname(c.get_module_path(obj=obj, simple=simple))
    get_module_dir = get_module_dirpath
    
    @classmethod
    def filepath(cls) -> str:
        '''
        removes the PWD with respect to where module.py is located
        '''
        return cls.get_module_path(simple=False)
    pythonpath = pypath =  filepath
    @classmethod
    def configpath(cls) -> str:
        '''
        removes the PWD with respect to where module.py is located
        '''
        return cls.get_module_config_path()
    cfgpath = configpath
    
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
    
    name = module_path
    module_name = module_path
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
        config = load_yaml(path)
        return config



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
        return cls.cmd(f'python3 sandbox.py')
    sand = sandbox
    @classmethod
    def save_yaml(cls, path:str,  data: dict, root:bool = False) -> Dict:
        '''
        Loads a yaml file
        '''
        path = cls.resolve_path(path, root=root)
            
        from commune.utils.dict import save_yaml
        if isinstance(data, Munch):
            data = cls.munch2dict(deepcopy(data))
            
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
    
    def resolve_config_path(cls, module= None) -> str:
        
        
        if module != None: 
            module_tree = cls.module_tree()
            path = module_tree[module].replace('.py', '.yaml')
        else:
            path = cls.__config_file__()
        assert isinstance(path, str)
        return path
    
    config_path = resolve_config_path
    
    @classmethod
    def load_config(cls, path:str=None, to_munch:bool = False, root:bool = False) -> Union[Munch, Dict]:
        '''
        Args:
            path: The path to the config file
            to_munch: If true, then convert the config to a munch
        '''
        path = cls.resolve_config_path(path)

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
            cache :bool = True,
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
    
        if cache:
            cls.cache[k] = v
        
        return data
    
    

        
    @classmethod
    def get(cls,
            key:str, 
            default: Any=None, 
            password: str=None, 
            mode:str = 'json',
            max_age:str = None,
            cache :bool = True,
            **kwargs) -> Any:
        
        '''
        Puts a value in sthe config, with the option to encrypt it

        Return the value
        '''
        if cache:
            if key in cls.cache:
                return cls.cache[key]
        
        verbose = kwargs.get('verbose', False)
        data = getattr(cls, f'get_{mode}')(key,default=default, **kwargs)
        if data == None: 
            data = default
        encrypted = c.is_encrypted(data)
        if encrypted:
            data = cls.decrypt(data, password=password)
        if isinstance(data, dict):
            if max_age != None:
                timestamp = data.get('timestamp', None)
                if timestamp != None:
                    age = c.get_age(timestamp)
                    if age > max_age:
                        if verbose:
                            c.print(f'{key} is too old, age: {int(age)} > {max_age}', color='red')
                        return default
        else:
            data = default
            
        if isinstance(data, dict):
            if 'data' in data:
                data = data['data']
        return data
    

        
    @classmethod
    def get_ts(cls,
            key:str, 
            default: Any=None, 
            password: str=None, 
            mode:str = 'json',
            max_age:str = None,
            cache :bool = True,
            **kwargs) -> Any:
        
        '''
        Puts a value in sthe config, with the option to encrypt it

        Return the value
        '''
        if cache:
            if key in cls.cache:
                return cls.cache[key]
        
        verbose = kwargs.get('verbose', False)
        data = getattr(cls, f'get_{mode}')(key,default=default, **kwargs)
        if data == None: 
            data = default
        encrypted = c.is_encrypted(data)
        if encrypted:
            data = cls.decrypt(data, password=password)
        if isinstance(data, dict):
            if max_age != None:
                timestamp = data.get('timestamp', None)
                if timestamp != None:
                    age = c.get_age(timestamp)
                    if age > max_age:
                        if verbose:
                            c.print(f'{key} is too old, age: {int(age)} > {max_age}', color='red')
                        return default
        return data['timestamp']
    
    @staticmethod
    def get_age(timestamp:int=0):
        return c.time() - timestamp
    
    @staticmethod
    def too_old(self, timestamp:int, max_age:int):
        return self.get_age(timestamp) > max_age
    
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
            v = cls.encrypt(v, password=password)

        cls.dict_put(config, k, v)
        cls.save_config(config=config)
   
   
    @classmethod
    def rmc(cls, k, password=None) -> Munch:
        '''
        Saves the config to a yaml file
        '''
        config = cls.config()
        c.dict_rm(config, k)
        cls.save_config(config=config)
   
    delc = rmc
    setc = putc
    @classmethod
    def encryptc(cls, k, password=None) -> Munch:
        '''
        Saves the config to a yaml file
        '''
        config = cls.config()
        assert k in config, f'key {k} not found in config'
        v = cls.dict_get(config, k)
        # assert isinstance(v,str), f'cannot encrypt {v} of type {type(v)}, strings only'
        if password:
            v = cls.encrypt(v,  password=password)

        cls.dict_put(config, k, v)
        cls.save_config(config=config)
        return v
   
    encc=encryptc
    @classmethod
    def decryptc(cls, k, password=None) -> Munch:
        '''
        Saves the config to a yaml file
        '''
        config = cls.config()
        v = config[k]
        if password:
            v = cls.decrypt(v,  password=password)

        if v != None:
            config[k] = v
            cls.save_config(config=config)
        
        return v
    
    decc = decryptc
    
    @classmethod
    def is_encryptedc(cls, k) -> Munch:
        '''
        Saves the config to a yaml file
        '''
        config = cls.getc(c)
        return c.is_encrypted(v)
    @classmethod
    def frontend(cls):
        c.cmd('yarn start', cwd=f'{c.repo_path}/frontend', verbose=True)
      
    @classmethod
    def popc(cls, key:str):
        config = cls.config()
        config.pop(key, None)
        cls.save_config(config=config)
        
    @classmethod  
    def getc(cls, key, password=None, default= None) -> Any:
        '''
        Saves the config to a yaml file
        '''
        
        data = cls.dict_get(cls.config(), key, default)
        if c.is_encrypted(data):
            if password == None:
                return data
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
                   config:dict = None,
                   kwargs:dict=None, 
                   to_munch:bool = True,
                   root:bool = False) -> Munch:
        '''
        Set the config as well as its local params
        '''
        kwargs = kwargs if kwargs != None else {}
        kwargs.pop('kwargs', None)
        if isinstance(config, str):
            try:
                config = cls.load_config(path=config)
            except FileNotFoundError as e:
                config = {}
            assert isinstance(config, dict), f'config must be a dict, not {type(config)}'
        elif isinstance(config, dict):
            default_config = cls.load_config()
            default_config.update(config)
            config = default_config

        elif config == None:
            config = cls.load_config()
            
        assert isinstance(config, dict), f'config must be a dict, not {config}'
        
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
                   kwargs:dict=None,
                   to_munch: bool = True,
                   add_attributes: bool = False,
                   save_config:bool = False) -> Munch:
        '''
        Set the config as well as its local params
        '''
        kwargs = kwargs if kwargs != None else {}
        kwargs = c.locals2kwargs(kwargs)
            

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
        return c.module('key').add_key(*args, **kwargs)
    # KEY LAND
    @classmethod
    def rename_key(cls, *args, **kwargs):
        return c.module('key').rename_key(*args, **kwargs)
    mv_key = rename_key
    @classmethod
    def add_keys(cls, *args, **kwargs):
        return c.module('key').add_keys(*args, **kwargs)
    @classmethod
    def key_exists(cls, *args, **kwargs):
        return c.module('key').key_exists(*args, **kwargs)
    @classmethod
    def ls_keys(cls, *args, **kwargs):
        return c.module('key').ls_keys(*args, **kwargs)
    @classmethod
    def rm_key(cls, *args, **kwargs):
        return c.module('key').rm_key(*args, **kwargs)
    @classmethod
    def key_encrypted(cls, *args, **kwargs):
        return c.module('key').key_encrypted(*args, **kwargs)

    @classmethod
    def encrypt_key(cls, *args, **kwargs):
        return c.module('key').encrypt_key(*args, **kwargs)
        

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
        module = c.module(module)
        module_filepath = module.filepath()
        c.print(f'Running {module_filepath}', color='green')
        cls.run_command(f'streamlit run {module_filepath} -- --fn {fn}', verbose=True)

    @staticmethod
    def stside(fn):
        import streamlit as st
        
        def wrapper(*args, **kwargs):
            with st.sidebar:
                return fn(*args, **kwargs)
        
        return wrapper
    @staticmethod
    def st_load_css(*args, **kwargs):
        c.module('streamlit').load_css(*args, **kwargs)

    @classmethod
    def cmd(cls, 
                    command:Union[str, list],
                    verbose:bool = False, 
                    env:Dict[str, str] = {}, 
                    output_text:bool = True,
                    sudo:bool = False,
                    password: bool = None,
                    color: str = 'white',
                    **kwargs) -> 'subprocess.Popen':
        '''
        Runs  a command in the shell.
        
        '''
        if isinstance(command, list):
            kwargs = c.locals2kwargs(locals())
            for idx,cmd in enumerate(command):
                c.print(f'Running {idx}/{len(command)}', color='green')
                kwargs['command'] = cmd
                c.cmd(**kwargs)
            command = command.split(' ')
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
            command = f'sudo {command}'
            
            
        process = subprocess.Popen(shlex.split(command),
                                    stdout=subprocess.PIPE, 
                                    # stderr=subprocess.PIPE, 
                                    env={**os.environ, **env}, **kwargs)

            
        new_line = b''
        stdout_text = ''
        line_count_idx = 0
        line_delay_period = 0
        last_time_line_printed = time.time()
 
        try:
            for ch in iter(lambda: process.stdout.read(1), b""):
                

                if  ch == b'\n':
                    line_count_idx += 1
                    stdout_text += (new_line+ch).decode()
                    if verbose:
            
                        c.print(new_line.decode(), color=color)
                    new_line = b''
                    continue
                
                new_line += ch
  
        except KeyboardInterrupt:
            kill_process(process)
        
        return stdout_text


    run_command = shell = cmd 
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
            c.print(f'Importing {object_name} from {module}')
        obj =  getattr(import_module(module), object_name)
        return obj
    
    get_object = importobj = import_object
    

    
    @classmethod
    def modules(cls, search=None)-> List[str]:
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
        
        
        
        if path.startswith('/'):
            return path
        elif path.startswith('~/'):
            return os.path.expanduser(path)
        elif path.startswith('./'):
            return os.path.abspath(path)
        else:
            # if it is a relative path, then it is relative to the module path
            # ex: 'data' -> '.commune/path_module/data'
            tmp_dir = c.tmp_dir() if root else cls.tmp_dir()

            if tmp_dir not in path:
                path = os.path.join(tmp_dir, path)
            if not os.path.isdir(path):
                if extension != None and extension != path.split('.')[-1]:
                    path = path + '.' + extension

            return path
    
    @classmethod
    def get_address(cls, module, **kwargs):
        return c.namespace(**kwargs).get(module, None)
    
    @classmethod
    def resolve_address(cls, address:str = None):
        if address == None:
            address = c.free_address()
        assert isinstance(address, str),  'address must be a string'
        return address
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
    def free_ports(cls, n=10, reserve:bool = False, random_selection:bool = False, **kwargs ) -> List[int]:
        free_ports = []
        avoid_ports = kwargs.pop('avoid_ports', [])
        for i in range(n):
            free_ports += [cls.free_port(reserve=reserve, 
                                         random_selection=random_selection, 
                                         avoid_ports=avoid_ports, **kwargs)]
            avoid_ports += [free_ports[-1]]
              
        return free_ports
    
    @classmethod
    def random_port(cls, *args, **kwargs):
        return cls.choice(cls.free_ports(*args, **kwargs))
    
    @staticmethod
    def random_int(*args):
        import random
        if len(args) == 1:
            return random.randint(0, args[0])
        elif len(args) == 2:
            return random.randint(args[0], args[1])
        else:
            raise ValueError('Invalid number of arguments')
    
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
    def free_address(cls, **kwargs):
        return f'{c.ip()}:{c.free_port(**kwargs)}'
    
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
    def kill_all_servers(bro, verbose: bool = True) -> {'bro': ['str','Text'], 'bro2': 'Text'}:
        '''
        Kill all of the servers
        '''
        for module in c.servers():
            if verbose:
                c.print(f'Killing {module}', color='red')
            c.kill_server(module)
            
    
    @classmethod
    def kill_all(cls, search= None):
        for module in c.servers():
            if search != None and search in module:
                cls.kill(module)
            
        


    @classmethod
    def restart_all_servers(cls, verbose: bool = True):
        '''
        Kill all of the servers
        '''
        for module in cls.servers():
            if verbose:
                c.print(f'Restarting {module}', color='red')
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
    def path2simple(cls, path:str, compress:bool = True,) -> str:

        # does the config exist

        simple_path =  path.split(deepcopy(cls.root_dir))[-1]

        if cls.path_config_exists(path):
            simple_path = os.path.dirname(simple_path)

        simple_path = simple_path.replace('.py', '')
        
        
        simple_path = simple_path.replace('/', '.')[1:]
        if compress:
            simple_path = cls.compress_name(simple_path, seperator='.')
        
        if simple_path.startswith('modules.'):
            simple_path = simple_path.replace('modules.', '')
        return simple_path
    

            
    @staticmethod
    def compress_name( name, seperator='.', suffixes = ['_module', 'module']):
        '''
        
        '''
        chunks = name.split(seperator)
        new_chunks = []
        for i, chunk in enumerate(chunks):
            if len(new_chunks)>0:
                if new_chunks[-1] == chunks[i]:
                    continue
                elif any([chunks[i].endswith(s) for s in suffixes]):
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
    def find_python_class(cls, path:str , class_index:int=0, search:str = None, start_lines:int=2000):
        import re
        
        # read the contents of the Python script file
        python_script = cls.readlines(path, end_line = start_lines, resolve=False)
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
            
        object_name = cls.find_python_class(path)
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
    def get_module(cls, path:str, verbose:bool = False, handle_error:bool=True) -> str:
        
        og_path = path
        path = cls.simple2path(path)
        path = cls.path2objectpath(path)
        return cls.import_object(path)


    @classmethod
    def module_tree(cls, search=None, 
                    mode='path', 
                    cache:bool = True,
                    update:bool = False,
                    verbose:bool = False,
                    max_age:int=1_000_000_000,) -> List[str]:
                
        if update and verbose:
            c.print('Building module tree', verbose=verbose)
        assert mode in ['path', 'object']
        module_tree = {}
        if mode == 'path':
            module_tree = {cls.path2simple(f):f for f in cls.get_module_python_paths()}

        elif mode == 'object':
            module_tree = {cls.path2simple(f):cls.path2objectpath(f) for f in cls.get_module_python_paths()}
        module_tree = {k:v for k,v in module_tree.items() if search is None or search in k}
        
        # to use functions like c. we need to replace it with module lol
        if cls.root_module_class in module_tree:
            module_tree[cls.module_path()] = module_tree.pop(cls.root_module_class)
        if cache or update:
            c.put('module_tree', module_tree, cache=cache)
        return module_tree
    
    available_modules = tree = module_tree
    @classmethod
    def list_modules(cls, search=None):
        modules = list(cls.module_tree(search).keys())
        return modules
    
    @classmethod
    def servers(cls, *args, **kwargs) -> List[str]:
        modules = list(c.namespace(*args, **kwargs).keys())
        return modules
    
    @classmethod
    def has_servera(cls, *args, **kwargs):
        return bool(len(c.servers(*args, **kwargs)) > 0)
        
    @classmethod
    def has_module(cls, module):
        return module in c.modules()
        
    
    @classmethod
    def valid_module(cls,module,**kwargs ):
        modules = c.servers(module, **kwargs)
        return bool(len(modules) > 0)
    
    @classmethod
    def tasks(cls, task = None, mode='pm2',**kwargs) -> List[str]:
        kwargs['network'] = 'local'
        kwargs['update'] = False
        modules = c.servers( **kwargs)
        tasks = getattr(cls, f'{mode}_list')(task)
        tasks = list(filter(lambda x: x not in modules, tasks))
        return tasks
    
    @classmethod
    def models(cls, *args, **kwargs) -> List[str]:
        models = c.servers(*args, **kwargs)
        models = [k for k in models if k.startswith('model')]
        return models
    @classmethod
    def datasets(cls, *args, **kwargs) -> List[str]:
        return [k for k in list(c.namespace(*args, **kwargs).keys()) if k.startswith('dataset')]
    
    @classmethod
    def datasets(cls, *args, **kwargs) -> List[str]:
        return [k for k in list(c.namespace(*args, **kwargs).keys()) if k.startswith('dataset')]
    @staticmethod
    def module_config_tree() -> List[str]:
        return [f.replace('.py', '.yaml')for f in  c.get_module_python_paths()]

    @staticmethod
    def is_imported(package:str) :
        return  bool(package in sys.modules)

    @classmethod
    def simple2path(cls, path) -> Dict[str, str]:
        module_tree = cls.module_tree()
        return module_tree[path]


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

        # find all of the python files
        for f in glob(c.root_path + '/**/*.py', recursive=True):
            if os.path.isdir(f):
                continue
            file_path, file_ext =  os.path.splitext(f)
   
            if file_ext == '.py':
                dir_path, file_name = os.path.split(file_path)
                dir_name = os.path.basename(dir_path)
                previous_dir_path = dir_path.split('/')[-2]
                
                if dir_name.lower() == file_name.lower():
                    # if the dirname is equal to the filename then it is a module
                    modules.append(f)
                # elif file_name.lower().endswith(dir_name.lower()):
                #     # if the dirname is equal to the filename then it is a module
                #     modules.append(f)
                elif file_name.lower().endswith('module'):
                    # if the dirname is equal to the filename then it is a module
                    modules.append(f)
                    
                elif 'module' in file_name.lower():
                    modules.append(f)
                elif any([os.path.exists(file_path+'.'+ext) for ext in ['yaml', 'yml']]):
                    modules.append(f)
                else:
                    # FIX ME
                    f_classes = cls.find_python_class(f, search=['commune.Module', 'c.Module'])
                    # f_classes = []
                    if len(f_classes) > 0:
                        modules.append(f)
        cls.module_python_paths = modules
        
        return modules

    @classmethod
    def dashboard(cls, *args, **kwargs):
        return cls.get_module('dashboard')(*args, **kwargs)

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
                      seperate_args:bool=False,
                      merge_kwargs :bool = True) -> dict:
        kwargs = {}
        locals_dict = locals_dict if locals_dict != None else {}
        assert isinstance(locals_dict, dict)
        kwargs.update(locals_dict)
        if merge_kwargs:
            kwargs.update(locals_dict.get('kwargs', {}))
        
        kwargs.pop('cls', None)
        kwargs.pop('self', None)

        if seperate_args:
            args = locals_dict.pop('args', [])
            return args, kwargs
        
        return kwargs
    

    get_kwargs = get_params = locals2kwargs 
        
    @classmethod
    def get_parents(cls, obj=None):
        
        if obj == None:
            obj = cls

        return list(obj.__mro__[1:-1])

    @classmethod
    def module_config_tree(cls):         
        return {m: c.simple2config(m) for m in c.modules()}
    
   
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
                             default:Any=None,
                             root: bool = False,
                             verbose: bool = False,
                             **kwargs):

        from commune.utils.dict import async_get_json
        path = cls.resolve_path(path=path, extension='json', root=root)
        try:
            data = await async_get_json(path, **kwargs)
        except Exception as e:
            if verbose:
                c.print(f'Failed to load json from {path} with error {e}')
            return default
        if isinstance(data, dict):
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
    
    
    def init_nn(self):
        import torch
        torch.nn.Module.__init__(self)
    
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
    def file_exists(cls, path:str, root:bool = False)-> bool:
        path = cls.resolve_path(path=path,  root=root)
        return os.path.exists(path)

        

    
    
    exists = exists_json = file_exists

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
        path = cls.resolve_path(path=path, root=root)
        return os.path.isdir(path)
        

    @classmethod
    def isfile(cls, path, root: bool = False):
        path = cls.resolve_path(path=path, root=root)
        return os.path.isfile(path)

    @classmethod
    def rm(cls, path, extension=None, root=False):
        path = cls.resolve_path(path=path, extension=extension, root=root)
        if os.path.exists(path):
            if os.path.isdir(path):
                cls.rmdir(path)
            else:
                os.remove(path)
            assert not os.path.exists(path)
            return {'success':True, 'message':f'{path} removed'}
        else:
            return {'success':False, 'message':f'{path} does not exist'}

    
    @classmethod
    def glob(cls,  path ='~/', files_only:bool = True, root:bool = False, recursive:bool=False):
        
        path = cls.resolve_path(path, extension=None, root=root)
        
        if os.path.isdir(path):
            path = os.path.join(path, '**')
            
        paths = glob(path, recursive=recursive)
        
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
    def walk(cls, path:str, module:str=False) -> List[str]:
        
        import os
        path_map = {}
        for root, dirs, files in os.walk(path):
            for f in files:
                path = os.path.join(root, f)
                path_map[path] = f
        return list(path_map.keys())
    
       
    ftree = walk
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
                network : str = None,
                namespace = None,
                virtual:bool = True, 
                wait_for_server:bool = False,
                trials = 3, 
                verbose: bool = False, 
                key = None,
                ignore_error:bool = False,
                **kwargs ):
        network = c.resolve_network(network)
        if key != None:
            key = cls.get_key(key)
            
        if (name == None and ip == None and port == None):
            return cls.root_module()
        
        if wait_for_server:
            cls.wait_for_server(name)
        
        if namespace == None :
            namespace = c.namespace(network, update=False)
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
            c.print(f'Connecting to {name} on {ip}:{port}', color='yellow')
        client= cls.get_client(ip=ip, port=int(port), virtual=virtual, key=key)
        
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
    def round_decimals(cls, x:Union[float, int], decimals: int=6, small_value: float=1.0e-9):
        import math
        """
        Rounds x to the number of {sig} digits
        :param x:
        :param sig: signifant digit
        :param small_value: smallest possible value
        :return:
        """
        x = float(x)
        return round(x, decimals)

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
       
        address =  c.connect('module').address
        return address
    
    
    addy = root_address
    anchor = root_module
    anchor_address = root_address

 
    
    @classmethod
    def connect_pool(cls, modules=None, *args, return_dict:bool=False, **kwargs):
        if modules == None:
            modules = c.servers(modules)
        
        module_clients =  cls.gather([cls.async_connect(m, ignore_error=True,**kwargs) for m in modules])
        if return_dict:
            return dict(zip(modules, module_clients))
        return module_clients

    client_module_path = 'module.server.client'
    server_module_path = 'module.server'
    @classmethod
    def get_client(cls, *args,virtual:bool = True, **kwargs):
        
        client_class = c.module(cls.client_module_path)
        client = client_class(*args, **kwargs)
        if virtual:
            client =  client.virtual()
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
    def port2module(cls, *args, **kwargs):
        namespace = c.namespace(*args, **kwargs)
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
        namespace = c.namespace(*args, **kwargs)
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
        
        registered_peer_addresses = []
        for peer_id, (peer_address, peer_info) in enumerate(peer_registry.items()):
            
            if isinstance(peer_info, dict):
                peer_name = f'{prefix}{peer_id}'
                peer_namespace = peer_info.get('namespace', None)
                if isinstance(peer_namespace, dict):
                    for name, address in peer_namespace.items():
                        if  not address in registered_peer_addresses:
                            remote_namespace[name+seperator+peer_name] = address
                            registered_peer_addresses.append(peer_address)
                else:
                    c.print(f'Peer {peer_name} has no namespace', color='red')
        
        c.put('remote_namespace', remote_namespace)
        
        

        return remote_namespace
        
        
    @staticmethod
    def check_response(x) -> bool:
        if isinstance(x, dict) and 'error' in x:
            return False
        else:
            return True
        
    @staticmethod
    async def async_get_peer_name(peer_address):
        peer = await c.async_connect(peer_address, namespace={}, timeout=5, virtual=False, ignore_error=True)
        if peer == None: 
            return peer
        module_name =  await peer(fn='module_name',  return_future=True)
        if c.check_response(module_name):
            return module_name
        else:
            return None
                
    @classmethod
    def local_namespace(cls, verbose:bool = False, **kwargs)-> dict:
        '''
        The module port is where modules can connect with each othe.
        When a module is served "module.serve())"
        it will register itself with the local_namespace dictionary.
        '''
        # from copy import deepcopy
        # update = False
        address2module = {}
        local_namespace = c.get('local_namespace', {})
        external_ip = cls.external_ip()
        local_namespace = {k:cls.default_ip + f":{v.split(':')[-1]}" for k,v in local_namespace.items()}
        return local_namespace
    
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
    def server_exists(cls, name:str, **kwargs) -> bool:
        return bool(name in cls.servers(**kwargs))
    
    @classmethod
    def get_port(cls, port:int = None, **kwargs)->int:
        port = port if port is not None and port != 0 else cls.free_port(**kwargs)
        while cls.port_used(port):
            port += 1   
        return port 
    
    resolve_port = get_port
    
    @classmethod
    def module_exists(cls, name:str, **kwargs) -> bool:
        namespace = c.namespace(**kwargs)
        return bool(name in namespace)
    
    
    @classmethod
    def wait_for_server(cls,
                          name: str ,
                          timeout:int = 600,
                          sleep_interval: int = 4) -> bool :
        
        start_time = cls.time()
        time_waiting = 0
        cls.local_namespace()

        while not cls.server_exists(name):
            cls.sleep(sleep_interval)
            time_waiting += sleep_interval
            c.print(f'Waiting for server {name} to start... {time_waiting} seconds', end='\r')

            if time_waiting > timeout:
                raise TimeoutError(f'Timeout waiting for server to start')
        return True
    

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
    def get_attributes(cls, search = None, obj=None):
        if obj is None:
            obj = cls
        if isinstance(obj, str):
            obj = c.module(obj)
        # assert hasattr(obj, '__dict__'), f'{obj} has no __dict__'
        attrs =  dir(obj)
        if search is not None:
            attrs = [a for a in attrs if search in a]
        return attrs

    @classmethod
    def global_namespace(cls, update=False) -> Dict:
        
        global_namespace = {
            **cls.local_namespace(),
            **cls.remote_namespace()
        }
        
        return global_namespace
    
    

    @classmethod
    def subspace_namespace(cls, netuid=None, **kwargs ) -> Dict:
        namespace = c.module('subspace')().namespace(**kwargs)
        return namespace

        
    @classmethod
    def name2address(cls, name:str, **kwargs) -> str:
        namespace = cls.namespace(**kwargs)
        address =  namespace.get(name, None)
        ip = c.ip()
    
        address = address.replace(c.default_ip, ip)
        assert ip in address, f'ip {ip} not in address {address}'
        return address
        
    @classmethod
    def namespace(cls,
                  search = None,
                  network:str=None,
                  verbose: bool = False,
                  update: bool = False,
                  max_staleness:int = 30,
                  **kwargs):
        
        network = cls.resolve_network(network)
        
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
    def namespace_options(cls,search=None) -> List[str]:
        namespace  = c.namespace()
        namespace_names = list(namespace.keys())
        namespace_addresses = list(namespace.values())
        namespace_options =  namespace_names + namespace_addresses
        if search:
            namespace_options = [o for o in namespace_options if search in o]
        return namespace_options
    
    
    
    @classmethod
    def resolve_server_name(cls, module:str = None, name:str = None, tag:str=None, tag_seperator:str='::', **kwargs):
        if module == None:
            module = cls 
        if name == None:
            if isinstance(module, str):
                module = c.module(module)
            if hasattr(module, 'module_path'):
                name = module.module_path()
            else:
                name = module.__name__
                
                
            assert name != None, f'Could not resolve name for module {module}'
        if tag != None:
            name = f'{name}{tag_seperator}{tag}'
        return name
    resolve_name = resolve_server_name
    
    @property
    def whitelist(self):
        whitelist = c.helper_whitelist
        is_module = c.is_root_module(self)
        if not is_module:
            whitelist += self.functions() + self.attributes()
        return whitelist
            
    
    
    
    @classmethod
    def serve(cls, 
              module:Any = None ,
              # name related
              name:str=None, 
              tag:str=None,
              # networking 
              address:str = None,
              ip:str=None, 
              port:int=None ,
              key = None, # key for server's identity
              refresh:bool = True, # refreshes the server's key
              whitelist:List[str] = None, # list of addresses that can connect to the server
              blacklist:List[str] = None, # list of addresses that cannot connect to the server
              wait_for_termination:bool = True, # waits for the server to terminate before returning
              wait_for_server:bool = False, # waits for the server to start before returning
              wait_for_server_timeout:int = 30, # timeout for waiting for the server to start
              wait_for_server_sleep_interval: int = 1, # sleep interval for waiting for the server to start
              verbose:bool = False, # prints out information about the server
              reserve_port:bool = False, # reserves the port for the server
              tag_seperator: str = '::', # seperator for the tag
              remote:bool = True, # runs the server remotely (pm2, ray)
              args:list = None,  # args for the module
              kwargs:dict = None,  # kwargs for the module
              
              ):
        '''
        Servers the module on a specified port
        '''
        kwargs  = kwargs if kwargs else {}
        args = args if args else []
        name = cls.resolve_server_name(module=module, name=name, tag=tag)
        tag = None
        if remote:
            remote_kwargs = cls.locals2kwargs(locals(), merge_kwargs=False)
            remote_kwargs['remote'] = False
            return cls.remote_fn('serve', name=name, kwargs=remote_kwargs, )
        
        if address != None and port == None:
            port = int(address.split(':')[-1])
        # ensure the port is free
        if port == None:
            port = cls.free_port(reserve=reserve_port)

        port = int(port)
    
        module = cls.resolve_module(module)
            
        self = module(*args, **kwargs)

        if whitelist == None:
            whitelist = self.whitelist
        if blacklist == None:
            blacklist = self.blacklist
    
        if self.server_exists(name): 
            c.print(f'Server {name} already exists', color='yellow')
            if refresh:
                if verbose:
                    c.print(f'Stopping existing server {name}', color='yellow')
                self.kill_server(name)
            else: 
                raise Exception(f'The server {name} already exists on port {existing_server_port}')

        # ensure that the module has a name
        for k in ['module_name', 'module_id', 'my_name', 'el_namo', 'name']:
            if k not in self.__dict__:
                self.__dict__[k] = name

            
        server = c.module('module.server')(ip=ip, 
                        port=port,
                        module = self,
                        name= name,
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
    def functions(cls, search = None, 
                  include_module=False):
        functions = cls.get_functions(include_module=include_module)  

        functions = list(set(functions))
        
        if isinstance(search, str):
            functions = [f for f in functions if search in f]
            
        return functions

    fns = functions
        


    @classmethod
    def get_function_signature_map(cls, obj=None, include_module:bool = False):
        function_signature_map = {}
        if isinstance(obj, str):
            obj = c.module(obj)
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
                function_signature_map[f] = {k:str(v) for k,v in cls.get_function_signature(getattr(cls, f )).items()}        
        
    
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
    


    @classmethod
    def get_peer_info(cls, peer: Union[str, 'Module']) -> Dict[str, Any]:
        if isinstance(peer, str):
            peer = cls.connect(peer)
            
        info = peer.info()
        return info
    
    
    def is_fn_allowed(self, fn_name:str) -> bool:
        whitelist = self.whitelist
        blacklist = self.blacklist
        if fn_name in whitelist and fn_name not in blacklist:
            return True
        else:
            return False
        
    def info(self , 
             include_schema: bool = False,
             include_namespace:bool = False,
             include_peers: bool = False) -> Dict[str, Any]:
        fns = [fn for fn in self.fns() if self.is_fn_allowed(fn)]
        attributes =[ attr for attr in self.attributes() if self.is_fn_allowed(attr)]
    
        info  = dict(
            address = self.address,
            functions =  fns, # get the functions of the module
            attributes = attributes, # get the attributes of the module
            name = self.module_name() if callable(self.module_name) else self.module_name, # get the name of the module
            path = self.module_path(), # get the path of the module
            chash = self.chash(), # get the hash of the module (code)
        )
        if include_peers:
            info['peers'] = self.peers()
        # EXTRA FEATURES THAT CAN BE ADDED, BUT ARE NOT INCLUDED BY DEFAULT
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

        return {k: v for k,v in cls.get_schema(*args,search=search,**kwargs).items()}
    @classmethod
    def get_schema(cls,
                                obj = None,
                                search = None,
                                code : bool = False,
                                docs: bool = False,
                                include_hidden:bool = False, 
                                include_module:bool = False,
                                defaults:bool = False,):
        
        obj = obj if obj else cls
        
        if isinstance(obj, str):
            obj = c.module(obj)
            
        function_schema_map = {}
        for fn in cls.get_functions(obj, include_module=include_module):
               
            if search != None :
                if search not in fn:
                    continue
            fn_obj = getattr(obj, fn )
            if callable(fn_obj):
                c.print(f'getting schema for {fn}')
                function_schema_map[fn] = cls.get_function_schema(fn, defaults=defaults, code=code, docs=docs)
        return function_schema_map
    
    
    @classmethod
    def bruh(cls):
        return 'fam'

    @classmethod
    def get_function_annotations(cls, fn):
        fn = cls.resolve_fn(fn)
        return fn.__annotations__
        
    @classmethod
    def get_function_schema(cls, fn:str,
                            defaults:bool=False,
                            code:bool = False,
                            docs:bool = False)->dict:
        '''
        Get function schema of function in cls
        '''
        import inspect
        fn_schema = {}
        if isinstance(fn, str):
            fn = getattr(cls, fn)
        fn_args = cls.get_function_args(fn)
        fn_schema['input']  = cls.get_function_annotations(fn=fn)
        
        if defaults:
            fn_schema['default'] = cls.get_function_defaults(fn=fn) 
            for k,v in fn_schema['default'].items(): 
                if k not in fn_schema['input'] and v != None:
                    fn_schema['input'][k] = type(v).__name__ if v != None else None
           
           
        for k,v in fn_schema['input'].items():
            v = str(v)
            if v.startswith('<class'):
                fn_schema['input'][k] = v.split("'")[1]
            elif v.startswith('typing.'):
                fn_schema['input'][k] = v.split(".")[1].lower()
            else:
                fn_schema['input'][k] = v
                

        
        fn_schema['output'] = fn_schema['input'].pop('return', {})
        
        if docs:         
            fn_schema['docs'] =  fn.__doc__ 
        if code:
            fn_schema['code'] = inspect.getsource(fn)
                

        fn_args = c.get_function_args(fn)
        fn_schema['type'] = 'static'
        for arg in fn_args:
            if arg not in fn_schema['input']:
                fn_schema['input'][arg] = 'NA'
            if arg in ['self', 'cls']:
                fn_schema['type'] = arg
                fn_schema['input'].pop(arg)
                

        return fn_schema
    

    @staticmethod
    def get_annotations(fn:callable) -> dict:
        return fn.__annotations__



    @classmethod
    def kill(cls, *modules,
             mode:str = 'pm2',
             verbose:bool = False,
             update : bool = True,
             **kwargs):

        kill_fn = getattr(cls, f'{mode}_kill')
        delete_modules = []
        for module in modules:
            killed_module =kill_fn(module, verbose=verbose, **kwargs)
            if isinstance(killed_module, list):
                delete_modules.extend(killed_module)
            elif isinstance(killed_module, str):
                delete_modules.append(killed_module)
            else:
                raise Exception(f'killed module {killed_module} is not a string or list, Somethings up')
        # update modules
        cls.update(network='local')
        return {'killed': delete_modules}

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
               name:Optional[str]=None,  
               refresh:bool=True,
               mode:str = 'pm2',
               tag:str=None, 
               tag_seperator: str = '::',
               verbose : bool = True, 
               device:str = None,
               update: bool = False,
               **extra_kwargs):
        '''
        Launch a module as pm2 or ray 
        '''
        if update:
            cls.update()
        kwargs = kwargs if kwargs else {}
        kwargs.update(extra_kwargs)
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
            c.print(f'[bold cyan]Launching[/bold cyan] [bold yellow]class:{module.__name__}[/bold yellow] [bold white]name[/bold white]:{name} [bold white]fn[/bold white]:{fn} [bold white]mode[/bold white]:{mode}', color='green')

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
        
            getattr(cls, f'{mode}_launch')(**launch_kwargs)
        else: 
            raise Exception(f'launch mode {mode} not supported')

        return name

    launch = deploy
    
    @classmethod
    def pm2_kill_all(cls, verbose:bool = True):
        for module in cls.pm2_list():
            cls.pm2_kill(module, verbose=verbose)
                
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
    def pm2_exists(cls, name:str):
        return name in cls.pm2_list()
    
    @staticmethod
    def pm2_start(path:str , 
                  name:str,
                  cmd_kwargs:str = None, 
                  refresh: bool = True,
                  verbose:bool = True,
                  force : bool = True,
                  interpreter : str = None,
                  **kwargs):
        if c.pm2_exists(name) and refresh:
            c.pm2_kill(name, verbose=verbose)
            
        cmd = f'pm2 start {path} --name {name}'
        if force:
            cmd += ' -f'
            
        if interpreter != None:
            cmd += f' --interpreter {interpreter}'
            
        if cmd_kwargs != None:
            cmd += f' -- '
            if isinstance(cmd_kwargs, dict):
                for k, v in cmd_kwargs.items():
                    cmd += f'--{k} {v}'
            elif isinstance(cmd_kwargs, str):
                cmd += f'{cmd_kwargs}'
                

        c.print(f'[bold cyan]Starting (PM2)[/bold cyan] [bold yellow]{name}[/bold yellow]', color='green')
            
        # c.print(f'[bold cyan]Starting (PM2)[/bold cyan] [bold yellow]{name}[/bold yellow]', color='green')
        return c.cmd(cmd, verbose=verbose,**kwargs)
        
    @classmethod
    
    def pm2_launch(cls, 
                   module:str = None,  
                   fn: str = 'serve',
                   name:Optional[str]=None, 
                   tag : str = None,
                   args : list = None,
                   kwargs: dict = None,
                   device:str=None, 
                   interpreter:str='python3', 
                   no_autorestart: bool = False,
                   verbose: bool = False , 
                   force:bool = True,
                   meta_fn: str = 'module_fn',
                   tag_seperator:str = '::',
                   refresh:bool=True ):
    

        if module == None:
            module = cls.module_name()
        elif hasattr(module, 'module_name'):
            module = module.module_name()
            
        # avoid these references fucking shit up
        args = args if args else []
        kwargs = kwargs if kwargs else {}
            
        # convert args and kwargs to json strings
        kwargs =  {
            'module': module,
            'fn': fn,
            'args': args,
            'kwargs': kwargs
            
        }
        kwargs_str = json.dumps(kwargs).replace('"', "'")
        name = c.resolve_server_name(module=module, name=name, tag=tag, tag_seperator=tag_seperator) 
        # build command to run pm2
        command = f" pm2 start {c.module_file()} --name {name} --interpreter {interpreter}"
        if no_autorestart:
            command = command + ' ' + '--no-autorestart'
        if force:
            command += ' -f '
        command = command + ''

        command = command +  f' -- --fn {meta_fn} --kwargs "{kwargs_str}"'
        env = {}
        if device != None:
            if isinstance(device, int):
                env['CUDA_VISIBLE_DEVICES']=str(device)
            if isinstance(device, list):
                env['CUDA_VISIBLE_DEVICES']=','.join(list(map(str, device)))
                
                
                
        if refresh:
            cls.pm2_kill(name)  

        if verbose:
            c.print(f'Launching {module} with command: {command}', color='green')
            
        stdout = cls.run_command(command, env=env, verbose=verbose)
        # c.print(f'STDOUT: \n {stdout}', color='green')
        return stdout
    
    
    @classmethod
    def register(cls, *args, **kwargs):
        return c.module('subspace')().register(*args,**kwargs)
    
    @classmethod
    def pm2_kill(cls, name:str, verbose:bool = True):
        output_list = []
        pm2_list = cls.pm2_list()
        
        if name in pm2_list:
            rm_list = [name]
        else:
            rm_list = [ p for p in pm2_list if p.startswith(name)]
        for n in rm_list:
            c.print(f'Killing {n}', color='red')
            cls.run_command(f"pm2 delete {n}", verbose=False)
            
        return name
    
    
    @classmethod
    def pm2_restart(cls, name:str = None, verbose:bool=False):
        pm2_list = cls.pm2_list()
            
        restarted_modules = []
        for module in pm2_list:
            if module.startswith(name) or name in ['all']:
                if verbose:
                    c.print(f'Restarting {module}', color='cyan')
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
            c.print(stdout,color='green')
        return stdout

    pm2_dir = os.path.expanduser('~/.pm2')
    @classmethod
    def pm2_logs(cls, module:str, start_line=0, end_line=-1, verbose=True, mode='cmd'):
        if mode == 'local':
            path = f'{cls.pm2_dir}/logs/{module}-out.log'.replace(':', '-')
            return c.get_text(path, start_line=start_line, end_line=end_line)
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
            c.print('Argparse Args: ',args, color='cyan')
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
        return c.module('model.transformer').learn(*args, **kwargs)
        
    
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
            module_class = c.module(module)
            
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
    def fn(cls, module:str, fn:str , args:list = None, kwargs:dict= None):
        module = c.module(module)
        is_self_method = bool(fn in module.self_methods())
        if is_self_method:
            module = module()
            fn = getattr(module, fn)
        else:
            fn =  getattr(module, fn)
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
            
        
        if len(args)>0 or len(kwargs)>0:
            return fn(*args, **kwargs)
        else:
            return fn()
    module_fn = fn
    
    @classmethod
    def module(cls,module: Any = None ,*args, **kwargs):
        '''
        Wraps a python class as a module
        '''
        
        if module is None:
            return cls
        if isinstance(module, str):
            modules = c.modules()
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
            def module_path(cls) -> str:
                return module_class.__name__.lower()
 
            @classmethod
            def functions(cls):
                return cls.get_functions(module)


        if is_class:
            return ModuleWrapper
        else:
            return ModuleWrapper()
        
        
            
        # return module


    m = module

    # UNDER CONSTRUCTION (USE WITH CAUTION)
    
    @classmethod
    def modulefn(cls, module, fn, *args, **kwargs):
        return getattr(c.module(module), fn)(*args, **kwargs)
        
    
    
    def setattr(self, k, v):
        setattr(self, k, v)
        

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
    def merge(cls, b, 
                        include_hidden:bool=True, 
                        allow_conflicts:bool=True, 
                        verbose: bool = False):
        
        '''
        Merge the functions of a python object into the current object (a)
        '''
        a =  cls
        
        for b_fn_name in dir(b):
            
            if include_hidden == False:
                #i`f the function name starts with __ then it is hidden
                if b_fn_name.startswith('__'):
                    continue
                
            # if the function already exists in the object, raise an error
            if  allow_conflicts:
                if hasattr(a, b_fn_name):
                    if verbose:
                        c.print(f'Warning: overriding function {b_fn_name} already exists in {a}', color='yellow')
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
                        c.print(error_fn_list, 'DEBUG')        
        return a
   
    @classmethod
    def nest_asyncio(cls):
        import nest_asyncio
        try:
            nest_asyncio.apply()
        except RuntimeError as e:
            c.print('Broooo, nest-asyncio doesnt work fam')
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
    def libs(cls):
        return list(cls.lib2version().values())
    
    @classmethod
    def ensure_lib(cls, lib:str, verbose:bool=False):
        if  cls.pip_exists(lib):
            return {'lib':lib, 'version':cls.version(lib), 'status':'exists'}
        elif cls.pip_exists(lib) == False:
            cls.pip_install(lib, verbose=verbose)
        return {'lib':lib, 'version':cls.version(lib), 'status':'installed'}
    
    ensure_package = ensure_lib
    @classmethod
    def pip_install(cls, lib:str, verbose:str=True):
        if lib in c.modules():
            c.print(f'Installing {lib} Module from local directory')
            lib = c.resolve_module(lib).dirpath()
            
        return cls.cmd(f'pip install {lib}', verbose=verbose)

    def install(self, lib:str, verbose:bool=True):
        return self.pip_install(lib, verbose=verbose)

    @classmethod
    def pip_exists(cls, lib:str, verbose:str=True):
        return bool(lib in cls.libs())
    
    
    @classmethod
    def lib2version(cls, lib:str = None) -> dict:
        lib2version = {}
        for l in cls.pip_list():
            name = l.split(' ')[0].strip()
            version = l.split(' ')[-1].strip()
            if len(name) > 0:
                lib2version[name] = version
            if lib != None and lib == name:
                return version
            
        return lib2version
    @classmethod
    def version(cls, lib:str=library_name):
        lines = [l for l in cls.cmd(f'pip list').split('\n') if l.startswith(lib)]
        if len(lines)>0:
            return lines[0].split(' ')[-1].strip()
        else:
            return f'No Library Found {lib}'
    
    @classmethod
    def get_external_ip(cls, *args, **kwargs) ->str:
        return cls.import_object('commune.utils.network.get_external_ip')(*args, **kwargs)

    @classmethod
    def external_ip(cls, *args, **kwargs) -> str:
        if not hasattr(cls, '__external_ip__'):
            cls.__external_ip__ =  cls.get_external_ip(*args, **kwargs)
        ip = cls.__external_ip__
        assert ip != None, 'External IP is None'
        assert ip not in ['0.0.0.0', '127.0.0.1'], 'External IP is'
        return ip 
    
    @classmethod
    def ip(cls,external=True, **kwargs) -> str:
        if external:
            ip =  cls.external_ip(**kwargs)

        else:
            ip =  '127.0.0.1'
        return ip
    
    @classmethod
    def resolve_ip(cls, ip=None, external:bool=True) -> str:
        if ip == None:
            if external:
                ip = c.external_ip()
            else:
                ip = '0.0.0.0'
        assert isinstance(ip, str)
        return ip
        
    @classmethod
    def get_external_ip(cls, *args, max_trials=4, **kwargs) ->str:
        if max_trials == 0:
            return None
        try:
            return cls.import_object('commune.utils.network.get_external_ip')(*args, **kwargs)
        except Exception as e:
            return cls.get_external_ip(*args, max_trials=max_trials-1, **kwargs)
            
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
    
    @staticmethod
    def format_data_size(x: Union[int, float], fmt:str='b', prettify:bool=False):
        assert type(x) in [int, float], f'x must be int or float, not {type(x)}'
        fmt2scale = {
            'b': 1,
            'kb': 1000,
            'mb': 1000**2,
            'gb': 1000**3,
            'GiB': 1024**3,
            'tb': 1000**4,
        }
            
        for i, f in enumerate(fmts):
            if fmt == f:
                break
            else:
                x = x/1000
                
        
        if prettify:
            return f'{x:.2f} {f}'
        else:
            return x
        

    @classmethod
    def most_free_gpu(cls, 
                      free_gpu_memory:dict = None,
                      mode : bool = 'int',
                      **kwargs) -> Union[int, Dict[str, int]]:
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
        if mode == 'tuple':
            return most_available_gpu_tuples[0]
        elif mode == 'dict': 
            return {most_available_gpu_tuples[0][0]: most_available_gpu_tuples[0][1]}
        elif mode == 'int':
            return most_available_gpu_tuples[0][0]
        elif mode == 'str':
            return str(most_available_gpu_tuples[0][0])
        else:
            raise ValueError(f'Invalid mode {mode}')
    
    

    @classmethod
    def most_free_gpus(cls, 
                       n:int=None,
                      free_gpu_memory:dict = None,
                      mode : str = 'dict',
                      fmt:str='b',
                      **kwargs) -> Union[int, Dict[str, int]]:
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

        if n == None:
            n = len(most_available_gpu_tuples)
        if mode == 'dict': 
            return {most_available_gpu_tuples[i][0]: c.format_data_size(most_available_gpu_tuples[i][1], fmt=fmt) for i in range(n)}
        elif mode == 'tuple':
            return [(i,c.format_data_size(most_available_gpu_tuples[i][0], fmt=fmt)) for i in range(n)]
        else:
            return [c.format_data_size(most_available_gpu_tuples[i][0], fmt=fmt) for i in range(n)]
        
    
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
                gpu_id = cls.most_free_gpu()
                
            device = f'cuda:{gpu_id}'
        
            if verbose:
                device_info = cls.gpu_info(gpu_id)
                c.print(f'Using device: {device} with {device_info["free"]} GB free memory', color='yellow')
        return device  
    
    @classmethod
    def param_keys(cls, model:'nn.Module' = None)->List[str]:
        model = c.resolve_model(model)
        return list(model.state_dict().keys())
    
    @classmethod
    def params_map(cls, model, fmt='b'):
        params_map = {}
        state_dict = c.resolve_model(model).state_dict()
        for k,v in state_dict.items():
            params_map[k] = {'shape': list(v.shape) ,
                             'size': cls.get_tensor_size(v, fmt=fmt),
                             'dtype': str(v.dtype),
                             'requires_grad': v.requires_grad,
                             'device': v.device,
                             'numel': v.numel(),
                             
                             }
            
        return params_map
    

    

    @classmethod
    def get_num_params(cls, model:'nn.Module' = None)->int:
        import numpy as np
        from torch import nn
        model = c.resolve_model(model)
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        num_params = sum([np.prod(p.size()) for p in model_parameters])
        return num_params

    get_model_params = get_num_params
    @classmethod
    def get_tensor_size(cls, tensor:'torch.Tensor' = None, fmt:str='b') -> float:
        if tensor is None:
            import torch
            tensor = torch.rand(1)
        tensor_size =  tensor.nelement() * tensor.element_size()
        return c.format_data_size(tensor_size, fmt=fmt)
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
            c.print('Updating...', color='yellow')
            modules = c.servers()
            c.print(f'Modules (n): {modules}', color='cyan')
            c.print(modules, color='purple')
            cls.update()
            cls.sleep(period)
            
    def model_size(self, **kwargs ):
        return self.get_model_size(model=self, **kwargs)
    
    @classmethod
    def model_shortcuts(cls, **kwargs):
        return  c.module('model.transformer').shortcuts
    
    

    
    @classmethod
    def get_empty_model(cls, model,
                        verbose: bool = False,
                        trust_remote_code:bool=True,
                        init_device:str = 'meta',
                        **kwargs):
        model = c.model_shortcuts().get(model, model)
        from transformers import  AutoModelForCausalLM, AutoModel, AutoConfig
        from accelerate import init_empty_weights
        
        kwargs['trust_remote_code'] = trust_remote_code
        model = c.module('model.transformer').shortcuts.get(model, model)

        if isinstance(model, str):
            if verbose:
                c.print(f'loading config model from {model}...')

            config = AutoConfig.from_pretrained(model, **kwargs)
            config.init_device=init_device
            config_dict = config.to_dict()
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(config,  **kwargs)
                
                
        return model
    
    @classmethod
    def init_empty_weights(cls, *args, **kwargs):
        from accelerate import init_empty_weights

        return init_empty_weights(*args, **kwargs)
        
        
    @classmethod
    def get_model_size(cls, 
                       model: 'nn.Module',
                       model_inflation_ratio: float = 1.0, 
                       fmt = 'b',
                       keys:List[str]=None):
        
        # get the size of the model by initializing an empty model
        model = c.resolve_model(model)
            
        params = {}
        size_in_bytes = 0 
        for name, param in model.state_dict().items():
            if keys != None and name not in keys:
                continue
            
            size_in_bytes += cls.get_tensor_size(param)
          
        return c.format_data_size(size_in_bytes * model_inflation_ratio, fmt=fmt)


    @classmethod
    def resolve_model(cls, model):
        if isinstance(model, str):
            model = c.get_empty_model(model)
        return model
        
    @classmethod
    def params_size_map(cls, 
                       model: str,
                       block_prefix:str = 'layers',
                       fmt= 'b',
                       keys:List[str]=None):
        
        
        
        # get the size of the model by initializing an empty model
        model = c.resolve_model(model)
        
        params = {}
        size_in_bytes = 0 
        
        for name, param in model.state_dict().items():
            params_size = c.format_data_size(cls.get_tensor_size(param), fmt=fmt)
            if name.startswith(block_prefix):
                
                idx = name.replace(block_prefix+'.','').split('.')[0]
                block_name = f'{block_prefix}.{idx}'
                if block_name not in params:
                    params[block_name] = 0
                params[block_name] += params_size
            else:
                params[name] = params_size
                        
        return params


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
    def error(cls, *args, **kwargs):
        logger = cls.resolve_logger()
        return logger.error(*args, **kwargs)
    
    @classmethod
    def debug(cls, *args, **kwargs):
        logger = cls.resolve_logger()
        return logger.debug(*args, **kwargs)
    
    @classmethod
    def warning(cls, *args, **kwargs):
        logger = cls.resolve_logger()
        return logger.warning(*args, **kwargs)
    
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
    dict_rm = dict_delete
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
        
        if hasattr(data, 'hex'):
            return data.hex()
        else:
            return bytes.decode(data, mode)
    
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

               
    @classmethod
    def get_keys(cls,*args, **kwargs ):
        return c.module('key').get_keys(*args, **kwargs )
    
    @classmethod
    def rm_keys(cls,*args, **kwargs ):
        return c.module('key').rm_keys(*args, **kwargs )
    
    @classmethod
    def key2address(cls,*args, **kwargs ):
        return c.module('key').key2address(*args, **kwargs )

    @classmethod
    def address2key(cls,*args, **kwargs ):
        return c.module('key').address2key(*args, **kwargs )
    
    @classmethod
    def get_key_for_address(cls, address:str):
         return c.module('key').get_key_for_address(address)

    @classmethod
    def key_info(cls, key:str = None, **kwargs):
        return c.module('key').key_info(key, **kwargs)
    
    @classmethod
    def get_key(cls,key:str = None ,mode='commune', **kwargs) -> None:
     
        mode2module = {
            'commune': 'key',
            'subspace': 'subspace.key',
            'substrate': 'web3.account.substrate',
            'evm': 'web3.account.evm',
            'aes': 'key.aes',
            }
        key = cls.resolve_keypath(key)
        if 'Keypair' in c.type_str(key):
            return key
        module = c.module(mode2module[mode])
        if hasattr(module, 'get_key'):
            key = module.get_key(key, **kwargs)
        else:
            key = module(key, **kwarg)

        return key
    
        
        

            
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
        password = c.resolve_password(password)
        data = c.python2str(data)
        

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
        password = c.resolve_password(password)
        key = c.module('key.aes')(password)
        
        if not c.is_encrypted(data, prefix=prefix):
            return data
        
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
                    c.print(f'Exception: Wrong Password, try another',color='red')
            else:
                raise Exception(f'could not decrypt data, try another pasword')
        
        return data
    enc = encrypt
    dec = decrypt
    cache = {}
    cache = {}
    @classmethod
    def put_cache(cls,k,v ):
        cls.cache[k] = v
    
    @classmethod
    def get_cache(cls,k, default=None, **kwargs):
        v = cls.cache.get(k, default)
        return v

    def auth(self,*args,  key=None, **kwargs):
        key = self.resolve_key(key)
        return self.module('subspace')().auth(*args, key=key, **kwargs)
    
    @classmethod
    def call(cls,  *args, loop=None, **kwargs) -> None:
        loop = cls.get_event_loop()
        return loop.run_until_complete(cls.async_call(*args, **kwargs))
    
    @classmethod
    async def async_call(cls,
                         module : str,
                         fn : str,
                         *args,
                         network : str = None,
                         key : str = None,
                         timeout : int = 4,
                         **kwargs) -> None:
        network = c.resolve_network(network)
        
        if isinstance(module, str) and fn == None:
            
            module, fn = '.'.join(module.split('.')[:-1]),  module.split('.')[-1],
            pool_mode = False
            
            while module.endswith('.'):
                pool_mode = True
                module = module[:-1]
            if pool_mode:
                module = c.servers(module)
            
        if fn == None:
            fn = 'forward'
            
        if isinstance(module, str):
            module = await cls.async_connect(module, network=network, key=key, virtual=False)
            
        
        return module.__call__(fn=fn, args=args, kwargs=kwargs, timeout=timeout)


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
            modules = c.servers(modules)
            
        modules = cls.shuffle(modules)[:n]
        assert isinstance(modules, list), 'modules must be a list'
        c.print(f'Calling {fn} on {len(modules)} modules', color='green')
        jobs = []
        for m in modules:
            job = cls.async_call(m, fn, *args, **kwargs)
            jobs.append(job)
        
        responses = await asyncio.gather(*jobs)
        
        is_error = lambda r: isinstance(r, dict) and 'error' in r
        successes  = [r for r in responses if not is_error(r)]
        errors = [r for r in responses if is_error(r)]
        
        if len(successes) == 0:
            c.print(f'ERRORS {errors}', color='red')
        return successes[0]
    

    @classmethod
    def resolve_fn_module(cls, fn, module=None ) -> str:
    
        if module == None and len(fn.split('.')) > 1:
            module = '.'.join(fn.split('.')[:-1])
            module = cls.connect(module)
        
        return  fn, module

    
    def resolve_key(self, key: str = None) -> str:
        if key == None:
            key = self.resolve_keypath(key)
        key = self.get_key(key)
        return key  
    
    
    @classmethod
    def type_str(cls, x):
        return type(x).__name__
                
    @classmethod  
    def keys(cls, *args, **kwargs):
        return c.module('key').keys(*args, **kwargs)
    


    
    @classmethod
    def set_key(self, key:str = None, **kwargs) -> None:
        if key == None:
            key = self.name()
        key = self.get_key(key)
        self.key = key
        return key
    
    @classmethod
    def resolve_keypath(cls, key = None):
        if key == None:
            key = cls.module_path()
        return key
    def create_key(cls , key = None):
        key = cls.resolve_keypath(key)
        return c.module('key').create_key(key)
    
    @classmethod
    def add_key(cls, key, *args,  **kwargs):
        return c.module('key').add_key(key, *args, **kwargs)
    

    def sign(self, data:dict  = None, key: str = None) -> bool:
        key = self.resolve_key(key)
        return key.sign(data) 
    

    def timestamp_to_iso(timestamp):
        import datetime
        # Convert timestamp to datetime object
        dt = datetime.datetime.fromtimestamp(timestamp)

        # Format datetime object as ISO date string
        iso_date = dt.date().isoformat()

        return iso_date

       
    
    @classmethod
    def verify(cls, auth, module='subspace', **kwargs ) -> bool:    
        return c.module(module)(**kwargs).verify(auth)
        
    
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
    def is_encrypted(cls, data, prefix='AESKEY'):
        if isinstance(data, str):
            if data.startswith(prefix):
                return True
        elif isinstance(data, dict):
            return bool(data.get('encrypted', False) == True)
        else:
            return False
        
        
    
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
    def network(cls) -> str:
        return c.resolve_network()
    net = network
    


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
        c.print(f'reserving {port}')
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
            # if zero then do all fam, tehe
            ports = list(reserved_ports.keys())
        elif len(ports) == 1 and isinstance(ports[0],list):
            ports = ports[0]
        ports = list(map(str, ports))
        reserved_ports = {rp:v for rp,v in reserved_ports.items() if not any([p in ports for p in [str(rp), int(rp)]] )}
        c.print(reserved_ports)
        cls.put(var_path, reserved_ports, root=root)
        return cls.reserved_ports()
    
    
    unresports = unreserve_ports
    @classmethod
    def fleet(cls, *tags, n=1, **kwargs):
        if len(tags) == 0:
            tags = list(range(n))
            
        for tag in tags: 
            cls.serve(tag=tag, **kwargs)



    
    @classmethod
    def client(cls, *args, **kwargs) -> 'Client':
        return c.module('module.client')(*args, **kwargs)
    
    # @classmethod
    # def serializer(cls, *args, **kwargs) -> 'Serializer':
    #     return c.module('module.server.serializer')(*args, **kwargs)

    
    @classmethod
    def serialize(cls, data, metadata=None,to_json = False, **kwargs):
        metadata = metadata or {}
        if not isinstance(data, dict):
            data = dict(value=data)
        serializer = c.module('serializer')
        proto_data =  serializer.serialize(data=data, metadata=metadata ,**kwargs)
        if to_json:
            proto_data = cls.proto2json(proto_data)
            
        return proto_data

    @classmethod
    def proto2json(cls, data):
        from google.protobuf.json_format import MessageToJson
        return MessageToJson(data)


    @classmethod
    def json2proto(cls, data):
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
                x =  [cls.determine_type(item.strip()) for item in list_items]
                if len(x) == 1 and x[0] == '':
                    x = []
                return x
       
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
    
    @classmethod
    def resolve_port_range(cls, port_range: list = None) -> list:
        return cls.get_port_range(port_range)
        return port_range

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
                             verbose:bool = True,
                             add_peer = True):
        
        peer_registry = await cls.async_get_json('peer_registry', default={}, root=True)


        peer_info = await cls.async_call(module=peer_address, 
                                              fn='info',
                                              include_namespace=True, 
                                              timeout=timeout)
        
        if add_peer:
            await cls.async_call(module=peer_address, 
                                              fn='add_peer',
                                              args=[cls.root_address],
                                              include_namespace=True, 
                                              timeout=timeout)
        

        if 'error' in peer_info:
            if verbose:
                c.print(f'Error adding peer {peer_address} due to {peer_info["error"]}',color='red')
            return None    
        else:
            if verbose:
                c.print(f'Successfully added peer {peer_address}', color='green')
        
            
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
            if 'success' in x and x['success'] == False:
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
    def resolve_network(cls, network=None):
        config = c.config()
        if network == None:
            network = config['network']
        return network
    get_network = resolve_network
    @classmethod
    def set_network(cls, network=None):
        config = c.config()
        assert network in config['networks'], f'Network {network} not found in {config["networks"]}'
        config['network'] = network
        c.save_config(config)
        return network
    setnet = set_network
    
    @classmethod
    def get_network(self, network=None):
        config = self.config()
        network = config['network']
        return network
    getnet = get_network
    resnet = resolve_network
    
    @classmethod
    def update(cls, 
               network: str = None,
               verbose:bool = True,
               namespace: bool = True,
               module_tree: bool = True,
               ):
        
        
        c.namespace(network=network,verbose=verbose, update=True)
        
    
        
        
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
    def readlines(self, path:str,
                  start_line:int = 0,
                  end_line:int = 0, 
                  root=False, 
                  resolve:bool = True) -> List[str]:
        # Get the absolute path of the file
        if resolve:
            path = self.resolve_path(path, root=root)
        # Read the contents of the file
        with open(path, 'r') as file:
            lines = file.readlines()
            if end_line == 0 :
                if start_line == 0 :
                    start_line = 0
                    end_line = len(lines)
                elif start_line > 0:
                    end_line = start_line
                    start_line = 0
                elif start_line < 0:
                    start_line = len(lines) + start_line
                    end_line = len(lines)
            
            assert start_line >= 0, f"start_line must be greater than or equal to 0"
            assert end_line > start_line, f"end_line must be less than or equal to {len(lines)}"
                
            lines = lines[start_line:end_line]
        lines = '\n'.join(lines)
        return lines
    
    
    @classmethod
    def get_text(cls, 
                 path: str, 
                 start_byte:int = 0,
                 end_byte:int = 0,
                 start_line :int= None,
                 end_line:int = None,
                  root=False, ) -> str:
        # Get the absolute path of the file
        
        path = cls.resolve_path(path, root=root)
        # Read the contents of the file
        with open(path, 'rb') as file:
        
                
            file.seek(0, 2) # this is done to get the fiel size
            file_size = file.tell()  # Get the file size
            if start_byte < 0:
                start_byte = file_size - start_byte
            if end_byte <= 0:
                end_byte = file_size - end_byte 
            chunk_size = end_byte - start_byte + 1
            file.seek(start_byte)
            content = file.read(chunk_size).decode()
            c.print(path)
            if start_line != None or end_line != None:
                if end_line == None:
                    end_line = len(content) 
                if start_line == None:
                    start_line = 0
                content = content.split('\n')
                content = '\n'.join(content[start_line:end_line])


        return content
    
    load_text = get_text


    @classmethod
    def free_gpu_memory(cls, 
                     max_gpu_ratio: float = 1.0 ,
                     reserved_gpus: bool = False,
                     buffer_memory: float = 0,
                     fmt = 'b') -> Dict[int, float]:
        import torch
        free_gpu_memory = {}
        
        buffer_memory = c.resolve_memory(buffer_memory)
        
        gpu_info_map = cls.gpu_map()
        gpus = [int(gpu) for gpu in gpu_info_map.keys()] 
        
        if  reserved_gpus != False:
            reserved_gpus = reserved_gpus if isinstance(reserved_gpus, dict) else cls.copy(cls.reserved_gpus())
            assert isinstance(reserved_gpus, dict), 'reserved_gpus must be a dict'
            
            for r_gpu, r_gpu_memory in reserved_gpus.items():
                gpu_info_map[r_gpu]['total'] -= r_gpu_memory
               
        for gpu_id, gpu_info in gpu_info_map.items():
            if int(gpu_id) in gpus or str(gpu_id) in gpus:
                gpu_memory = max(gpu_info['total']*max_gpu_ratio - gpu_info['used'] - buffer_memory, 0)
                if gpu_memory <= 0:
                    continue
                free_gpu_memory[gpu_id] = c.format_data_size(gpu_memory, fmt=fmt)
        
        assert sum(free_gpu_memory.values()) > 0, 'No free memory on any GPU, please reduce the buffer ratio'

                
        return cls.copy(free_gpu_memory)
    
    
    free_gpus = free_gpu_memory

    @classmethod
    def mkdir( cls, path = 'bro', exist_ok:bool=True):
        """ Makes directories for path.
        """
        path = cls.resolve_path(path)
        return os.makedirs( path , exist_ok=exist_ok) 
        
    def rm_module(self, module):
        module = module.replace('.','/')
        module_path = os.path.join(c.modules_path, module)
        self.rm(module_path)

    @staticmethod
    def repo2module( repo, module = None):
        if module == None:
            module = os.path.basename(repo).replace('.git','').replace(' ','_').replace('-','_').lower()
        
        c.new_module(module=module, repo=repo)
        return {'module':module, 'repo':repo, 'status':'success'}
        
    
    def rm_module_code(cls, module):
        module = module.replace('.','/')
        module_path = c.resolve_module_path(module)
        cls.rm(module_path)
    @classmethod
    def new_module( cls,
                   module : str = None,
                   repo : str = None,
                   base : str = 'base',
                   overwrite : bool  = False,
                   module_type : str ='dir'):
        """ Makes directories for path.
        """
        if module == None: 
            assert repo != None, 'repo must be specified if module is not specified'
            module = os.path.basename(repo).replace('.git','').replace(' ','_').replace('-','_').lower()
        module_path = 'path'
        module = module.replace('.','/')
        assert c.has_module(module) == False or overwrite, f'Module {module} already exists'
        module_path = os.path.join(c.modules_path, module)
        
        
        if overwrite: 
            c.rm(module_path)
        
        if repo != None:
            # Clone the repository
            c.cmd(f'git clone {repo} {module_path}', verbose=True)
            # Remove the .git directory
            c.cmd(f'rm -rf {module_path}/.git', verbose=True)
        if module == None:
            assert repo != None, 'repo must be specified if module is not specified'
            module = os.path.basename(repo).replace('.git','').replace(' ','_').replace('-','_').lower()
        
        if module_type == 'dir':
            c.mkdir(module_path, exist_ok=True)
            c.print(f'Created module {module} at {module_path}')
        else:
            raise ValueError(f'Invalid module_type: {module_type}, options are dir, file')
        
        base_module = c.module(base)
        base_code = base_module.code()
        base_config = base_module.config()
        module = module.replace('/','_') # replace / with _ for the class name
        
        module_config_path = f'{module_path}/{module}.yaml'

        module_code_path =f'{module_path}/{module}.py'
        module_code_lines = []
        class_name = module[0].upper() + module[1:] # capitalize first letter
        class_name = ''.join([m.capitalize() for m in module.split('_')])
        for code_ln in base_code.split('\n'):
            if all([ k in code_ln for k in ['class','c.Module', ')', '(']]):
                indent = code_ln.split('class')[0]
                code_ln = f'{indent}class {class_name}(c.Module):'
            module_code_lines.append(code_ln)
        module_code = '\n'.join(module_code_lines)
        c.put_text(module_code_path, module_code)
        c.save_yaml(module_config_path, base_config)
        
        c.update()
        
    make_dir= mkdir
    @classmethod
    def max_gpu_memory(cls, memory:Union[str,int] = None,
                       mode:str = 'most_free', 
                       min_memory_ratio = 0.0,
                       reserve:bool = False, 
                       buffer_memory = '5gb',
                       free_gpu_memory: dict = None,
                       saturate:bool = False,
                       fmt:str = 'b',
                       decimals:int = 3,
                       **kwargs):
        
        
        memory = cls.resolve_memory(memory)
        min_memory = min_memory_ratio * memory
        buffer_memory = c.resolve_memory(buffer_memory) # to bytes
        
        assert memory > 0, f'memory must be greater than 0, got {memory}'
        free_gpu_memory = free_gpu_memory if free_gpu_memory else cls.free_gpu_memory(**kwargs)
        total_gpu_memory = sum(free_gpu_memory.values())
        # free_gpu_memory = {k:v for k,v in free_gpu_memory.items() if v > min_memory}
        gpus = list(free_gpu_memory.keys()) 
        total_gpu_memory = total_gpu_memory - buffer_memory*len(gpus)
        
        
        
        assert memory < total_gpu_memory, f'model size {memory} is larger than total gpu memory {total_gpu_memory}, over gpus {gpus}'
        unallocated_memory = memory
        # max_memory = {}
        max_memory = {}
        
        
        free_gpu_memory = {k:v-buffer_memory for k,v in free_gpu_memory.items()}
        
        
        selected_gpus = []
        gpu = None
        gpu_memory = 0
        while unallocated_memory > 0:
            if gpu_memory == 0:
                gpu = cls.most_free_gpu(free_gpu_memory=free_gpu_memory)
                gpu_memory =  free_gpu_memory[gpu]
            
            c.print({'unallocated_memory':unallocated_memory,'gpu': gpu, 'max_memory':max_memory, 'gpu_memory':gpu_memory, 'free_gpu_memory':free_gpu_memory})

            if gpu in max_memory:
                continue
            
            if gpu_memory < min_memory:
                continue
                
  
            allocated_memory = min(gpu_memory, unallocated_memory)
            c.print(f'Allocated {allocated_memory} to gpu {gpu}')
            unallocated_memory -= allocated_memory
            max_memory[gpu] = allocated_memory
            free_gpu_memory[gpu] -= allocated_memory
            gpu_memory = free_gpu_memory[gpu]
        max_memory = {k:int(v) for k,v in max_memory.items() if v > 0}
        
        if reserve:
            
            cls.reserve_gpu_memory(max_memory)
            
            
        if saturate:
            free_gpu_memory = cls.free_gpu_memory()
            max_memory = {gpu:free_gpu_memory[gpu] for gpu in max_memory.keys()}
            
            
        max_memory = {k:c.round_decimals(c.format_data_size(v, fmt=fmt), decimals=decimals) for k,v in max_memory.items()}
        
        return max_memory
            
    @classmethod
    def resolve_module_path(cls, module=None):
        module_path = 'module'
        if isinstance(module, str):
            module_path = c.modules_path + '/' + module.replace('.','/')
        
        return module_path

    @classmethod
    def resolve_module(cls, module=None):
        if module == None:
            module = cls
        if isinstance(module, str):
            module = c.module(module)
        
        return module
            
            
    @classmethod
    def resolve_memory(cls, memory: Union[str, int, float]) -> str:
                    
        scale_map = {
            'kb': 1e3,
            'mb': 1e6,
            'gb': 1e9,
            'b': 1,
        }
        if isinstance(memory, str):
            scale_found = False
            for scale_key, scale_value in scale_map.items():
                
                
                if isinstance(memory, str) and memory.lower().endswith(scale_key):
                    memory = int(int(memory[:-len(scale_key)].strip())*scale_value)
                    
    
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
                
        c.print(f'unreserving {gpu_memory}')
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
    
    # @classmethod
    # def remote(cls, name:str = None, remote :str = False,**remote_kwargs):
    #     def decorator(fn):
    #         if name is None:
    #             name = fn.__name__
    #         def inner_function(**kwargs):
    #             remote = kwargs.pop('remote', remote)
    #             if remote:
    #                 kwargs['remote'] = False
    #                 return cls.launch(fn=fn, kwargs=kwargs, name=name, **remote_kwargs)
    #             else:
    #                 return fn(**kwargs)
                    
    #         # Return the inner function (wrapper)
    #         return inner_function
    
    #     # Return the decorator function
    #     return decorator


    @classmethod
    def remote_fn(cls, 
                    fn: str='train', 
                    module: str = None,
                    args : list = None,
                    kwargs : dict = None, 
                    tag: str = None,
                    refresh : bool =True,
                    tag_seperator : str = '::',
                    name : str =None):
        
        if len(fn.split('.'))>1:
            module = '.'.join(fn.split('.')[:-1])
            fn = fn.split('.')[-1]
            
        kwargs = kwargs if kwargs else {}
        args = args if args else []
        
        

        if name == None:
            prefix = cls.resolve_module(module).module_path()
            name = f'{prefix}{tag_seperator}{fn}'
    
        if 'remote' in kwargs:
            kwargs['remote'] = False
            
        cls.launch(fn=fn, 
                   module = module,
                    kwargs=kwargs,
                    refresh=refresh,
                    name=name)

    rfn = remote_fn
    @classmethod
    def choice(cls, options:Union[list, dict])->list:
        import random
        options = c.copy(options) # copy to avoid changing the original
        if isinstance(options, dict):
            options = list(options.values())

        assert isinstance(options, list),'options must be a list'
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
    def gather(cls,jobs:list, mode='asyncio', loop=None, timeout = None)-> list:
        if not isinstance(jobs, list):
            jobs = [jobs]
        assert isinstance(jobs, list)
        
        
        
        if mode == 'asyncio':
            loop = loop if loop != None else cls.get_event_loop()
            if timeout is not None:
                jobs = [asyncio.wait_for(job, timeout=timeout) for job in jobs]
            results = loop.run_until_complete(asyncio.gather(*jobs))
            
        else:
            raise NotImplementedError
        
        return results
    @classmethod
    def addresses(cls, *args, **kwargs) -> List[str]:
        return list(c.namespace(*args,**kwargs).values())

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
    def is_ss58(address):
        # Check address length
        if len(address) != 47:
            return False
        
        # Check prefix
        network_prefixes = ['1', '2', '5', '7']  # Add more prefixes as needed
        if address[0] not in network_prefixes:
            return False
        
        # Verify checksum
        encoded = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
        address_without_checksum = address[:-1]
        checksum = address[-1]
        address_hash = 0
        for char in address_without_checksum:
            address_hash = address_hash * 58 + encoded.index(char)
        
        # Calculate the expected checksum
        expected_checksum = encoded[address_hash % 58]
        
        # Compare the expected checksum with the provided checksum
        if expected_checksum != checksum:
            return False
        
        return True
 
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

        
    @classmethod
    def mv(cls, path1, path2):
        import shutil
        shutil.move(path1, path2)
        return path2

        
        
    @classmethod
    def cp(cls, path1, path2):
        import shutil
        # what if its a folder?
        assert os.path.exists(path1), path1
        assert not os.path.exists(path2), path2
        
        if os.path.isdir(path1):
            shutil.copytree(path1, path2)
        elif os.path.isfile(path1):
            shutil.copy(path1, path2)
        else:
            raise ValueError(f'path1 is not a file or a folder: {path1}')
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
        return c.module('model.transformer').learn(*args, **kwargs)
        
    @classmethod
    def mine(cls,*args, **kwargs):
        kwargs['remote'] = kwargs.get('remote', True)
        return c.module('bittensor').mine(*args, **kwargs)
    
    @classmethod
    def train_fleet(cls, *args, **kwargs):
        kwargs['remote'] = kwargs.get('remote', True)
        return c.module('model.transformer').train_fleet(*args, **kwargs)
    
    @classmethod
    def miners(cls, *args, **kwargs):
        return c.module('bittensor').miners(*args, **kwargs)
    
    @classmethod
    def check_miners(cls, *args, module='bittensor', **kwargs):
        return c.module(module).check_miners( *args, **kwargs)
    
    
    @classmethod
    def shuffle(cls, x:list)->list:
        import random
        random.shuffle(x)
        return x
    

    @classmethod
    def pull(cls, stash:bool = True):
        if stash:
            cls.cmd('git stash')
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
    def docker_ps(cls, sudo=True):
        return cls.cmd('docker ps', sudo=True)
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
            
        c.print('Added the key fam')
        
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
        
    ssh_path = os.path.expanduser('~/.ssh/id_rsa.pub')

    @classmethod
    def resolve_ssh_path(cls, ssh_path=None):
        if ssh_path is None:
            ssh_path = cls.ssh_path
        return os.path.expanduser(ssh_path)
    @classmethod
    def ssh_pubkey(cls,ssh_path=None):
        ssh_path = cls.resolve_ssh_path(ssh_path)
        return cls.get_text(ssh_path)
    @classmethod
    def generate_ssh_key_pair(cls, path=None,
                            passphrase=None):
        c.ensure_lib('paramiko')
        path = cls.resolve_ssh_path(path)
        import paramiko
        key = paramiko.RSAKey.generate(bits=2048)

        # Save the private key to a file
        key.write_private_key_file(path, password=passphrase)

        # Save the public key to a file
        with open(path, "w") as pub_key_file:
            pub_key_file.write(f"{key.get_name()} {key.get_base64()}")
        
        return cls.ssh_pubkey(path) 

    @classmethod
    def ssh_key(cls, key_file=os.path.expanduser('~/.ssh/id_rsa'),
                            passphrase=None):
        c.ensure_lib('paramiko')
        import paramiko
        key = paramiko.RSAKey.generate(bits=2048)

        # Save the private key to a file
        key.write_private_key_file(key_file, password=passphrase)

        # Save the public key to a file
        ssh_key_path = f"{key_file}.pub"
        with open(ssh_key_path, "w") as pub_key_file:
            pub_key_file.write(f"{key.get_name()} {key.get_base64()} Generated by Python")
        
        c.print(f"SSH key pair generated and saved to {ssh_key_path}")

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
        c.print({'msg': f"Started miner {name} on port {port}"})
        
        
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
    def torch(cls):
        return cls.import_module('torch')

    @classmethod
    def tensor(cls, *args, **kwargs):
        return cls.import_object('torch.tensor')(*args, **kwargs)

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

    @classmethod
    def fn2str(cls,search = None,  code = True, defaults = True, **kwargs):
        schema = cls.schema(search=search, code=code, defaults=defaults)
        fn2str = {}
        for k,v in schema.items():
            fn2str[k] = c.python2str(v)
            
        return fn2str
    
        
    @classmethod
    def module2fn2str(self, code = True, defaults = False, **kwargs):
        module2fn2str = {  }
        for module in c.modules():
            try:
                module_class = c.module(module)
                if hasattr(module_class, 'fn2str'):
                    module2fn2str[module] = module_class.fn2str(code = code,                                          defaults = defaults, **kwargs)
            except:
                pass
        return module2fn2str


    @classmethod
    def stwrite(self, *args, **kwargs):
        import streamlit as st
        st.write(*args, **kwargs)
        
    # TAG CITY     
        
    def set_tag(self, tag:str,default_tag:str='base'):
        if tag == None:
            tag = default_tag
        self.tag = tag
        return default_tag
        
    def resolve_tag(self, tag:str=None, default_tag='base'):
        if tag == None:
            tag = default_tag
        return tag
    
    @classmethod
    def python2types(cls, d:dict)-> dict:
        return {k:str(type(v)).split("'")[1] for k,v in d.items()}
    
    @staticmethod
    def echo(x):
        return x
    
    @staticmethod
    def get_files_code(directory):
        import os
        code_dict = {}

        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, directory)

                with open(file_path, 'r') as f:
                    code = f.read()
                    code_dict[relative_path] = code

        return code_dict
    
    @classmethod
    def pool(cls , n=5):
        for i in range(n):
            cls.deploy(tag=str(i))
        

    @classmethod
    def classify_methods(cls, obj= None):
        obj = obj or cls
        method_type_map = {}
        for attr_name in dir(obj):
            method_type = None
            try:
                method_type = cls.classify_method(getattr(obj, attr_name))
            except Exception as e:
                continue
        
            if method_type not in method_type_map:
                method_type_map[method_type] = []
            method_type_map[method_type].append(attr_name)
        
        return method_type_map


    @classmethod
    def resolve_fn(cls,fn, obj=None, ensure_exists:bool=True):
        if obj is None:
            obj = cls
        if isinstance(fn, str):
            if hasattr(obj, fn):
                fn = getattr(obj, fn)  
            else:
                if ensure_exists:
                    raise Exception(f"Object {obj} does not have attribute {fn}")
        return fn
    
    @classmethod
    def get_function_args(cls, fn):
        fn = cls.resolve_fn(fn)
        args = inspect.getfullargspec(fn).args
        return args
    
    fn_args = get_fn_args =  get_function_args
    
    @classmethod
    def classify_method(cls, fn):
        fn = cls.resolve_fn(fn)
        args = cls.get_function_args(fn)
        if len(args) == 0:
            return 'static'
        elif args[0] == 'self':
            return 'self'
        else:
            return 'class'
    
    @classmethod
    def build(cls, *args,**kwargs): 
        return c.module('subspace').build(*args, **kwargs)

    @classmethod
    def play(cls):
        c.print(c.rm_key('brodfdf'))
        

    @staticmethod
    def get_parents(obj) -> List[str]:
        cls = resolve_class(obj)
        return list(cls.__mro__[1:-1])

    @staticmethod
    def get_parent_functions(cls) -> List[str]:
        parent_classes = get_parents(cls)
        function_list = []
        for parent in parent_classes:
            function_list += get_functions(parent)

        return list(set(function_list))

    @classmethod
    def is_property(cls, fn: 'Callable') -> bool:
        fn = cls.resolve_fn(fn,ensure_exists=False)
        return isinstance(fn, property)

    @classmethod
    def get_functions(cls, obj: Any = None,
                      include_module:bool = False, 
                      include_parents:bool=False, 
                      include_hidden:bool = False) -> List[str]:
        '''
        Get a list of functions in a class
        
        Args;
            obj: the class to get the functions from
            include_parents: whether to include the parent functions
            include_hidden: whether to include hidden functions (starts and begins with "__")
        '''
        
        if obj == None:
            obj = cls
        
        if isinstance(obj, str):
            obj = c.module(obj)
        
        if cls.is_root_module(obj):
            include_module = True
        
        
        functions = []
        parent_functions = [] 
        for fn_name in dir(obj):
            
            # skip hidden functions if include_hidden is False
            if (include_hidden==False) and (fn_name.startswith('__') and fn_name.endswith('__')):
                
                if fn_name != '__init__':
                    continue
    
            # if the function is in the parent class, skip it
            if  (fn_name in parent_functions) and (include_parents==False):
                continue

            # if the function is a property, skip it
            if hasattr(type(obj), fn_name) and \
                isinstance(getattr(type(obj), fn_name), property):
                continue
            
            # if the function is callable, include it
            if callable(getattr(obj, fn_name)):
                functions.append(fn_name)
                
        if not include_module:
            module_functions = c.get_functions(obj=c)
            new_functions = []
            for f in functions:
                if f == '__init__':
                    new_functions.append(f)
                if f not in module_functions:
                    new_functions.append(f)
            functions = new_functions
            
            
        return functions



    @classmethod
    def get_class_methods(cls: Union[str, type], obj = None)-> List[str]:
        '''
        Gets the class methods in a class
        '''
        if obj is None:
            obj = cls
            
        functions =  c.get_functions(cls)
        signature_map = {}
        for f in functions:
            if f.startswith('__'):
                continue
            signature_map[f] = cls.get_function_args(getattr(cls, f)) 

        return [k for k, v in signature_map.items() if 'self' not in v]

    @classmethod
    def get_self_methods(cls: Union[str, type], obj=None):
        '''
        Gets the self methods in a class
        '''
        obj = obj or cls
        functions =  c.get_functions(obj)
        signature_map = {f:cls.get_function_args(getattr(obj, f)) for f in functions}
        return [k for k, v in signature_map.items() if 'self' in v]
    
    self_methods = self_fns = get_self_methods

    @classmethod
    def get_static_methods(cls: Union[str, type], obj=None):
        '''
        Gets the self methods in a class
        '''
        obj = obj or cls
        functions =  c.get_functions(obj)
        signature_map = {f:cls.get_function_args(getattr(obj, f)) for f in functions}
        return [k for k, v in signature_map.items() if not ('self' in v or 'cls' in v)]
    
    static_meethods = static_fns = get_static_methods
    
    @classmethod
    def get_method_type(cls, fn):
        return cls.get_function_signature( fn)
        

    @classmethod
    def get_function_signature(cls, fn) -> dict: 
        '''
        get the signature of a function
        '''
        if isinstance(fn, str):
            fn = getattr(cls, fn)
        
        import inspect
        return dict(inspect.signature(fn)._parameters)

    @staticmethod
    def get_function_input_variables(fn)-> dict:
        return list(c.get_function_signature(fn).keys())

    @classmethod
    def get_function_defaults(cls, fn):
        import inspect
        
        fn = cls.resolve_fn(fn)
        function_defaults = dict(inspect.signature(fn)._parameters)
        for k,v in function_defaults.items():
            if v._default != inspect._empty and  v._default != None:
                function_defaults[k] = v._default
            else:
                function_defaults[k] = None

        return function_defaults


        
        
    @staticmethod
    def is_class(obj):
        '''
        is the object a class
        '''
        return type(obj).__name__ == 'type'


    @staticmethod
    def resolve_class(obj):
        '''
        resolve class of object or return class if it is a class
        '''
        if c.is_class(obj):
            return obj
        else:
            return obj.__class__
    @staticmethod
    def is_full_function(fn_schema):

        for mode in ['input', 'output']:
            if len(fn_schema[mode]) > 0:
                for value_key, value_type in fn_schema[mode].items():
                    if value_type == None:
                        return None
            else:
                return None
        return fn_schema 

    @staticmethod
    def try_n_times(fn, max_trials:int=10, args:list=[],kwargs:dict={}):
        assert isinstance(fn, callable)
        for t in range(max_trials):
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception as e:
                continue
        raise(e)

    @staticmethod
    def has_fn(obj, fn_name):
        return callable(getattr(obj, fn_name, None))

    @staticmethod
    def try_fn_n_times(fn, kwargs:Dict, try_count_limit: int = 10):
        '''
        try a function n times
        '''
        try_count = 0
        return_output = None
        while try_count < try_count_limit:
            try:
                return_output = fn(**kwargs)
                break
            except RuntimeError:
                try_count += 1
        return return_output
    
    
    @classmethod
    def jload(cls, json_string):
        import json
        return json.loads(json_string)
    
    @classmethod
    def bro(cls, x):
        return x
    
    
    @classmethod
    def giturl(cls):
        return c.cmd('git remote -v').split('\n')[0].split('\t')[1].split(' ')[0]
    

    @classmethod
    def my_modules(cls, *args, **kwargs):
        return c.module('subspace')().my_modules(*args, **kwargs)
    @classmethod
    def my_stake(cls, *args, **kwargs):
        return c.module('subspace')().my_stake(*args, **kwargs)

    @classmethod
    def my_tokens(cls, *args, **kwargs):
        return c.module('subspace')().my_tokens(*args, **kwargs)
    
    my_value = my_tokens
    
    @classmethod
    def partial(cls, fn, *args, **kwargs):
        from functools import partial
        return partial(fn, *args, **kwargs)
        
        
    @staticmethod
    def sizeof( obj):
        import sys
        type_str = c.type_str(obj)
        sizeof = 0
        if isinstance(obj, dict):
            for k,v in obj.items():
                sizeof +=  c.sizeof(k) + c.sizeof(v)
        elif isinstance(obj, list):
            for v in obj:
                sizeof += c.sizeof(v)
        elif any([k.lower() in c.type_str(obj).lower() for k in ['torch', 'Tensor'] ]):

            sizeof += c.get_tensor_size(obj)
        else:
            sizeof += sys.getsizeof(obj)
                
        return sizeof
    
    @classmethod
    def code(cls, module = None, *args, **kwargs):
        module = cls.resolve_module(module)
        return c.get_text( module.pypath(), *args, **kwargs)
    pycode = code
    @classmethod
    def codehash(cls,  *args, **kwargs):
        code = cls.code(*args, **kwargs)
        return c.hash(code)
    chash = pyhash = codehash
    @classmethod
    def match_module_hash(cls, hash:str, module:str=None, *args, **kwargs):
        '''
        match the hash of a module
        '''
        
        if module != None:
            module = c.module(module)
        else:
            module = cls
        return module.script_hash(*args, **kwargs) == hash
    
    @classmethod
    def find_code_line(self, search, *args, **kwargs):
        code = self.code(*args, **kwargs)
        found_lines = []
        for i, line in enumerate(code.split('\n')):
            if search in line:
                found_lines.append({'idx': i, 'text': line})
        if len(found_lines) == 0:
            return None
        elif len(found_lines) == 1:
            return found_lines[0]
        return found_lines
    
    
    
    
    def tokenizer(self, tokenizer='gpt2', *args, **kwargs):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(tokenizer, *args, **kwargs)
    
    def tokenize(self, text, tokenizer='gpt2', *args, **kwargs):
        return self.tokenizer(tokenizer, *args, **kwargs).encode(text)
    def detokenize(self, tokens, tokenizer='gpt2', *args, **kwargs):
        return self.tokenizer(tokenizer, *args, **kwargs).decode(tokens)

    
    def generate_completions(self, past_tokens = 10, future_tokens = 10, tokenizer:str='gpt2', mode:str='lines', **kwargs):
        code = self.code()
        code_lines = code.split('\n')
        c.tokenizer()
        if mode == 'lines':
            code_lines
        else:
            raise ValueError(f'unknown mode {mode}')
        return 
    
    
    ## SUBSPACE FNS
    
    @classmethod
    def register(cls, *args, **kwargs):
        return c.module('subspace')().register(*args, **kwargs)
    
    @classmethod
    def transfer(cls, *args, **kwargs):
        return c.module('subspace')().transfer(*args, **kwargs)
    @classmethod
    def update_module(cls, *args, **kwargs):
        return c.module('subspace')().update_module(*args, **kwargs)
    
    @classmethod
    def vote(cls, *args, **kwargs):
        return c.module('subspace')().vote(*args, **kwargs)
    
    @classmethod
    def stake(cls, *args, **kwargs):
        return c.module('subspace')().stake(*args, **kwargs)
    
    @classmethod
    def snap(cls, *args, **kwargs):
        return c.module('subspace')().snap(*args, **kwargs)   

    @classmethod
    def build_spec(cls, *args, **kwargs): 
        return c.module('subspace').build_spec(*args, **kwargs) 
    
    @classmethod
    def unstake(cls, *args, **kwargs):
        return c.module('subspace')().unstake(*args, **kwargs)
    
    @classmethod
    def my_modules(cls, *args, **kwargs):
        return c.module('subspace')().my_modules(*args, **kwargs)
    
    @classmethod
    def my_keys(cls, *args, **kwargs):
        return c.module('subspace')().my_modules(*args, **kwargs)
    

    @classmethod
    def register_loop(cls, *args, **kwargs):
        return c.module('subspace')().register_loop(*args, **kwargs)
    
    rloop = register_loop
    
    @classmethod
    def balance(cls, *args, **kwargs):
        return c.module('subspace')().balance(*args, **kwargs)
    get_balance = balance
    
    @classmethod
    def my_balances(cls, *args, **kwargs):
        return c.module('subspace')().my_balances(*args, **kwargs)

    @classmethod
    def my_keys(cls, *args, **kwargs):
        return c.module('subspace')().my_keys(*args, **kwargs)
    
    @classmethod
    def key_info(self, *args, **kwargs):
        return c.module('key').key_info(*args, **kwargs)
    @classmethod
    def key_info_map(self, *args, **kwargs):
        return c.module('key').key_info_map(*args, **kwargs)
    @classmethod   
    def infer_device_map(cls, 
                         model:str, 
                         max_memory: dict = None,
                         block_prefix : str = 'model.layers',
                         buffer_memory:float = '10gb', # 10GB buffer (bytes)
                         verbose: bool = False,
                         **kwargs,
                         ):
        model = c.resolve_model(model)
        param_size_map = c.params_size_map(model, block_prefix=block_prefix, **kwargs)
        
        free_gpu_memory = c.free_gpu_memory() if max_memory == None else max_memory
        buffer_memory  = c.resolve_memory(buffer_memory)
        device_map = {}
        gpu = None
        gpu_memory = 0
        unallocated_memory = sum(param_size_map.values())
        allocated_gpu_memory = {}
        
        gpu = None
        
        
        
        for param_key, param_size in param_size_map.items():            
            # find the most free gpu if gpu is None or if the gpu has less memory than the buffer memory
        
            if (gpu == None) or (free_gpu_memory[gpu] < buffer_memory) or (free_gpu_memory[gpu] < param_size):
                gpu = c.most_free_gpu( fmt='b', free_gpu_memory=free_gpu_memory)
                llocated_gpu_memory[gpu] = 0
    
   
            
            allocated_gpu_memory[gpu] += param_size
            free_gpu_memory[gpu] -= param_size
            unallocated_memory -= param_size
            device_map[param_key] = gpu
            
        c.print(allocated_gpu_memory, c.free_gpu_memory())
        assert unallocated_memory == 0, f'unallocated memory {unallocated_memory} != 0'
                
        return device_map
        
        
    @classmethod
    def snap(cls, *args, **kwargs):
        return c.module('subspace')().snap(*args, **kwargs)

    @classmethod
    def save(cls, *args, **kwargs):
        return c.module('subspace')().save(*args, **kwargs)
    
    def key2balance(self,  *args, **kwargs):
        return c.module('subspace')().key2balance( *args, **kwargs)
    
    def key2value(self,  *args, **kwargs):
        return c.module('subspace')().key2value( *args, **kwargs)

    def key2stake(self,  *args, **kwargs):
        return c.module('subspace')().key2balance( *args, **kwargs)

    def live_keys(self,  *args, **kwargs):
        return c.module('subspace')().live_keys( *args, **kwargs)
    def dead_keys(self,  *args, **kwargs):
        return c.module('subspace')().dead_keys( *args, **kwargs)


    @classmethod
    def my_balance(cls, *args, **kwargs):
        return c.module('subspace')().my_balance(*args, **kwargs)

    @classmethod
    def nodes(cls, *args, **kwargs):
        return c.module('subspace')().nodes(*args, **kwargs)
    @classmethod
    def kill_nodes(cls, *args, **kwargs):
        return c.module('subspace')().kill_nodes(*args, **kwargs)
    

    @classmethod
    def cj(cls, *args, **kwargs):
        return c.module('subspace')().cj(*args, **kwargs)
    j = cj
    
    @classmethod
    def watchdog(cls, *args, **kwargs):
        return c.module('subspace')().watchdog(*args, **kwargs)
    watch = watchdog
    @classmethod
    def n(self, *args, **kwargs):
        return c.module('subspace')().n(*args, **kwargs)
    
    @classmethod
    def upgrade_proto(cls, verbose:bool = True):
        c.cmd('pip install --upgrade protobuf', verbose=verbose)
        c.cmd('pip install --upgrade grpcio-tools', verbose=verbose)
        
    @classmethod
    def fix_proto(cls):
        cls.upgrade_proto()
        cls.build_proto()
        
    @classmethod
    def subnets(cls, *args, **kwargs):
        return c.module('subspace')().subnets(*args, **kwargs)
    
    @classmethod
    def subnet(cls, *args, **kwargs):
        return c.module('subspace')().subnet(*args, **kwargs)

    @classmethod
    def networth(cls, *args, **kwargs):
        return c.module('subspace')().networth(*args, **kwargs)
    total_tokens = networth
    @classmethod
    def key2balance(cls, *args, **kwargs):
        return c.module('subspace')().key2balance(*args, **kwargs)
    
    @classmethod
    def key2tokens(cls, *args, **kwargs):
        return c.module('subspace')().key2tokens(*args, **kwargs)
    @classmethod
    def key2stake(cls, *args, **kwargs):
        return c.module('subspace')().key2tokens(*args, **kwargs)

    @classmethod
    def key2stake(cls, *args, **kwargs):
        return c.module('subspace')().key2tokens(*args, **kwargs)
    
    
        
    
    @classmethod
    def build_proto(cls, *args, **kwargs):
        src_dir = c.root_path + '/module/server/proto'
        proto_path = src_dir + '/server.proto'
        cmd = f"python3 -m grpc.tools.protoc {proto_path}  -I {src_dir}  --python_out={src_dir} --grpc_python_out={src_dir}"
        c.cmd(cmd, verbose=True)
        
    @classmethod
    def update_network(cls, *args, **kwargs):
        return c.module('subspace')().update_network(*args, **kwargs)
    
    @classmethod
    def market_cap(cls, *args, **kwargs):
        return c.module('subspace')().market_cap(*args, **kwargs)
    mcap = market_cap
    @classmethod
    def n(cls, *args, **kwargs):
        return c.module('subspace')().n(*args, **kwargs)

    def stats(self, *args, **kwargs):
        return c.module('subspace')().stats(*args, **kwargs)

    def my_stats(self, *args, **kwargs):
        return c.module('subspace')().my_stats(*args, **kwargs)

    @classmethod
    def register_dead_keys(cls, *args, **kwargs):
        return c.module('subspace')().register_dead_keys(*args, **kwargs)
    

    @classmethod
    def shortcuts(cls) -> Dict[str, str]:
        return c.getc('shortcuts')

    @classmethod
    def add_shortcut(cls, shortcut, name) -> Dict[str, str]:
        shortcuts =  c.getc('shortcuts')
        shortcuts[shortcut] = name
        c.putc('shortcuts', shortcuts)
        return {'success': True, 'msg': f'added shortcut {shortcut} -> {name}'}

    @classmethod
    def resolve_shortcut(cls, name:str) -> str:
        return c.getc('shortcuts').get(name, name)
        
Module = c

Module.run(__name__)
    
