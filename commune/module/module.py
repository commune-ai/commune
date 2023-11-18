

import inspect
import numpy as np
import os
import concurrent
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
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# AGI BEGINS 
class c:
    descrition = """This is a module"""
    base_module = 'module'
    encrypted_prefix = 'ENCRYPTED'
    homepath = os.path.expanduser('~')
    root_module_class = 'c' # WE REPLACE THIS THIS Module at the end, kindof odd, i know, ill fix it fam, chill out dawg, i didnt sleep with your girl
    default_port_range = [50050, 50150] # the port range between 50050 and 50150
    default_ip = local_ip = loopback = '0.0.0.0'
    address = '0.0.0.0:8888' # the address of the server (default)
    root_path  = root  = os.path.dirname(os.path.dirname(__file__)) # the path to the root of the library
    libpath = os.path.dirname(root_path) # the path to the library
    datapath = os.path.join(libpath, 'data') # the path to the data folder
    modules_path = os.path.join(root_path, 'modules') # the path to the modules folder
    repo_path  = os.path.dirname(root_path) # the path to the repo
    library_name = libname = lib = root_dir = root_path.split('/')[-1] # the name of the library
    pwd = os.getenv('PWD') # the current working directory from the process starts 
    console = Console() # the consolve
    helper_whitelist = ['info', 'schema','server_name', 'is_admin'] # whitelist of helper functions to load
    whitelist = [] # whitelist of functions to load
    blacklist = [] # blacklist of functions to not to access for outside use
    server_mode = 'http' # http, grpc, ws (websocket)
    default_network = 'local' # local, subnet
    cache = {} # cache for module objects
    home = os.path.expanduser('~') # the home directory
    __ss58_format__ = 42 # the ss58 format for the substrate address

    def __init__(self, config:Dict=None, **kwargs):
        self.set_config(config=config,kwargs=kwargs)  

    @classmethod
    def init(cls, *args, **kwargs):
        return cls(*args, **kwargs)

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
    def module_dirpath(self) -> str:
        return  os.path.dirname(self.module_file())

    @classmethod
    def __module_dir__(cls) -> str :
        # get the directory of the module
        return os.path.dirname(cls.module_file())
    
    @classmethod
    def get_module_path(cls, obj=None,  simple:bool=False) -> str:
        import inspect
        # odd case where the module is a module in streamlit
        obj = cls.resolve_module(obj)
        try:
            module_path =  inspect.getfile(obj)
        except Exception as e:
            if 'source code not available' in str(e):
                return cls.class_name()
            else: 
                raise e
     
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
    def config_path(cls) -> str:
        '''
        removes the PWD with respect to where module.py is located
        '''
        return cls.config_path()
    cfgpath = config_path = config_path

    
    @classmethod
    def dirpath(cls) -> str:
        '''
        removes the PWD with respect to where module.py is located
        '''
        return os.path.dirname(cls.filepath())


    @classmethod
    def dlogs(cls, *args, **kwargs):
        return c.module('docker').logs(*args, **kwargs)

    @classmethod
    def images(cls, *args, **kwargs):
        return c.module('docker').images(*args, **kwargs)
    
    @classmethod
    def module_path(cls, simple:bool=True) -> str:
        # get the module path
        
        path = cls.get_module_path(simple=simple)
        path = path.replace('modules.', '')
        return path
    
    path  = name = module_name =  module_path
    
    @classmethod
    def module_class(cls) -> str:
        return cls.__name__
    @classmethod
    def class_name(cls, obj= None) -> str:
        obj = obj if obj != None else cls
        return obj.__name__
    def get_class_name(cls, obj = None) -> str:
        obj = obj if obj != None else cls
        if not cls.is_class(obj):
            obj = type(obj)
        
        return obj.__name__
    

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
    def config_path(cls) -> str:
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
    
    get_yaml = load_yaml


    @classmethod
    def fn2code(cls, search=None, module=None)-> Dict[str, str]:
        module = module if module else cls
        functions = module.fns(search)
        fn_code_map = {}
        for fn in functions:
            c.print(f'fn: {fn}')
            try:
                fn_code_map[fn] = module.fn_code(fn)
            except Exception as e:
                c.print(f'Error: {e}', color='red')
        return fn_code_map
    
            

    @classmethod
    def fn_code(cls,fn:str, detail:bool=False, ) -> str:
        '''
        Returns the code of a function
        '''
        
        
        code_text = inspect.getsource(getattr(cls, fn))
        text_lines = code_text.split('\n')
        if 'classmethod' in text_lines[0] or 'staticmethod' in text_lines[0] or '@' in text_lines[0]:
            text_lines.pop(0)

        assert 'def' in text_lines[0], 'Function not found in code'
        start_line = cls.find_code_line(search=text_lines[0])
        fn_code = '\n'.join([l[len('    '):] for l in code_text.split('\n')])
        if detail:
            fn_code =  {
                'text': fn_code,
                'start_line': start_line ,
                'end_line':  start_line + len(text_lines)
            }
                
        return fn_code

    @classmethod
    def sandbox(cls):
        
        c.cmd(f'python3 {c.libpath}/sandbox.py')
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
    
    put_yaml = save_yaml

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
    def config_path(cls) -> str:
        path = cls.module_file().replace('.py', '.yaml')
        return path
    
    
    @classmethod
    def load_config(cls, path:str=None, to_munch:bool = False, root:bool = False) -> Union[Munch, Dict]:
        '''
        Args:
            path: The path to the config file
            to_munch: If true, then convert the config to a munch
        '''

        if path == None: 
            path = cls.config_path()
        else:
            module_tree = cls.module_tree()
            path = module_tree[path].replace('.py', '.yaml')
            
        config = cls.load_yaml(path)

        # convert to munch
        if config == None:
            config = {}

        # convert to munch
        if to_munch:
            config =  cls.dict2munch(config)
        
        return config
    
    
    default_config = load_config

    @classmethod
    def encrypt_path(cls, path:str, key=None, prefix='ENCRYPTED') -> str:
        '''
        Encrypts the path
        '''
        path = cls.resolve_path(path)
        text = c.get_text(path)
        encrypted_text = prefix + c.encrypt(text, key=key)
        c.put_text(path, encrypted_text)

        return {'success': True, 'msg': f'encrypted {path}'}
        

    @classmethod
    def decrypt_path(cls, path:str, key=None, prefix='ENCRYPTED') -> str:
        '''
        Encrypts the path
        '''
        path = cls.resolve_path(path)
        text = c.get_text(path)
        assert text.startswith(prefix), f'path {path} is not encrypted'
        text = text[len(prefix):]
        encrypted_text = c.decreypt(text, key=key)
        c.put_text(path, encrypted_text)

        return {'success': True, 'msg': f'encrypted {path}'}
        

    def is_encrypted_path(self, path:str, prefix='ENCRYPTED') -> bool:
        '''
        Encrypts the path
        '''
        path = self.resolve_path(path)
        text = c.get_text(path)
        return text.startswith(prefix)


    
    @classmethod
    def put(cls, 
            k, 
            v, 
            mode: bool = 'json',
            key : str = None,
            encrypt: bool = False,
            ):
        '''
        Puts a value in the config
        '''
        
        if encrypt:
            data = c.encrypt(v, key=key, return_dict=True)
        
        data = {'data': v, 'encrypted': encrypt, 'timestamp': c.timestamp()}            
        # default json 
        getattr(cls,f'put_{mode}')(k, data)
    
        return data
    
    

        
    @classmethod
    def get(cls,
            k:str, 
            default: Any=None, 
            mode:str = 'json',
            max_age:str = None,
            cache :bool = False,
            full :bool = False,
            key: 'Key' = None,
            **kwargs) -> Any:
        
        '''
        Puts a value in sthe config, with the option to encrypt it

        Return the value
        '''
        if cache:
            if k in cls.cache:
                return cls.cache[k]
            

        verbose = kwargs.get('verbose', False)
        data = getattr(cls, f'get_{mode}')(k,default=default, **kwargs)
        if data == None: 
            data = default
        encrypted = c.is_encrypted(data)
        if encrypted:
            data = cls.decrypt(data, key=key)
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
            
        if not full:
            if isinstance(data, dict):
                if 'data' in data:
                    data = data['data']

        # local cache
        if cache:
            cls.cache[k] = data

        return data
    

    @classmethod
    def obj_age(cls, item:dict) -> int:
        return c.timestamp() - int(cls.get_json(item).get('timestamp', 0))

    @classmethod
    def get_many(cls,
            *k, 
            default: Any=None, 
            mode:str = 'json',
            max_age:str = None,
            cache :bool = False,
            full :bool = False,
            **kwargs) -> Any:
        
        '''
        Puts a value in sthe config, with the option to encrypt it

        Return the value
        '''
        data_map = {k: cls.get(k, default=default, mode=mode, max_age=max_age, cache=cache, full=full, **kwargs) for k in k}
        return data_map



    @staticmethod
    def get_age(timestamp:int=0):
        return c.time() - timestamp
    
    @staticmethod
    def too_old(self, timestamp:int, max_age:int):
        return self.get_age(timestamp) > max_age
    
    
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

        return {'success': True, 'msg': f'config({k} = {v})'}
   
   
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
    def frontend(cls):
        return c.compose('frontend')
      
    @classmethod
    def popc(cls, key:str):
        config = cls.config()
        config.pop(key, None)
        cls.save_config(config=config)

    @classmethod
    def hasc(cls, key:str):
        config = cls.config()
        return key in config

    @classmethod
    def keysc(cls):
        config = cls.config()
        return list(config.keys())
        
    @classmethod  
    def getc(cls, key, default= None, password=None) -> Any:
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
            config = cls.config()
        
        path = path if path else cls.config_path()
        
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
        path = path if path else self.config_path()
        return self.path_exists(path)

    @classmethod
    def get_config(cls, 
                   config:dict = None,
                   kwargs:dict=None, 
                   module = None,
                   to_munch:bool = True) -> Munch:
        '''
        Set the config as well as its local params
        '''
        if not cls.has_config():
            config =  {}
        else:
            if config == None:
                config = cls.load_config()
            elif isinstance(config, str):
                
                config = cls.load_config(path=config)
                assert isinstance(config, dict), f'config must be a dict, not {type(config)}'
            elif isinstance(config, dict):
                default_config = cls.load_config()
                config = {**default_config, **config}
            else:
                raise ValueError(f'config must be a dict, str or None, not {type(config)}')
            
        assert isinstance(config, dict), f'config must be a dict, not {config}'
        
        # SET THE CONFIG FROM THE KWARGS, FOR NESTED FIELDS USE THE DOT NOTATION, 
        # for example  model.name=bert is the same as config[model][name]=bert

        kwargs = kwargs if kwargs != None else cls.init_kwargs()

        # merge kwargs with itself (CAUTION THIS DOES NOT WORK IF KWARGS WAS MEANT TO BE A VARIABLE LOL)
        kwargs.update(kwargs.pop('kwargs', {}))
        for k,v in kwargs.items():
            cls.dict_put(config,k,v )
            
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

        # in case they passed in a locals() dict, we want to resolve the kwargs and avoid ambiguous args
        kwargs = c.locals2kwargs(kwargs)

        if 'config' in kwargs:
            config = kwargs.pop('config')
            
        # get the config
        config =  self.get_config(config=config,kwargs=kwargs, to_munch=to_munch)


        # add the config attributes to the class (via munch -> dict -> class )
        if add_attributes:
            self.__dict__.update(self.munch2dict(config))
        self.config = config 
        self.kwargs = kwargs
        
        if save_config:
            self.save_config(config=config)
    
        return self.config

    @classmethod
    def flatten_dict(cls, x = {'a': {'b': 1, 'c': {'d': 2, 'e': 3}, 'f': 4}}):
        from commune.utils.dict import deep2flat
        return deep2flat(x)

    @classmethod
    def start_node(cls, *args, **kwargs):
        return c.module('subspace').start_node(*args, **kwargs)


    @classmethod
    def start_telemetry(cls, *args, **kwargs):
        return c.module('subspace').start_telemetry(*args, **kwargs)

    @classmethod
    def start_local_node(cls, *args, **kwargs):
        return c.module('subspace').start_local_node(*args, **kwargs)

    @classmethod
    def start_chain(cls, *args, **kwargs):
        c.module('subspace').start_chain(*args, **kwargs)
        return {'success': True, 'msg': 'started chain'}
    @classmethod
    def kill_chain(cls, *args, **kwargs):
        c.module('subspace').kill_chain(*args, **kwargs)
        return {'success': True, 'msg': 'killed chain'}
    def seconds_per_epoch(self, *args, **kwargs):
        return c.module('subspace')().seconds_per_epoch(*args, **kwargs)

    # KEY LAND
    @classmethod
    def add_key(cls, *args, **kwargs):
        return c.module('key').add_key(*args, **kwargs)
    
    @classmethod
    def getmem(self, *args, **kwargs):
        return c.module('key').getmem(*args, **kwargs)
    mem = getmem

    # KEY LAND
    @classmethod
    def mv_key(cls, *args, **kwargs):
        return c.module('key').mv_key(*args, **kwargs)

    @classmethod
    def mems(cls, *args, **kwargs):
        return c.module('key').mems(*args, **kwargs)

    # KEY LAND
    @classmethod
    def switch_key(cls, *args, **kwargs):
        return c.module('key').switch_key(*args, **kwargs)

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
    def gradio(self, *args, **kwargs):
        return c.module('gradio')(*args, **kwargs)
    
    
    @classmethod
    def st(cls, module = None, fn='dashboard', port=8501, kwargs:dict=None):
        if module == None: 
            module = cls.module_path()
        module = c.module(module)
        module_filepath = module.filepath()
        c.print(f'Running {module_filepath}', color='green')
        # add port to the command
        port = c.get_port(port)
        cmd = f'streamlit run {module_filepath}'
        if port != None:
            cmd += f' --server.port {port}'
        
        if kwargs == None:
            kwargs = {}

        kwargs_str = json.dumps(kwargs)
        kwargs_str = kwargs_str.replace('"', "'")

        cmd += f' -- --fn {fn} --kwargs "{kwargs_str}"'

        c.cmd(cmd, verbose=True)

    

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
    def rcmd(cls, *args, **kwargs):
        return c.module('remote').cmd(*args, **kwargs)
    
    @classmethod
    def cmd(cls, *args, **kwargs):
        return c.module('os').cmd(*args, **kwargs)
    run_command = shell = cmd 

    @classmethod
    def import_module(cls, import_path:str) -> 'Object':
        from importlib import import_module
        return import_module(import_path)


    @classmethod
    def import_object(cls, key:str, verbose: bool = False)-> Any:
        
        '''
        
        Import an object from a string with the format of {module_path}.{object}
        Examples: import_object("torch.nn"): imports nn from torch
        
        '''
        from importlib import import_module
        module = '.'.join(key.split('.')[:-1])
        object_name = key.split('.')[-1]
        if verbose:
            c.print(f'Importing {object_name} from {module}')
        obj =  getattr(import_module(module), object_name)
        return obj
    
    imp = get_object = importobj = import_object



    @classmethod
    def module_exists(cls, module:str) -> bool:
        '''
        Returns true if the module exists
        '''
        return module in c.modules()


    
    @classmethod
    def modules(cls, search=None)-> List[str]:
        '''
        List of module paths with respect to module.py file
        
        Assumes the module root directory is the directory containing module.py
        '''
        module_list = list(cls.module_tree().keys())
        if search != None:
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
                port=int(port)
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
    def makedirs(cls, *args, **kwargs):
        import os
        return os.makedirs(*args, **kwargs)

    @classmethod
    def resolve_path(cls, path:str, extension:Optional[str]= None, root:bool = False, file_type:str = 'json'):
        '''
        Resolves path for saving items that relate to the module
        
        The path is determined by the module path 
        
        '''
        if path == None:
            path = cls.tmp_dir()
        
        if path.startswith('/'):
            path = path
        elif path.startswith('~/'):
            path =  os.path.expanduser(path)
        elif path.startswith('./'):
            path = os.path.abspath(path)
        else:
            # if it is a relative path, then it is relative to the module path
            # ex: 'data' -> '.commune/path_module/data'
            tmp_dir = c.tmp_dir() if root else cls.tmp_dir()

            if tmp_dir not in path:
                path = os.path.join(tmp_dir, path)
            if not os.path.isdir(path):
                if extension != None and extension != path.split('.')[-1]:
                    path = path + '.' + extension
            
        if not os.path.exists(path) and os.path.exists(path + f'.{file_type}'):
            path = path + f'.{file_type}'       
                 
        return path
    
    @classmethod
    def resolve_address(cls, address:str = None):
        if address == None:
            address = c.free_address()
        assert isinstance(address, str),  'address must be a string'
        return address
    @classmethod
    def get_available_ports(cls, port_range: List[int] = None , ip:str =None) -> int:
        port_range = cls.resolve_port_range(port_range)
        ip = ip if ip else c.default_ip
        
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
                c.print(f'Port {port} is open', color='green')
                open_ports.append(port)
            else:
                c.print(f'Port {port} is closed', color='red')
        return open_ports

    @classmethod
    def resolve_port(cls, port:int=None, **kwargs):
        
        '''
        
        Resolves the port and finds one that is available
        '''
        if port == None or port == 0:
            port = c.free_port(port, **kwargs)
            
        if c.port_used(port):
            port = c.free_port(port, **kwargs)
            
        return int(port)

    @classmethod
    def has_free_ports(self, n:int = 1, **kwargs):

        return len(self.free_ports(n=n, **kwargs)) > 0
    
    @classmethod
    def free_ports(cls, n=10, reserve:bool = False, random_selection:bool = False, **kwargs ) -> List[int]:
        free_ports = []
        avoid_ports = kwargs.pop('avoid_ports', [])
        for i in range(n):
            try:
                free_ports += [cls.free_port(reserve=reserve, 
                                            random_selection=random_selection, 
                                            avoid_ports=avoid_ports, **kwargs)]
            except Exception as e:
                c.print(f'Error: {e}', color='red')
                break
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
            
            
            
        ip = ip if ip else c.default_ip

        if random_selection:
            ports = c.shuffle(ports)
            
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
            return port
        elif mode == 'bash':
            return c.run_command(f'kill -9 $(lsof -ti:{port})', bash=True, verbose=True)

    @classmethod
    def restart_servers(cls, module:str=None, mode:str = 'server'):
        '''
        Kill the server by the name
        '''

        fn = getattr(cls, f'{mode}_restart')
        for module in c.servers(module,network='local'):
            try:
                c.print(f'Restarting {module}', color='red')
                fn(module)
            except Exception as e:
                c.print(f'Error: {e}', color='red')
                continue

    @classmethod
    def pm2_restart_all(cls):
        '''
        Kill the server by the name
        '''
        for p in c.pm2_list():
            c.print(f'Restarting {p}', color='red')
            c.pm2_restart(p)

        c.update()


    @staticmethod
    def kill_all_servers( *args, **kwargs):
        '''
        Kill all of the servers
        '''
        for module in c.servers(*args, **kwargs):
            c.kill(module)

        # c.update(network='local')
            
    
    @classmethod
    def kill_all(cls, network='local'):
        futures = []
        for s in c.servers(network=network):
            futures += [c.submit(c.kill, args=[s], return_future=True)]

        results = c.wait(futures)
        c.update_namespace(network=network)

        return {'namespace': c.namespace(network=network)}

        
        
            
    @classmethod
    def restart_peers(cls, timeout=20):
        futures = []
        for p in cls.peers():
            futures += [c.submit(c.restart_server, args=[p], return_future=True, timeout=timeout)]

        results = c.wait(futures,timeout=timeout)
        return results



    @classmethod
    def restart_all_servers(cls, verbose: bool = True):
        '''
        Kill all of the servers
        '''
        for module in cls.servers():
            if verbose:
                c.print(f'Restarting {module}', color='red')
            cls.server_restart(module)
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

        # compress nae
        chunks = simple_path.split('.')
        new_chunks = []
        for i, chunk in enumerate(chunks):
            if len(new_chunks)>0:
                if new_chunks[-1] == chunks[i]:
                    continue
                elif any([chunks[i].endswith(s) for s in ['_module', 'module']]):
                    continue
            new_chunks.append(chunk)
        simple_path = '.'.join(new_chunks)
        
        # remove the modules prefix
        if simple_path.startswith('modules.'):
            simple_path = simple_path.replace('modules.', '')

        # remove any files to compress the name even further for
        if len(simple_path.split('.')) > 2:
            
            if simple_path.split('.')[-1].endswith(simple_path.split('.')[-2]):
                simple_path = '.'.join(simple_path.split('.')[:-1])
        return simple_path
    


    @classmethod
    def path2localpath(cls, path:str) -> str:
        local_path = path.replace(cls.repo_path, cls.root_dir)
        return local_path
    @classmethod
    def path2config(cls, path:str, to_munch=False)-> dict:
        path = cls.path2config_path(path=path)
        return cls.load_config(path, to_munch=to_munch)
    
    @classmethod
    def path2config_path(cls, path:str):
        return path.replace('.py', '.yaml')
    @classmethod
    def simple2config_path(cls,  path:str):
        return cls.path2config_path(cls.simple2path(path))
    @classmethod
    def simple2config(cls, path:str, to_munch=False)-> dict:
        return cls.load_config(cls.simple2config_path(path), to_munch=to_munch)
    
    
    @classmethod
    def import_path(cls):
        return cls.path2objectpath(cls.module_file())
    
    @classmethod
    def object_path(cls):
        return cls.path2objectpath(cls.module_path(simple=False))

    @classmethod
    def find_classes(cls, module=None):
        if module == None:
            module = cls.module_path()
        module = c.module(module)
        filepath = module.filepath()
        code = c.get_text(filepath)
        classes = []
        for line in code.split('\n'):
            if all([s in line for s in ['class ', '(', '):']]):
                classes.append(line.split('class ')[-1].split('(')[0].strip())
        object_path = cls.path2objectpath(filepath)

        return ['.'.join(object_path.split('.')[:-1]+[c]) for c in classes]

    
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
            has_class_bool = all([key_element in line for key_element in key_elements])

            if has_class_bool:
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
        return c.import_object(path)


    @classmethod
    def get_module(cls, path:str) -> str:
        path = cls.simple2path(path)
        path = cls.path2objectpath(path)
        return c.import_object(path)


    @classmethod
    def module_tree(cls, search=None, 
                    mode='path', 
                    cache:bool = True,
                    update:bool = False,
                    verbose:bool = False) -> List[str]:
                
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
            c.put('module_tree', module_tree)
        return module_tree
    
    available_modules = tree = module_tree
    @classmethod
    def list_modules(cls, search=None):
        modules = list(cls.module_tree(search).keys())
        return modules
    



    @classmethod
    def get_tags(cls, module, *args, **kwargs):
        servers =  c.servers(module, *args, **kwargs)
        return [s.split('::')[-1] if len(s.split('::'))>1 else None  for s in servers]

    @classmethod
    def has_config(cls) -> bool:
        config_path = cls.config_path()
        return c.exists(config_path)
  
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
    def datasets(cls, **kwargs) -> List[str]:
        return c.servers('data',  **kwargs)
    datas = datasets
    
    @staticmethod
    def module_config_tree() -> List[str]:
        return [f.replace('.py', '.yaml')for f in  c.get_module_python_paths()]

    @staticmethod
    def is_imported(package:str) :
        return  bool(package in sys.modules)

    @classmethod
    def simple2path(cls, path) -> Dict[str, str]:
        module_tree = c.module_tree()
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
                if file_name.lower().endswith(dir_name.lower()):
                    # if the dirname is equal to the filename then it is a module
                    modules.append(f)
                if file_name.lower().endswith('module'):
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


    tree_folders_path = 'module_tree_folders'
    @classmethod
    def add_tree(cls, tree_path:str, **kwargs):
        path = cls.tree_folders_path
        tree_folder = c.get(path, [])
        tree_folder += [tree_path]
        assert os.path.isdir(tree_path)
        assert isinstance(tree_folder, list)
        c.put(path, tree_folder, **kwargs)
        return {'module_tree_folders': tree_folder}
    
    @classmethod
    def ls_trees(cls):
        path = tree_folders_path
        tree_folders = c.get(path, [])
        return tree_folders
    @classmethod
    def rm_tree(cls, tree_path:str, **kwargs):
        path = cls.tree_folders_path
        tree_folder = c.get(tree_path, [])
        tree_folder = [f for f in tree_folder if f != tree_path ]
        c.put(path, tree_folder)
        return {'module_tree_folders': tree_folder}

    @classmethod
    def dash(cls, *args, **kwargs):
        c.print('FAM')
        if cls.module_path() == 'module':
            return cls.st('dashboard')
        else:
            return cls.st()
    
    @classmethod
    def dashboard(cls):
        return c.module('dashboard').dashboard()
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

    @staticmethod
    def timeit(fn):
        def wrapper(*args, **kwargs):
            t = c.time()
            result = fn(*args, **kwargs)
            c.print(f'Finished {fn.__name__} in {c.time() - t:.2f} seconds')
            return result
        
        return wrapper
    
    @staticmethod
    def remotewrap(fn):
        '''
        WARNNG IN PROGRSS, USE WITH CAUTION
        '''
        
        def wrapper(self, *args, **kwargs):
            
            c.remote_fn(module=self, fn=fn.__name__, args=args, kwargs=kwargs)
            result = fn(self, *args, **kwargs)
            c.print(f'Finished {fn.__name__} in {c.time() - t:.2f} seconds')
            # return result
        
        return wrapper
    
    
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
            assert isinstance(args, list), f'args must be a list, got {type(args)}'
            return args, kwargs

        assert isinstance(kwargs, dict), f'kwargs must be a dict, got {type(kwargs)}'
        
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
        return f'{c.cache_path()}/{cls.module_path()}'
    storage_dir = tmp_dir
    
    @classmethod
    def refresh_storage(cls):
        c.rm(cls.tmp_dir())

    @classmethod
    def refresh_tmp_dir(cls):
        c.rm(cls.tmp_dir())
        c.makedirs(cls.tmp_dir())
        

    ############ JSON LAND ###############

    @classmethod
    def cache_path(cls):
        return os.path.expanduser(f'~/.{cls.library_name}')

    @classmethod
    def tilde_path(cls):
        return os.path.expanduser('~')

        
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
            data = await async_get_json(path, default=default, **kwargs)
        except Exception as e:
            if verbose:
                c.print(f'Failed to load json from {path} with error {e}')
            return default
        if isinstance(data, dict):
            if 'data' in data and 'meta' in data:
                data = data['data']
        
        return data

    load_json = get_json

    data_path = repo_path + '/data'

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
                 cache: bool = False,
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
        
        exists =  os.path.exists(path)
        if not exists and not path.endswith('.json'):
            exists = os.path.exists(path + '.json')
        
        return exists

        

    @classmethod
    def docs(cls):
        # Markdown input
        markdown_text = "## Hello, *Markdown*!"


        path = cls.filepath().replace('.py', '_docs.md')
        markdown_text =  cls.get_text(path=path)
        return markdown_text



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


    def rm_many(cls, paths:List[str]):
        paths = c.ls(paths)

    


        # for path in paths:
        #     cls.rm(path)

    @classmethod
    def rm(cls, path, extension=None, root=False, mode = 'json'):
        path = cls.resolve_path(path=path, extension=extension, root=root)

        if not os.path.exists(path) and os.path.exists(path+'.json'):
            path += f'.{mode}'

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
    def glob(cls,  path =None, files_only:bool = True, root:bool = False, recursive:bool=True):
        
        path = cls.resolve_path(path, extension=None, root=root)
        
        if os.path.isdir(path):
            path = os.path.join(path, '**')
            
        paths = glob(path, recursive=recursive)
        
        if files_only:
            paths =  list(filter(lambda f:os.path.isfile(f), paths))
        return paths

    @classmethod
    def get_file_size(cls, path:str):
        path = cls.resolve_path(path)
        return os.path.getsize(path)
         
    @classmethod
    def ls_json(cls, path:str = '', recursive:bool = True):
        return [os.path.basename(p).replace('.json', '')for p in cls.ls(path, recursive=recursive)]
    

    @classmethod
    def ls(cls, path:str = '', 
           recursive:bool = False,
           root:bool = False,
           return_full_path:bool = True):
        """
        provides a list of files in the path 

        this path is relative to the module path if you dont specifcy ./ or ~/ or /
        which means its based on the module path
        """
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
        return cls.namespace_local().get(name, {})

    @classmethod
    def connect(cls,
                module:str, 
                network : str = None,
                namespace = None,
                mode = server_mode,
                virtual:bool = True, 
                verbose: bool = False, 
                prefix_match: bool = False,
                key = None,
                return_future:bool = False,):

        kwargs = c.locals2kwargs(locals())
        return_future = kwargs.pop('return_future', False)
        future = cls.async_connect(**kwargs)

        if return_future:
            return future
        return c.gather(future)

    @classmethod
    async def async_connect(cls, 
                module:str, 
                network : str = None,
                namespace = None,
                mode = server_mode,
                virtual:bool = False, 
                verbose: bool = True, 
                prefix_match: bool = False,
                key = None,
                **kwargs ):

        """
        Connects to a server by the name of the module
        :param module: name of the module
        """

        network = c.resolve_network(network)
        key = cls.get_key(key)
        if c.is_address(module):
            address = module
        else:
            namespace = namespace if namespace != None else c.namespace(module, network=network)
            modules = list(namespace.keys())
            if prefix_match == True:
                module = c.choice(modules)
            else:
                modules = [m for m in modules if m==module]
                
            assert len(modules) > 0, f'No modules with {module} found in namespace {namespace.keys()}'
            address = namespace[module]

        port = address.split(':')[-1]
        ip = address.replace(f':{port}', '')

        # CONNECT TO THE MODULE
        if 'None' in address:
            raise Exception(f'Invalid address {address}')

        if ip == c.ip():
            ip = '0.0.0.0'

        client= c.get_client(ip=ip, port=int(port), key=key, mode=mode, virtual=virtual, **kwargs)

        return client
     
    @classmethod
    def root_address(cls, name:str='module',
                    network : str = 'local',
                    timeout:int = 100, 
                    sleep_interval:int = 1,
                    **kwargs):

        """
        Root module
        """
        try:
            if not c.server_exists(name, network=network):
                c.serve(name, network=network, wait_for_server=True, **kwargs)
            address = c.call('module', 'address', network=network, timeout=timeout)
            ip = c.ip()
            address = ip+':'+address.split(':')[-1]
        except Exception as e:
            c.print(f'Error: {e}', color='red')
            address = None
        return address
    addy = root_address

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
    def connect_pool(cls, modules=None, *args, return_dict:bool=False, **kwargs):
        if modules == None:
            modules = c.servers(modules)
        
        module_clients =  cls.gather([cls.async_connect(m, ignore_error=True,**kwargs) for m in modules])
        if return_dict:
            return dict(zip(modules, module_clients))
        return module_clients

    @classmethod
    def get_client(cls, ip:str = None, port:int = None ,virtual:bool = True, mode=server_mode, **kwargs):
        '''
        Returns a client to a server
        '''
        client = c.module(f'server.{mode}.client')(ip=ip, port=port,**kwargs)
        # if virtual turn client into a virtual client, making it act like if the server was local
        if virtual:
            client = c.virtual_client(client)
        
        return client

    
   
    nest_asyncio_enabled : bool = False
    @classmethod
    def nest_asyncio(cls):
        assert not cls.nest_asyncio_enabled, 'Nest Asyncio already enabled'
        import nest_asyncio
        nest_asyncio.apply()
        nest_asyncio_enabled = True
        

    

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
        
        
    @staticmethod
    def check_response(x) -> bool:
        if isinstance(x, dict) and 'error' in x:
            return False
        else:
            return True
    
    @classmethod
    def check_connection(cls, *args, **kwargs):
        return c.gather(cls.async_check_connection(*args, **kwargs))

    @classmethod
    def module2connection(cls,modules = None, network=None):
        if modules == None:
            modules = c.servers(network=network)
        connections = c.gather([ c.async_check_connection(m) for m in modules])

        module2connection = dict(zip(modules, connections))
    
        return module2connection


    @classmethod
    def dead_servers(cls, network=None):
        module2connection = cls.module2connection(network=network)
        dead_servers = [m for m, c in module2connection.items() if not c]
        return dead_servers


        


    @classmethod
    async def async_check_connection(cls, module, timeout=5, **kwargs):
        try:
            module = await c.async_connect(module, return_future=False, virtual=False, **kwargs)
        except Exception as e:
            return False
        server_name =  await module(fn='server_name',  return_future=True)
        if c.check_response(server_name):
            return True
        else:
            return False

    
    
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
    def is_address(cls, address:str) -> bool:
        if not isinstance(address, str):
            return False
        conds = []
        conds.append(isinstance(address, str))
        conds.append(':' in address)
        conds.append(cls.is_number(address.split(':')[-1]))
    
        return all(conds)
    
    @classmethod
    def is_module(cls, obj=None) -> bool:
        
        if obj is None:
            obj = cls
        if all([hasattr(obj, k) for k in ['module_class', 'root_module_class', 'set_config', '']]):
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
    def get_port(cls, port:int = None)->int:
        port = port if port is not None and port != 0 else cls.free_port()
        while cls.port_used(port):
            port += 1   
        return port 
    
    resolve_port = get_port

    @property
    def server_name(self):
        if not hasattr(self, 'config') or not (isinstance(self.config, Munch)):
            self.config =  Munch({})

        config = self.config

        if 'server_name' in self.config:
            name =  config['server_name']
        else:
            name = self.module_path()
            if self.tag !=None: 
                name = f'{name}::{self.tag}'
            config['server_name'] = name
            self.config = config
            
        return name
        
    @server_name.setter
    def server_name(self, v):
        self.config['server_name'] = v
        return self.config['server_name']
    
    @classmethod
    def wait_for_server(cls,
                          name: str ,
                          network: str = 'local',
                          timeout:int = 600,
                          sleep_interval: int = 1, 
                          verbose:bool = False) -> bool :
        
        time_waiting = 0
        logs = []
        while not c.server_exists(name, network=network):
            c.sleep(sleep_interval)
            time_waiting += sleep_interval
            c.print(f'Waiting for server {name} to start')
            new_logs = list(set(c.logs(name, mode='local').split('\n')))
            print_logs = [l for l in new_logs if l not in logs]

            if verbose:
                if len(print_logs) > 0:
                    logs.extend(print_logs)
                    logs = list(set(logs))
                    c.print('\n'.join(print_logs))
            if time_waiting > timeout:
                raise TimeoutError(f'Timeout waiting for server to start')
        return True
        
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
    def virtual_client(cls, module): 
        virtual_client =  c.import_object('commune.modules.client.virtual.VirtualClient')
        return virtual_client(module)
    
    # NAMESPACE::MODULE
    namespace_module = 'module.namespace'
    @classmethod
    def name2address(cls, name:str, network:str='local') -> str:
        return c.module("namespace").name2address(name=name, network=network)
    @classmethod
    def servers(cls, *args, **kwargs) -> List[str]:
        return c.module("namespace").servers(*args, **kwargs)

    @classmethod
    def rservers(cls, *args, **kwargs) -> List[str]:
        return c.module("remote").servers(*args, **kwargs)

    
    @classmethod
    def get_address(cls, module, **kwargs):
        address = c.module("namespace").get_address(module, **kwargs)
        return address
    @classmethod
    def get_port(cls, module, **kwargs):
        address =  cls.get_address(module, **kwargs)
        if address == None:
            return None
        return int(address.split(':')[-1])
    @classmethod
    def server_infos(cls, *args, **kwargs) -> List[str]:
        return c.module("namespace").server_infos(*args, **kwargs)
    @classmethod
    def server2info(cls, *args, **kwargs) -> List[str]:
        return c.module("namespace").server2info(*args, **kwargs)
    
    @classmethod
    def has_server(cls, *args, **kwargs):
        return c.module("namespace").has_server(*args, **kwargs)
    @classmethod
    def server_exists(cls, name:str, network:str = 'local',  prefix_match:bool=False, **kwargs) -> bool:

        return c.module("namespace").server_exists(name=name, network=network,  prefix_match=prefix_match, **kwargs)
    @classmethod
    def register_server(cls, name: str, address:str, network='local')-> dict:
        return c.module("namespace").register_server(name=name, address=address, network=network)

    @classmethod
    def deregister_server(cls, name: str, network:str = 'local')-> dict:
        return c.module("namespace").deregister_server(name=name, network=network)
    @classmethod
    def add_server(cls, *args, **kwargs):
        return c.module("namespace").add_server(*args, **kwargs)
    @classmethod
    def add_servers(cls, *args, **kwargs):
        return c.module("namespace").add_servers(*args, **kwargs)

    @classmethod
    def readd_servers(cls, *args, **kwargs):
        return c.module("namespace").readd_servers(*args, **kwargs)
    @classmethod
    def rm_server(cls, *args, **kwargs):
        return c.module("namespace").rm_server(*args, **kwargs)

    @classmethod
    def remote_servers(cls, *args, **kwargs):
        return c.module("namespace").remote_servers(*args, **kwargs)

    @classmethod
    def namespace(cls,
                  search:str = None,
                  network:str='local',
                  update: bool = False, **kwargs):
        return c.module("namespace").namespace(search=search, network=network, update=update, **kwargs)
    @classmethod
    def rm_namespace(cls, *args, **kwargs):
        return c.module("namespace").rm_namespace(*args, **kwargs)

    @classmethod
    def empty_namespace(cls, *args, **kwargs):
        return c.module("namespace").empty_namespace(*args, **kwargs)

    @classmethod
    def add_namespace(cls, *args, **kwargs):
        return c.module("namespace").empty_namespace(*args, **kwargs)


    
    @classmethod
    def update_namespace(cls, network:str='local',**kwargs):
        return c.module("namespace").update_namespace(network=network, **kwargs)
    
    @classmethod
    def put_namespace(cls,network:str, namespace:dict, **kwargs):
        namespace = c.module("namespace").put_namespace(network=network, namespace=namespace, **kwargs)
        return namespace
    
    @classmethod
    def rm_namespace(cls,network:str, **kwargs):
        namespace = c.module("namespace").rm_namespace(network=network, **kwargs)
        return namespace
    

    
    @classmethod
    def resolve_server_name(cls, module:str = None, tag:str=None, name:str = None,  tag_seperator:str='::', **kwargs):
        

        # if name is not specified, use the module as the name such that module::tag
        if name == None:
            # module::tag
            module = cls.module_path() if module == None else module
            if tag_seperator in module: 
                module, tag = module.split(tag_seperator)
            name = module
            if tag in ['None','null'] :
                tag = None
            if tag != None:
                name = f'{name}{tag_seperator}{tag}'
        assert isinstance(name, str), f'Invalid name {name}'
        return name
    resolve_name = resolve_server_name
    
    @property
    def whitelist(self):
        if hasattr(self, '_whitelist'):
            return self._whitelist
        whitelist = c.helper_whitelist
        is_module = c.is_root_module(self)
        # we want to expose the helper functions
        if not is_module:
            whitelist += self.functions() + self.attributes()
        return whitelist
    
    @whitelist.setter
    def whitelist(self, whitelist:List[str]):
        self._whitelist = whitelist + self.helper_functions
        return whitelist
    bl = blacklist = []
    
    @classmethod
    def save_serve_kwargs(cls,server_name:str,  kwargs:dict, network:str = 'local'):
        serve_kwargs = c.get(f'serve_kwargs/{network}', {})
        serve_kwargs[server_name] = kwargs
        c.put(f'serve_kwargs/{network}', serve_kwargs)
        return serve_kwargs
    
    @classmethod
    def load_serve_kwargs(cls, server_name:str, network:str = 'local'):
        serve_kwargs = c.get(f'serve_kwargs/{network}', {})
        return serve_kwargs.get(server_name, {})

    @classmethod
    def has_serve_kwargs(cls, server_name:str, network='local'):
        serve_kwargs = c.get(f'serve_kwargs/{network}', {})
        return server_name in serve_kwargs

    @classmethod
    def serve(cls, 
              module:Any = None ,
              tag:str=None,
              network = 'local',
              port :int = None, # name of the server if None, it will be the module name
              server_name:str=None, # name of the server if None, it will be the module name
              kwargs:dict = None,  # kwargs for the module
              refresh:bool = True, # refreshes the server's key
              wait_for_server:bool = False , # waits for the server to start before returning
              remote:bool = True, # runs the server remotely (pm2, ray)
              server_mode:str = server_mode,
              tag_seperator:str='::',
              update:bool = False,
              max_workers:int = None,
              mode:str = "thread",
              public: bool = False,
              verbose:bool = False,
              **extra_kwargs
              ):

        
        # RESOLVE THE KWARGS
        kwargs = kwargs or {}
        if 'kwargs' in kwargs:
            kwargs = kwargs['kwargs']
        kwargs = {**kwargs, **extra_kwargs} # ADD THE EXTRA KWARGS
        extra_kwargs = {} # EMPTY THE EXTRA KWARGS

        if module == None:
            module = cls.module_path()

        # module::tag
        if tag_seperator in module:
            module, tag = module.split(tag_seperator)

        if 'tag' in kwargs:
            tag = kwargs['tag']
    
        server_name = cls.resolve_server_name(module=module, name=server_name, tag=tag, tag_seperator=tag_seperator)


        if tag_seperator in server_name:
            tag = server_name.split(tag_seperator)[-1] 
        
        address = c.get_address(server_name, network=network)
        if address != None and ':' in address:
            port = int(address.split(':')[-1])
        if port == None:
            port = c.free_port()
        # NOTE REMOVE THIS FROM THE KWARGS REMOTE

        if remote:
            remote_kwargs = cls.locals2kwargs(locals(), merge_kwargs=False)
            remote_kwargs.pop('extra_kwargs') # REMOVE THE extra_kwargs
    
            remote_kwargs['remote'] = False # SET THIS TO FALSE
            remote_kwargs.pop('address') # WE INTRODUCED THE ADDRES
            c.save_serve_kwargs(server_name, remote_kwargs) # SAVE THE RESULTS
            c.print(f'Serving {server_name} remotely {remote_kwargs}', color='yellow')
            response = cls.remote_fn('serve',name=server_name, kwargs=remote_kwargs)
            if wait_for_server:
                cls.wait_for_server(server_name, network=network)
            address = c.ip() + ':' + str(remote_kwargs['port'])
            return {'success':True, 'name': server_name, 'address':address}
        
        module_class = cls.resolve_module(module)
        kwargs.update(extra_kwargs)

  
        # this automatically adds 
        self = module_class(**kwargs)
        self.tag = tag
        self.server_name = server_name
        self.key = server_name

        address = c.get_address(server_name, network=network)
        if address != None and ':' in address:
            port = address.split(':')[-1]   


        if c.server_exists(server_name, network=network): 
            
            if refresh:
                c.print(f'Stopping existing server {server_name}', color='yellow') 
                c.deregister_server(server_name, network=network)
                if c.pm2_exists(server_name): 
                    c.kill(server_name)
            else:  
                return {'success':True, 'message':f'Server {server_name} already exists'}

        

        c.module(f'server.{server_mode}')(module=self, 
                                          name=server_name, 
                                          port=port, 
                                          network=network, 
                                          max_workers=max_workers, 
                                          mode=mode, 
                                          public=public)
        
        response =  {'success':True, 'address':  f'{c.default_ip}:{port}' , 'name':server_name, 'module':module}

        return response

    serve_module = serve
    
    @classmethod
    def functions(cls, search: str=None , include_parents:bool = False):
        functions = cls.get_functions(include_parents=include_parents)  
        functions = list(set(functions))
        if isinstance(search, str):
            functions = [f for f in functions if search in f]
        return functions

    fns = functions
    
    @classmethod
    def get_function_signature_map(cls, obj=None, include_parents:bool = False):
        function_signature_map = {}
        if isinstance(obj, str):
            obj = c.module(obj)
        obj = obj if obj else cls
        for f in cls.get_functions(obj = obj, include_parents=include_parents):
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
    def function_signature_map(self, include_parents:bool = False):
        return self.get_function_signature_map(obj=self, include_parents=include_parents)
    
    @property
    def function_default_map(self, include_parents=False):
        return self.get_function_default_map(obj=self, include_parents=False)
        
    @classmethod
    def get_function_default_map(cls, obj:Any= None, include_parents=False) -> Dict[str, Dict[str, Any]]:
        obj = obj if obj else cls
        default_value_map = {}
        function_signature = cls.get_function_signature_map(obj=obj,include_parents=include_parents)
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
    


    
    def is_fn_allowed(self, fn_name:str) -> bool:
        whitelist = self.whitelist
        blacklist = self.blacklist
        if fn_name in whitelist and fn_name not in blacklist:
            return True
        else:
            return False

    def set_server_name(self, name:str, **kwargs):
        if hasattr(self, 'server_name'):
            c.deregister_server(name)
        self.server_name = name
        c.print(f'Server name set to {name}', color='yellow')
        c.register_server(name, self.address, **kwargs)
        return {'success':True, 'message':f'Server name set to {name}'}
        
    @classmethod
    def dummy_gen(cls):
        for i in range(10):
            c.print(i)
            yield i
        
    def info(self , 
             schema: bool = True,
             namespace:bool = False,
             peers: bool = False) -> Dict[str, Any]:
        fns = [fn for fn in self.fns() if self.is_fn_allowed(fn)]
        attributes =[ attr for attr in self.attributes() if self.is_fn_allowed(attr)]
        info  = dict(
            address = self.address.replace(c.default_ip, c.ip(update=False)),
            functions =  fns, # get the functions of the module
            attributes = attributes, # get the attributes of the module
            name = self.server_name() if callable(self.server_name) else self.server_name, # get the name of the module
            path = self.module_path(), # get the path of the module
            chash = self.chash(), # get the hash of the module (code)
        )
        info['hash'] = c.hash(info)

        if hasattr(self, 'key'):
            auth = self.key.sign(info, return_json=True)
            info['signature'] = auth['signature']
            info['ss58_address'] = auth['address']
        if schema:
            schema = self.schema(defaults=True)
            info['schema'] = {fn: schema[fn] for fn in fns}

        return info
    
    help = info


    

    @classmethod
    def schema(cls,search: str = None,
                    code : bool = False,
                    docs: bool = True,
                    include_parents:bool = False,
                     defaults:bool = True) -> 'Schema':

        kwargs = c.locals2kwargs(locals())
        return {k: v for k,v in cls.get_schema(**kwargs).items()}
    
    @classmethod
    def init_schema(cls):
        return cls.fn_schema('__init__')

    @classmethod
    def init_kwargs(cls):
        kwargs =  cls.fn_defaults('__init__')
        kwargs.pop('self')
        if 'config' in kwargs:
            if kwargs['config'] != None:
                kwargs.update(kwargs.pop('config'))
            del kwargs['config']
        if 'kwargs' in kwargs:
            if kwargs['kwargs'] != None:
                kwargs = kwargs.pop('kwargs')
            del kwargs['kwargs']

        return kwargs

    @classmethod
    def get_schema(cls,
                                search = None,
                                module = None,
                                code : bool = False,
                                docs: bool = True,
                                include_parents:bool = False,
                                defaults:bool = True,):
        
        module = module if module else cls
        
        if isinstance(module, str):
            module = c.module(module)
            
        function_schema_map = {}
        for fn in cls.get_functions(module, include_parents=include_parents):
               
            if search != None :
                if search not in fn:
                    continue
            module_fn = getattr(module, fn )
            if callable(module_fn):
                function_schema_map[fn] = cls.fn_schema(fn, defaults=defaults, code=code, docs=docs)
    
        return function_schema_map

    @classmethod
    def get_function_annotations(cls, fn):
        fn = cls.get_fn(fn)
        return fn.__annotations__
        
    @classmethod
    def fn_schema(cls, fn:str,
                            defaults:bool=True,
                            code:bool = False,
                            docs:bool = True)->dict:
        '''
        Get function schema of function in cls
        '''
        import inspect
        fn_schema = {}
        fn = cls.get_fn(fn)
        fn_args = cls.get_function_args(fn)
        fn_schema['input']  = cls.get_function_annotations(fn=fn)
        
        if defaults:
            fn_schema['default'] = cls.fn_defaults(fn=fn) 
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
            fn_schema['code'] = cls.fn_code(fn)
 
        fn_args = c.get_function_args(fn)
        fn_schema['type'] = 'static'
        for arg in fn_args:
            if arg not in fn_schema['input']:
                fn_schema['input'][arg] = 'NA'
            if arg in ['self', 'cls']:
                fn_schema['type'] = arg
                fn_schema['input'].pop(arg)
                if 'default' in fn_schema:
                    fn_schema['default'].pop(arg, None)
                

        return fn_schema
    

    @staticmethod
    def get_annotations(fn:callable) -> dict:
        return fn.__annotations__

    @classmethod
    def kill(cls, module,
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
                c.deregister_server(m)

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
        
    killpre = kill_prefix



    @classmethod
    def kill_many(cls, search:str, network='local', parallel=False, **kwargs):
        servers = c.servers(network=network)
        servers = [s for s in servers if  search in s]

        futures = []
        for s in servers:
            future = c.submit(c.kill, kwargs={'module':s, **kwargs}, mode='process', return_future = True)
            futures.append(future)

        results = c.wait(futures)
            
        return {'success':True, 'message':f'Killed servers with prefix {search}', 'results': results}
        
    delete = kill_server = kill
    def destroy(self):
        self.kill(self.server_name)
        return path
    
    def self_destruct(self):
        c.kill(self.server_name)    
        
    def self_restart(self):
        c.restart(self.server_name)
        
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
    def launch(cls, 
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
                server_name = line.split('│')[2].strip()
                # fixes odd issue where there is a space between the name and the front 
                server_name = server_name.split(' ')[-1]
                module_list += [server_name]
                
        
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
            module = cls.module_path()
        elif hasattr(module, 'module_path'):
            module = module.module_path()
            
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
            
        stdout = c.cmd(command, env=env, verbose=verbose)
        
        return stdout

    @classmethod
    def register(cls,  
                 module = None,
                 tag:str = None,
                 key : str = None,
                 stake : int = None,
                 subnet:str = 'commune',
                 refresh:bool =False,
                 wait_for_server:bool = False,
                 **kwargs ):
        subspace = c.module('subspace')()

        # resolve module name and tag if they are in the server_name
        if isinstance(module, str) and  '::' in module:
            module, tag = module.split('::')
        server_name = cls.resolve_server_name(module=module, tag=tag)

        if not c.key_exists(server_name):
            c.add_key(server_name)
        if c.server_exists(server_name, network='local') and refresh == False:
            c.print(f'Server already Exists ({server_name})')
            address = c.get_address(server_name)
        
        else:
            module = cls.resolve_module(module)
            serve_info =  module.serve(
                                server_name=server_name, 
                                wait_for_server=wait_for_server, 
                                refresh=refresh, 
                                tag=tag,
                                **kwargs)
            server_name = serve_info['name']
            address = serve_info['address']

        return subspace.register(name=server_name, address=address, subnet=subnet, key=key, stake=stake)

    @classmethod
    def key_stats(cls, *args, **kwargs):
        return c.module('subspace')().key_stats(*args, **kwargs)

    @classmethod
    def key2stats(cls, *args, **kwargs):
        return c.module('subspace')().key_stats(*args, **kwargs)

    
    r = reg = register
    @classmethod
    def pm2_kill(cls, name:str, verbose:bool = False, prefix_match:bool = True):
        pm2_list = cls.pm2_list()
        if name in pm2_list:
            rm_list = [name]
        else:
            if prefix_match:
                rm_list = [ p for p in pm2_list if p.startswith(name)]
            else:
                raise Exception(f'pm2 process {name} not found')
        if len(rm_list) == 0:
            if verbose:
                c.print(f'ERROR: No pm2 processes found for {name}',  color='red')
            return []
        for n in rm_list:
            if verbose:
                c.print(f'Killing {n}', color='red')
            cls.cmd(f"pm2 delete {n}", verbose=False)
            cls.pm2_rm_logs(n)
        return rm_list
    
    @staticmethod
    def detailed_error(e) -> dict:
        import traceback
        tb = traceback.extract_tb(e.__traceback__)
        file_name = tb[-1].filename
        line_no = tb[-1].lineno
        line_text = tb[-1].line
        response = {
            'error': str(e),
            'file_name': file_name,
            'line_no': line_no,
            'line_text': line_text
        }   
        return response
    
    @classmethod
    def pm2_restart(cls, name:str, verbose:bool = False, prefix_match:bool = True):
        pm2_list = cls.pm2_list()
        if name in pm2_list:
            rm_list = [name]
        else:
            if prefix_match:
                rm_list = [ p for p in pm2_list if p.startswith(name)]
            else:
                raise Exception(f'pm2 process {name} not found')

        if len(rm_list) == 0:
            if verbose:
                c.print(f'ERROR: No pm2 processes found for {name}',  color='red')
            return []
        for n in rm_list:
            c.print(f'Restarting {n}', color='cyan')
            c.cmd(f"pm2 restart {n}", verbose=False)
            cls.pm2_rm_logs(n)  
        return rm_list
       
    @classmethod
    def pm2_restart_prefix(cls, name:str = None, verbose:bool=False):
        pm2_list = cls.pm2_list()
            
        restarted_modules = []
        
        for module in pm2_list:
            if module.startswith(name) or name in ['all']:
                if verbose:
                    c.print(f'Restarting {module}', color='cyan')
                c.cmd(f"pm2 restart {module}", verbose=verbose)
                restarted_modules.append(module)
        
        return restarted_modules
            
    
    @classmethod
    def restart(cls, name:str, mode:str='pm2', verbose:bool = False, prefix_match:bool = True):
        refreshed_modules = getattr(cls, f'{mode}_restart')(name, verbose=verbose, prefix_match=prefix_match)
        return refreshed_modules

    def restart_self(self):
        """
        Helper function to restart the server
        """
        c.restart_server(self.server_name)


    def kill_self(self):
        """
        Helper function to kill the server
        """
        c.kill(self.server_name)

    refresh = reset = restart
    @classmethod
    def pm2_status(cls, verbose=True):
        stdout = cls.run_command(f"pm2 status")
        if verbose:
            c.print(stdout,color='green')
        return stdout

    pm2_dir = os.path.expanduser('~/.pm2')
    @classmethod
    def pm2_logs_path_map(cls, name=None):
        pm2_logs_path_map = {}
        for l in c.ls(f'{cls.pm2_dir}/logs/'):
            key = '-'.join(l.split('/')[-1].split('-')[:-1]).replace('-',':')
            pm2_logs_path_map[key] = pm2_logs_path_map.get(key, []) + [l]

    
        for k in pm2_logs_path_map.keys():
            pm2_logs_path_map[k] = {l.split('-')[-1].split('.')[0]: l for l in list(pm2_logs_path_map[k])}

        if name != None:
            return pm2_logs_path_map.get(name, {})

        return pm2_logs_path_map

    @classmethod
    def pm2_rm_logs( cls, name):
        pm2_logs_map = cls.pm2_logs_path_map(name)

        for k in pm2_logs_map.keys():
            c.rm(pm2_logs_map[k])

    @classmethod
    def pm2_logs(cls, 
                module:str, 
                tail: int =100, 
                verbose: bool=True ,
                mode: str ='cmd'):

        if mode == 'local':
            text = ''
            for m in ['out','error']:

                # I know, this is fucked 
                path = f'{cls.pm2_dir}/logs/{module.replace("/", "-")}-{m}.log'.replace(':', '-').replace('_', '-')
                try:
                    text +=  c.get_text(path, tail=tail)
                except Exception as e:
                    c.print(e)
                    continue
            
            return text
        elif mode == 'cmd':
            return cls.run_command(f"pm2 logs {module}", verbose=verbose)
        else:
            raise NotImplementedError(f'mode {mode} not implemented')
    @staticmethod
    def memory_usage(fmt='gb'):
        fmt2scale = {'b': 1e0, 'kb': 1e1, 'mb': 1e3, 'gb': 1e6}
        import os, psutil
        process = psutil.Process()
        scale = fmt2scale.get(fmt)
        return (process.memory_info().rss // 1024) / scale

    @classmethod
    def argparse(cls, verbose: bool = False):
        import argparse
        parser = argparse.ArgumentParser(description='Argparse for the module')
        parser.add_argument('-fn', '--fn', dest='function', help='fn', type=str, default="__init__")
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

            c.print(args)
            if args.function == '__init__':
                return cls(*args.args, **args.kwargs)     
            else:
                fn = getattr(cls, args.function)
                fn_type = cls.classify_method(fn)

                if fn_type == 'self':
                    self = cls(*args.args, **args.kwargs)

                    return

                return getattr(cls, args.function)(*args.args, **args.kwargs)     

    
    
    
    @classmethod
    def learn(cls, *args, **kwargs):
        return c.module('model.hf').learn(*args, **kwargs)
        
    
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

    # def resource_usage(self):
    #     resource_dict =  self.config.get('actor', {}).get('resources', None)
    #     resource_dict = {k.replace('num_', ''):v for k,v in resource_dict.items()}
    #     resource_dict['memory'] = self.memory_usage(mode='ratio')
    #     return  resource_dict
    

    @classmethod
    def transfer_fn_code(cls, module1= 'module',
                        fn_prefix = 'ray_',
                        module2 = 'ray',
                        refresh = False):

        module1 = c.module(module1)
        module2 = c.module(module2)
        module1_fn_code_map = module1.fn2code(fn_prefix)
        module2_code = module2.code()
        module2_fns = module2.fns()
        filepath = module2.filepath()
        for fn_name, fn_code in module1_fn_code_map.items():
            c.print(f'adding {fn_name}')
            c.print('fn_code', fn_code)
            if fn_name in module2_fns:
                if refresh:
                    module2_code = module2_code.replace(module2_fns[fn_name], '')
                else:
                    c.print(f'fn_name {fn_name} already in module2_fns {module2_fns}')

            module2_code += '\n'
            module2_code += '\n'.join([ '    ' + line for line in fn_code.split('\n')])
            module2_code += '\n'
        c.print('module2_code', module2_code)
        c.put_text(filepath, module2_code)

        return {'success': True, 'module2_code': module2_code, 'module2_fns': module2_fns, 'module1_fn_code_map': module1_fn_code_map}
    @classmethod
    def ensure_ray_context(cls, ray_config:dict = None):
        ray_config = ray_config if ray_config != None else {}
        
        if cls.ray_initialized():
            ray_context = cls.get_ray_context()
        else:
            ray_context =  cls.ray_init(init_kwargs=ray_config)
        
        return ray_context

    @classmethod
    def get_server_name(cls, name:str=None, tag:str=None, seperator:str='.'):
        name = name if name else cls.__name__.lower()
        if tag != None:
            name = tag + seperator + name
        return name
  

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

    module_cache = {}
    
    @classmethod
    def module(cls,module: Any = 'module' , **kwargs):
        '''
        Wraps a python class as a module
        '''
        if module is None:
            module = cls.module_path()
        modules = c.modules()
        assert module in modules, f'{module} does not exist'
        if module in c.module_cache:
            module_class = c.module_cache[module]
        else:
            module_class =  c.get_module(module,**kwargs)
            c.module_cache[module] = module_class
        
        return module_class
        


    m = mod = module

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
    def merge(cls, a = None, b= None, 
                        include_hidden:bool=True, 
                        allow_conflicts:bool=True, 
                        verbose: bool = False):
        
        '''
        Merge the functions of a python object into the current object (a)
        '''
        if a == None:
            a =  cls

        assert a != None, 'a cannot be None'
        assert b != None, 'b cannot be None'
        
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
            pass
        
        
    # JUPYTER NOTEBOOKS
    @classmethod
    def jupyter(cls):
        cls.nest_asyncio()
    enable_jupyter = jupyter
        
    @classmethod
    def int_to_ip(cls, *args, **kwargs):
        return c.import_object('commune.utils.network.int_to_ip')(*args, **kwargs)
        
    @classmethod
    def ip_to_int(cls, *args, **kwargs):
        return c.import_object('commune.utils.network.ip_to_int')(*args, **kwargs)

    @classmethod
    def ip_version(cls, *args, **kwargs):
        return c.import_object('commune.utils.network.ip_version')(*args, **kwargs)
    
    @classmethod
    def pip_list(cls, lib=None):
        pip_list =  c.cmd(f'pip list', verbose=False, bash=True).split('\n')
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

    required_libs = []
    @classmethod
    def ensure_libs(cls, *libs, verbose:bool=False):
        assert len(libs) > 0, 'No libraries specified'
        if len(libs) == 1:
            if isinstance(libs[0], list):
                libs = libs[0]
        elif len(libs) == 0:
            raise Exception('No libraries specified')
        elif len(libs) > 1:
            libs = list(libs)
        else:
            raise Exception('No libraries specified, WTF WAS THIS')

            
        if libs == None:
            libs = cls.required_libs
        r = []
        for lib in libs:
            r.append(cls.ensure_lib(lib, verbose=verbose))
            c.print(r[-1])
        return r
    
    @classmethod
    def ensure_env(cls):
        c.ensure_libs(cls.libs)
    
    ensure_package = ensure_lib
    @classmethod
    def pip_install(cls, 
                    lib:str= None,
                    upgrade:bool=True ,
                    verbose:str=True,
                    ):
        

        if lib in c.modules():
            c.print(f'Installing {lib} Module from local directory')
            lib = c.resolve_module(lib).dirpath()
        if lib == None:
            lib = c.libpath

        if c.exists(lib):
            cmd = f'pip install -e'
        else:
            cmd = f'pip install'
            if upgrade:
                cmd += ' --upgrade'
        return cls.cmd(cmd, verbose=verbose)

    def install(self, lib:str = None, verbose:bool=True, upgrade=True):
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
        lines = [l for l in cls.cmd(f'pip list', verbose=False).split('\n') if l.startswith(lib)]
        if len(lines)>0:
            return lines[0].split(' ')[-1].strip()
        else:
            return f'No Library Found {lib}'
    
    @classmethod
    def external_ip(cls, *args, **kwargs) -> str:
        ip = c.module('network').get_external_ip(*args, **kwargs)
        if ip == None or len(ip) == 0:
            ip = c.default_ip
        return ip
    

    
    @classmethod
    def ip(cls, update:bool = False, **kwargs) -> str:
        if not update:
            ip = c.get('ip', None)
            if ip != None:
                return ip
        
        ip =  cls.external_ip(**kwargs)
        if ip == None:
            ip = '0.0.0.0'
        if update:
            c.put('ip', ip)
        return ip
    @classmethod
    def queue(cls, size:str=-1, *args,  mode='queue', **kwargs):
        if mode == 'queue':
            return c.import_object('queue.Queue')(size, *args, **kwargs)
        elif mode in ['multiprocessing', 'mp', 'process']:
            return c.module('process')(size, *args, **kwargs)
        elif mode == 'ray':
            return c.import_object('ray.util.queue.Queue')(size, *args, **kwargs)
        elif mode == 'redis':
            return c.import_object('redis.Queue')(size, *args, **kwargs)
        elif mode == 'rabbitmq':
            return c.import_object('pika.Queue')(size, *args, **kwargs)
        else:
            raise NotImplementedError(f'mode {mode} not implemented')

    
    @classmethod
    def resolve_ip(cls, ip=None, external:bool=True) -> str:
        if ip == None:
            if external:
                ip = c.external_ip()
            else:
                ip = '0.0.0.0'
        assert isinstance(ip, str)
        return ip

    @staticmethod
    def is_class(module: Any) -> bool:
        return type(module).__name__ == 'type' 
    
    
    @classmethod
    def upnpc_create_port_map(cls, port:int):
        return c.import_object('commune.utils.network.upnpc_create_port_map')(port=port)

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
    def gpu_info_map(cls) -> Dict[int, Dict[str, float]]:
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

    @classmethod
    def gpu_total_map(cls) -> Dict[int, Dict[str, float]]:
        import torch
        return {k:v['total'] for k,v in c.gpu_info_map().items()}
    

    @classmethod
    def gpu_total(cls, idx=0, fmt='b') -> Dict[int, Dict[str, float]]:
        import torch
        return c.format_data_size(c.gpu_total_map()[idx])
    
    gpu_map =gpu_info_map
 
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
            
        assert fmt in fmt2scale.keys(), f'fmt must be one of {fmt2scale.keys()}'
        scale = fmt2scale[fmt] 
        x = x/scale 
        
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
    def update_loop(cls, period=2, ):
        while True:
            c.print('Updating...', color='yellow')
            modules = c.servers()
            c.print(f'Modules (n): {modules}', color='cyan')
            c.print(modules, color='purple')
            c.update()
            c.sleep(period)
            
    @classmethod
    def model_shortcuts(cls, **kwargs):
        return  c.module('hf').getc('shortcuts')
    @classmethod
    def resolve_model_shortcut(cls, model):
        model_shortcuts = c.model_shortcuts()
        return model_shortcuts.get(model,model)
            
    @classmethod
    def add_model_shortcut(cls, *args, **kwargs):
        return  c.module('hf').add_model_shortcut(*args, **kwargs)    
    @classmethod
    def rm_model_shortcut(cls, *args, **kwargs):
        return  c.module('hf').rm_model_shortcut(*args, **kwargs)
    
    def add_remote(self, *args, **kwargs):
        return c.module('namespace').add_remote(*args, **kwargs)

    @classmethod
    def model_options(cls):
        return list(c.model_shortcuts().keys())

    @classmethod
    def shortcut2model(cls, shortcut:str):
        return c.model_shortcuts()[shortcut]

    @staticmethod
    def get_trainable_params(model:'nn.Module')->List[str]:
        return c.module('model').get_trainable_params(model)
    @classmethod
    def model_gpu_memory(cls, model:str, num_shard = 2):
        model_size = cls.get_model_size(model)
        size_per_shard = model_size/num_shard
        free_gpu_memory = cls.free_gpu_memory()
        model_gpu_memory = {}
        for i in range(num_shard):
            for gpu_id in c.copy(list(free_gpu_memory.keys())):
                gpu_memory  = free_gpu_memory[gpu_id]
                if gpu_memory > size_per_shard:
                    model_gpu_memory[gpu_id] = size_per_shard 
                    free_gpu_memory.pop(gpu_id)
                    break
        return model_gpu_memory

    @classmethod
    def model_gpus(cls, model, num_shard=2):
        return list(cls.model_gpu_memory(model,num_shard).keys())
        


            

    
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
        model = c.model_shortcuts().get(model, model)

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

    model_size = get_model_size
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
    def logs(cls, *args, **kwargs):
        return cls.pm2_logs(*args, **kwargs)


    @classmethod
    def logmap(cls, *args, **kwargs):
        logmap = {}
        for m in c.servers(*args,**kwargs):
            logmap[m] = c.logs(m)
        return logmap

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
            try:
                return console.print(*text, **kwargs)
            except Exception as e:
                print(e)
                # print(*text, **kwargs)

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
    def test(cls, modules=['server', 'key', 'namespace', 'executor'], verbose:bool=False):
        test_results = []
        for module_name in modules:
            c.print('#'*300)
            c.print(f'[bold cyan]Testing {module_name}[/bold cyan]', color='yellow')

            module = c.module(module_name)
            assert hasattr(module, 'test'), f'Module {module_name} does not have a test function'
            module_test_results = module.test()
            test_results.append(module_test_results)
            c.print(f'Test Results: {module_test_results}', color='white')
        return test_results
        
               
    @classmethod
    def import_bittensor(cls):
        try:
            import bittensor
        except RuntimeError:
            cls.new_event_loop()
            import bittensor
        return bittensor
         

    # TIME LAND
    
    @classmethod  
    def time( cls, t=None) -> float:
        import time
        if t is not None:
            return time.time() - t
        else:
            return time.time()

    @classmethod
    def datetime(cls):
        import datetime
        # UTC 
        return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    @classmethod
    def time2datetime(cls, t:float):
        import datetime
        return datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")
    time2date = time2datetime

    @classmethod
    def datetime2time(cls, x:str):
        import datetime
        c.print(x)
        return datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timestamp()
    date2time =  datetime2time

    @classmethod
    def delta_t(cls, t):
        return t - c.time()
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
        dict_put = c.import_object('commune.utils.dict.dict_put')
        return dict_put(*args, **kwargs)
    @classmethod
    def dict_get(cls, *args, **kwargs):
        dict_get = c.import_object('commune.utils.dict.dict_get')
        return dict_get(*args, **kwargs)
    @classmethod
    def dict_delete(cls, *args, **kwargs):
        dict_delete = c.import_object('commune.utils.dict.dict_delete')
        return dict_delete(*args, **kwargs)
    dict_rm = dict_delete
    @classmethod
    def dict_has(cls, *args, **kwargs):
        dict_has = c.import_object('commune.utils.dict.dict_has')
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
    def str2bytes(cls, data: str, mode: str = 'hex') -> bytes:
        if mode in ['utf-8']:
            return bytes(data, mode)
        elif mode in ['hex']:
            return bytes.fromhex(data)
    
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

    tostr = string = python2str
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
            
    @classmethod
    def restart_server(cls, module:str, **kwargs) -> None:
        return c.serve(module, port=port, **kwargs)
    
    server_restart = restart_server
    
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
    k2a = key2address

    @classmethod
    def is_key(self, key:str) -> bool:
        return c.module('key').is_key(key)

    @classmethod
    def root_key(cls):
        return c.get_key()

    @classmethod
    def address2key(cls,*args, **kwargs ):
        return c.module('key').address2key(*args, **kwargs )
    
    @classmethod
    def get_key_for_address(cls, address:str):
         return c.module('key').get_key_for_address(address)

    # @classmethod
    # def key_info(cls, key:str = None, **kwargs):
    #     return c.module('key').key_info(key, **kwargs)
    
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
            key = module(key, **kwargs)

        return key
    
    
    def idcard(self) -> str:
        seed = str(c.timestamp())
        idcard = self.key.sign(seed)
        return c.python2str(idcard)
    
    def verify_idcard(self, idcard:str = None) -> bool:
        if idcard == None:
            idcard = self.idcard()
        idcard = c.str2bytes(idcard)
        return self.key.verify(idcard)
    

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


    @classmethod
    def encrypt(cls, 
                data: Union[str, bytes],
                key: str = None, 
                prefix = encrypted_prefix) -> bytes:

        key = c.get_key(key)
        path = None
        if c.exists(data):
            path = data
            data =  c.get_text(data)

        data = c.python2str(data)

        c.print(f'Encrypting {data} with {key}', color='cyan')

        encrypted_data = key.encrypt(data)

        if path != None:
            c.put_text(path, encrypted_data)

        return encrypted_data
    

    @classmethod
    def decrypt(cls, 
                data: Union[str, bytes],
                key: str = None, 
                prefix = encrypted_prefix) -> bytes:

        key = c.get_key(key)
        path = None
        if c.exists(data):
            c.print(f'Decrypting from {data} as it exists', color='cyan')
            path = data
            data =  c.get_text(path)

        c.print(data, 'FA', path)   
        data = key.decrypt(data)


        if path != None:
            c.put_text(path, c.python2str(data))

        return data
    
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
    def call(cls,  *args , n: int=1, return_future:bool=False, remote:bool = False,  **kwargs) -> None:
        if n == 1:
            futures = c.async_call(*args,**kwargs)
        else:
            futures = [ c.async_call(*args,**kwargs) for i in range(n)]
        if return_future:
            return futures
        
        results =  c.gather(futures)
        return results
    
    @classmethod
    async def async_call(cls,
                module : str, 
                fn : str = 'info',
                *args,
                timeout : int = 10,
                prefix_match:bool = False,
                network:str = None,
                key:str = None,
                ignore_error = False,
                kwargs = None,
                **extra_kwargs
                ) -> None:

        kwargs = kwargs or {}
        kwargs.update(extra_kwargs)    
        try:
            module = c.connect(module, prefix_match=prefix_match, network=network, virtual=False, key=key)
            future =  module.async_forward(fn=fn, kwargs=kwargs, args=args)
            result = await asyncio.wait_for(future, timeout=timeout)
        except Exception as e:
            if ignore_error:
                result = c.detailed_error(e)
            else:
                raise e
        
        return result

    @classmethod
    def live_modules(cls, **kwargs):
        return cls.call_pool(fn='address', **kwargs)

    @classmethod
    def call_pool(cls, 
                    modules, 
                    fn = 'info',
                    *args, 
                    network =  'local',
                    timeout = 10,
                    n=None,
                    **kwargs):
        
        args = args or []
        kwargs = kwargs or {}
        
        if isinstance(modules, str) or modules == None:
            modules = c.servers(modules, network=network)
        if n == None:
            n = len(modules)
        modules = cls.shuffle(modules)[:n]
        assert isinstance(modules, list), 'modules must be a list'
        c.print(f'[bold cyan]Calling {fn} on {len(modules)} modules [/bold cyan]', color='yellow')
        jobs = []
        
        for m in modules:
            job_kwargs = {'module':  m, 'fn': fn, **kwargs}
            job = c.submit(c.call, kwargs=kwargs, args=[m, fn, *args] , timeout=timeout, return_future=True)
            jobs.append(job)
        responses = c.wait(jobs, timeout=timeout)
        return responses
    
    @classmethod
    def resolve_fn(cls,fn, init_kwargs=None ):
        if isinstance(fn, str):
            if '.' in fn:
                module = '.'.join(fn.split('.')[:-1])
                module = c.module(module)
            else:
                module = c.module(fn)
            fn = fn.split('.')[-1]
            fn_obj = getattr(module, fn)
            method_type = c.classify_method(fn_obj)
            if method_type == 'self':
                if init_kwargs is None:
                    init_kwargs = {}
                module = module(**init_kwargs)
            fn = getattr(module, fn)
        
        assert callable(fn), f'{fn} is not callable'
        return fn
    
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
    def keys(cls, search = None, *args, **kwargs):
        if search == None:
            search = cls.module_path()
            if search == 'module':
                search = None
        return c.module('key').keys(search, *args, **kwargs)

    @classmethod  
    def get_mem(cls, *args, **kwargs):
        return c.module('key').get_mem(*args, **kwargs)
    


    
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
    
    @classmethod
    def loadmems(self, *args, **kwargs):
        return c.module('key').loadmems(*args, **kwargs)

    def savemems(self, *args, **kwargs):
        return c.module('key').savemems(*args, **kwargs)
    
    
    @classmethod
    def save_keys(cls, *args,  **kwargs):
        c.print('saving keys')
        return c.module('key').save_keys(*args, **kwargs)

    @classmethod
    def load_keys(cls, *args,  **kwargs):
        return c.module('key').load_keys(*args, **kwargs)
    
    @classmethod
    def load_key(cls, *args,  **kwargs):
        return c.module('key').load_key(*args, **kwargs)


    def sign(self, data:dict  = None, key: str = None, **kwargs) -> bool:
        key = self.resolve_key(key)
        signature =  key.sign(data, **kwargs)
        return signature
    

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
    
    @classmethod
    def start(cls, *args, **kwargs):
        return cls(*args, **kwargs)
    

    @classmethod
    def is_encrypted(cls, data, prefix=encrypted_prefix):
        if isinstance(data, str):
            if data.startswith(prefix):
                return True
        elif isinstance(data, dict):
            return bool(data.get('encrypted', False) == True)
        else:
            return False
        
        
    
    @classmethod
    def rm_user(cls, user: str = None):
        self.users.pop(user, None)  
        
    
    
    @classmethod
    def network(cls) -> str:
        return c.resolve_network()
    
    
    net = network
    
    @classmethod
    def networks(cls, *args, **kwargs) -> List[str]:
        return c.module('namespace').networks( *args, **kwargs)

    @classmethod
    def network2namespace(self, *args, **kwargs) -> str:
        return c.module("namespace").network2namespace(*args, **kwargs)
    all = network2namespace
    
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
        cls.put(var_path, reserved_ports, root=root)
        return cls.reserved_ports()
    
    
    unresports = unreserve_ports
    @classmethod
    def fleet(cls,n=2, tag=None, max_workers=10, parallel=False, timeout=20,  **kwargs):

        c.update()
        if tag == None:
            tag = ''

        if parallel:
            executor = c.module('executor')(max_workers=max_workers, mode='thread')
            futures = []
            for i in range(n):
                server_kwargs={'tag':tag + str(i), **kwargs}
                future = executor.submit(fn=cls.serve, kwargs=server_kwargs, timeout=timeout, return_future=True)
                futures = futures + [future]
            
            results =  c.wait(futures, timeout=timeout)
            for result in results:
                c.register_server(name=result['name'], address=result['address'])

        else:
            results = []
            for i in range(n):
                c.print(f'Launching {tag}')
                server_kwargs={'tag':tag + str(i), **kwargs}
                result = cls.serve(**server_kwargs)
                results = results + [result]

        return results
        
                
        

    @classmethod
    def kill_fleet(cls, tag=None, network='local', **kwargs):

        path = cls.resolve_server_name(tag=tag)
        servers = c.servers(path, network=network)
        executor = c.module('executor')(mode='process')
        for server in servers:
            futures += [executor.submit(fn=cls.kill_server, kwargs={'server_name':p, 'network':network})]

        return c.wait(futures)

    @classmethod
    def executor(cls, max_workers:int=None, mode:str="thread", **kwargs):
        return c.module(f'executor').executor(max_workers=max_workers, mode=mode,  **kwargs)

    @classmethod
    def submit(cls, 
                fn, 
                args:list = [], 
                kwargs: dict = {}, 
                timeout:int = 20, 
                return_future:bool=False,
                init_args : list = [],
                init_kwargs:dict= {},
                executor = None,
                module: str = None,
                mode:str='thread',
                max_workers : int = None,
                ):


        fn = c.get_fn(fn)
        executor = c.executor(max_workers=max_workers, mode=mode) if executor == None else executor
        args = c.copy(args)
        kwargs = c.copy(kwargs)
        init_kwargs = c.copy(init_kwargs)
        init_args = c.copy(init_args)
        if module == None:
            module = cls
        else:
            module = c.module(module)
        if isinstance(fn, str):
            method_type = c.classify_method(getattr(module, fn))
        elif callable(fn):
            method_type = c.classify_method(fn)
        else:
            raise ValueError('fn must be a string or a callable')
        
        if method_type == 'self':
            module = module(*init_args, **init_kwargs)

        future = executor.submit(fn=fn, args=args, kwargs=kwargs, timeout=timeout)
        
        if return_future:
            return future
        else:
            return c.wait(future, timeout=timeout)

    @classmethod
    def submit_batch(cls,  fn:str, batch_kwargs: List[Dict[str, Any]], return_future:bool=False, timeout:int=10, module = None,  *args, **kwargs):
        n = len(batch_kwargs)
        module = cls if module == None else module
        executor = c.executor(max_workers=n)
        futures = [ executor.submit(fn=getattr(module, fn), kwargs=batch_kwargs[i], timeout=timeout) for i in range(n)]
        if return_future:
            return futures
        return c.wait(futures)

    @classmethod
    def regfleet(cls,module = None, tag:str=None, n:int=2, timeout=40 , stake=None, multithread:bool=False, **kwargs):
        subspace = c.module('subspace')()
        if tag == None:
            tag = ''
        if module == None:
            module = cls.module_path()
        server_names = []
        if stake == None:
            stake = subspace.min_stake()
            c.print('No stake provided, using min stake, which is {}'.format(stake), color='yellow')
        if multithread:
            executor = c.module('executor')(max_workers=n)
            futures = []
            for i in range(n):
                server_name = module +"::" + tag + str(i)
                if c.is_registered(server_name):
                    c.print(f'Server {server_name} already exists, skipping', color='yellow')
                    continue
                future = executor.submit(fn=cls.register,  kwargs={'module':module, 'tag':tag+str(i), 'stake': stake,  **kwargs}, timeout=timeout)
                futures = futures + [future]
            return c.wait(futures, timeout=timeout)
        else:
            for i in range(n):
                
                try:
                    server_name = module +"::" + tag + str(i)

                    if c.is_registered(server_name):
                        c.print(f'Server {server_name} already exists, skipping', color='yellow')
                        continue
                    r = cls.register(module=module, tag=tag+str(i), stake=stake,  **kwargs)
                except Exception as e:
                    c.print(e)
                    r = {'success':False, 'error':c.detailed_error(e)}
                c.print(r)
                server_names.append(r)
            return {'servers':server_names}

    @classmethod
    def servefleet(cls,module = None, tag:str=None, n:int=2, refresh=False, **kwargs):
        subspace = c.module('subspace')()
        if tag == None:
            tag = ''
        server_names = []
        for i in range(n):
            r = cls.serve(module=module, tag=tag+str(i), refresh=refresh,  **kwargs)
            server_names.append(r)
        return {'servers':server_names}
    
    @classmethod
    def client(cls, *args, **kwargs) -> 'Client':
        return c.module('module.client')(*args, **kwargs)
    
    @classmethod
    def serialize(cls, x, **kwargs):
        serializer = c.serializer()
        return serializer.serialize(x, **kwargs)

    @classmethod
    def serializer(cls, *args, **kwargs):
        return  c.module('serializer')(*args, **kwargs)
    
    @classmethod
    def deserialize(cls, x, **kwargs):
        return c.serializer().deserialize(x, **kwargs)

    @classmethod
    def proto2json(cls, data):
        from google.protobuf.json_format import MessageToJson
        return MessageToJson(data)

    @classmethod
    def process(cls, *args, **kwargs):
        return c.module('process').process(*args, **kwargs)

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
        return c.import_object('commune.launchpad.Launchpad')()
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
    def check_module(cls, module:str):
        return c.connect(module)
    
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
    def is_error(cls, x:dict):
        return not cls.is_success(x)


    @staticmethod
    def is_number(value):
        try:
            int(value)
        except Exception:
            return False
        return True
    
    @classmethod
    def resolve_network(cls, network=None):

        network_shortcuts = {
            'r': 'remote',
            'l': 'local',
            'g': 'global',
            's': 'subspace',
            'bt': 'bittensor',
            'auto': 'autonolous',
        }

        network = network_shortcuts.get(network, network)
        
        if network == None:
            network = cls.get_network()

        return network

    get_network = resolve_network
    @classmethod
    def set_network(cls, network:str):
        old_network = c.network()
        network = c.resolve_network(network)
        c.put('network', network)

        return {'success': True, 'msg': f'from {old_network} -> {network}'}
    
    setnet = set_network

    @classmethod
    def switch_network(cls):
        network = cls.get_network()
        if network == 'subspace':
            network = 'local'
        else:
            network = 'subspace'
        return cls.set_network(network)

    switchnet = switch_network
    
    @classmethod
    def get_network(self):
        return c.get('network', self.default_network)

    getnet = get_network
    resnet = resolve_network
    
    @classmethod
    def update(cls, 
               network: str = None,
               ):
        
        # update local namespace
        c.ip(update=True)
        c.namespace(network=network, update=True)
        servers = c.servers(network=network)
        c.server_infos(update=True, network='local')

        

        return {'success': True, 'servers': servers}

    @classmethod
    def sync(cls, *args, **kwargs):
            
        return c.module('subspace')().sync(*args, **kwargs)
        

    @classmethod
    def run_jobs(cls, jobs: List, mode ='asyncio',**kwargs):
        if mode == 'asyncio':
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(asyncio.gather(*jobs))
            return results
        else:
            raise ValueError(f"Invalid mode: {mode}")
        

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
    def put_text(cls, path:str, text:str, root=False, key=None) -> None:
        # Get the absolute path of the file
        path = cls.resolve_path(path, root=root)
        if key != None:
            text = c.get_key(key).encrypt(text)
        # Write the text to the file
        with open(path, 'w') as file:
            file.write(text)

        # get size
        text_size = len(text)*8
    

        return {'success': True, 'msg': f'Wrote text to {path}', 'size': text_size}
            
            
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
                 tail = None,
                 start_byte:int = 0,
                 end_byte:int = 0,
                 start_line :int= None,
                 end_line:int = None ) -> str:
        # Get the absolute path of the file
        path = cls.resolve_path(path)

        # Read the contents of the file
        with open(path, 'rb') as file:

            file.seek(0, 2) # this is done to get the fiel size
            file_size = file.tell()  # Get the file size
            if start_byte < 0:
                start_byte = file_size - start_byte
            if end_byte <= 0:
                end_byte = file_size - end_byte 

            if end_byte < start_byte:
                end_byte = start_byte + 100
            chunk_size = end_byte - start_byte + 1

            file.seek(start_byte)

            content_bytes = file.read(chunk_size)

            # Convert the bytes to a string
            try:
                content = content_bytes.decode()
            except UnicodeDecodeError as e:
                if hasattr(content_bytes, 'hex'):
                    content = content_bytes.hex()
                else:
                    raise e

            if tail != None:
                content = content.split('\n')
                content = '\n'.join(content[-tail:])
    
            elif start_line != None or end_line != None:
                
                content = content.split('\n')
                if end_line == None or end_line == 0 :
                    end_line = len(content) 
                if start_line == None:
                    start_line = 0
                if start_line < 0:
                    start_line = start_line + len(content)
                if end_line < 0 :
                    end_line = end_line + len(content)
                content = '\n'.join(content[start_line:end_line])
            else:
                content = content_bytes.decode()
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
    def mkdir( cls, path = 'bro', exist_ok:bool = True):
        """ Makes directories for path.
        """
        path = cls.resolve_path(path)
        if os.path.exists(path):
            return  {'success': True, 'msg': f'Directory {path} already exists'}
        os.makedirs( path , exist_ok=exist_ok) 
        assert os.path.exists(path), f'Failed to create directory {path}'
        return  {'success': True, 'msg': f'Created directory {path}'}

    @staticmethod
    def repo2module( repo, module = None):
        if module == None:
            module = os.path.basename(repo).replace('.git','').replace(' ','_').replace('-','_').lower()
        
        c.new_module(module=module, repo=repo)
        return {'module':module, 'repo':repo, 'status':'success'}
        
    

    def new_modules(self, *modules, **kwargs):
        for module in modules:
            self.new_module(module=module, **kwargs)
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
        if c.has_module(module) and overwrite==False:
            return {'success': False, 'msg': f' module {module} already exists, set overwrite=True to overwrite'}
        module_path = os.path.join(c.modules_path, module)
        
        if overwrite and c.module_exists(module_path): 
            c.rm(module_path)
        
        if repo != None:
            # Clone the repository
            c.cmd(f'git clone {repo} {module_path}')
            # Remove the .git directory
            c.cmd(f'rm -rf {module_path}/.git')



        # Create the module name if it doesn't exist, infer it from the repo name 
        if module == None:
            assert repo != None, 'repo must be specified if module is not specified'
            module = os.path.basename(repo).replace('.git','').replace(' ','_').replace('-','_').lower()
        
        # currently we are using the directory name as the module name
        if module_type == 'dir':
            c.mkdir(module_path, exist_ok=True)
        else:
            raise ValueError(f'Invalid module_type: {module_type}, options are dir, file')
        

        base_module = c.module(base)
        base_code = base_module.code()
        base_config = base_module.config()
        module = module.replace('/','_') # replace / with _ for the class name
        
        # define the module code and config paths
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

        return {'success': True, 'msg': f' created a new repo called {module}'}
        
    make_dir= mkdir

    @classmethod
    def filepath2text(cls, path:str = None):
        if path == None:
            path = c.root_path
        filepath2text = {}
        for filepath in c.glob(path):
            filepath2text[filepath] = c.get_text(filepath)
        return filepath2text
        

    @classmethod
    def model_max_gpu_memory(cls, model, *args, **kwargs):
        model_size = c.get_model_size(model)
        return c.max_gpu_memory(model_size,  *args, **kwargs)

    @classmethod
    def model_max_gpus(cls, model, *args, **kwargs):
        return list(c.model_max_gpu_memory(model,  *args, **kwargs).keys())

    infer_gpus = model_max_gpus


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
            
            if gpu in max_memory:
                continue
            
            if gpu_memory < min_memory:
                continue
                
  
            allocated_memory = min(gpu_memory, unallocated_memory)
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
    def resolve_module(cls, module=None):
        if module == None:
            module = cls
        if isinstance(module, str):
            module = c.module(module)
        return module


    thread_map = {}

    @classmethod
    def get_fn(cls, fn:str, seperator='.'):
        if isinstance(fn, str):
            if seperator in fn:
                # module{sperator}fn
                fn_splits = fn.split(seperator)
                # incase you have multiple seperators in the  name
                module = seperator.join(fn_splits[:-1])
                fn = fn_splits[-1]
                # get the model
                module = c.module(module)
            else:
                module = cls
            # get the module function
            if hasattr(module, fn):
                fn = getattr(module, fn)
            else:
                return None
        # assert callable(fn), 'Is not callable'
        return fn
    

            
            
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
                    locals = None,
                    name : str =None,
                    tag: str = None,
                    refresh : bool =True,
                    tag_seperator : str = '::',):

        if locals != None:
            kwargs = c.locals2kwargs(locals)
        
        if len(fn.split('.'))>1:
            module = '.'.join(fn.split('.')[:-1])
            fn = fn.split('.')[-1]
            
        kwargs = kwargs if kwargs else {}
        args = args if args else []
        
        

        if name == None:
            module_path = cls.resolve_module(module).module_path()
            name = f"{module_path}{tag_seperator}{fn}"
            if tag != None:
                name = f'{name}{tag_seperator}{tag}'

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
        if len(options) == 0:
            return None
        if isinstance(options, dict):
            options = list(options.values())

        assert isinstance(options, list),'options must be a list'
        return random.choice(options)

    @classmethod
    def sample(cls, options:list, n=2):
        if isinstance(options, int):
            options = list(range(options))
        options = c.shuffle(options)
        return options[:n]
        

    @classmethod
    def chown(cls, path:str = None, sudo:bool =True):
        path = cls.resolve_path(path)
        user = c.env('USER')
        cmd = f'chown -R {user}:{user} {path}'
        c.cmd(cmd , sudo=sudo, verbose=True)
        return {'success':True, 'message':f'chown cache {path}'}

    @classmethod
    def chown_cache(cls, sudo:bool = True):
        return c.chown(c.cache_path(), sudo=sudo)
        
    @classmethod
    def colors(cls):
        return ['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white', 'bright_black', 'bright_red', 'bright_green', 'bright_yellow', 'bright_blue', 'bright_magenta', 'bright_cyan', 'bright_white']
    colours = colors
    @classmethod
    def random_color(cls):
        import random
        return random.choice(cls.colors())
    randcolor = randcolour = colour = color = random_colour = random_color

    @classmethod
    def random_float(cls, min=0, max=1):
        import random
        return random.uniform(min, max)


    @classmethod
    def random_ratio_selection(cls, x:list, ratio:float = 0.5)->list:
        
        
        import random
        if type(x) in [float, int]:
            x = list(range(int(x)))
        assert len(x)>0
        if ratio == 1:
            return x
        assert ratio > 0 and ratio <= 1
        random.shuffle(x)
        k = max(int(len(x) * ratio),1)
        return x[:k]

    default_tag = 'base'
    @property
    def tag(self):
        tag = None
        if not hasattr(self, 'config') or not isinstance(self.config, dict):
            self.config = c.dict2munch({})
        if 'tag' in self.config:
            tag = self.config['tag']
        return tag
    @tag.setter
    def tag(self, value):
        if not hasattr(self, 'config') or not isinstance(self.config, dict):
            self.config = c.dict2munch({})
        self.config['tag'] = value
        return value
        
    
    @classmethod
    def tags(cls):
        return ['alice', 'bob', 'chris', 'dan', 'fam', 'greg', 'elon', 'huck']
    
    @classmethod
    def rand_tag(cls):
        return cls.choice(cls.tags())

    @classmethod
    def as_completed(cls , futures:list, timeout:int=10, **kwargs):
        return concurrent.futures.as_completed(futures, timeout=timeout)
    @staticmethod
    def wait(futures:list, timeout:int = None, generator:bool=False, return_dict:bool = True) -> list:
        
        import concurrent.futures
        futures = [futures] if not isinstance(futures, list) else futures
        future2idx = {future:i for i,future in enumerate(futures)}

        results = []

        # wait for the futures as they complete

        results = []
        results = [None]*len(futures)

        if timeout == None and hasattr(futures[0], 'timeout'):
            timeout = futures[0].timeout


        if generator:
            def get_results():
                for future in concurrent.futures.as_completed(futures, timeout=timeout):
                    if return_dict:
                        idx = future2idx[future]
                        yield {'idx': idx, 'result': future.result()}
                    else:
                        yield future.result()
        else:
            def get_results():
                try:
                    for future in concurrent.futures.as_completed(futures, timeout=timeout):
                        idx = future2idx[future]
                        results[idx] = future.result()
                except Exception as e:
                    print(e)
                return results
            
        return get_results()

    @staticmethod
    def as_completed( futures, timeout=10, **kwargs):
        import concurrent.futures
        return concurrent.futures.as_completed(futures, timeout=timeout, **kwargs)

    
    @classmethod
    def gather(cls,jobs:list, mode='asyncio', loop=None, timeout = 20)-> list:
        if not isinstance(jobs, list):
            singleton = True
            jobs = [jobs]
        else:
            singleton = False
        assert isinstance(jobs, list)
        if mode == 'asyncio':
            if loop == None:
                loop = c.get_event_loop()
            results = loop.run_until_complete(asyncio.wait_for(asyncio.gather(*jobs), timeout=timeout))
        else:
            raise NotImplementedError

        if singleton:
            return results[0]
        return results

    @classmethod
    def split_gather(cls,jobs:list, n=3,  **kwargs)-> list:
        if len(jobs) < n:
            return c.gather(jobs, **kwargs)
        gather_jobs = [asyncio.gather(*job_chunk) for job_chunk in c.chunk(jobs, num_chunks=n)]
        gather_results = c.gather(gather_jobs, **kwargs)
        results = []
        for gather_result in gather_results:
            results += gather_result
        
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
        path1 = cls.resolve_path(path1)
        path2 = cls.resolve_path(path2)
        assert os.path.exists(path1), path1
        if not os.path.isdir(path2):
            path2_dirpath = os.path.dirname(path2)
            if not os.path.isdir(path2_dirpath):
                os.makedirs(path2_dirpath, exist_ok=True)
        shutil.move(path1, path2)
        return path2

        
        
    @classmethod
    def cp(cls, path1:str, path2:str, refresh:bool = False):
        import shutil
        # what if its a folder?
        assert os.path.exists(path1), path1
        if refresh == False:
            assert not os.path.exists(path2), path2
        
        path2_dirpath = os.path.dirname(path2)
        if not os.path.isdir(path2_dirpath):
            os.makedirs(path2_dirpath, exist_ok=True)
            assert os.path.isdir(path2_dirpath), f'Failed to create directory {path2_dirpath}'

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
        return c.module('model.hf').learn(*args, **kwargs)
        
    @classmethod
    def mine(cls,*args, **kwargs):
        kwargs['remote'] = kwargs.get('remote', True)
        return c.module('bittensor').mine(*args, **kwargs)
    
    @classmethod
    def train_fleet(cls, *args, **kwargs):
        kwargs['remote'] = kwargs.get('remote', True)
        return c.module('model.hf').train_fleet(*args, **kwargs)
    
    @classmethod
    def miners(cls, *args, **kwargs):
        return c.module('bittensor').miners(*args, **kwargs)
    
    @classmethod
    def check_miners(cls, *args, module='bittensor', **kwargs):
        return c.module(module).check_miners( *args, **kwargs)
    
    
    @classmethod
    def shuffle(cls, x:list)->list:
        import random
        if len(x) == 0:
            return x
        random.shuffle(x)
        return x
    
    @classmethod
    def pull(cls, stash:bool = False, cwd=None):
        return c.module('git').pull(stash=stash, cwd=cwd)

    @classmethod
    def rpull(cls, stash:bool = False, cwd=None):
        return c.module('remote').pull(stash=stash, cwd=cwd)

    @classmethod
    def push(cls, cwd=None):
        return c.module('git').push(cwd=cwd)


    @classmethod
    def push(cls, msg='update', cwd=None):
        return c.module('git').push(msg=msg, cwd=cwd)

    # @classmethod
    # def status(cls,  cwd=None):
    #     return c.module('git').status(cwd=cwd)

    @classmethod
    def make_pull(cls):
        return cls.cmd('make pull')

    @staticmethod
    def retry(fn, trials:int = 3, verbose:bool = True): 
        def wrapper(*args, **kwargs):
            for i in range(trials):
                try:
                    c.print(fn)
                    return fn(*args, **kwargs)
                except Exception as e:
                    if verbose:
                        c.print(c.detailed_error(e), color='red')
                        c.print(f'Retrying {fn.__name__} {i+1}/{trials}', color='red')

        return wrapper
    
    
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
    def ssh_authorized_keys(cls, authorized_keys_file:str='~/.ssh/authorized_keys'):
        authorized_keys_file = os.path.expanduser(authorized_keys_file)
        return cls.get_text(authorized_keys_file)

    @staticmethod
    def get_public_key_from_file(public_key_file='~/.ssh/id_rsa.pub'):
        public_key_file = os.path.expanduser(public_key_file)
        os.path.exists(public_key_file), f'public key file {public_key_file} does not exist'
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
    def reverse_map(x:dict)->dict:
        '''
        reverse a dictionary
        '''
        return {v:k for k,v in x.items()}

    @classmethod
    def pd(cls):
        '''
        import pandas
        '''
        return cls.import_module('pandas')

    @classmethod
    def df(cls, *args, **kwargs):
        df =  c.import_object('pandas.DataFrame')
        if len(args) > 0 or len(kwargs) > 0:
            df = df(*args, **kwargs)
        return df

    @classmethod
    def torch(cls):
        return cls.import_module('torch')

    @classmethod
    def tensor(cls, *args, **kwargs):
        return c.import_object('torch.tensor')(*args, **kwargs)

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
    def fn2hash(cls, *args, mode='sha256', **kwargs):
        fn2hash = {}
        for k,v in cls.fn2str(*args, **kwargs).items():
            fn2hash[k] = c.hash(v,mode=mode)
        return fn2hash
    
        
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
            tag = self.tag
        if tag == None:
            tag = default_tag
        assert tag != None
        return tag
    def resolve_tag_path(self, tag=None): 
        tag = self.resolve_tag(tag)
        return self.resolve_path(tag)
    
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
    def pool(cls , n=5, **kwargs):
        for i in range(n):
            cls.serve(tag=str(i), **kwargs)
        

    def self_methods(self):
        return c.get_self_methods(self)

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
    def get_function_args(cls, fn):
        fn = cls.get_fn(fn)
        args = inspect.getfullargspec(fn).args
        return args
    

    
    fn_args = get_fn_args =  get_function_args
    
    @classmethod
    def classify_method(cls, fn):
        fn = cls.get_fn(fn)
        args = cls.get_function_args(fn)
        if len(args) == 0:
            return 'static'
        elif args[0] == 'self':
            return 'self'
        else:
            return 'class'
    
    @classmethod
    def build(cls, *args, **kwargs): 
        return c.module('docker').build(*args, **kwargs)
    build_image = build
    @classmethod
    def has_gpus(cls): 
        return bool(len(c.gpus())>0)
    
    @classmethod
    def up(cls): 
        docker = c.module('docker')
        path = docker.get_compose_path('commune')
        compose = docker.get_compose(path)

        # create temporary compose file to toggle gpu options
        if not c.has_gpus():
            del compose['services']['commune']['deploy']
        tmp_path = path.replace('docker-compose', 'docker-compose-tmp')
        c.save_yaml(tmp_path, compose)

        docker.compose(tmp_path, compose = compose)
        c.rm(tmp_path)
        # return c.compose('commune')

    @classmethod
    def compose(cls, *args, **kwargs):
        return c.module('docker').compose(*args, **kwargs)


    @classmethod
    def ps(cls, *args, **kwargs):
        return c.module('docker').ps(*args, **kwargs)
 

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
        '''
        is the function a property
        '''
        fn = cls.get_fn(fn)

        return isinstance(fn, property)


    @classmethod
    def property_fns(cls) -> bool:
        '''
        Get a list of property functions in a class
        '''
        return [fn for fn in dir(cls) if cls.is_property(fn)]

    @classmethod
    def get_functions(cls, obj: Any = None,
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
    
        functions = []
        parent_functions = [] 

        if include_parents:
            dir_list = dir(obj)
        else:
            # this only has atrributes for the child class
            dir_list = obj.__dict__.keys()

        for fn_name in dir_list:
            fn_obj = getattr(obj, fn_name)
            if not callable(fn_obj):
                continue
            
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
    def get_class_methods(cls: Union[str, type], obj=None):
        '''
        Gets the self methods in a class
        '''
        obj = obj or cls
        functions =  c.get_functions(obj)
        signature_map = {f:cls.get_function_args(getattr(obj, f)) for f in functions}
        return [k for k, v in signature_map.items() if 'cls' in v]
    
    class_methods = class_fns = get_class_methods

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
    def fn_defaults(cls, fn):

        """
        Gets the function defaults
        """

        import inspect
        
        fn = cls.get_fn(fn)
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

    @classmethod
    def has_fn(cls,fn_name, obj = None):
        if obj == None:
            obj = cls
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
        return json.loads(json_string.replace("'", '"'))
    
    @classmethod
    def bro(cls, x):
        return x
    
    
    @classmethod
    def giturl(cls):
        return c.cmd('git remote -v', verbose=False).split('\n')[0].split('\t')[1].split(' ')[0]
    url = giturl

    @classmethod
    def my_modules(cls, *args, **kwargs):
        return c.module('subspace')().my_modules(*args, **kwargs)
    @classmethod
    def my_stake(cls, *args, **kwargs):
        return c.module('subspace')().my_stake(*args, **kwargs)

    @classmethod
    def my_staketo(cls, *args, **kwargs):
        return c.module('subspace')().my_staketo(*args, **kwargs)

    @classmethod
    def my_stakefrom(cls, *args, **kwargs):
        return c.module('subspace')().my_stakefrom(*args, **kwargs)

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
    def filesize(cls, filepath:str):
        filepath = cls.resolve_path(filepath)
        return os.path.getsize(filepath)
    
    @classmethod
    def code(cls, module = None, *args, **kwargs):
        module = cls.resolve_module(module)
        path = module.pypath()
        text =  c.get_text( module.pypath(), *args, **kwargs)
        return text
        

    @classmethod
    def get_text_line(cls, module = None, *args, **kwargs):
        module = cls.resolve_module(module)
        return c.get_text_line( module.pypath(), *args, **kwargs)
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
    def find_code_line(cls, search:str, code:str = None):
        if code == None:
            code = cls.code() # get the code
        found_lines = [] # list of found lines
        for i, line in enumerate(code.split('\n')):
            if search in line:
                found_lines.append({'idx': i, 'text': line})
        if len(found_lines) == 0:
            return None
        elif len(found_lines) == 1:
            return found_lines[0]['idx']
        return found_lines
    
    @classmethod
    def fn_info(cls, fn) -> dict:
        r = {}
        code = cls.fn_code(fn)
        lines = code.split('\n')
        start_line = cls.find_code_line(lines[0])
        end_line = start_line + len(lines)
        has_docs = bool('"""' in code or "'''" in code)
        filepath = cls.filepath()

        return {
            'start_line': start_line,
            'end_line': end_line,
            'has_docs': has_docs,
            'code': code,
            'n_lines': len(lines),
            'hash': c.hash(code),
            'path': filepath
        }



        return r


    @classmethod
    def get_code_line(cls, idx:int = 0, code:str = None ):
        if code == None:
            code = cls.code() # get the code
        lines = code.split('\n')
        assert idx < len(lines), f'idx {idx} is out of range for {len(lines)}'
        return lines[idx]

    
    
    def ensure_self_attr(self, attr, default=None):
        if not hasattr(self, attr):
            setattr(self, attr, default)
    @classmethod
    def ensure_class_attr(cls, attr, default=None):
        if not hasattr(cls, attr):
            setattr(cls, attr, default)

    tokenizer_cache = {}
    @classmethod
    def tokenizer(cls, tokenizer='gpt2', cache = True,  **kwargs):
        if cache and tokenizer in cls.tokenizer_cache:
            return cls.tokenizer_cache[tokenizer]
        from transformers import AutoTokenizer
        tokenizer_obj =  AutoTokenizer.from_pretrained(tokenizer,**kwargs)
        if cache:
            cls.tokenizer_cache[tokenizer] = tokenizer_obj
        return tokenizer_obj
        
    @classmethod
    def tokenize(cls, text, tokenizer='gpt2', *args, **kwargs):
        return cls.tokenizer(tokenizer, *args, **kwargs).encode(text)
    @classmethod
    def detokenize(cls, tokens, tokenizer='gpt2', *args, **kwargs):
        return cls.tokenizer(tokenizer, *args, **kwargs).decode(tokens)

    @classmethod
    def num_tokens(cls, text, **kwargs):
        return len(cls.tokenize(text, **kwargs))

    
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
    def is_registered(cls, *args, **kwargs):
        return c.module('subspace')().is_registered(*args, **kwargs)
    
    @classmethod
    def transfer(cls, *args, **kwargs):
        return c.module('subspace')().transfer(*args, **kwargs)

    send = transfer

    @classmethod
    def block(self, *args, **kwargs):
        return c.module('subspace')().block

    @classmethod
    def total_supply(self, *args, **kwargs):
        return c.module('subspace')().total_supply(*args, **kwargs)

    @classmethod
    def update_module(cls, *args, **kwargs):
        return c.module('subspace')().update_module(*args, **kwargs)

    def update_servers(self, *args, **kwargs):
        subspace = c.module('subspace')()
        for name, address in c.namespace(network='localf').items():
            subspace.update_module(name, address)
        return subspacec 
    
    @classmethod
    def vote(cls, *args, **kwargs):
        return c.module('subspace')().vote(*args, **kwargs)
    
    @classmethod
    def stake(cls, *args, **kwargs):
        return c.module('subspace')().stake(*args, **kwargs)
    

    @classmethod
    def multistake(cls, *args, **kwargs):
        return c.module('subspace')().multistake(*args, **kwargs)

    @classmethod
    def random_word(cls, *args, n=2, seperator='_', **kwargs):
        return seperator.join(c.module('key').generate_mnemonic(*args, **kwargs).split(' ')[:n])

    @classmethod
    def remove_number_from_word(cls, word:str) -> str:
        while word[-1].isdigit():
            word = word[:-1]
        return word

    @classmethod
    def multiunstake(cls, *args, **kwargs):
        return c.module('subspace')().multiunstake(*args, **kwargs)

    @classmethod
    def repo_url(cls, *args, **kwargs):
        return c.module('git').repo_url(*args, **kwargs)    

    @classmethod
    def get_stake(cls, *args, **kwargs):
        return c.module('subspace')().get_stake(*args, **kwargs)
    
    @classmethod
    def get_staketo(cls, *args, **kwargs):
        return c.module('subspace')().get_staketo(*args, **kwargs)
    
    @classmethod
    def get_stakefrom(cls, *args, **kwargs):
        return c.module('subspace')().get_stakefrom(*args, **kwargs)
    
    @classmethod
    def stake_multiple(cls, *args, **kwargs):
        return c.module('subspace')().stake_multiple(*args, **kwargs)

    @classmethod
    def stake_spread(cls, *args, **kwargs):
        return c.module('subspace')().stake_spread(*args, **kwargs)
    
    @classmethod
    def snap(cls, *args, **kwargs):
        return c.module('subspace')().build_snapshot(*args, **kwargs)   

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
        return c.module('subspace')().my_keys(*args, **kwargs)
    wallets = my_keys

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
    def key_info(cls, *args, **kwargs):
        return c.module('key').key_info(*args, **kwargs)

    @classmethod
    def key2mem(cls, *args, **kwargs):
        return c.module('key').key2mem(*args, **kwargs)
    @classmethod
    def key_info_map(cls, *args, **kwargs):
        return c.module('key').key_info_map(*args, **kwargs)
    
    @property
    def key(self):
        if not hasattr(self, '_key'):
            c.print(self.server_name, 'FAM')
            self._key = c.get_key(self.server_name, create_if_not_exists=True)
        return self._key

    @staticmethod
    def is_valid_ss58_address(address:str):
        return c.module('key').is_valid_ss58_address(address)

    @key.setter
    def key(self, key):
        self._key = c.get_key(key, create_if_not_exists=True)
        return self._key

    @classmethod
    def node_keys(cls, *args, **kwargs):
        return c.module('subspace').node_keys(*args, **kwargs)

    @classmethod
    def add_node(cls, *args, **kwargs):
        return c.module('subspace').add_node(*args, **kwargs)

    @classmethod
    def add_node_key(cls, *args, **kwargs):
        return c.module('subspace').add_node_key(*args, **kwargs)
    

    @classmethod   
    def infer_device_map(cls, 
                         model:str, 
                         max_memory: dict = None,
                         block_prefix : str = 'model.layers',
                         buffer_memory:float = '1gb', # 10GB buffer (bytes)
                         quantize:str = None, #
                         verbose: bool = False,
                         **kwargs,
                         ):
        if quantize in ['int8']: 
            quantize_factor = 0.5
        elif quantize in ['int4']:
            quantize_factor = 0.25
        elif quantize == None: 
            quantize_factor = 1
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
                allocated_gpu_memory[gpu] = 0
            
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
    def upgrade(cls, lib):
        c.cmd(f'pip install --upgrade {lib}', verbose=True)
        
    @classmethod
    def fix_proto(cls):
        cls.upgrade_proto()
        cls.build_proto()

    # SUBSPACE LAND
    @classmethod
    def register_servers(cls, *args, **kwargs):
        return c.module('subspace')().register_servers(*args, **kwargs)
    reg_servers = register_servers

    @classmethod
    def registered_servers(cls, *args, **kwargs):
        return c.module('subspace')().registered_servers(*args, **kwargs)
    reged_servers = registered_servers    
        
    @classmethod
    def unregistered_servers(cls, *args, **kwargs):
        return c.module('subspace')().unregistered_servers(*args, **kwargs)
    unreged_servers = unregistered_servers
        
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
    def update_global(cls, *args, **kwargs):
        return c.module('subspace')().update_global(*args, **kwargs)
    

    @classmethod
    def market_cap(cls, *args, **kwargs):
        return c.module('subspace')().market_cap(*args, **kwargs)
    mcap = market_cap
    @classmethod
    def n(cls, *args, **kwargs):
        return c.module('subspace')().n(*args, **wkwargs)
    @classmethod
    def stats(cls, *args, **kwargs):
        t = c.timer()
        return c.module('subspace')().stats(*args, **kwargs)

    @classmethod
    def vstats(cls, *args, **kwargs):
        return c.module('vali').all_stats(*args, **kwargs)
    @classmethod
    def valis(cls, network=None):
        return c.servers('vali', network=network)

    @classmethod
    def restart_valis(cls, search=None, network= 'local'):
        namespace = c.namespace('vali', network=network)
        for name, address in namespace.items():
            if search != None:
                if search not in name:
                    continue
                c.restart(name)

        return namespace

        

    @classmethod
    def check_valis(cls, *args, **kwargs):
        return c.module('vali').check_valis(*args, **kwargs)
    
    @classmethod
    def check_servers(cls, *args, **kwargs):
        return c.module('subspace')().check_servers(*args, **kwargs)

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
        name2shortcut = c.reverse_map(shortcuts)
        if name in name2shortcut:
            del shortcuts[name2shortcut[name]]
        shortcuts[shortcut] = name
        c.putc('shortcuts', shortcuts)
        return {'success': True, 'msg': f'added shortcut ({shortcut} -> {name})'}

    @classmethod
    def resolve_shortcut(cls, name:str) -> str:
        return c.getc('shortcuts').get(name, name)
    
    @classmethod
    def model_menu(cls):
        return c.model_shortcuts()
    
    @classmethod
    def talk(cls , *args, module = 'model', num_jobs=1, timeout=6, **kwargs):
        jobs = []
        for i in range(num_jobs):
            model = c.connect(module, virtual=False)
            c.print('Selecting: ', model)
            job = model.async_forward(fn='talk', args=args, kwargs=kwargs)
            jobs += [job]

        results = c.gather(jobs, timeout=timeout)
        for r in results:
            if c.is_success(r):
                if isinstance(r, str) and len(r) > 0:
                    return r

        return 'Im sorry I dont know how to respond to that, can you rephrase that?'

    chat = talk

    def x(self, y=1):
        c.print('fam', y)

    @classmethod
    def ask(cls, *args, **kwargs):
        return c.module('model.hf').talk(*args, **kwargs)

    @classmethod
    def containers(cls):
        return c.module('docker').containers()

    @staticmethod
    def chunk(sequence:list = [0,2,3,4,5,6,67,],
            chunk_size:int=None,
            num_chunks:int= None):
        assert chunk_size != None or num_chunks != None, 'must specify chunk_size or num_chunks'
        if chunk_size == None:
            chunk_size = len(sequence) // num_chunks

        if chunk_size > len(sequence):
            return [sequence]
        if num_chunks == None:
            num_chunks = len(sequence) // chunk_size


        chunks = [[] for i in range(num_chunks)]
        for i, element in enumerate(sequence):
            idx = i % num_chunks
            chunks[idx].append(element)
        return chunks
    @classmethod
    def batch(cls, x: list, batch_size:int=8): 
        return c.chunk(x, chunk_size=batch_size)
    
    @classmethod 
    def chmod_scripts(cls):
        c.cmd(f'chmod +x {c.libpath}/scripts/*', verbose=True, bash=True)

    def install_docker_gpus(self):
        self.chmod_scripts()
        c.cmd('./scripts/nvidia_docker_setup.sh', cwd=self.libpath, verbose=True, bash=True)

    def install_docker(self):
        self.chmod_scripts()
        c.cmd('./scripts/install_docker.sh', cwd=self.libpath, verbose=True, bash=True)

    @classmethod
    def install_rust(cls, sudo=True) :
        cls.chmod_scripts()
        c.cmd('./scripts/install_rust_env.sh', cwd=cls.libpath, verbose=True, bash=True, sudo=sudo)

    @classmethod
    def install_npm(cls, sudo=False) :
        c.cmd('apt install npm', sudo=sudo)

    @classmethod
    def install_pm2(cls, sudo=True) :
        c.cmd('npm install pm2 -g', sudo=sudo)

    @classmethod
    def install_python(cls, sudo=True) :
        c.cmd('apt install -y python3-dev python3-pip', verbose=True, bash=True, sudo=sudo)

    @classmethod
    def cachefn(cls, func, max_age=60, update=False, cache=True, cache_folder='cachefn'):
        import functools
        path_name = cache_folder+'/'+func.__name__
        def wrapper(*args, **kwargs):
            fn_name = func.__name__
            cache_params = {'max_age': max_age, 'cache': cache}
            for k, v in cache_params.items():
                cache_params[k] = kwargs.pop(k, v)

            
            if not update:
                result = cls.get(fn_name, default=None, **cache_params)
                if result != None:
                    return result

            result = func(*args, **kwargs)
            
            if cache:
                cls.put(fn_name, result, cache=cache)
            return result
        return wrapper

    @classmethod
    def ss58_encode(cls, data:Union[str, bytes], ss58_format=42, **kwargs):
        from scalecodec.utils.ss58 import ss58_encode
        if type(data) is str:
            data = bytes.fromhex(data.replace('0x', ''))
        return ss58_encode(data, ss58_format=ss58_format, **kwargs)


    @classmethod
    def ss58_decode(cls, data:Union[str, bytes],**kwargs):
        from scalecodec.utils.ss58 import ss58_decode
        return ss58_decode(data,  **kwargs)


    @classmethod
    def random_tmp_file_path(cls, prefix='randomtempfile_utc'):
        return f"/tmp/{prefix}{c.time()}"

    @classmethod
    def name2compose(self, **kwargs):
        return c.module('docker').name2compose(**kwargs)



    @classmethod
    def generator(cls):
        for i in range(10):
            yield i


    @classmethod
    def run_generator(cls):
        """
        
        """
        for i in cls.generator():
            c.print(i)
    @classmethod
    def is_generator(cls, obj):
        """
        Is this shiz a generator dawg?
        """
        import inspect
        return inspect.isgeneratorfunction(obj)
    


    @classmethod
    def module2docpath(cls):
        tree = c.tree()
        module2docpath = {}
        for m, p in tree.items():

            dirpath = os.path.dirname(p)
            docpaths = [f for f in c.ls(dirpath) if f.endswith('.md')]
            if len(docpaths) > 1:
                [c.print(f) for f in docpaths]
            if len(docpaths) > 0:
                
                doc_name = docpaths[0].split('/')[-1].split('.')[0]
                if not (doc_name.startswith(m.replace('.','_')) or doc_name.endswith('_doc')):
                    continue
                module2docpath[m] = docpaths[0]

                
            
        return module2docpath
    @classmethod
    def hello(cls):
        c.print('hello')


    thread_map = {}
    @classmethod
    def thread(cls,fn: Union['callable', str],  
                    args:list = None, 
                    kwargs:dict = None, 
                    daemon:bool = True, 
                    tag = None,
                    start:bool = True,
                    tag_seperator:str=':'):

        if isinstance(fn, str):
            fn = c.get_fn(fn)
        if args == None:
            args = []
        if kwargs == None:
            kwargs = {}

        assert callable(fn), f'target must be callable, got {fn}'
        assert  isinstance(args, list), f'args must be a list, got {args}'
        assert  isinstance(kwargs, dict), f'kwargs must be a dict, got {kwargs}'
        
        import threading
        t = threading.Thread(target=fn, args=args, kwargs=kwargs)
        
        # set the time it starts
        t.__dict__['start_time'] = c.time()
        
        t.daemon = daemon
        if start:
            t.start()
        fn_name = fn.__name__
        if tag != None:
            tag = str(tag)
            name = fn_name + tag_seperator + tag
        else:
            name = fn_name + tag_seperator
        cnt = 0
        while name in cls.thread_map:
            cnt += 1
            name = fn_name + tag_seperator + tag + str(cnt)

        cls.thread_map[name] = t

        return t

    @classmethod
    def join_threads(cls, threads:[str, list]):

        threads = self.thread_map
        for t in threads.values():
            # throw error if thread is not in thread_map
            t.join()

    @classmethod
    def threads(cls, *args, **kwargs):
        return list(cls.thread_map(*args, **kwargs).keys())

    @classmethod
    def resolve_key_address(cls, key):
        key2address = c.key2address()
        if key in key2address:
            address = key2address[key]
        else:
            address = key
        return address
    
    ##################################
    # USER LAND
    ##################################
    @classmethod
    def add_user(cls, address, role='user', **kwargs):
        return c.module('user').add_user(address, role=role, **kwargs)
    @classmethod
    def users(cls, *args, **kwargs):
        return c.module('user').user(*args, **kwargs)
    @classmethod
    def is_user(cls, address):
        return c.module('user').is_user(address)
    @classmethod
    def is_user(self, address):
        return c.module('user').is_user(address)
    @classmethod
    def get_user(cls, address):
        return c.module('user').get_user(address)
    @classmethod
    def update_user(cls, *args, **kwargs):
        return c.module('user').update_user(*args, **kwargs)
    @classmethod
    def get_role(cls, *args, **kwargs):
        return c.module('user').get_role(*args, **kwargs)
    @classmethod
    def refresh_users(cls):
        return c.module('user').refresh_users()
    @classmethod
    def user_exists(cls, address):
        return address in cls.get('users', {})
    @classmethod
    def is_root_key(cls, address:str)-> str:
        return address == c.root_key().ss58_address
    @classmethod
    def is_admin(cls, *args, **kwargs):
        return c.module('user').is_admin(*args, **kwargs)
    @classmethod
    def admins(cls):
        return c.module('user').admins()
    @classmethod
    def add_admin(cls, address):
        return  c.module('user').add_admin(address)
    @classmethod
    def rm_admin(cls, address):
        return  c.module('user').rm_admin(address)
    @classmethod
    def num_roles(cls, role:str):
        return c.module('user').num_roles(role)
    @classmethod
    def rm_user(cls, address):
        return c.module('user').rm_user(address)
    ##################################
    # REPLICA LAND
    ##################################
    @classmethod
    def replicas(cls, network:str=None, **kwargs) -> List[str]:
        servers = c.servers(cls.module_path(),network=network, **kwargs)
        return servers

    @classmethod
    def restart_replicas(cls, network:str=None, **kwargs):
        for m in cls.replicas(network=network, **kwargs):
            c.print(m)
            c.restart(m)

    @classmethod
    def restart_many(cls, search:str = None, network = None, **kwargs):
        servers = c.servers(search, network=network)
        for m in servers:
            c.restart(m, **kwargs)
        return servers

        
    
    @classmethod
    def kill_replicas(self, network:str=None, **kwargs):
        for m in cls.replicas(network=network, **kwargs):
            c.kill(m)

    @classmethod
    def gc(cls):
        import gc
        gc.collect()
        return {'success': True, 'msg': 'garbage collected'}

    def __repr__(self) -> str:
        return f'<{self.class_name()} tag={self.tag}>'
    def __str__(self) -> str:
        return f'<{self.class_name()} tag={self.tag}>'

    @classmethod
    def emoji(cls,  name:str):
        emojis = []
        for k,v in c.emojis.items():
            if name in k:
                emojis += [v]
   
        return c.choice(emojis)
        
    emojis = {'dank': '🔥',
            'error': '💥',
            'white': '🕊️',
            'cool': '😎',
            'success': '✨',
            'sad': '😢',
            'time': '🕒',
            'count': '🔢',
            'output': '📤',
            'input': '📥',
            'party': '🥳',
            'fireworks': '🎆',
            'explosion': '💣',
            'alien': '👽',
            'rocket': '🚀',
            'money': '💰',
            'victory': '✌️',
            'unicorn': '🦄',
            'rainbow': '🌈',
            'music': '🎵',
            'pizza': '🍕',
            'taco': '🌮',
            'sunglasses': '😎',
            'flame': '🔥',
            'diamond': '💎',
            'savage': '😈',
            'laughing': '😂',
            'ninja': '🥷',
            'skull': '💀',
            'thumbs_up': '👍',
            'thumbs_down': '👎',
            'crown': '👑',
            'cyber_eye': '👁️‍🗨️',
            'data_stream': '🌐', 
            'brain': '🧠',
            'robot': '🤖',
            'lightning': '⚡',
            'heart': '❤️',
            'heartbreak': '💔',
            'heartpulse': '💗',
            'green_heart': '💚',
            'blue_heart': '💙',
            'purple_heart': '💜',
            'yellow_heart': '💛',
            'orange_heart': '🧡',
            'error': '💥',
            'cross': '❌',
            'check': '✅',
            'wrong': '❌',
            'right': '✅',
            'correct': '✅',
            'incorrect': '❌',
            'checkmark': '✅',
            'check_mark': '✅',
            'checkered_flag': '🏁',
            'warning': '⚠️',
            'warning_sign': f'⚠️',
            'question': '❓',
            'happy': '😀',
            'sad': '😢',
            'angry': '😠',
            'angry_face': '😠',
            'angry_face_with_horns': '👿',
            'devil': '😈',
            'red_circle': '🔴',
            'green_circle': '🟢',
            'blue_circle': '🔵',
            'yellow_circle': '🟡',
            'orange_circle': '🟠',
            'purple_circle': '🟣',
            'black_circle': '⚫',
            'white_circle': '⚪',
            'brown_circle': '🟤',
            'red_square': '🟥',
            'green_square': '🟩',
            'blue_square': '🟦',
            'yellow_square': '🟨',
            'orange_square': '🟧',
            'purple_square': '🟪',
            'black_square': '⬛',
            'white_square': '⬜',
            'brown_square': '🟫',

            
    }
    
    
    
    @staticmethod
    def tqdm(*args, **kwargs):
        from tqdm import tqdm
        return tqdm(*args, **kwargs)
    
    # PEER LAND
    @classmethod
    def peers(cls, network:str='local', tag=None):
        module = cls.module_path()
        servers = c.servers(network=network)
        peers = [s for s in servers if s.startswith(module)]
        return peers

    @classmethod
    def random_peer(cls, network:str='local', tag=None):
        peers = cls.peers(network=network, tag=tag)
        return c.choice(peers)

    @classmethod
    def random_peer_address(cls, network:str='local', tag=None):
        random_peer = cls.random_peer(network=network, tag=tag)
        address = c.namespace(network=network).get(random_peer)
        return address

    @classmethod
    def random_peers(cls, network:str='local', n=2, tag=None):
        peers = cls.peers(network=network, tag=tag)
        return c.shuffle(peers)[:n]


    @classmethod
    def play(cls):
        c.module('music').play()

    @classmethod
    def type(cls,x ):
        return type(x).__name_
        

    def set_api_key(self, api_key:str, cache:bool = True):
        import os
        api_key = os.getenv(str(api_key), None)
        if api_key == None:
            api_key = self.get_api_key()

        
        self.api_key = api_key
        if cache:
            self.add_api_key(api_key)

        assert isinstance(api_key, str)


    ## API MANAGEMENT ##

    @classmethod
    def add_api_key(cls, api_key:str):
        assert isinstance(api_key, str)
        api_keys = cls.get('api_keys', [])
        api_keys.append(api_key)
        api_keys = list(set(api_keys))
        cls.put('api_keys', api_keys)
        return {'api_keys': api_keys}


    @classmethod
    def add_api_keys(cls, api_keys:str):
        api_keys = list(set(api_keys + cls.get('api_keys', [])))
        cls.put('api_keys', api_keys)
        return {'api_keys': api_keys}

    @classmethod
    def set_api_keys(cls, api_keys:str):
        api_keys = list(set(api_keys))
        cls.put('api_keys', api_keys)
        return {'api_keys': api_keys}


    @classmethod
    def rm_api_key(cls, api_key:str):
        assert isinstance(api_key, str)
        api_keys = cls.get('api_keys', [])
        for i in range(len(api_keys)):
            if api_key == api_keys[i]:
                api_keys.pop(i)
                break   

        cls.put('api_keys', api_keys)
        return {'api_keys': api_keys}


    @classmethod
    def get_api_key(cls):
        api_keys = cls.api_keys()
        if len(api_keys) == 0:
            return None
        else:
            return c.choice(api_keys)

    @classmethod
    def api_keys(cls):
        return cls.get('api_keys', [])
    

    @classmethod
    def rm_api_keys(self):
        self.put('api_keys', [])
        return {'api_keys': []}

    @classmethod
    def send_api_keys(cls, module:str, network='local'):
        api_keys = cls.api_keys()
        assert len(api_keys) > 0, 'no api keys to send'
        module = c.connect(module, network=network)
        return module.add_api_keys(api_keys)

    @classmethod
    def loop(cls, interval=60, network=None, remote:bool=True, local:bool=True, save:bool=True):
        if remote:
            kwargs = c.locals2kwargs(locals())
            kwargs['remote'] = False
            c.remote_fn('loop', kwargs=kwargs, name='loop')
            return {'success': True, 'msg': 'looping on remote'}
        start_time = 0
        subspace = c.module('subspace')()
        while True:
            current_time = c.timestamp()
            elapsed = current_time - start_time
            if elapsed > interval:
                c.print('SYNCING AND UPDATING THE SERVERS_INFO')
                c.print(c.server_infos(update=True, network='local'))
                # subspace.sync(network=network, remote=remote, local=local, save=save)
                start_time = current_time
            c.sleep(interval)

    
    def load_state(self, update:bool=False, netuid=0, network='main', state=None, _self = None):
        
        if _self != None:
            self = _self
        
        import streamlit as st
        
        self.key = c.get_key()

        t = c.timer()
        @st.cache_data(ttl=60*60*24, show_spinner=False)
        def get_state():
            subspace = c.module('subspace')()
            state =  subspace.state_dict(update=update)
            return state
        
        if state == None:
            state = get_state()
        self.state =  state
        self.netuid = 0
        self.subnets = self.state['subnets']

        self.modules = self.state['modules'][self.netuid]
        self.name2key = {k['name']: k['key'] for k in self.modules}
        self.key2name = {k['key']: k['name'] for k in self.modules}

        self.namespace = c.namespace()

        self.keys  = c.keys()
        self.key2index = {k:i for i,k in enumerate(self.keys)}

        self.namespace = {m['name']: m['address'] for m in self.modules}
        self.module_names = [m['name'] for m in self.modules]
        self.block = self.state['block']
        for i, m in enumerate(self.modules):
            self.modules[i]['stake'] = self.modules[i]['stake']/1e9
            self.modules[i]['emission'] = self.modules[i]['emission']/1e9

        self.key_info = {
            'ss58_address': self.key.ss58_address,
            'balance': self.state['balances'].get(self.key.ss58_address,0),
            'stake_to': self.state['stake_to'][self.netuid].get(self.key.ss58_address,{}),
            'stake': sum([v[1] for v in self.state['stake_to'][self.netuid].get(self.key.ss58_address)]),
        }

        self.key_info['balance']  = self.key_info['balance']/1e9
        self.key_info['stake_to'] = {k:v/1e9 for k,v in self.key_info['stake_to']}
        self.key_info['stake'] = sum([v for k,v in self.key_info['stake_to'].items()])
        # convert keys to names 
        for k in ['stake_to']:
            self.key_info[k] = {self.key2name.get(k, k): v for k,v in self.key_info[k].items()}

        self.subnet_info = self.state['subnets'][0]
        balances = self.state['balances']
        self.total_balance = sum(balances.values())/1e9
        for k in ['stake', 'emission', 'min_stake']:
            self.subnet_info[k] = self.subnet_info[k]/1e9
    
    

      
    @classmethod
    def function2streamlit(cls, 
                           module = None,
                           fn:str = '__init__',
                           fn_schema = None, 
                           extra_defaults:dict=None,
                           cols:list=None,
                           skip_keys = ['self', 'cls'],
                           salt = None,
                            mode = 'pm2'):
        import streamlit as st
        
        key_prefix = f'{module}.{fn}'
        if salt != None:
            key_prefix = f'{key_prefix}.{salt}'
        if module == None:
            module = cls
            
        elif isinstance(module, str):
            module = c.module(module)
        extra_defaults = {} if extra_defaults is None else extra_defaults
        kwargs = {}

        if fn_schema == None:

            fn_schema = module.schema(defaults=True, include_parents=True)[fn]
            if fn == '__init__':
                config = module.config(to_munch=False)
                extra_defaults = config
            fn_schema['default'].pop('self', None)
            fn_schema['default'].pop('cls', None)
            fn_schema['default'].update(extra_defaults)
            fn_schema['default'].pop('config', None)
            fn_schema['default'].pop('kwargs', None)
            
        fn_schema['input'].update({k:str(type(v)).split("'")[1] for k,v in extra_defaults.items()})
        if cols == None:
            cols = [1 for i in list(range(int(len(fn_schema['input'])**0.5)))]
        if len(cols) == 0:
            return kwargs
        cols = st.columns(cols)

        for i, (k,v) in enumerate(fn_schema['default'].items()):
            
            optional = fn_schema['default'][k] != 'NA'
            fn_key = k 
            if fn_key in skip_keys:
                continue
            if k in fn_schema['input']:
                k_type = fn_schema['input'][k]
                if 'Munch' in k_type or 'Dict' in k_type:
                    k_type = 'Dict'
                if k_type.startswith('typing'):
                    k_type = k_type.split('.')[-1]
                fn_key = f'**{k} ({k_type}){"" if optional else "(REQUIRED)"}**'
            col_idx  = i 
            if k in ['kwargs', 'args'] and v == 'NA':
                continue
            

            col_idx = col_idx % (len(cols))
            if type(v) in [float, int] or c.is_number(v):
                kwargs[k] = cols[col_idx].number_input(fn_key, v, key=f'{key_prefix}.{k}')
            elif v in ['True', 'False']:
                kwargs[k] = cols[col_idx].checkbox(fn_key, v, key=f'{key_prefix}.{k}')
            else:
                kwargs[k] = cols[col_idx].text_input(fn_key, v, key=f'{key_prefix}.{k}')
        kwargs = cls.process_kwargs(kwargs, fn_schema)       
        
        return kwargs

   

    @classmethod
    def process_kwargs(cls, kwargs:dict, fn_schema:dict):
        
        for k,v in kwargs.items():
            if v == 'None':
                v = None
            
            if isinstance(v, str):
                if v.startswith('[') and v.endswith(']'):
                    if len(v) > 2:
                        v = eval(v)
                    else:
                        v = []

                elif v.startswith('{') and v.endswith('}'):

                    if len(v) > 2:
                        v = c.jload(v)
                    else:
                        v = {}               
                elif k in fn_schema['input'] and fn_schema['input'][k] == 'str':
                    if v.startswith("f'") or v.startswith('f"'):
                        v = c.ljson(v)
                    else:
                        v = v

                elif fn_schema['input'][k] == 'float':
                    v = float(v)

                elif fn_schema['input'][k] == 'int':
                    v = int(v)

                elif k == 'kwargs':
                    continue
                elif v == 'NA':
                    assert k != 'NA', f'Key {k} not in default'
                elif v in ['True', 'False']:
                    v = eval(v)
                elif c.is_number(v):
                    v = eval(v)
                else:
                    v = v
            
            kwargs[k] = v

        return kwargs
    
 
Module = c
Module.run(__name__)
    

