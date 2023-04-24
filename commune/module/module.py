

import inspect
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


boot_peers = ['162.157.13.236:9057', '162.157.13.236:9255', '162.157.13.236:9451']


class Module:
    
    # port range for servers
    default_port_range = [50050, 50150] 
    
    user = None
    
    # default ip
    default_ip = '0.0.0.0'
    
    address = None
    key = None
    # the root path of the module (assumes the module.py is in ./module/module.py)
    root_path  = root = os.path.dirname(os.path.dirname(__file__))
    repo_path  = os.path.dirname(root_path)
    
    # get the current working directory  (doesnt have /)
    pwd = os.getenv('PWD')
    
    # get the root directory (default commune)
    # Please note that this assumes that {root_dir}/module.py is where your module root is
    root_dir = root_path.split('/')[-1]
    console = Console()
    boot_peers = boot_peers
    
    
    def __init__(self, 
                 config:Dict=None, 
                 save_if_not_exists:bool=False, 
                 add_attributes: bool = False,
                 key: str = None,
                 boot_peers = None,
                 *args, 
                 **kwargs):
        # set the config of the module (avoid it by setting config=False)
        # self.new_event_loop(nest_asyncio=True)
        self.set_config(config=config, save_if_not_exists=save_if_not_exists, add_attributes=add_attributes)  
        # do you want a key fam
        if key is not None:
            self.set_key(key)
    def getattr(self, k:str)-> Any:
        return getattr(self,  k)
    @classmethod
    def getclassattr(cls, k:str)-> Any:
        return getattr(cls,  k)
    
    
    
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
    def module_path(cls) -> str:
        # get the module path
        if not hasattr(cls, '_module_path'):
            cls._module_path = cls.get_module_path(simple=True)
        return cls._module_path



    
        
    @classmethod
    def module_class(cls) -> str:
        return cls.__name__

    
    @classmethod
    def class_name(cls) -> str:
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
        return cls.load_config( *args, **kwargs)
    
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
        '''
        Loads a yaml file
        '''
        path = cls.resolve_path(path, root=root)
        
        from commune.utils.dict import load_yaml
        return load_yaml(path)


    @classmethod
    def get_fn_code(cls, 
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
    def fn_code(cls, fn ) -> str:
        '''
        Returns the code of a function
        '''
        return cls.get_fn_code(fn)

    @classmethod
    def sandbox(cls):
        return cls.cmd('python3 commune/sandbox.py')
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
    def load_config(cls, path:str=None,  to_munch:bool = False, save_if_not_exists:bool = False) -> Union[Munch, Dict]:
        '''
        Args:
            path: The path to the config file
            to_munch: If true, then convert the config to a munch
        '''
        module_tree = cls.module_tree()
        if path in module_tree: 
            path = module_tree[path].replace('.py', '.yaml')
        
        path = path if path else cls.__config_file__()
            
        if save_if_not_exists:    
            if not os.path.exists(__config_file__):
                cls.save_config(config=cls.minimal_config(), path=__config_file__)
               
        try:
            
            config = cls.load_yaml(path)
        except FileNotFoundError as e:
            config = cls.minimal_config()
        
        if to_munch:
            config =  cls.dict2munch(config)
            
        
        return config
    
    get_config = load_config

    @classmethod
    def put_config(cls, key, value) -> Munch:
        '''
        Saves the config to a yaml file
        '''
        config = self.get_config()
        cls.dict_put(config, key, value)
        self.set_config(config=config, key=key, value=value)
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
    
    put_config = save_config
    


    def set_config(self, 
                   config:Optional[Union[str, dict]]=None, 
                   kwargs:dict={},
                   save_if_not_exists:bool = False,
                   to_munch: bool = True,
                   add_attributes: bool = False) -> Munch:
        '''
        Set the config as well as its local params
        '''
        
        from commune.utils.dict import munch2dict, dict2munch
        

        default_config =  self.get_config()
        
        # ensure to include the inner kwargs if that is provided (Which isnt great practice lol)
        kwargs  = {**kwargs, **kwargs.pop('kwargs', {}) }
        
        # ensure there are no inner_args to avoid ambiguous args 
        inner_args = kwargs.pop('args', [])
        assert len(inner_args) == 0, f'Please specify your keywords for this to act nicely, args: {inner_args}'
    
        if type(config) in [dict]:
            config = config
            config_kwargs = config.pop('kwargs', None)
            if isinstance(config_kwargs, dict):
                kwargs = {**config_kwargs, **kwargs}
            config.pop('args', None)
            config.pop('self', None)
            config.pop('cls', None)
        elif type(config) in [Munch]:
            config = munch2dict(config)
        elif type(config) in [str, type(None)]:
            config = self.load_config(path=config, save_if_not_exists=False)
        
        
        for k,v in kwargs.items():
            self.dict_put(config, k, v)
        
        # overwrite the default config with the new config
        config = {**default_config, **config}
        if add_attributes:
            self.__dict__.update(config)
            
        #  add the config after in case the config has a config attribute lol
        if to_munch:
            self.config = dict2munch(config)
        
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
    def st(cls, module = None):
        module = cls.get_module(module)
        module_filepath = module.filepath()
        cls.run_command(f'streamlit run {module_filepath}', verbose=True)




    @classmethod
    def run_command(cls, 
                    command:str,
                    verbose:bool = True, 
                    env:Dict[str, str] = {}, 
                    output_text:bool = True,
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
                        
                        cls.print(new_line.decode(), color=color)
                    new_line = b''
                    continue
                
                new_line += c
  
        except KeyboardInterrupt:
            kill_process(process)
            
             
        kill_process(process)
        return stdout_text


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
        cls.print(f'Importing {object_name} from {module}')
        obj =  getattr(import_module(module), object_name)
        return obj
    
    get_object = import_object

    
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
    
    @staticmethod
    def port_used(port:int, ip:str ='0.0.0.0'):
        '''7um
        Check if port is available
        '''
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    
    @classmethod
    def port_available(cls, port:int, ip:str ='0.0.0.0'):
        return not cls.port_used(port=port, ip=ip)
        

    @classmethod
    def get_used_ports(cls, ports:List[int] = None, ip:str = '0.0.0.0', port_range:Tuple[int, int] = None):
        '''
        Get availabel ports out of port range
        
        Args:
            ports: list of ports
            ip: ip address
        
        '''
        port_range = cls.resolve_port_range(port_range=port_range)
        if ports == None:
            ports = list(range(*port_range))
        
        used_ports = []
        for port in ports: 
            if cls.port_used(port=port, ip=ip):
                used_ports.append(port)
        
        return used_ports
    

   
    @classmethod
    def resolve_path(cls, path:str, extension:Optional[str]= None, root:bool = False):
        '''
        Resolves path for saving items that relate to the module
        
        The path is determined by the module path 
        
        '''
        
        tmp_dir = Module.tmp_dir() if root else cls.tmp_dir()
        
        if path.startswith('/'):
            return path
        elif path.startswith('~/'):
            return os.path.expanduser(path)
        elif path.startswith('./'):
            return path.replace('./', cls.pwd + '/')
        else:
            tmp_dir = cls.tmp_dir()
            if tmp_dir not in path:
                path = os.path.join(tmp_dir, path)
            if not os.path.isdir(path):
                if extension != None and extension != path.split('.')[-1]:
                    path = path + '.' + extension

            return path
    
    
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
    @classmethod
    def resolve_port(cls, port:int=None, find_available:bool = True):
        
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
    def get_available_port(cls, port_range: List[int] = None , ip:str =None) -> int:
        
        '''
        
        Get an availabldefe port within the {port_range} [start_port, end_poort] and {ip}
        '''
        port_range = cls.resolve_port_range(port_range)
        ip = ip if ip else cls.default_ip
        
        # return only when the port is available
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
                        print('KILLED')
            return port
        elif mode == 'bash':
            return cls.run_command('kill -9 $(lsof -ti:{port})')

    @classmethod
    def kill_server(cls, module:str, mode:str = 'pm2'):
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

    @classmethod
    def kill_all_servers(cls, verbose: bool = True):
        '''
        Kill all of the servers
        '''
        for module in cls.servers():
            if verbose:
                cls.print(f'Killing {module}', color='red')
            cls.kill_server(module)
            
    
    @classmethod
    def kill_all(cls):
        cls.kill_all_servers()


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
        return cls.path2objectpath(cls.__module_file__())
    
    
    

    

    @classmethod
    def get_classes_from_python_path(cls, path:str, class_index=-1):
        import re
        
        # read the contents of the Python script file
        python_script = cls.get_text(path)
        class_names  = []
        lines = python_script.split('\n')
        for line in lines:
            key_elements = ['class ', '(', '):']
            if all([key_element in line for key_element in key_elements]) and \
                    'ModuleWrapper' not in line and 'key_elements' not in line:
                class_name = line.split('class ')[-1].split('(')[0].strip()
                class_names.append(class_name)
                
        # return the class names
        return class_names

    @classmethod
    def path2objectpath(cls, path:str) -> str:
        object_name = cls.get_classes_from_python_path(path)
        if len(object_name) == 0:
            return None
        object_name = object_name[0]
        path = path.replace(cls.repo_path+'/', '').replace('.py','.').replace('/', '.') 
        path = path + object_name
        print(path,object_name,  'fam')
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
    def get_module(cls, path:str, verbose:bool = False, handle_error:bool=False) -> str:
        
        try:
            
            path = cls.simple2path(path)
            path = cls.path2objectpath(path)
            assert path is not None, f'Could not find path for {path}'
            if verbose:
                cls.print(f'Found {path}', verbose=verbose)
        except KeyError as e:
            if handle_error:
                cls.print(f'{e}', verbose=verbose)
            else:
                raise e
            
        print()
        
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
        return list(cls.module_tree().keys())
    @classmethod
    def modules(cls, *args, **kwargs) -> List[str]:
        return list(cls.namespace(*args, **kwargs).keys())
    @classmethod
    def models(cls, *args, **kwargs) -> List[str]:
        return [k for k in list(cls.namespace(*args, **kwargs).keys()) if k.startswith('model')]
    @classmethod
    def datasets(*args, **kwargs) -> List[str]:
        return [k for k in list(cls.namespace(*args, **kwargs).keys()) if k.startswith('dataset')]
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
        data = await async_get_json(path, **kwargs)
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
    async def async_put_json(cls, path:str, 
                 data:Dict, 
                 meta = None,
                 root: bool = False, 
                 **kwargs) -> str:
        
        from commune.utils.dict import async_put_json
        if meta != None:
            data = {'data':data, 'meta':meta}
        path = cls.resolve_path(path=path, extension='json', root=root)
        cls.print(path,'bro')
        await async_put_json(path=path, data=data, **kwargs)
        return path
    
    save_json = put_json
    
    @classmethod
    def exists(cls, path:str, extension = 'json', root:bool = False)-> bool:
        path = cls.resolve_path(path=path, extension=extension, root=root)
        return os.path.exists(path)

    @classmethod
    def rm_json(cls, path=None, root:bool = False):
        from commune.utils.dict import rm_json

        if path in ['all', '**']:
            return [cls.rm_json(f) for f in cls.glob(files_only=False)]
        
        path = cls.resolve_path(path=path, extension='json', root=root)

        return rm_json(path )

    @classmethod
    def glob(cls,  path ='~/', files_only:bool = True, root:bool = False):
        
        path = cls.resolve_path(path, extension=None, root=root)
        
        if os.path.isdir(path):
            path = os.path.join(path, '**')
            
        paths = glob(path, recursive=True)
        if len(paths) == 0:
            paths = glob(path+'/**', recursive=True)
        if files_only:
            paths =  list(filter(lambda f:os.path.isfile(f), paths))
        return paths
         
    @classmethod
    def ls_json(cls, path:str, recursive:bool = True):
        return cls.ls(path, recursive=recursive)
    

    @classmethod
    def ls(cls, path:str = './', 
           recursive:bool = False,
           root:bool = False):
        path = cls.resolve_path(path, extension=None, root=root)

        ls_files = cls.lsdir(path) if not recursive else cls.walk(path)
        return [os.path.expanduser(os.path.join(path,f)) for f in ls_files]
    
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
    def bittensor(cls, *args, **kwargs):
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
    def root_module(cls, name:str='module',
                    timeout:int = 100, 
                    sleep_interval:int = 1,
                    return_info = False,
                    refresh:bool = False,
                    **kwargs):
        if not cls.server_exists(name) or refresh:
            cls.launch(name=name, **kwargs)
            cls.wait_for_server(name, timeout=timeout, sleep_interval=sleep_interval)
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
        module = cls.connect(name)
        return module.server_info['address']
    
    anchor = root_module
    anchor_address = root_address

        
    @classmethod
    async def async_connect(cls, 
                name:str=None, 
                ip:str=None, 
                port:int=None , 
                network : str = 'local',
                virtual:bool = True, 
                wait_for_server:bool = False,
                **kwargs ):
        
        if (name == None and ip == None and port == None):
            return cls.root_module()
            
            
        if wait_for_server:
            cls.wait_for_server(name)

        # subspce namespce
        if network == 'subspace': 
            network = self.resolve_network(network)
            namespace = network.namespace()
            server_info = namespace[name]
            ip = server_info['ip']
            port = server_info['port']
            client_kwargs = {'ip':ip, 
                                'port':port}
                        
            client_kwargs['network'] = network
        
        # local namespace    
        if network == 'local':
            
            if isinstance(name, str):
                if len(name.split(':')) == 2:
                    port = int(name.split(':')[1])
                    ip = name.split(':')[0]
                else:
                    
                    namespace = cls.namespace()
                    servers = list(namespace.keys())
                    
                    if name not in servers:
                        found_server_bool = False
                        for s in servers:
                            if s.startswith(name):
                                name = s
                                found_server_bool = True
                        assert found_server_bool, f'No server found with name {name} in {servers}'
                    assert name in namespace, f'No server found with name {name} in {servers}'
                    module_address = namespace[name]
                    cls.print(module_address, color='blue')
                    ip = ':'.join(module_address.split(':')[:-1])
                    port = int(module_address.split(':')[-1])
                    
            if ip == None:
                ip = cls.default_ip
            client_kwargs = dict(ip=ip, port=int(port))


        ip = client_kwargs['ip']
        port = client_kwargs['port']
        assert isinstance(port, int) , 'Port must be specified as an int'
        assert isinstance(ip, str) , 'IP must be specified as a string'
    

        Client = cls.import_object('commune.server.client.Client')
        client_module = Client( **kwargs,**client_kwargs)

        cls.print(f'Connecting to {name} on {ip}:{port}', color='yellow')

        if virtual:
            return client_module.virtual()
        
        return client_module
   
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
            ip = cls.external_ip()
        peer_addresses = []
        for port in used_local_ports:
            peer_addresses.append(f'{ip}:{port}')
            
        return peer_addresses
            
    
    @classmethod
    def get_local_namespace(cls, 
                            ip:str = None, 
                            save:bool = False,
                            timeout:int  = 3) -> Dict:
        peer_registry = {}
        peer_addresses = cls.get_peer_addresses()
        peer = ['']
        jobs = []
        for address in peer_addresses:
            cls.print(f'Connecting to {address}', color='yellow')
            ip, port = address.split(':')
            jobs += [cls.async_connect(ip=ip, port=port, timeout=timeout)]
        loop = cls.get_event_loop()
        peers = loop.run_until_complete(asyncio.gather(*jobs))
        
        for peer, peer_address in zip(peers, peer_addresses):

            peer_name = peer.module_name

            peer_registry[peer_name] = peer_address
            
        if save:
            Module.save_json('local_namespace', peer_registry)
        return peer_registry

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
    def address2module(cls, *args, **kwargs):
        namespace = cls.namespace(*args, **kwargs)
        port2module =  {}
        for name, address in namespace.items():
            port2module[address] = name
        return port2module
    address2name = address2module
        
        
    @classmethod
    def remote_namespace(cls,  seperator = '<R>'):
        peer_registry = cls.peer_registry()  
        namespace = {}          
        for peer_id, (peer_address, peer_info) in enumerate(peer_registry.items()):
            
            if isinstance(peer_info, dict):
                peer_name = f'peer{peer_id}'
                if 'namespace' in peer_info:
                    if peer_info['namespace'] == None or isinstance(peer_info['namespace'], str):
                        cls.print(f'Peer {peer_name} has no namespace', color='red')
                        continue
                for name, address in peer_info['namespace'].items():
                    namespace[name+seperator+peer_name] = address
            
        return namespace
            
    @classmethod
    def local_namespace(cls, update:bool = True)-> dict:
        '''
        The module port is where modules can connect with each othe.
        When a module is served "module.serve())"
        it will register itself with the local_namespace dictionary.
        '''
        # from copy import deepcopy
        
        address2module = {}
            
        try:
            local_namespace = Module.get_json('local_namespace', handle_error=True, default={})
        except json.JSONDecodeError as e:
            print('Error decoding server registry, resetting to empty dict')
            update = True
            
        if update:
            local_namespace = cls.get_local_namespace(save=True)
         
        return local_namespace

    
  
    @classmethod
    def servers(cls, search:str = None) -> List[str]:

        servers = list(cls.local_namespace().keys())
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
    def register_server(cls, name: str, 
                        ip: str,
                        port: int,
                        netuid: Union[str, int] = None,
                        context: str =  None,
                        network: str = None,
                        key: 'Key' = None)-> dict:
        local_namespace = cls.local_namespace()    
        
        server_info = dict(
                key=key,
                name=name,
                context = context,
                ip = ip,
                port = port  
            )    
        
        local_namespace[name] = server_info
        cls.put_json(path='local_namespace', data=local_namespace, root=True) 

        # only serve module if you have a network
        if key != None:
            # if the key is not None, we want to add the password to the key
            key = cls.get_key(key)
            # if the key is not None, we want to add the password to the key
            network = cls.resolve_network(network)
            server_info['netuid'] = netuid

            network.register(**register_kwargs)
            
        return local_namespace
  
  
    @classmethod
    def is_module(cls, obj=None) -> bool:
        
        if obj is None:
            obj = cls
        if hasattr(obj, 'module_class'):
            module_class = obj.module_class()
            if module_class == 'Module':
                return True
            
        return False

    @classmethod
    def new_event_loop(cls, nest_asyncio:bool = True) -> 'asyncio.AbstractEventLoop':
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
  
        if nest_asyncio:
            cls.nest_asyncio()

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
    def module_exists(cls, name:str, **kwargs) -> bool:
        namespace = cls.namespace(**kwargs)
        return bool(name in namespace)
    
    @classmethod
    def wait_for_server(cls,
                          name: str ,
                          timeout:int = 30,
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
    
    
    
    whitelist_functions: List[str] = []
    blacklist_functions: List[str] = []

    def attributes(self):
        return list(self.__dict__.keys())
    @classmethod
    def namespace(cls,
                  search = None, 
                  network:str='global',
                  include_peers:bool = True, 
                  **kwargs):
        
        if search in ['local', 'global', 'subspace']:
            network = search
            search = None
        namespace = {}
        if network == 'subspace' :
            subspace = cls.subspace(**kwargs)
            namespace.update(cls.subspace_namespace())
        elif network in ['local', 'global']:
            include_peers = True if network == 'global' else False
            namespace.update(cls.local_namespace())
            if network == 'global':
                namespace.update(cls.remote_namespace())
        else:
            raise ValueError(f'network must be either "subspace" or "local"')
        if search:
            namespace = {k:v for k,v in namespace.items() if search in k}
        return namespace
    
    
    @classmethod
    def subspace_namespace(cls, search = None):
        raise NotImplementedError()
    

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
              network: 'Network' = None,
              netuid= None,
              context= '',
              key = None,
              tag:str=None, 
              replace:bool = True, 
              whitelist_functions:List[str] = None,
              blacklist_functions:List[str] = None,
              wait_for_termination:bool = True,
              wait_for_server:bool = False,
              wait_for_server_timeout:int = 30,
              wait_for_server_sleep_interval: int = 1,
              
              *args, 
              **kwargs ):
        '''
        Servers the module on a specified port
        '''

        if module == None:
            self = cls(*args, **kwargs)
        elif isinstance(module, str):
            self = cls.get_module(module)(*args, **kwargs)
        else:
            self = module
             
        whitelist_functions = whitelist_functions if whitelist_functions else cls.whitelist_functions()
        blacklist_functions = blacklist_functions if blacklist_functions else cls.blacklist_functions()
    
        # resolve the module id
        
        # if the module is a class, then use the module_tag 
        # Make sure you have the module tag set
        
        module_name = name if name != None else self.default_module_name()

        '''check if the server exists'''
        if self.server_exists(module_name): 
            if replace:
                self.kill_server(module_name)
            else: 
                raise Exception(f'The server {module_name} already exists on port {existing_server_port}')

        for k in ['module_name', 'module_id', 'my_name', 'el_namo']:
            self.__dict__[k] = module_name

        Server = cls.import_object('commune.server.server.Server')
        
        self.save_kwargs('serve', locals())

        server = Server(ip=ip, 
                        port=port,
                        module = self )
        
        # register the server
        self.server_info = server.info
        self.ip = server.ip
        self.port = server.port
        self.address = self.ip_address = self.ip_addy =  server.address
        
        
        # register the server
        server_info = cls.register_server(name=module_name, 
                                          context=context,
                                          ip=self.ip,
                                          port=self.port,
                                          network=network,
                                          netuid=netuid)

 
        self.set_key(key)
            
        # serve the server
        server.serve(wait_for_termination=wait_for_termination)
        
        if wait_for_server:
            cls.wait_for_server(name=module_name, timeout=wait_for_server_timeout, sleep_interval=wait_for_server_sleep_interval)
        
    @classmethod
    def functions(cls, search = None, include_module=False):
        functions = cls.get_functions(include_module=include_module)  
        
        if isinstance(search, str):
            functions = [f for f in functions if search in f]
        return functions

        
    @classmethod
    def get_functions(cls, obj:Any=None, include_module:bool = False,) -> List[str]:
        '''
        List of functions
        '''
        if isinstance(obj, str):
            obj = cls.get_module(obj)
        from commune.utils.function import get_functions
        obj = obj if obj != None else cls

        
        if cls.is_module(obj):
            include_module = True
            
    
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
        
        function_schema_map = peer.function_schema_map()
        server_info = peer.server_info
        info  = dict(
            module_name = peer.module_name,
            server_info = peer.server_info,
            function_schema = function_schema_map,
            intro =function_schema_map.get('__init__', 'No Intro Available'),
            examples =function_schema_map.get('examples', 'No Examples Available'),
            public_ip =  server_info if not isinstance(server_info, dict) else server_info['external_ip'] + ':' + str(server_info['port']) ,

        )
        
        return info
    
    def info(self, include_schema: bool = False ) -> Dict[str, Any]:
        function_schema_map = self.function_schema_map()
        info  = dict(
            name = self.module_name,
            ip = self.ip,
            port = self.port,
            address = self.address,
            function_schema = function_schema_map,
            functions =  list(function_schema_map.keys()),
            intro =function_schema_map.get('__init__', 'No Intro Available'),
            examples =function_schema_map.get('examples', 'No Examples Available'),
        )
        return info



    def peer_info(self) -> Dict[str, Any]:
        self.info()
    @classmethod
    def schema(cls, *args,  **kwargs): 
        function_schema_map = cls.get_function_schema_map(*args, **kwargs)
        return {k:v for k,v in function_schema_map.items()}
    
    @classmethod
    def get_function_schema_map(cls,
                                obj = None,
                                include_hidden:bool = False, 
                                include_module:bool = False,
                                include_docs: bool = False):
        
        obj = obj if obj else cls
        if isinstance(obj, str):
            obj = cls.module(obj)
        function_schema_map = {}
        for fn in cls.get_functions(obj, include_module=include_module):
            # if not include_hidden:
            #     if (fn.startswith('__') and fn.endswith('__')) or fn.startswith('_'):
            #         if fn != '__init__':
            #             continue
            
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
                               
                if include_docs:         
                    function_schema_map[fn] = {
                        'schema': fn_schema,
                        'docs': getattr(obj, fn ).__doc__
                    }
                else:
                    function_schema_map[fn] = fn_schema
        return function_schema_map
    
    def function_schema_map(cls, include_hidden:bool = False, include_module:bool = False):
        function_schema_map = {}
        for fn in cls.functions(include_module=include_module):
            if not include_hidden:
                if (fn.startswith('__') and fn.endswith('__')) or fn.startswith('_'):
                    continue
            if callable(getattr(cls, fn )):
                function_schema_map[fn] = {}
                fn_schema = {}
                for fn_k, fn_v in getattr(cls, fn ).__annotations__.items():
                    
                    fn_v = str(fn_v)
                    
                    if fn_v == inspect._empty:
                        fn_schema[fn_k]= 'Any'
                    elif fn_v.startswith('<class'):
                        fn_schema[fn_k] = fn_v.split("'")[1]
                    else:
                        fn_schema[fn_k] = fn_v
                                        
                function_schema_map[fn] = {
                    'schema': fn_schema,
                    'docs': getattr(cls, fn ).__doc__
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
    
    def module_schema(self, 
                      
                      include_hidden:bool = False, 
                      include_module:bool = False):
        module_schema = {
            'module_name':self.module_name,
            'server':self.server_info,
            'function_schema':self.function_schema_map(include_hidden=include_hidden, include_module=include_module),
        }
        return module_schema
    
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
    def kill(cls, *modules, mode:str = 'pm2', verbose:bool = True):
        servers = cls.servers()
        for module in modules:
            delete_modules = [server for server in servers if  module in server]
            for d_m in delete_modules:
                if verbose:
                    cls.print(f'Killing {d_m}...')
                if mode == 'pm2':
                    cls.pm2_kill(d_m)
                elif mode == 'ray':
                    cls.ray_kill(d_m)

        return delete_modules

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
    def launch(cls, 
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
               shortcut = None,
               device = None,
               **extra_kwargs):
        '''
        Launch a module as pm2 or ray 
        '''

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
                
        cls.print(f'[bold cyan]Launching[/bold cyan] [bold yellow]class:{module.__name__}[/bold yellow] [bold white]name[/bold white]:{name} [bold white]fn[/bold white]:{fn} [bold white]mode[/bold white]:{mode}', color='green')
            
        if fn == 'serve':
            kwargs['tag'] = kwargs.get('tag', tag)
            kwargs['name'] = kwargs.get('name', name)

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
            launch_fn = getattr(cls, f'pm2_launch')
            stdout = launch_fn(**launch_kwargs)
            return launch_kwargs
            
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
            return launch_fn(**launch_kwargs)
        else: 
            raise Exception(f'launch mode {mode} not supported')

    deploy = launch
    
    @classmethod
    def pm2_kill_all(cls, verbose:bool = True):
        for module in cls.pm2_list():
            
            cls.pm2_kill(module)
            if verbose:
                cls.print(f'[red] Killed {module}[/red]')
    @classmethod
    def pm2_list(cls, verbose:bool = False) -> List[str]:
        output_string = cls.run_command('pm2 status', verbose=False)
        module_list = []
        for line in output_string.split('\n'):
            if '│ default     │ ' in line:
                module_name = line.split('│')[2].strip()
                module_list += [module_name]
                
        return module_list
            
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
                   verbose: bool = True , 
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
            
        
        module_path = module.__module_file__()
        
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
    def pm2_kill(cls, name:str, verbose:bool = False):
        output_list = []
        pm2_list = cls.pm2_list()
        for module in pm2_list:
            if module.startswith(name):
                output_str = cls.run_command(f"pm2 delete {module}", verbose=verbose)
                output_list.append(output_str)
        return output_list
    @classmethod
    def pm2_restart(cls, name:str, verbose:bool=False):
        return cls.run_command(f"pm2 restart {name}", verbose=verbose)
        stdout = cls.run_command(f"pm2 status")
        if verbose:
            cls.print(stdout, color='orange')
            
            
    def restart_self(self, mode:str='pm2'):
        assert hasattr(self, 'module_name'), 'self.module_name must be defined to restart'
        return self.restart(self.module_name)
    @classmethod
    def restart(cls, name:str = None, mode:str='pm2', verbose:bool = False):
        if name == None:
            name = cls.default_module_name()
        if mode == 'pm2':
            return cls.pm2_restart(name, verbose=verbose)
        elif mode == 'ray':
            return cls.ray_restart(name)
        else:
            raise Exception(f'mode {mode} not supported')
        

    @classmethod
    def pm2_status(cls, verbose=True):
        stdout = cls.run_command(f"pm2 status")
        if verbose:
            cls.print(stdout,color='green')
        return stdout


    @classmethod
    def pm2_logs(cls, module:str,verbose=True):
        return cls.run_command(f"pm2 logs {module}", verbose=verbose)

    @classmethod
    def argparse(cls):
        import argparse
        parser = argparse.ArgumentParser(description='Gradio API and Functions')
        parser.add_argument('-fn', '--fn', dest='function', help='run a function from the module', type=str, default="__init__")
        parser.add_argument('-kwargs', '--kwargs', dest='kwargs', help='key word arguments to the function', type=str, default="{}")  
        parser.add_argument('-args', '--args', dest='args', help='arguments to the function', type=str, default="[]")  
        args = parser.parse_args()
        cls.print(args, color='cyan')
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
    def module(cls,
               module: Any = None ,
               init_module:bool=False , 
               serve:bool=False, 
               **kwargs):
        '''
        Wraps a python class as a module
        '''
        
        if module is None:
            return cls.root_module()
        if isinstance(module, str):
            modules = cls.module_list()
            if module in modules:
                return cls.get_module(module,**kwargs)
            elif module in self.servers():
                return self.connect(module,**kwargs)
    

        # serve the module if the bool is True
        is_class = cls.is_class(module)
        cls.print(f'is_class: {is_class}')
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
            def __module_file__(cls): 
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
                return Module.get_functions(module)
            # class Module(Module):
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
                        allow_conflicts:bool=True):
        
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
                    cls.print(f'Warning: overriding function {b_fn_name} already exists in {a}', color='yellow')
            else:
                assert not hasattr(a, b_fn_name), f'function {b_fn_name} already exists in {a}'
                
            # get the function from the python object
            try: 
                b_fn = getattr(b, b_fn_name)
            except NotImplementedError as e:
                print(e)
            if callable(b_fn):
                setattr(a, b_fn_name, b_fn)  
                
        return a
   



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
        if not hasattr(cls, '__external_ip__'):
            self.__external_ip__ =  cls.get_external_ip(*args, **kwargs)
            
        return self.__external_ip__
        
    
    @classmethod
    def get_external_ip(cls, *args, **kwargs) ->str:
        return cls.import_object('commune.utils.network.get_external_ip')(*args, **kwargs)

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


    
    ### GPU LAND
    
    @classmethod
    def gpus(cls) -> List[int]:
        import torch
        available_gpus = [int(i) for i in range(torch.cuda.device_count())]
        return available_gpus
    
    @classmethod
    def cuda_available(cls) -> bool:
        import torch
        return torch.cuda.is_available()
    @classmethod
    def gpu_map(cls) -> Dict[int, Dict[str, float]]:
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
    
    gpu_info = gpu_memory_map = gpu_map
 
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
    def get_num_params(cls, model:'nn.Module' = None)->int:
        import numpy as np
        from torch import nn
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        num_params = sum([np.prod(p.size()) for p in model_parameters])
        return num_params

    @classmethod
    def get_tensor_size(cls, tensor:'torch.Tensor'):
        import torch
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
    def available_models(cls):
        return cls.get_module('model.transformer').models()
    
    def model_size(self, **kwargs ):
        return self.get_model_size(model=self, **kwargs)
    
    
    @classmethod
    def get_empty_model(cls, model):
        from transformers import  AutoModelForCausalLM, AutoModel, AutoConfig
        from accelerate import init_empty_weights
        print(f'loading config model from {model}...')
        model = cls.shortcuts.get(model, model)

        if isinstance(model, str):
            model_config = AutoConfig.from_pretrained(model)
            model_config_dict = model_config.to_dict()
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(model_config)
                
        return model
        
    @classmethod
    def get_model_size(cls, model, model_inflation_ratio: float = 1.0, keys:List[str]=None):
        
        if isinstance(model, str):
            model = cls.get_empty_model(model)
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
    
    
    @classmethod
    def whitelist_functions(cls, mode='sudo') -> List[str]:
        access_control = cls.get_json('access_control',default={})
        access_control['whitelist'] = access_control.get('whitelist', {})
        access_control['whitelist'][mode]= access_control['whitelist'].get(mode, [])
        whitelist_functions = access_control['whitelist'][mode]
        if len(whitelist_functions) == 0:
            whitelist_functions = cls.functions(include_module=True)
            access_control['whitelist'][mode] = whitelist_functions
        return access_control['whitelist'][mode]
    
    @classmethod
    def blacklist_functions(cls, mode='sudo') -> List[str]:
        access_control = cls.get_json('access_control',default={})
        access_control['blacklist'] = access_control.get('blacklist', {})
        access_control['blacklist'][mode]= access_control['blacklist'].get(mode, [])
        whitelist_functions = access_control['blacklist'][mode]
        if len(whitelist_functions) == 0:
            whitelist_functions = []
            access_control['blacklist'][mode] = whitelist_functions
        return access_control['blacklist'][mode]

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
    def test(cls, *args, **kwargs):
        self = cls(*args, **kwargs)
               
               
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
        assert isinstance(input, str)
        try:
            output_dict = json.loads(input)
        except json.JSONDecodeError as e:
            return input

        return output_dict
    
    
    def is_json_serializable(self, value):
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
            raise cls.get_key(*args, **kwargs)
            # return self.get_module('subspace').get_wallet(*args, **kwargs)  
               
    
    @classmethod
    def get_key(cls, *args,mode='subspace', **kwargs) -> None:

        if mode == 'subspace':
            return cls.get_module('subspace.key')(*args, **kwargs)
        if mode == 'substrate':
            return cls.get_module(f'web3.account.substrate')(*args, **kwargs)
        elif mode == 'evm':
            return cls.get_module('web3.account.evm')(*args, **kwargs)
        elif  mode == 'aes':
            return cls.get_module('crypto.key.aes')(*args, **kwargs)
        else:
            raise ValueError('Invalid mode for key')
        
            
    @classmethod
    def hash(cls, 
             data: Union[str, bytes], 
             mode: str = 'keccak', 
             **kwargs) -> bytes:
        if not hasattr(cls, 'hash_module'):
            cls.hash_module = cls.get_module('crypto.hash')()
        return cls.hash_module(data, mode=mode, **kwargs)
    
    @classmethod
    def decrypt(cls, data: str, password= None) -> Any:
        key = cls.get_key(mode='aes', key=password)
        data = key.decrypt(data)
        
        return cls.str2python(data)



    @classmethod
    def encrypt(cls, data: Union[str, bytes], password: str = None) -> bytes:
        data = cls.python2str(data)
        key = cls.get_key(mode='aes', key=password)
        return key.encrypt(data)
    @classmethod
    async def async_call(cls, module:str, fn: str ,  *args, **kwargs) -> None:
        # call a module
        module = await cls.async_connect(module)
        fn = getattr(module, fn)
        if inspect.iscoroutinefunction(fn):
            return await fn(*args, **kwargs)
        elif callable(fn):
            return fn(*args, **kwargs)
        else:
            return fn


    @classmethod
    def call(cls,  *args, **kwargs) -> None:
        loop = cls.get_event_loop()
        return loop.run_until_complete(cls.async_call(*args, **kwargs))
    
        
    @classmethod
    def call_pool(cls, module:str, fn: str ,  *args, **kwargs) -> None:
        # call a module
        
        module_pool = list(cls.namespace(module).keys())
        jobs = []
        for module in module_pool:
            jobs += [cls.async_call(module, fn, *args, **kwargs)]
        
        loop = cls.get_event_loop()
        module_results = loop.run_until_complete(asyncio.gather(*jobs))
        
        results_map = {}
        for module_name, module_result in zip(module_pool, module_results):
            results_map[module_name] = module_result
            
        return results_map

    pool = call_pool
            
    def resolve_key(self, key: str) -> str:
        if key == None:
            if not hasattr(self, 'key'):
                self.set_key(key)
            key = self.key
        elif isinstance(key, str):
            key = self.get_key(key)
            
        return key  
                
                
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
    
    def verify(self, data:dict  = None, key: str = None) -> bool:
        key = self.resolve_key(key)
        
        return key.verify(data)
        
    
    def get_auth(self, 
                 data:dict  = None, 
                 key: str = None,
                 return_dict:bool = True,
                 encrypt: bool = False,
                 ) -> dict:
        
        key = self.resolve_key(key)
        if data == None:
            # default data  
            data = {'utc_timestamp': self.time()}

        sig_dict = key.sign(data, return_dict=return_dict)

        if encrypt:
            sig_dict['data'] = key.encrypt(sig_dict['data'])

        sig_dict['encrypted'] = encrypt
            
        
        
        return sig_dict
    
    
    @classmethod
    def start(cls, *args, **kwargs):
        return cls(*args, **kwargs)
    

      
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
        
        assert fn in self.whitelist_functions(), f'AuthFail: Function {fn} not in whitelist'
        assert fn not in self.blacklist_functions(), f'AuthFail: Function {fn} in blacklist'
        
        # # check if user is in the list of users
        # is_user = self.is_user(auth)
        
        # # check the data
        # data = auth['data']
        
        # expiration  = self.time() - staleness
        # is_user = bool(data['timestamp'] > expiration)
            
        return True
        
        
        
    def is_user(self, auth: dict = None) -> bool:
        assert isinstance(auth, dict), 'Auth must be provided'
        for k in ['signature', 'data', 'public_key']:
            assert k in auth, f'Auth must have key {k}'
            
        user_address = self.verify(user, auth)
        if not hasattr(self, 'users'):
            self.users = {}
        return bool(user_address in self.users)
        
        
    
    def add_user(self, user: str = None, info: dict = None):
        if not hasattr(self, 'users'):
            self.users = {}
        if info == None:
            info = {}
        info.update({'timestamp': self.time()})
        self.users[user] = info
    
    def rm_user(user: str = None):
        if not hasattr(self, 'users'):
            self.users = {}
        self.users.pop(user, None)
        
        
        
    @property
    def users(self):
        if not hasattr(self, '_users'):
            self._users = {}
        return self._users
    
    @users.setter
    def users(self, value: dict):
        self._users = value
        
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
    def deploy_fleet(cls, modules=None):
        if isinstance(modules, str):
            modules = [modules]
        modules = modules if modules else ['model.transformer', 'dataset.text.huggingface', 'dataset.text.bittensor']
            
 
        for module in modules:
            
            module_class = cls.get_module(module)
            assert hasattr(module_class,'deploy_fleet'), f'{module} does not have a deploy_fleet method'
            cls.get_module(module).deploy_fleet()
    # # ARRAY2BYTES
    # @classmethod
    # def array2bytes(self, data: np.array) -> bytes:
    #     if isinstance(data, np.array):
    #         data = data.astype(np.float64)
    #     return data.tobytes()
    
    
    # SUBSPACE BABY 
    @classmethod
    def subtensor(self, *args, **kwargs):
        import bittensor
        return bittensor.subtensor(*args, **kwargs)
    
    @classmethod
    def subspace(cls, *args, **kwargs):
        subspace = cls.get_module('subspace')(*args, **kwargs)
        return subspace
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

    key = None 
    @classmethod
    def client(cls, *args, **kwargs) -> 'Client':
        return cls.import_object('commune.server.Client')(*args, **kwargs)
    
    @classmethod
    def server(cls, *args, **kwargs) -> 'Server':
        return cls.import_object('commune.server.Server')(*args, **kwargs)
    
    @classmethod
    def serializer(cls, *args, **kwargs) -> 'Serializer':
        return cls.import_object('commune.server.Serializer')(*args, **kwargs)

    @classmethod
    def copy(cls, data: Any) -> Any:
        import copy
        return copy.deepcopy(data)
    
    @classmethod
    def launchpad(cls):
        return cls.import_object('commune.launchpad.Launchpad')()
    @classmethod
    def determine_type(cls, x):
        if x.lower() == 'null':
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
    default_port_range = [50050, 50150]
    @classmethod
    def set_port_range(cls, *port_range: list):
        if len(port_range) ==0 :
            port_range = cls.default_port_range
        assert len(port_range) == 2, 'Port range must be a list of two integers'        
        for port in port_range:
            assert isinstance(port, int), f'Port {port} range must be a list of integers'
        assert port_range[0] < port_range[1], 'Port range must be a list of integers'
                
        data = dict(port_range =port_range)
        cls.put_json('port_range', data, root=True)
        cls.port_range = data['port_range']
        return data['port_range']
    
    
    
    
    @classmethod
    def get_port_range(cls, port_range: list = None) -> list:

        if not Module.exists('port_range'):
            cls.set_port_range(port_range)
            
        if port_range == None:
            port_range = Module.get_json('port_range')['port_range']
            
        if len(port_range) == 0:
            port_range = cls.default_port_range
            
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
    def ansible(cls, *args, fn='shell', **kwargs):
        ansible_module = cls.get_module('ansible')()
        return getattr(ansible_module, fn)(*args, **kwargs)
        
    @classmethod
    def add_peer(cls, *args, **kwargs):
        loop = cls.get_event_loop()
        return loop.run_until_complete(cls.async_add_peer(*args, **kwargs))
       
    add_peers = add_peer
    
    
    @classmethod
    async def async_add_peer(cls, *peer_addresses,timeout:int=1):
        
        if len(peer_addresses) == 1:
            if isinstance(peer_addresses[0], list):
                peer_addresses = peer_addresses[0]
        
        peer_registry = await Module.async_get_json('peer_registry', default={})

        if len(peer_addresses) == 0:
            peer_addresses = cls.boot_peers
            
        peer_jobs = []
        # get the server registry for each peer
        for peer_address in peer_addresses:
            peer_job = cls.async_call(module=peer_address, fn='namespace', timeout=timeout)
            peer_jobs.append(peer_job)
            
        # wait for all jobs to complete
        peer_namespaces = await asyncio.gather(*peer_jobs)
        
        
        cls.print(f'Adding peer  to registry, {peer_addresses}')

        #  add each peer to the registry
        for peer_address, peer_namespace in zip(peer_addresses, peer_namespaces):
            # TODO : ADD PEER NAME
            peer_registry[peer_address] = dict(name=None, 
                                               namespace=peer_namespace,
                                               address = peer_address)
            
        await Module.async_put_json('peer_registry', peer_registry)
        
        return peer_registry
    

    @staticmethod
    def is_number(value):
        try:
            int(value)
        except ValueError:
            return False
        return True

        

    
    @classmethod
    def rm_peer(cls, peer_address: str):
        peer_registry = Module.get_json('peer_registry', default={})
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
            
      
    @classmethod
    def update_peers(cls, peers: list = None):
        if peers == None:
            peers = cls.peers()
            
        cls.print(peers)
        cls.add_peers(peers)
        
        
        
    @classmethod
    def update(cls, 

               min_update_delay: int = 60,
               force:bool = False, 
               verbose:bool = True,
               
               ):
        update_info = Module.get_json('update_info', default={})
        last_update_time = update_info.get('last_update', 0)
        current_time = cls.time()
        update_delay = (current_time - last_update_time)
        update_bool = (update_delay > min_update_delay) or force
        if update_bool:
            if verbose:
                cls.print('Updating server registry')
            cls.update_peers()
            cls.update_local_namespace()
            update_info['last_update'] = cls.time()
            cls.put_json('update_info', update_info, root=True)
            
        else:
            if verbose:
                cls.print(f'Server registry is up to date, skipping update, last update was {update_delay}s ', )
            
           
        cls.root_module()

        
        
    @classmethod
    def peer_registry(cls, update: bool = False):
        if update:
            cls.update_peers()
        return Module.get_json('peer_registry', default={})
    
    

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
    def put_text(cls, path:str, text:str) -> None:
        # Get the absolute path of the file
        path = os.path.abspath(path)

        # Write the text to the file
        with open(path, 'w') as file:
            file.write(text)

    @classmethod
    def get_text(cls, path:str) -> None:
        # Get the absolute path of the file
        path = os.path.abspath(path)

        # Read the contents of the file
        with open(path, 'r') as file:
            content =  file.read()
            
        return content


    @classmethod
    def get_file_class(cls, path=None, ignore_error:bool = False):
        if cls == Module:
            return 'Module'
        # Get the file path of the module
        module_file_path = os.path.abspath(module.__file__)

        # Read the contents of the file
        with open(module_file_path, 'r') as file:
            file_contents = file.read()
            
        file_content = cls.get_file_contents(obj)
        for k in ['(commune.Module)']:
            if k in file_content:
                return file_content.split(k)[0].split('class')[-1]
        if ignore_error:
            return None
        else: 
            raise ValueError('Could not find class name in file')
        
 
    @classmethod
    def free_gpu_memory(cls, 
                     gpus:List[int] = None,
                     max_gpu_ratio: float = 0.9 ,
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
        if gpus == None :
            gpus = cls.gpus()
            
        if isinstance(gpus, int):
            gpus = [gpus]
        
        gpus = [int(gpu) for gpu in gpus] 
    
        for gpu_id, gpu_info in cls.gpu_map().items():
            if int(gpu_id) in gpus:
                
                gpu_memory = min(gpu_info['free'], gpu_info['total']*max_gpu_ratio)
                free_gpu_memory[gpu_id] = cls.copy(gpu_memory /scale)
                if fmt == '%':
                    free_gpu_memory[gpu_id] = (free_gpu_memory[gpu_id]/gpu_info['total']) * 100
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
    
    



    @classmethod
    def max_gpu_memory(cls, model:str, max_gpu_ratio:float=0.8, fmt='b', model_inflation_ratio:float=1.2):
        
        if type(model) in [float, int]:
            model_size = model
        else:
            model_size = cls.get_model_size(model, model_inflation_ratio=model_inflation_ratio)
        free_gpu_memory = cls.free_gpu_memory(fmt=fmt, max_gpu_ratio=max_gpu_ratio)
        gpus = list(free_gpu_memory.keys()) 
        cls.print(free_gpu_memory)
        total_gpu_memory = sum(free_gpu_memory.values())
        assert model_size < total_gpu_memory, f'model size {model_size} is larger than total gpu memory {total_gpu_memory}, over gpus {gpus}'
        unallocated_model_memory = model_size
        # max_memory = {}
        max_memory = {}
        
        
        while unallocated_model_memory > 0:
            most_free_gpu, most_free_gpu_memory = cls.most_free_gpu(free_gpu_memory=deepcopy(free_gpu_memory), return_tuple=True)
            assert most_free_gpu not in max_memory 
            cls.print(most_free_gpu, most_free_gpu_memory)
            allocated_memory = min(most_free_gpu_memory, unallocated_model_memory)
            unallocated_model_memory -= allocated_memory
            max_memory[most_free_gpu] = allocated_memory
            free_gpu_memory[most_free_gpu] -= allocated_memory
            
        
        return max_memory
            

    @classmethod
    def free_gpus(cls, *args, **kwargs):
        return cls.free_gpu_memory(*args, **kwargs)
    
    
    @classmethod
    def total_free_gpu_memory(cls, gpus = None, max_gpu_ratio=0.2, fmt='b'):
        free_gpu_memory = cls.free_gpu_memory(gpus=gpus, max_gpu_ratio=max_gpu_ratio, fmt=fmt)
        total_free_memory = sum(free_gpu_memory.values())
        return total_free_memory
    
    total_free_gpus = total_free_gpu_memory
    
    @classmethod
    def help(cls, fn:str = None):
        schema = cls.schema() 
        if fn != None: 
            return {k:v for k,v in schema.items() if fn in k}
            
            
        return schema
    
    @classmethod
    def git_pull(cls, stash: bool = True):
        cmd = 'git pull'
        if stash:
            cmd = 'git stash; '+ cmd
        return cls.run_command(cmd)
    
    @classmethod
    def pip(cls, library, version = None, upgrade = False):

        if upgrade:
            cmd = f'pip install --upgrade {library}'
    
        elif version != None:
            cmd = f'pip install {library}=={version}'
    
        return cls.run_command(cmd, verbose=True)
    

    @classmethod
    def pip_upgrade(cls, library, version = None):
        cls.pip(library, version = version, upgrade = True)


    @staticmethod
    def symlink( src, dst):
        os.symlink(src, dst)

        import os

        src = '/usr/bin/python'
        dst = '/tmp/python'

        # This creates a symbolic link on python in tmp directory
        os.symlink(src, dst)
        # @classmethod
        
        return dst
    
    def ping(self):
        class_name = self.__class__.__name__
        address = self.address
        return f'pong from {class_name} at {address}'
    
    @classmethod
    def init_empty_weights(cls, model_class,  *args, fn=None, **kwargs):
        from accelerate import init_empty_weights
        with init_empty_weights():
            if isinstance(fn, str):
                if hasattr(model_class, fn):
                    model = getattr(model_class, fn)(*args, **kwargs)
            model = model_class(*args, **kwargs)
        
        return model

    @classmethod
    def hash(cls, *args, **kwargs):
        return cls.module('crypto.hash').hash(*args,**kwargs)

    @classmethod
    def merge_classes(cls, *classes):
        # Get the dictionary of attributes for each class
        if len(classes) == 1:
            classes = [cls, classes[0]]
        assert len(classes) == 2, 'Only two classes can be merged'
        attrs1 = class1.__dict__
        attrs2 = class2.__dict__

        # Merge the two dictionaries
        merged_attrs = {**attrs1, **attrs2}

        # Define the __init__ method for the merged class
        def __init__(self, *args, **kwargs):
            class1.__init__(self, *args, **kwargs)

        # Add the __init__ method to the merged attributes
        merged_attrs['__init__'] = __init__

        # Define additional class methods for the merged class
        @classmethod
        def class_method(cls):
            print("This is a class method of the merged class")

        # Add the additional class methods to the merged attributes
        merged_attrs['class_method'] = class_method

        # Create a new class
        class_merged = type('MergedClass', (class1, class2), merged_attrs)

        # Return the merged class
        return class_merged

    @classmethod
    def signin(cls, 
               user:str = 'Alice on Chains', 
               password:str = 'Obama is a lizard',
               seperator:str = '::',
               mode: str = 'subspace'):
        cls.user = user
        
        # For the lazy fucks who want to us user<seperator>password
        user_splits = user.split(seperator)
        if len(user_splits)>1:
            # take the last chunk as the password , 
            # NOTE: DO NOT INCLUDE "::" in your username
            user = user_splits[0]
            password = seperator.join(user_splits[1:])
            
        key_seed = seperator.join([user, password])
        cls.key = cls.get_key(key_seed ,mode=mode)
        cls.address = cls.key.ss58_address
        return cls.key
        

if __name__ == "__main__":
    Module.run()
    

# pm2 start commune/module/module.py --name module --interpreter python3