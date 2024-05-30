import os
import inspect
import concurrent
import threading
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
import nest_asyncio
import random

nest_asyncio.apply()

# AGI BEGINS 
class c:
    whitelist = ['info',
                'schema',
                'server_name',
                'is_admin',
                'namespace',
                'whitelist', 
                'blacklist',
                'fns'] # whitelist of helper functions to load
    cost = 1
    description = """This is a module"""
    base_module = 'module' # the base module
    encrypted_prefix = 'ENCRYPTED' # the prefix for encrypted values
    giturl = git_url = 'https://github.com/commune-ai/commune.git' # tge gutg
    homepath = home_path = os.path.expanduser('~') # the home path
    root_module_class = 'c' # WE REPLACE THIS THIS Module at the end, kindof odd, i know, ill fix it fam, chill out dawg, i didnt sleep with your girl
    default_port_range = [50050, 50150] # the port range between 50050 and 50150
    default_ip = local_ip = loopback = '0.0.0.0'
    address = '0.0.0.0:8888' # the address of the server (default)
    root_path  = root  = os.path.dirname(os.path.dirname(__file__)) # the path to the root of the library
    libpath = lib_path = os.path.dirname(root_path) # the path to the library
    libname = lib_name = lib = root_path.split('/')[-1] # the name of the library
    datapath = os.path.join(root_path, 'data') # the path to the data folder
    modules_path = os.path.join(lib_path, 'modules') # the path to the modules folder
    repo_path  = os.path.dirname(root_path) # the path to the repo
    console = Console() # the consolve
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
    def pwd(cls):
        pwd = os.getenv('PWD') # the current wor king directory from the process starts 
        return pwd

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
        config =  self.config(config=config,kwargs=kwargs, to_munch=to_munch)

        # add the config attributes to the class (via munch -> dict -> class )
        if add_attributes:
            self.__dict__.update(self.munch2dict(config))

        self.config = config 
        self.kwargs = kwargs

        if save_config:
            self.save_config(config=config)
            
        return self.config


    @property
    def key(self):
        if not hasattr(self, '_key'):
            self._key = c.get_key(self.server_name, create_if_not_exists=True)
        return self._key
    
    @key.setter
    def key(self, key: 'Key'):
        self._key = c.get_key(key, create_if_not_exists=True)
        return self._key
    
    @classmethod
    def call(cls, *args, **kwargs) -> None:
        return c.module('client').call( *args, **kwargs)

    @classmethod
    async def async_call(cls, *args,**kwargs):
        return c.call(*args, **kwargs)


    @classmethod
    def call_search(cls,*args, **kwargs) -> None:
        return c.m('client').call_search(*args, **kwargs)

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
        # odd case where the module is a module in streamlit
        obj = cls.resolve_module(obj)
        module_path =  inspect.getfile(obj)
        # convert into simple
        if simple:
            module_path = cls.path2simple(module_path)
        return module_path
    

    @classmethod
    def filepath(cls, obj=None) -> str:
        '''
        removes the PWD with respect to where module.py is located
        '''
        obj = cls.resolve_module(obj)
        module_path =  inspect.getfile(obj)
        return module_path
    
    @classmethod
    def gitbranch(cls) -> str:
        return c.cmd('git branch').split('\n')[0].replace('* ', '')

    @classmethod
    def gitpath(cls ,root='https://github.com/commune-ai/commune/tree/'):
        branch = cls.gitbranch()
        root = root + branch + '/'
        filepath = cls.filepath().replace(c.repo_path + '/', '')
        return root + filepath


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
    folderpath = dirname = dirpath

    @classmethod
    def dlogs(cls, *args, **kwargs):
        '''
        logs of the docker contianer
        '''
        return c.module('docker').logs(*args, **kwargs)

    @classmethod
    def images(cls, *args, **kwargs):
        """
        images
        """
        return c.module('docker').images(*args, **kwargs)
    
    @classmethod
    def module_path(cls, simple:bool=True) -> str:
        # get the module path
        
        obj = cls.resolve_module(cls)
        module_path =  inspect.getfile(obj)
        # convert into simple
        if simple:
            module_path = cls.path2simple(module_path)
        return module_path
    
    path  = name = module_name =  module_path
    
    @classmethod
    def module_class(cls) -> str:
        return cls.__name__
    @classmethod
    def class_name(cls, obj= None) -> str:
        obj = obj if obj != None else cls
        return obj.__name__
    classname = class_name
    @classmethod
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
        minimal_config = {'module': cls.__name__}
        return minimal_config


    @classmethod
    def config_path(cls) -> str:
        return cls.get_module_path(simple=False).replace('.py', '.yaml')

    @classmethod
    def dict2munch(cls, x:dict, recursive:bool=True)-> Munch:
        '''
        Turn dictionary into Munch
        '''
        if isinstance(x, dict):
            for k,v in x.items():
                if isinstance(v, dict) and recursive:
                    x[k] = c.dict2munch(v)
            x = Munch(x)
        return x 

    @classmethod
    def munch2dict(cls, x:Munch, recursive:bool=True)-> dict:
        '''
        Turn munch object  into dictionary
        '''
        if isinstance(x, Munch):
            x = dict(x)
            for k,v in x.items():
                if isinstance(v, Munch) and recursive:
                    x[k] = c.munch2dict(v)

        return x 

    @classmethod
    def munch(cls, x:Dict) -> Munch:
        '''
        Converts a dict to a munch
        '''
        return cls.dict2munch(x)
    
    @classmethod
    def load_yaml(cls, path:str=None, default={}, **kwargs) -> Dict:
        '''f
        Loads a yaml file
        '''
        import yaml
        path = cls.resolve_path(path)

        try:
            with open(path, 'r') as file:
                data = yaml.safe_load(file)
        except:
            data = default

        return data
        
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
    def fn_code(cls,fn:str, 
                detail:bool=False, 
                seperator: str = '/'
                ) -> str:
        '''
        Returns the code of a function
        '''
        if isinstance(fn, str):
            if seperator in fn:
                module_path, fn = fn.split(seperator)
                module = c.module(module_path)
                fn = getattr(module, fn)
            else:
                fn = getattr(cls, fn)
        
        
        code_text = inspect.getsource(fn)
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
    
    fncode = fn_code
    @classmethod
    def sandbox(cls):
        c.cmd(f'python3 {c.libpath}/sandbox.py', verbose=True)
        return 
    sand = sandbox

    @classmethod
    def save_yaml(cls, path:str,  data: dict) -> Dict:
        '''
        Loads a yaml file
        '''
        path = cls.resolve_path(path)
            
        from commune.utils.dict import save_yaml
        if isinstance(data, Munch):
            data = cls.munch2dict(deepcopy(data))
            
        return save_yaml(data=data , path=path)
    
    put_yaml = save_yaml

    
    @classmethod
    def config_path(cls) -> str:
        path = cls.module_file().replace('.py', '.yaml')
        return path
    
    @classmethod
    def load_config(cls, path:str=None, to_munch:bool = False) -> Union[Munch, Dict]:
        '''
        Args:
            path: The path to the config file
            to_munch: If true, then convert the config to a munch
        '''

        if path == None: 
            path = cls.config_path()
        else:
            module_tree = c.tree()
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

    encrypted_prefix = 'ENCRYPTED'
    @classmethod
    def encrypt_file(cls, path:str, password=None, key=None,) -> str:
        key = c.get_key(key)
        text = cls.get_text(path)
        r = key.encrypt(text, password=password)
        return cls.put_text(path, r)
        
    @classmethod
    def decrypt_file(cls, path:str, key=None, password=None, **kwargs) -> str:
        key = c.get_key(key)
        text = cls.get_text(path)
        r = key.decrypt(text, password=password, key=key, **kwargs)
        return cls.put_text(path, r)


    def is_encrypted_path(self, path:str, prefix=encrypted_prefix) -> bool:
        '''
        Encrypts the path
        '''
        path = self.resolve_path(path)
        text = c.get_text(path)
        return text.startswith(prefix)

    @classmethod
    def put(cls, 
            k: str, 
            v: Any,  
            mode: bool = 'json',
            encrypt: bool = False, 
            verbose: bool = False, 
            password: str = None, **kwargs) -> Any:
        '''
        Puts a value in the config
        '''
        encrypt = encrypt or password != None
        
        if encrypt or password != None:
            v = c.encrypt(v, password=password)

        if not c.jsonable(v):
            v = c.serialize(v)    
        
        data = {'data': v, 'encrypted': encrypt, 'timestamp': c.timestamp()}            
        
        # default json 
        getattr(cls,f'put_{mode}')(k, data)

        if verbose:
            c.print(f'put {k} = {v}')

        data_size = c.sizeof(v)
    
        return {'k': k, 'data_size': data_size, 'encrypted': encrypt, 'timestamp': c.timestamp()}
    
    @classmethod
    def get(cls,
            k:str, 
            default: Any=None, 
            mode:str = 'json',
            max_age:str = None,
            cache :bool = False,
            full :bool = False,
            key: 'Key' = None,
            update :bool = False,
            password : str = None,
            **kwargs) -> Any:
        
        '''
        Puts a value in sthe config, with the option to encrypt it

        Return the value
        '''
        if cache:
            if k in cls.cache:
                return cls.cache[k]

        data = getattr(cls, f'get_{mode}')(k,default=default, **kwargs)
            

        if password != None:
            assert data['encrypted'] , f'{k} is not encrypted'
            data['data'] = c.decrypt(data['data'], password=password, key=key)

        data = data or default
        
        if isinstance(data, dict):
            if update:
                max_age = 0
            if max_age != None:
                timestamp = data.get('timestamp', None)
                if timestamp != None:
                    age = int(c.time() - timestamp)
                    if age > max_age: # if the age is greater than the max age
                        c.print(f'{k} is too old ({age} > {max_age})', color='red')
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
    setc = putc
    @classmethod
    def rmc(cls, k, password=None) -> Munch:
        '''
        Saves the config to a yaml file
        '''
        config = cls.config()
        c.dict_rm(config, k)
        cls.save_config(config=config)
   
    delc = rmc
    

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
    def getc(cls, key, default= None, password=None) -> Any:
        '''
        Saves the config to a yaml file
        '''

        config = cls.config()
        data = cls.dict_get(config, key, default)
        
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
        
        assert isinstance(config, dict), f'config must be a dict, not {config}'

        config = cls.save_yaml(data=config , path=path)

        return config
    
    
    def config_exists(self, path:str=None) -> bool:
        '''
        Returns true if the config exists
        '''
        path = path if path else self.config_path()
        return self.path_exists(path)

    @classmethod
    def config(cls, 
                   config:dict = None,
                   kwargs:dict=None, 
                   to_munch:bool = True) -> Munch:
        '''
        Set the config as well as its local params
        '''
        # THIS LOADS A YAML IF IT EXIST, OR IT USES THE INIT KWARGS IF THERE IS NO YAML
        if cls.has_config():
            default_config = cls.load_config(to_munch=False)
        else: 
            default_config = cls.init_kwargs()

        if config == None:
            config =  default_config
        elif isinstance(config, str):
            config = cls.load_config(path=config)
            assert isinstance(config, dict), f'config must be a dict, not {type(config)}'
        
        if isinstance(config, dict):
            config = {**default_config, **config}
        else:
            raise ValueError(f'config must be a dict, str or None, not {type(config)}')
                
        # SET THE CONFIG FROM THE KWARGS, FOR NESTED FIELDS USE THE DOT NOTATION, 
        # for example  model.name=bert is the same as config[model][name]=bert
        # merge kwargs with itself (CAUTION THIS DOES NOT WORK IF KWARGS WAS MEANT TO BE A VARIABLE LOL)

        config = c.locals2kwargs(config)

        if kwargs != None:
            kwargs = c.locals2kwargs(kwargs)
            for k,v in kwargs.items():
                cls.dict_put(config,k,v )
        #  add the config after in case the config has a config attribute lol
        if to_munch:
            config = cls.dict2munch(config)


        return config
 
    cfg = get_config = config

    @classmethod
    def flatten_dict(cls, x = {'a': {'b': 1, 'c': {'d': 2, 'e': 3}, 'f': 4}}):
        from commune.utils.dict import deep2flat
        return deep2flat(x)


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
    
    @classmethod
    def module_info(cls, *args, **kwargs):
        return c.module('subspace')().module_info(*args, **kwargs)
    
    minfo = module_info

    @classmethod
    def pwd2key(cls, *args, **kwargs):
        return c.module('key').pwd2key(*args, **kwargs)

    password2key = pwd2key        
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
    def start_app(cls,
           module:str = 'module', 
           fn='app', 
           port=8501, 
           public:bool = False, 
           remote:bool = False):
        kwargs = c.locals2kwargs(locals())
        return c.module('app')().start(**kwargs)
    
    app = start_app
 
    @staticmethod
    def st_load_css(*args, **kwargs):
        c.module('streamlit').load_css(*args, **kwargs)

    @classmethod
    def rcmd(cls, *args, **kwargs):
        return c.module('remote').cmd(*args, **kwargs)
    @classmethod
    def rpwd(cls, *args, **kwargs):
        return c.module('remote')().pwd(*args, **kwargs)
    pw = rpwd
    @classmethod
    def cmd(cls, *args,**kwargs):
        return c.module('os').cmd( *args, **kwargs)
    run_command = shell = cmd 

 

    @classmethod
    def sys_path(cls, *args, **kwargs):
        return c.module('os').sys_path(*args, **kwargs)

    @classmethod
    def import_module(cls, import_path:str) -> 'Object':
        from importlib import import_module
        try:
            return import_module(import_path)
        except Exception as e:
            import sys
            sys.path.append(c.pwd())
            return import_module(import_path)
        
    def can_import_module(self, module:str) -> bool:
        '''
        Returns true if the module is valid
        '''
        try:
            c.import_module(module)
            return True
        except:
            return False

    @classmethod
    def import_object(cls, key:str, verbose: bool = False)-> Any:
        '''
        Import an object from a string with the format of {module_path}.{object}
        Examples: import_object("torch.nn"): imports nn from torch
        '''
        module = '.'.join(key.split('.')[:-1])
        object_name = key.split('.')[-1]
        if verbose:
            c.print(f'Importing {object_name} from {module}')
    
        obj =  getattr(c.import_module(module), object_name)
        return obj
    
    imp = get_object = importobj = import_object



    @classmethod
    def module_exists(cls, module:str, **kwargs) -> bool:
        '''
        Returns true if the module exists
        '''
        return module in c.modules(**kwargs)


    
    @classmethod
    def modules(cls, search=None, mode='local', tree='commune', **kwargs)-> List[str]:
        if any([str(k) in ['subspace', 's'] for k in [mode, search]]):
            module_list = c.module('subspace')().modules(search=search, **kwargs)
        else:
            module_list = list(c.tree(search=search, tree=tree, **kwargs).keys())
            if search != None:
                module_list = [m for m in module_list if search in m]
        return module_list


    def mean(self, x:list=[0,1,2,3,4,5,6,7,8,9,10]):
        if not isinstance(x, list):
            x = list(x)
        return sum(x) / len(x)
    
    def median(self, x:list=[0,1,2,3,4,5,6,7,8,9,10]):
        if not isinstance(x, list):
            x = list(x)
        x = sorted(x)
        n = len(x)
        if n % 2 == 0:
            return (x[n//2] + x[n//2 - 1]) / 2
        else:
            return x[n//2]
    
    def stdev(self, x:list= [0,1,2,3,4,5,6,7,8,9,10], p=2):
        if not isinstance(x, list):
            x = list(x)
        mean = c.mean(x)
        return (sum([(i - mean)**p for i in x]) / len(x))**(1/p)
    
    # def test_stats(self, x:list):c
    #     mean = self.mean(x)
    #     stdev = self.stdev(x)
    #     return {'mean': mean, 'stdev': stdev}


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
        return os.makedirs(*args, **kwargs)

    @classmethod
    def resolve_path(cls, path:str = None, extension:Optional[str]= None, file_type:str = 'json'):
        '''
        ### Documentation for `resolve_path` class method
        
        #### Purpose:
        The `resolve_path` method is a class method designed to process and resolve file and directory paths based on various inputs and conditions. This method is useful for preparing file paths for operations such as reading, writing, and manipulation.
        
        #### Parameters:
        - `path` (str, optional): The initial path to be resolved. If not provided, a temporary directory path will be returned.
        - `extension` (Optional[str], optional): The file extension to append to the path if necessary. Defaults to None.
        - `root` (bool, optional): A flag to determine whether the path should be resolved in relation to the root directory. Defaults to False.
        - `file_type` (str, optional): The default file type/extension to append if the `path` does not exist but appending the file type results in a valid path. Defaults to 'json'.
        
        #### Behavior:
        - If `path` is not provided, the method returns a path to a temporary directory.
        - If `path` starts with '/', it is returned as is.
        - If `path` starts with '~/', it is expanded to the user’s home directory.
        - If `path` starts with './', it is resolved to an absolute path.
        - If `path` does not fall under the above conditions, it is treated as a relative path. If `root` is True, it is resolved relative to the root temp directory; otherwise, relative to the class's temp directory.
        - If `path` is a relative path and does not contain the temp directory, the method joins `path` with the appropriate temp directory.
        - If `path` does not exist as a directory and an `extension` is provided, the extension is appended to `path`.
        - If `path` does not exist but appending the `file_type` results in an existing path, the `file_type` is appended.
        - The parent directory of `path` is created if it does not exist, avoiding any errors when the path is accessed later.
        
        #### Returns:
        - `str`: The resolved and potentially created path, ensuring it is ready for further file operations. 
        
        #### Example Usage:
        ```python
        # Resolve a path in relation to the class's temporary directory
        file_path = MyClassName.resolve_path('data/subfolder/file', extension='txt')
        
        # Resolve a path in relation to the root temporary directory
        root_file_path = MyClassName.resolve_path('configs/settings'
        ```
        
        #### Notes:
        - This method relies on the `os` module to perform path manipulations and checks.
        - This method is versatile and can handle various input path formats, simplifying file path resolution in the class's context.
        '''
    
        if path == None:
            return cls.storage_dir()
        


        if path.startswith('/'):
            path = path
        elif path.startswith('~/'):
            path =  os.path.expanduser(path)
        elif path.startswith('.'):
            path = os.path.abspath(path)
        else:
            # if it is a relative path, then it is relative to the module path
            # ex: 'data' -> '.commune/path_module/data'
            storage_dir = cls.storage_dir()

            if storage_dir not in path:
                path = os.path.join(storage_dir, path)
            if not os.path.isdir(path):
                if extension != None and extension != path.split('.')[-1]:
                    path = path + '.' + extension
            
        if not os.path.exists(path) and os.path.exists(path + f'.{file_type}'):
            path = path + f'.{file_type}' 

        dirpath = os.path.dirname(path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)      
                 
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
    def random_int(start_value=100, end_value=None):
        if end_value == None: 
            end_value = start_value
            start_value, end_value = 0 , start_value
        
        assert start_value != None, 'start_value must be provided'
        assert end_value != None, 'end_value must be provided'
        return random.randint(start_value, end_value)
    
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
        port = None
        for port in ports: 
            if port in avoid_ports:
                continue
            
            if cls.port_available(port=port, ip=ip):
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

    def kill_port_range(self, start_port = None, end_port = None, timeout=5, n=0):
        if start_port != None and end_port != None:
            port_range = [start_port, end_port]
        else:
            port_range = c.port_range()
        
        if n > 0:
            port_range = [start_port, start_port + n]
        assert isinstance(port_range[0], int), 'port_range must be a list of ints'
        assert isinstance(port_range[1], int), 'port_range must be a list of ints'
        assert port_range[0] < port_range[1], 'port_range must be a list of ints'
        futures = []
        for port in range(*port_range):
            c.print(f'Killing port {port}', color='red')
            try:
                self.kill_port(port) 
            except Exception as e:
                c.print(f'Error: {e}', color='red')


    def check_used_ports(self, start_port = 8501, end_port = 8600, timeout=5):
        port_range = [start_port, end_port]
        used_ports = {}
        for port in range(*port_range):
            used_ports[port] = self.port_used(port)
        return used_ports
    @classmethod
    def kill_port(cls, port:int, mode='bash')-> str:
        return c.module('os').kill_port(port=port, mode=mode)

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
            c.print(result, verbose=verbose)
            progress.update(1)
            results_list += [result]
        servers = c.servers(network=network, update=True)
        new_n = len(servers)
        c.print(f'Killed {n - new_n} servers, with {n} remaining {servers}', color='red')
        return results_list
    @classmethod
    def path2simple(cls, *args, **kwargs ) -> str:
        return c.module('tree').path2simple(*args, **kwargs)  
    @classmethod
    def path2objectpath(cls, path:str = None, tree=None) -> str:
        return c.module('tree').path2objectpath(path=path, tree=tree)
    
    @classmethod
    def tree_paths(cls, *args, **kwargs) -> List[str]:
        return c.module('tree').tree_paths(*args, **kwargs)  
    
    
    @classmethod
    def tree_names(cls):
        return c.module('tree').tree_names()
    
    def file2classes(self, path:str = None, search:str = None, start_lines:int=2000):
        return self.find_python_classes(path=path, search=search, start_lines=start_lines)

    @classmethod
    def find_classes(cls, path):
        code = c.get_text(path)
        classes = []
        for line in code.split('\n'):
            if all([s in line for s in ['class ', '(', '):']]):
                classes.append(line.split('class ')[-1].split('(')[0].strip())
        return [c for c in classes]
    

    @classmethod
    def find_functions(cls, path):
        code = c.get_text(path)
        functions = []
        for line in code.split('\n'):
            if line.startswith('def '):
                if all([s in line for s in ['def ', '(', '):']]):
                    functions.append(line.split('def ')[-1].split('(')[0].strip())
        return functions
    
    @classmethod
    def find_python_classes(cls, path:str , class_index:int=0, search:str = None, start_lines:int=2000):
        import re
        path = cls.resolve_path(path)
        if os.path.isdir(path):
            file2classes = {}
            for f in c.glob(path):
                if f.endswith('.py'):
                    try:
                        file2classes[f] = cls.find_python_classes(f, class_index=class_index, search=search, start_lines=start_lines)
                    except Exception as e:
                        c.print(f'Error: {e}', color='red')
            return file2classes
        # read the contents of the Python script file
        python_script = cls.readlines(path, end_line = start_lines, resolve=False)
        class_names  = []
        lines = python_script.split('\n')

        # c.print(python_script)
        
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
    def url2text(cls, *args, **kwargs):
        return c.module('web').url2text(*args, **kwargs).text

    module_cache = {}
    @classmethod
    def get_module(cls, 
                   path:str = 'module',  
                   cache=True,
                   trials = 3,
                   tree = 'commune',
                   verbose = 0,
                   ) -> str:
        """
        params: 
            path: the path to the module
            cache: whether to cache the module
            tree: the tree to search for the module
        """
        path = path or 'module'
        module = None
        cache_key = f'{tree}_{path}'
        t0 = c.time()

        if cache and cache_key in c.module_cache:
            module = c.module_cache[cache_key]
            if module != None:
                return module

        try:
            module = c.simple2object(path)
        except Exception as e:
            raise e
            if trials == 0:
                raise Exception(f'Could not find {path} in {c.modules(path)} modules')
            c.print(f'Could not find {path} in {c.modules(path)} modules, so we are updating the tree', color='red')
            module = c.get_module(path, cache=cache , verbose=verbose, trials=trials-1)                
        if cache:
            c.module_cache[cache_key] = module

        if verbose:
            c.print(f'Loaded {path} in {c.time() - t0} seconds', color='green')

        return module

    @classmethod
    def is_dir_module(cls, path:str) -> bool:
        """
        determine if the path is a module
        """
        filepath = cls.simple2path(path)
        if path.replace('.', '/') + '/' in filepath:
            return True
        if ('modules/' + path.replace('.', '/')) in filepath:
            return True
        return False
    

    @classmethod
    def timefn(cls, fn, *args, trials=1, **kwargs):
        if trials > 1:
            responses = []
            for i in range(trials):
                responses += [cls.timefn(fn, *args, trials=1, **kwargs)]
            return responses
        if isinstance(fn, str):
            if '/' in fn:
                module, fn = fn.split('/')
                module = c.module(module)
            else:
                module = cls
            if module.classify_fn(fn) == 'self':
                module = cls()
            fn = getattr(module, fn)
        
        t1 = c.time()
        result = fn(*args, **kwargs)
        t2 = c.time()

        return {'time': t2 - t1}


    def search_dict(self, d:dict = 'k,d', search:str = {'k.d': 1}) -> dict:
        search = search.split(',')
        new_d = {}

        for k,v in d.items():
            if search in k.lower():
                new_d[k] = v
        
        return new_d
    


    @classmethod
    def tree(cls, *args, **kwargs) -> List[str]:
        return c.module('tree').tree(*args,  **kwargs) 

    @classmethod
    def tree2path(cls, *args, **kwargs) -> List[str]:
        return c.module('tree').tree2path( *args, **kwargs)

    
    @classmethod
    def trees(cls):
        return c.m('tree').trees()
    
    @classmethod
    def add_tree(cls, *args, **kwargs):
        return c.m('tree').add_tree(*args, **kwargs)

    @classmethod
    def add_tree(cls, *args, **kwargs):
        return c.m('tree').add_tree(*args, **kwargs)
    
    @classmethod
    def rm_tree(cls, *args, **kwargs):
        return c.m('tree').rm_tree(*args, **kwargs)

    def repo2module(self, repo:str, name=None, template_module='demo', **kwargs):
        if not repo_path.startswith('/') and not repo_path.startswith('.') and not repo_path.startswith('~'):
            repo_path = os.path.abspath('~/' + repo_path)
        assert os.path.isdir(repo_path), f'{repo_path} is not a directory, please clone it'
        c.add_tree(repo_path)
        template_module = c.module(template_module)
        code = template_module.code()

        # replace the template module with the name
        name = name or repo_path.split('/')[-1] 
        assert not c.module_exists(name), f'{name} already exists'
        code_lines = code.split('\n')
        for i, line in enumerate(code_lines):
            if 'class' in line and 'c.Module' in line:
                class_name = line.split('class ')[-1].split('(')[0]
                code_lines[i] = line.replace(class_name, name)
                break
        code = '\n'.join(code_lines)

        module_path = repo_path + '/module.py'

        # write the module code
        c.put_text(code, module_path)

        # build the tree
        c.build_tree(update=True)


    @classmethod
    def simple2path(cls, path:str, **kwargs) -> str:
        return c.module('tree').simple2path(path, **kwargs)
    
    @classmethod
    def simple2objectpath(cls, path:str,path2objectpath = {'tree': 'commune.tree.tree.Tree'}, **kwargs) -> str:
        if path in path2objectpath:
            object_path =  path2objectpath[path]
        else:
            object_path =  c.module('tree').simple2objectpath(path, **kwargs)
        return object_path
    @classmethod
    def simple2object(cls, path:str, **kwargs) -> str:
        path = c.simple2objectpath(path, **kwargs)
        try:
            return c.import_object(path)
        except Exception as e:
            c.print(path)
            raise e
        

    
    @classmethod
    def python_paths(cls, path:str = None, recursive=True, **kwargs) -> List[str]:
        if path == None:
            path = c.homepath
        return  glob(path + '/**/*.py', recursive=recursive, **kwargs)
    
    @classmethod
    def get_module_python_paths(cls, 
                                path : str= None, 
                                search:str=None,
                                end_line=200, 
                                ) -> List[str]:
        '''
        Search for all of the modules with yaml files. Format of the file
        '''

        path = path or c.libpath

        search_glob = path +'/**/*.py' 
        modules = []

        # find all of the python files
        for f in glob(search_glob, recursive=True):

            initial_text = c.readlines(f, end_line=end_line)

            commune_in_file = 'import commune as c' in initial_text 
            is_commune_root = 'class c:' in initial_text
            if not commune_in_file and not is_commune_root:
                continue
            if os.path.isdir(f):
                continue

            modules.append(f)
        # we ar caching t
        return modules


    available_modules  = module_tree = tree
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
        return c.exists(cls.config_path())
  
    @classmethod
    def has_module(cls, module):
        return module in c.modules()
        
    @classmethod
    def Vali(cls, *args, **kwargs):
        return c.module('vali')
    
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
    def infer_device_map(cls, *args, **kwargs):
        return cls.infer_device_map(*args, **kwargs)
    
    @classmethod
    def datasets(cls, **kwargs) -> List[str]:
        return c.servers('data',  **kwargs)
    datas = datasets

    @staticmethod
    def is_imported(package:str) :
        return  bool(package in sys.modules)



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
    


    def timefn(cls, fn, *args,  **kwargs):
        def wrapper(*args, **kwargs):
            t1 = c.time()
            result = fn(*args, **kwargs)
            t2 = c.time()
            c.print(f'{fn.__name__} took {t2-t1} seconds')
            return result
        return wrapper

    @classmethod
    def timeit(cls, fn, *args, include_result=False, **kwargs):

        t = c.time()
        if isinstance(fn, str):
            fn = cls.get_fn(fn)
        result = fn(*args, **kwargs)
        response = {
            'latency': c.time() - t,
            'fn': fn.__name__,
            
        }
        if include_result:
            c.print(response)
            return result
        return response

    @staticmethod
    def remotewrap(fn, remote_key:str = 'remote'):
        '''
        calls your function if you wrap it as such

        @c.remotewrap
        def fn():
            pass
            
        # deploy it as a remote function
        fn(remote=True)
        '''
    
        def remotewrap(self, *args, **kwargs):
            remote = kwargs.pop(remote_key, False)
            if remote:
                return c.remote_fn(module=self, fn=fn.__name__, args=args, kwargs=kwargs)
            else:
                return fn(self, *args, **kwargs)
        
        return remotewrap
    
    # def local
    @classmethod
    def local_node_urls(cls):
        return c.module('subpsace').local_node_urls()

    def locals2hash(self, kwargs:dict = {'a': 1}, keys=['kwargs']) -> str:
        kwargs.pop('cls', None)
        kwargs.pop('self', None)
        return c.dict2hash(kwargs)

    @classmethod
    def dict2hash(cls, d:dict) -> str:
        for k in d.keys():
            assert c.jsonable(d[k]), f'{k} is not jsonable'
        return c.hash(d)

    
    @classmethod
    def locals2kwargs(cls,locals_dict:dict, kwargs_keys=['kwargs']) -> dict:
        kwargs = locals_dict or {}
        kwargs.pop('cls', None)
        kwargs.pop('self', None)

        assert isinstance(kwargs, dict), f'kwargs must be a dict, got {type(kwargs)}'
        
        # These lines are needed to remove the self and cls from the locals_dict
        for k in kwargs_keys:
            kwargs.update( locals_dict.pop(k, {}) or {})

        return kwargs
    
    get_kwargs = get_params = locals2kwargs 
        
    @classmethod
    def get_parents(cls, obj=None):
        
        if obj == None:
            obj = cls

        return list(obj.__mro__[1:-1])

    @classmethod
    def storage_dir(cls):
        return f'{c.cache_path()}/{cls.module_path()}'
    tmp_dir = cache_dir   = storage_dir
    
    @classmethod
    def refresh_storage(cls):
        c.rm(cls.storage_dir())

    @classmethod
    def refresh_storage_dir(cls):
        c.rm(cls.storage_dir())
        c.makedirs(cls.storage_dir())
        

    ############ JSON LAND ###############

    @classmethod
    def cache_path(cls):
        path = os.path.expanduser(f'~/.{cls.libname}')
        return path

    @classmethod
    def tilde_path(cls):
        return os.path.expanduser('~')

    @classmethod
    def get_json(cls, 
                path:str,
                default:Any=None,
                verbose: bool = False,**kwargs):
        from commune.utils.dict import async_get_json
        path = cls.resolve_path(path=path, extension='json')

        c.print(f'Loading json from {path}', color='green', verbose=verbose)

        try:
            data = cls.get_text(path, **kwargs)
        except Exception as e:
            return default
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception as e:
                c.print(e)
                return default
        if isinstance(data, dict):
            if 'data' in data and 'meta' in data:
                data = data['data']
        return data
    @classmethod
    async def async_get_json(cls,*args, **kwargs):
        return  cls.get_json(*args, **kwargs)

    load_json = get_json

    data_path = repo_path + '/data'

    @classmethod
    def put_torch(cls, path:str, data:Dict,  **kwargs):
        import torch
        path = cls.resolve_path(path=path, extension='pt')
        torch.save(data, path)
        return path
    
    def init_nn(self):
        import torch
        torch.nn.Module.__init__(self)
    
    @classmethod
    async def async_put_json(cls,*args,**kwargs) -> str:
        return cls.put_json(*args, **kwargs) 
    
    @classmethod
    def put_json(cls, 
                 path:str, 
                 data:Dict, 
                 meta = None,
                 verbose: bool = False,

                 **kwargs) -> str:
        if meta != None:
            data = {'data':data, 'meta':meta}
        path = cls.resolve_path(path=path, extension='json')
        # cls.lock_file(path)
        c.print(f'Putting json from {path}', color='green', verbose=verbose)
        if isinstance(data, dict):
            data = json.dumps(data)
        c.put_text(path, data)
        return path
    
    save_json = put_json
    
    @classmethod
    def file_exists(cls, path:str)-> bool:
        path = cls.resolve_path(path=path)
        exists =  os.path.exists(path)
        if not exists and not path.endswith('.json'):
            exists = os.path.exists(path + '.json')
        return exists

    exists = exists_json = file_exists 

    @classmethod
    def readme(cls):
        # Markdown input
        markdown_text = "## Hello, *Markdown*!"
        path = cls.filepath().replace('.py', '_docs.md')
        markdown_text =  cls.get_text(path=path)
        return markdown_text
    
    docs = readme



    @classmethod
    def rm_json(cls, path=None):
        from commune.utils.dict import rm_json

        if path in ['all', '**']:
            return [cls.rm_json(f) for f in cls.glob(files_only=False)]
        
        path = cls.resolve_path(path=path, extension='json')

        return rm_json(path )
    
    @classmethod
    def rmdir(cls, path):
        import shutil
        return shutil.rmtree(path)

    @classmethod
    def isdir(cls, path):
        path = cls.resolve_path(path=path)
        return os.path.isdir(path)
        

    @classmethod
    def isfile(cls, path):
        path = cls.resolve_path(path=path)
        return os.path.isfile(path)


    def rm_many(cls, paths:List[str]):
        paths = c.ls(paths)

    
        # for path in paths:
        #     cls.rm(path)

    @classmethod
    def rm_all(cls):
        for path in cls.ls():
            cls.rm(path)
        return {'success':True, 'message':f'{cls.storage_dir()} removed'}
        

    @classmethod
    def rm(cls, path, extension=None, mode = 'json'):
        
        assert isinstance(path, str), f'path must be a string, got {type(path)}'
        path = cls.resolve_path(path=path, extension=extension)

        # incase we want to remove the json file
        mode_suffix = f'.{mode}'
        if not os.path.exists(path) and os.path.exists(path+mode_suffix):
            path += mode_suffix

        if not os.path.exists(path):
            return {'success':False, 'message':f'{path} does not exist'}
        if os.path.isdir(path):
            c.rmdir(path)
        else:
            os.remove(path)
        assert not os.path.exists(path), f'{path} was not removed'

        return {'success':True, 'message':f'{path} removed'}
    
    @classmethod
    def rm_all(cls):
        storage_dir = cls.storage_dir()
        if c.exists(storage_dir):
            cls.rm(storage_dir)
        assert not c.exists(storage_dir), f'{storage_dir} was not removed'
        c.makedirs(storage_dir)
        assert c.is_dir_empty(storage_dir), f'{storage_dir} was not removed'
        return {'success':True, 'message':f'{storage_dir} removed'}

    def is_dir_empty(self, path:str):
        return len(self.ls(path)) == 0


    @classmethod
    def glob(cls,  path =None, files_only:bool = True, recursive:bool=True):
        path = cls.resolve_path(path, extension=None)
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
    def ls(cls, path:str = '', 
           recursive:bool = False,
           search = None,
           return_full_path:bool = True):
        """
        provides a list of files in the path 

        this path is relative to the module path if you dont specifcy ./ or ~/ or /
        which means its based on the module path
        """
        path = cls.resolve_path(path, extension=None)
        try:
            ls_files = cls.lsdir(path) if not recursive else cls.walk(path)
        except FileNotFoundError:
            return []
        if return_full_path:
            ls_files = [os.path.abspath(os.path.join(path,f)) for f in ls_files]

        ls_files = sorted(ls_files)
        if search != None:
            ls_files = list(filter(lambda x: search in x, ls_files))
        return ls_files
    
    @classmethod
    def lsdir(cls, path:str) -> List[str]:
        path = os.path.abspath(path)
        return os.listdir(path)

    @classmethod
    def abspath(cls, path:str) -> str:
        return os.path.abspath(path)


    @classmethod
    def walk(cls, path:str, module:str=False) -> List[str]:
        
        path_map = {}
        for root, dirs, files in os.walk(path):
            for f in files:
                path = os.path.join(root, f)
                path_map[path] = f
        return list(path_map.keys())
    
       
    @classmethod
    def __str__(cls):
        return cls.__name__

    @classmethod
    def get_server_info(cls,name:str) -> Dict:
        return cls.namespace_local().get(name, {})

    @classmethod
    def connect(cls,
                module:str, 
                network : str = 'local',
                mode = 'http',
                virtual:bool = True, 
                **kwargs):
        
        return c.module( f'client').connect(module=module, 
                                       virtual=virtual, 
                                       network=network,
                                       **kwargs)

    @classmethod
    async def async_connect(cls, *args, **kwargs):
        return c.connect(*args, **kwargs)
     
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
            address = c.call('module/address', network=network, timeout=timeout)
            ip = c.ip()
            address = ip+':'+address.split(':')[-1]
        except Exception as e:
            c.print(f'Error: {e}', color='red')
            address = None
        return address
    addy = root_address

    @property
    def key_address(self):
        return self.key.ss58_address

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
    def nest_asyncio(cls):
        import nest_asyncio
        nest_asyncio.apply()


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
        if '://' in address:
            return True
        conds = []
        conds.append(len(address.split('.')) >= 3)
        conds.append(isinstance(address, str))
        conds.append(':' in address)
        conds.append(cls.is_int(address.split(':')[-1]))
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
    def is_root(cls, obj=None) -> bool:
        
        if obj is None:
            obj = cls
        if hasattr(obj, 'module_class'):
            module_class = obj.module_class()
            if module_class == cls.root_module_class:
                return True
            
        return False
    is_module_root = is_root_module = is_root
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
        try:
            loop = asyncio.get_event_loop()
        except Exception as e:
            loop = c.new_event_loop(nest_asyncio=nest_asyncio)
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
        if callable(self.config):
            self.set_config()
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
            time_waiting += sleep_interval
            c.sleep(sleep_interval)
            logs.append(f'Waiting for {name} to start')
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
            attrs = [a for a in attrs if search in a and callable(a)]
        return attrs
    
    # NAMESPACE::MODULE
    namespace_module = 'module.namespace'
    @classmethod
    def name2address(cls, name:str, network:str='local') -> str:
        return c.module("namespace").name2address(name=name, network=network)
    @classmethod
    def servers(cls, *args, **kwargs) -> List[str]:
        return c.module("namespace").servers(*args, **kwargs)
    @classmethod
    def server2key(self, *args, **kwargs):
        servers = c.servers()
        key2address = c.key2address()
        server2key = {s:key2address[s] for s in servers}
        return server2key

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
    def infos(cls, *args, **kwargs) -> List[str]:
        return c.module("namespace").infos(*args, **kwargs)
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
                  update: bool = False,
                   **kwargs):
        namespace =  c.module("namespace").namespace(search=search, network=network, update=update, **kwargs)
        return namespace
    

    get_namespace = namespace
    @classmethod
    def rm_namespace(cls, *args, **kwargs):
        """
        remove the namespace
        """
        return c.module("namespace").rm_namespace(*args, **kwargs)



    @classmethod
    def empty_namespace(cls, *args, **kwargs):
        """
        empty the namespace
        """
        return c.module("namespace").empty_namespace(*args, **kwargs)

    @classmethod
    def add_namespace(cls, *args, **kwargs):
        return c.module("namespace").empty_namespace(*args, **kwargs)

    @classmethod
    def update_namespace(cls, network:str='local',**kwargs):
        return c.module("namespace").update_namespace(network=network, **kwargs)
    
    
    @classmethod
    def build_namespace(cls, network:str='local',**kwargs):
        return c.module("namespace").update_namespace(network=network, **kwargs)
    
    @classmethod
    def update_subnet(cls, *args, **kwargs):
        return c.module("subspace")().update_subnet(*args, **kwargs)
    
    @classmethod
    def subnet_params(cls, *args, **kwargs):
        return c.module("subspace")().subnet_params(*args, **kwargs)

    @classmethod
    def my_subnets(cls, *args, **kwargs):
        return c.module("subspace")().my_subnets(*args, **kwargs)
    
    @classmethod
    def global_params(cls, *args, **kwargs):
        return c.module("subspace")().global_params(*args, **kwargs)
    
    @classmethod
    def subnet_names(cls, *args, **kwargs):
        return c.module("subspace")().subnet_names(*args, **kwargs)
    
    @classmethod
    def put_namespace(cls,network:str, namespace:dict, **kwargs):
        namespace = c.module("namespace").put_namespace(network=network, namespace=namespace, **kwargs)
        return namespace
    
    @classmethod
    def rm_namespace(cls,network:str, **kwargs):
        namespace = c.module("namespace").rm_namespace(network=network, **kwargs)
        return namespace
    

    def add_fn(self, fn, name=None):
        if name == None:
            name = fn.__name__
        assert not hasattr(self, name), f'{name} already exists'

        setattr(self, name, fn)

        return {
            'success':True ,
            'message':f'Added {name} to {self.__class__.__name__}'
        }
    
    add_attribute = add_attr = add_function = add_fn
    
    @classmethod
    def resolve_server_name(cls, 
                            module:str = None, 
                            tag:str=None, 
                            name:str = None,  
                            tag_seperator:str='::', 
                            **kwargs):
        """
        Resolves the server name
        """
        # if name is not specified, use the module as the name such that module::tag
        if name == None:
            module = cls.module_path() if module == None else module

            # module::tag
            if tag_seperator in module:
                module, tag = module.split(tag_seperator)
            if tag_seperator in module: 
                module, tag = module.split(tag_seperator)
            name = module
            if tag in ['None','null'] :
                tag = None
            if tag != None:
                name = f'{name}{tag_seperator}{tag}'

        # ensure that the name is a string
        assert isinstance(name, str), f'Invalid name {name}'
        return name
    resolve_name = resolve_server_name
    
    
    @classmethod
    def serve(cls, 
              module:Any = None ,
              kwargs:dict = None,  # kwargs for the module
              tag:str=None,
              server_network = 'local',
              port :int = None, # name of the server if None, it will be the module name
              server_name:str=None, # name of the server if None, it will be the module name
              name = None, # name of the server if None, it will be the module name
              refresh:bool = True, # refreshes the server's key
              remote:bool = True, # runs the server remotely (pm2, ray)
              tag_seperator:str='::',
              max_workers:int = None,
              free: bool = False,
              mnemonic = None, # mnemonic for the server
              key = None,
              **extra_kwargs
              ):
        if c.is_module(module):
            cls = module
        if module == None:
            module = cls.module_path()
        kwargs = kwargs or {}
        kwargs.update(extra_kwargs or {})
        if kwargs.get('debug', False):
            remote = False
        name = server_name or name # name of the server if None, it will be the module name
        name = cls.resolve_server_name(module=module, name=name, tag=tag, tag_seperator=tag_seperator)
        if tag_seperator in name:
            module, tag = name.split(tag_seperator)
        # RESOLVE THE PORT FROM THE ADDRESS IF IT ALREADY EXISTS
        if port == None:
            # now if we have the server_name, we can repeat the server
            address = c.get_address(name, network=server_network)
            port = int(address.split(':')[-1]) if address else c.free_port()

        # NOTE REMOVE THIS FROM THE KWARGS REMOTE
        if remote:
            remote_kwargs = c.locals2kwargs(locals())  # GET THE LOCAL KWARGS FOR SENDING TO THE REMOTE
            remote_kwargs['remote'] = False  # SET THIS TO FALSE TO AVOID RECURSION
            # REMOVE THE LOCALS FROM THE REMOTE KWARGS THAT ARE NOT NEEDED
            for _ in ['extra_kwargs', 'address']:
                remote_kwargs.pop(_, None) # WE INTRODUCED THE ADDRES
            cls.remote_fn('serve',name=name, kwargs=remote_kwargs)
            address = c.ip() + ':' + str(remote_kwargs['port'])
            return {'success':True, 
                    'name': name, 
                    'address':address, 
                    'kwargs':kwargs
                    }        
        module_class = c.module(module)

        kwargs.update(extra_kwargs)
        if mnemonic != None:
            c.add_key(server_name, mnemonic)
            
        self = module_class(**kwargs)
        self.server_name = name
        self.tag = tag

        address = c.get_address(name, network=server_network)
        if address != None and ':' in address:
            port = address.split(':')[-1]   

        if c.server_exists(server_name, network=server_network) and not refresh: 
            return {'success':True, 'message':f'Server {server_name} already exists'}
        c.module(f'server')(module=self, 
                                          name=name, 
                                          port=port, 
                                          network=server_network, 
                                          max_workers=max_workers, 
                                          free=free, 
                                          key=key)

        return  {'success':True, 
                     'address':  f'{c.default_ip}:{port}' , 
                     'name':name, 
                     'kwargs': kwargs,
                     'module':module}

    serve_module = serve
    
    @classmethod
    def functions(cls, search: str=None , include_parents:bool = False, module=None):
        if module != None:
            cls = c.module(module)
        functions = cls.get_functions(include_parents=include_parents, search=search)  
        return functions

    fns = functions

    def hasfn(self, fn:str):
        return hasattr(self, fn) and callable(getattr(self, fn))
    
    @classmethod
    def fn_signature_map(cls, obj=None, include_parents:bool = False):
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
    
    function_signature_map = fn_signature_map


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
            yield i
        
    def info(self , 
             module = None,
             features = ['schema', 'namespace', 'commit_hash', 'hardware','attributes','functions'], 
             lite_features = ['name', 'address', 'schema', 'ss58_address', 'description'],
             lite = True,
             cost = False,
             **kwargs
             ) -> Dict[str, Any]:
        '''
        hey, whadup hey how is it going
        '''
        if lite:
            features = lite_features
            
        if module != None:
            if isinstance(module, str):
                module = c.module(module)()
            self = module  
            
        info = {}

        if 'schema' in features:
            info['schema'] = self.schema(defaults=True, include_parents=True)
            info['schema'] = {k: v for k,v in info['schema'].items() if k in self.whitelist}
        if 'namespace' in features:
            info['namespace'] = c.namespace(network='local')
        if 'hardware' in features:
            info['hardware'] = c.hardware()
        if 'attributes' in features:
            info['attributes'] = attributes =[ attr for attr in self.attributes()]
        if 'functions' in features:
            info['functions']  = [fn for fn in self.whitelist]
        if 'name' in features:
            info['name'] = self.server_name() if callable(self.server_name) else self.server_name # get the name of the module
        if 'path' in features:
            info['path'] = self.module_path() # get the path of the module
        if 'address' in features:
            info['address'] = self.address.replace(c.default_ip, c.ip(update=False))
        if 'ss58_address' in features:    
            info['ss58_address'] = self.key.ss58_address
        if 'code_hash' in features:
            info['code_hash'] = self.chash() # get the hash of the module (code)
        if 'commit_hash' in features:
            info['commit_hash'] = c.commit_hash()
        if 'description' in features:
            info['description'] = self.description

        c.put_json('info', info)
        if cost:
            if hasattr(self, 'cost'):
                info['cost'] = self.cost
        return info
        
    help = info

    def metadata(self):
        schema = self.schema()
        return {fn: schema[fn] for fn in self.whitelist if fn not in self.blacklist and fn in schema}


    
    @classmethod
    def hardware(cls, fmt:str = 'gb', **kwargs):
        return c.module('os').hardware(fmt=fmt, **kwargs)

    @classmethod
    def init_schema(cls):
        return cls.fn_schema('__init__')

    @classmethod
    def init_kwargs(cls):
        kwargs =  cls.fn_defaults('__init__')
        kwargs.pop('self', None)
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
    def schema(cls,
                search = None,
                module = None,
                fn = None,
                docs: bool = True,
                include_parents:bool = False,
                defaults:bool = True, cache=False) -> 'Schema':

        if '/' in str(search):
            module, fn = search.split('/')
            cls = c.module(module)
        if isinstance(module, str):
            if '/' in module:
                module , fn = module.split('/')
            module = c.module(module)

        module = module or cls
        schema = {}
        fns = module.get_functions(include_parents=include_parents)
        for fn in fns:
            if search != None and search not in fn:
                continue
            if callable(getattr(module, fn )):
                schema[fn] = cls.fn_schema(fn, defaults=defaults,docs=docs)        

        # sort by keys
        schema = dict(sorted(schema.items()))

        return c.copy(schema)
        

    @classmethod
    def get_function_annotations(cls, fn):
        fn = cls.get_fn(fn)
        if not hasattr(fn, '__annotations__'):
            return {}
        return fn.__annotations__
        
    @classmethod
    def fn_schema(cls, fn:str,
                            defaults:bool=True,
                            code:bool = False,
                            docs:bool = True, 
                            version=2)->dict:
        '''
        Get function schema of function in cls
        '''
        fn_schema = {}
        fn = cls.get_fn(fn)
        fn_schema['input']  = cls.get_function_annotations(fn=fn)
        
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


        if defaults:
            fn_schema['default'] = cls.fn_defaults(fn=fn) 
            for k,v in fn_schema['default'].items(): 
                if k not in fn_schema['input'] and v != None:
                    fn_schema['input'][k] = type(v).__name__ if v != None else None
           
        if version == 1:
            pass
        elif version == 2:
            defaults = fn_schema.pop('default', {})
            fn_schema['input'] = {k: {'type':v, 'default':defaults.get(k)} for k,v in fn_schema['input'].items()}
        else:
            raise Exception(f'Version {version} not implemented')
                

        return fn_schema
    

    @staticmethod
    def get_annotations(fn:callable) -> dict:
        return fn.__annotations__

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
    def kill_many(cls, servers, search:str = None, network='local', parallel=True,  timeout=10, **kwargs):


        servers = c.servers(network=network)
        servers = [s for s in servers if  search in s]
        futures = []
        for s in servers:
            future = c.submit(c.kill, kwargs={'module':s, **kwargs}, imeout=timeout)
            futures.append(future)
        results = []
        for r in c.as_completed(futures, timeout=timeout):
            results += [r.result()]

        return results
            
        return 
        
    delete = kill_server = kill
    def destroy(self):
        return self.kill(self.server_name)
        
    
    def self_destruct(self):
        c.kill(self.server_name)    
        
    def self_restart(self):
        c.restart(self.server_name)
        

    @classmethod
    def get_shortcut(cls, shortcut:str) -> dict:
        return cls.shortcuts().get(shortcut)
 
    @classmethod
    def rm_shortcut(cls, shortcut) -> str:
        shortcuts = cls.shortcuts()
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
               verbose : bool = False, 
               update: bool = False,
               **extra_launch_kwargs):
        '''
        Launch a module as pm2 or ray 
        '''
        if update:
            cls.update()

        kwargs = kwargs or {}
        args = args or []

        # if module is not specified, use the current module
        module = module or cls 
        if isinstance(module, str):
            module = c.module(module) 
            
        # resolve the name
        if name == None:
            # if the module has a module_path function, use that as the name
            if hasattr(module, 'module_path'):
                name = module.module_path()
            else:
                name = module.__name__.lower() 
            # resolve the tag
            if tag != None:
                name = f'{name}{tag_seperator}{tag}'
 
        c.print(f'[bold cyan]Launching[/bold cyan] [bold yellow]class:{module.__name__}[/bold yellow] [bold white]name[/bold white]:{name} [bold white]fn[/bold white]:{fn} [bold white]mode[/bold white]:{mode}', color='green', verbose=verbose)

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
    
        return  getattr(cls, f'{mode}_launch')(**launch_kwargs)
    
    @classmethod
    def register(cls,  *args, **kwargs ):
        return  c.module('subspace')().register(*args, **kwargs)

    @classmethod
    def key_stats(cls, *args, **kwargs):
        return c.module('subspace')().key_stats(*args, **kwargs)

    @classmethod
    def key2stats(cls, *args, **kwargs):
        return c.module('subspace')().key_stats(*args, **kwargs)

    @staticmethod
    def detailed_error(e) -> dict:
        import traceback
        tb = traceback.extract_tb(e.__traceback__)
        file_name = tb[-1].filename
        line_no = tb[-1].lineno
        line_text = tb[-1].line
        response = {
            'success': False,
            'error': str(e),
            'file_name': file_name,
            'line_no': line_no,
            'line_text': line_text
        }   
        return response
    
    @classmethod
    def pm2_kill_many(cls, search=None, verbose:bool = True, timeout=10):
        return c.module('pm2').kill_many(search=search, verbose=verbose, timeout=timeout)
    
    @classmethod
    def pm2_kill_all(cls, verbose:bool = True, timeout=10):
        return cls.pm2_kill_many(search=None, verbose=verbose, timeout=timeout)
                
    @classmethod
    def pm2_servers(cls, search=None,  verbose:bool = False) -> List[str]:
        return  c.module('pm2').servers(verbose=verbose)
    pm2ls  = pm2_list = pm2_servers
    # commune.run_command('pm2 status').stdout.split('\n')[5].split('    │')[0].split('  │ ')[-1]commune.run_command('pm2 status').stdout.split('\n')[5].split('    │')[0].split('  │ ')[-1] 
    
    @classmethod
    def pm2_exists(cls, name:str) -> bool:
        return c.module('pm2').exists(name=name)
    
    @classmethod
    def pm2_start(cls, *args, **kwargs):
        return c.module('pm2').start(*args, **kwargs)
    
    @classmethod
    def pm2_launch(cls, *args, **kwargs):
        return c.module('pm2').launch(*args, **kwargs)
                              
    @classmethod
    def pm2_restart(cls, name:str, verbose:bool = False, prefix_match:bool = True):
        return c.module('pm2').restart(name=name, verbose=verbose, prefix_match=prefix_match)
    @classmethod
    def pm2_restart_prefix(cls, name:str = None, verbose:bool=False):
        return c.module('pm2').restart_prefix(name=name, verbose=verbose)  
    
    @classmethod
    def pm2_kill(cls, name:str, verbose:bool = False, prefix_match:bool = True):
        return c.module('pm2').kill(name=name, verbose=verbose, prefix_match=prefix_match)
    
    @classmethod
    def restart(cls, name:str, mode:str='pm2', verbose:bool = False, prefix_match:bool = True):
        refreshed_modules = getattr(cls, f'{mode}_restart')(name, verbose=verbose, prefix_match=prefix_match)
        return refreshed_modules

    def restart_self(self):
        """
        Helper function to restart the server
        """
        return c.restart(self.server_name)

    update_self = restart_self

    def kill_self(self):
        """
        Helper function to kill the server
        """
        return c.kill(self.server_name)

    refresh = reset = restart
    @classmethod
    def pm2_status(cls, verbose=True):
        return c.module('pm2').status(verbose=verbose)

    @classmethod
    def pm2_logs_path_map(cls, name=None):
        return c.module('pm2').logs_path_map(name=name)
    @classmethod
    def pm2_rm_logs( cls, name):
        return c.module('pm2').rm_logs(name=name)

    @classmethod
    def pm2_logs(cls, 
                module:str, 
                tail: int =20, 
                verbose: bool=True ,
                mode: str ='cmd',
                **kwargs):
        return c.module('pm2').logs(module=module,
                                     tail=tail, 
                                     verbose=verbose, 
                                     mode=mode, 
                                     **kwargs)
    
    
    @staticmethod
    def memory_usage(fmt='gb'):
        fmt2scale = {'b': 1e0, 'kb': 1e1, 'mb': 1e3, 'gb': 1e6}
        import psutil
        process = psutil.Process()
        scale = fmt2scale.get(fmt)
        return (process.memory_info().rss // 1024) / scale

    @classmethod
    def argparse(cls, verbose: bool = False, version=1):
        if version == 1:
            parser = argparse.ArgumentParser(description='Argparse for the module')
            parser.add_argument('-fn', '--fn', dest='function', help='The function of the key', type=str, default="__init__")
            parser.add_argument('-kwargs', '--kwargs', dest='kwargs', help='key word arguments to the function', type=str, default="{}") 
            parser.add_argument('-p', '-params', '--params', dest='params', help='key word arguments to the function', type=str, default="{}") 
            parser.add_argument('-i','-input', '--input', dest='input', help='key word arguments to the function', type=str, default="{}") 
            parser.add_argument('-args', '--args', dest='args', help='arguments to the function', type=str, default="[]")  
            args = parser.parse_args()
            if verbose:
                c.print('Argparse Args: ',args, color='cyan')
            args.kwargs = json.loads(args.kwargs.replace("'",'"'))
            args.params = json.loads(args.params.replace("'",'"'))
            args.inputs = json.loads(args.input.replace("'",'"'))

            # if you pass in the params, it will override the kwargs
            if len(args.params) > len(args.kwargs):
                args.kwargs = args.params
            args.args = json.loads(args.args.replace("'",'"'))
        elif version == 2:
            args = c.parseargs()

        return args

    @classmethod
    def run(cls, name:str = None, verbose:bool = False, version=1) -> Any: 
        is_main =  name == '__main__' or name == None or name == cls.__name__
        if not is_main:
            return {'success':False, 'message':f'Not main module {name}'}
        args = cls.argparse(version=version)

        if args.function == '__init__':
            return cls(*args.args, **args.kwargs)     
        else:
            fn = getattr(cls, args.function)
            fn_type = cls.classify_fn(fn)

            if fn_type == 'self':
                module = cls(*args.args, **args.kwargs)
            else:
                module = cls

            return getattr(module, args.function)(*args.args, **args.kwargs)     
    
    @classmethod
    def learn(cls, *args, **kwargs):
        return c.module('model.hf').learn(*args, **kwargs)
    
    @classmethod
    def commit_hash(cls, libpath:str = None):
        if libpath == None:
            libpath = c.libpath
        return c.cmd('git rev-parse HEAD', cwd=libpath, verbose=False).split('\n')[0].strip()

    
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
    def transfer_multiple(cls, *args, **kwargs):
        return c.module('subspace')().transfer_multiple(*args, **kwargs)   

            
    @classmethod
    def transfer_stake(cls, *args, **kwargs):
        return c.module('subspace')().transfer_stake(*args, **kwargs)   


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
        c.put_text(filepath, module2_code)

        return {'success': True, 'module2_code': module2_code, 'module2_fns': module2_fns, 'module1_fn_code_map': module1_fn_code_map}

    @classmethod
    def get_server_name(cls, name:str=None, tag:str=None, seperator:str='.'):
        name = name if name else cls.__name__.lower()
        if tag != None:
            name = tag + seperator + name
        return name
  

    @classmethod
    def fn(cls, module:str, fn:str , args:list = None, kwargs:dict= None):
        module = c.module(module)
        is_self_method = bool(fn in module.self_functions())

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
    def module(cls,module: Any = 'module' , **kwargs):
        '''
        Wraps a python class as a module
        '''
        shortcuts = c.shortcuts()
        if module in shortcuts:
            module = shortcuts[module]
        module_class =  c.get_module(module,**kwargs)
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
   
        
    # JUPYTER NOTEBOOKS
    @classmethod
    def enable_jupyter(cls):
        cls.nest_asyncio()
    
    jupyter = enable_jupyter
    
        
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
    def pip_libs(cls):
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
    def ensure_libs(cls, libs: List[str] = None, verbose:bool=False):
        if hasattr(cls, 'libs'):
            libs = cls.libs
        results = []
        for lib in libs:
            results.append(cls.ensure_lib(lib, verbose=verbose))
        return results
    
    @classmethod
    def install(cls, libs: List[str] = None, verbose:bool=False):
        return cls.ensure_libs(libs, verbose=verbose)
    
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

    @classmethod
    def pip_exists(cls, lib:str, verbose:str=True):
        return bool(lib in cls.pip_libs())
    
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
    def version(cls, lib:str=libname):
        lines = [l for l in cls.cmd(f'pip list', verbose=False).split('\n') if l.startswith(lib)]
        if len(lines)>0:
            return lines[0].split(' ')[-1].strip()
        else:
            return f'No Library Found {lib}'
    
    @classmethod
    def external_ip(cls, *args, **kwargs) -> str:
        return c.module('network').external_ip(*args, **kwargs)
    
    @classmethod
    def ip(cls,  max_age=10000, update:bool = False, **kwargs) -> str:
        ip = c.get('ip', None, max_age=max_age, update=update)
        if ip == None:
            ip =  c.module('network').external_ip(**kwargs)
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
        os.environ[key] = value
        return value 

    @classmethod
    def get_env(cls, key:str)-> None:
        '''
        Pay attention to this function. It sets the environment variable
        '''
        return  os.environ[key] 

    env = get_env
    


    def forward(self, a=1, b=2):
        return a+b
    
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
    def resolve_device(cls, *args, **kwargs):
        return c.module('gpu').resolve_device(*args, **kwargs)

    # CPU LAND

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
    def resolve_model_shortcut(cls, model):
        model_shortcuts = c.model_shortcuts()
        return model_shortcuts.get(model,model)
            
    @classmethod
    def add_model_shortcut(cls, *args, **kwargs):
        return  c.module('hf').add_model_shortcut(*args, **kwargs)    
    @classmethod
    def rm_model_shortcut(cls, *args, **kwargs):
        return  c.module('hf').rm_model_shortcut(*args, **kwargs)
    
    @classmethod
    def add_remote(self, *args, **kwargs):
        return c.module('namespace').add_remote(*args, **kwargs)
    
    @classmethod
    def model_options(cls):
        return list(c.model_shortcuts().keys())

    @classmethod
    def resolve_model(cls, model):
        if isinstance(model, str):
            model = c.get_empty_model(model)
        return model

    ### DICT LAND ###

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
    def from_json(cls, json_str:str) -> 'Module':
        import json
        return cls.from_dict(json.loads(json_str))
    

    ### LOGGER LAND ###
    @classmethod
    def resolve_logger(cls, logger = None):
        if not hasattr(cls,'logger'):
            from loguru import logger
            cls.logger = logger.opt(colors=True)
        if logger is not None:
            cls.logger = logger
        return cls.logger

    @classmethod
    def resolve_console(cls, console = None, **kwargs):
        if hasattr(cls,'console'):
            return cls.console
    
        
        import logging
        from rich.logging import RichHandler
        from rich.console import Console
        logging.basicConfig(level=logging.DEBUG, handlers=[RichHandler()])   
            # print the line number
        console = Console()
        cls.console = console
        return console
    
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
              verbose:bool = True,
              console: Console = None,
              flush:bool = False,
              **kwargs):
              
        if not verbose:
            return 
        if color == 'random':
            color = cls.random_color()
        if color:
            kwargs['style'] = color

        console = cls.resolve_console(console)
        try:
            if flush:
                console.print(**kwargs, end='\r')
            console.print(*text, **kwargs)
        except Exception as e:
            print(e)
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
    def status(cls, *args, **kwargs):
        console = cls.resolve_console()
        return console.status(*args, **kwargs)
    @classmethod
    def log(cls, *args, **kwargs):
        console = cls.resolve_console()
        return console.log(*args, **kwargs)
    
    @classmethod
    def test_fns(cls, *args, **kwargs):
        return [f for f in cls.functions(*args, **kwargs) if f.startswith('test_')]
    
    @classmethod
    def test(cls, module=None, timeout=60, trials=3):
        module = module or cls.module_path()
        if c.module_exists(module + '.test'):
            c.print('FOUND TEST MODULE', color='yellow')
            module = module + '.test'
        cls = c.module(module)
        self = cls()
        future2fn = {}
        for fn in self.test_fns():
            c.print(f'testing {fn}')
            f = c.submit(getattr(self, fn), timeout=timeout)
            future2fn[f] = fn
        fn2result = {}
        for f in c.as_completed(future2fn, timeout=timeout):
            fn = future2fn[f]
            result = f.result()
            c.print(f'{fn} result: {result}')
            assert result['success'], f'{fn} failed, {result}'
            fn2result[fn] = result
        return fn2result

    ### TIME LAND ###
    
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
        return datetime.datetime.utcnow().strftime("%Y-%m-%d_%H:%M:%S")

    @classmethod
    def time2datetime(cls, t:float):
        import datetime
        return datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d_%H:%M:%S")
    time2date = time2datetime

    @classmethod
    def datetime2time(cls, x:str):
        import datetime
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
            if isinstance(data, str):
                return data
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
    
    @classmethod
    def is_file_module(cls, module = None) -> bool:
        if module != None:
            cls = c.module(module)
        dirpath = cls.dirpath()
        filepath = cls.filepath()
        return bool(dirpath.split('/')[-1] != filepath.split('/')[-1].split('.')[0])
    
    @classmethod
    def is_folder_module(cls,  module = None) -> bool:
        if module != None:
            cls = c.module(module)
        return not cls.is_file_module()

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
        return c.serve(module, **kwargs)
    
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
    
    @classmethod
    def key_addresses(cls,*args, **kwargs ):
        return c.module('key').addresses(*args, **kwargs )
    k2a = key2address

    @classmethod
    def is_key(self, key:str) -> bool:
        return c.module('key').is_key(key)

    @classmethod
    def root_key(cls):
        return c.get_key()

    @classmethod
    def root_key_address(cls) -> str:
        return c.root_key().ss58_address
    
    @classmethod
    def root_keys(cls, search='module', address:bool = False):
        keys = c.keys(search)
        if address:
            key2address = c.key2address(search)
            keys = [key2address.get(k) for k in keys]
        return keys
    
    @classmethod
    def root_addys(cls):
        return c.root_keys(address=True)
    

    def transfer2roots(self, amount:int=1,key:str=None,  n:int=10):
        destinations = c.root_addys()[:n]
        c.print(f'Spreading {amount} to {len(destinations)} keys', color='yellow')
        return c.transfer_many(destinations=destinations, amounts=amount, n=n, key=key)


    def add_root_keys(self, n=1, tag=None, **kwargs):
        keys = []
        for i in range(n):
            key_path = 'module' + '::'+ (tag if tag != None else '') + str(i)
            c.add_key(key_path, **kwargs)
            keys.append(key_path)
        return {'success': True, 'keys': keys, 'msg': 'Added keys'}

    @classmethod
    def asubmit(cls, fn:str, *args, **kwargs):
        
        async def _asubmit():
            kwargs.update(kwargs.pop('kwargs',{}))
            return fn(*args, **kwargs)
        return _asubmit()

    
    @classmethod
    def address2key(cls,*args, **kwargs ):
        return c.module('key').address2key(*args, **kwargs )
    
    
    @classmethod
    def key_addresses(cls,*args, **kwargs ):
        return list(c.module('key').address2key(*args, **kwargs ).keys())
    

    @classmethod
    def get_key_for_address(cls, address:str):
         return c.module('key').get_key_for_address(address)

    @classmethod
    def get_key_address(cls, key):
        return c.get_key(key).ss58_address

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

    @classmethod
    def id(self):
        return self.key.ss58_address
    
    @property
    def ss58_address(self):
        if not hasattr(self, '_ss58_address'):
            self._ss58_address = self.key.ss58_address
        return self._ss58_address
    
    @ss58_address.setter
    def ss58_address(self, value):
        self._ss58_address = value
        return self._ss58_address
    
    
    
    
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
    def hash_map(cls):
        return {
            'code': cls.chash(),
            'commit': cls.commit_hash(),
        }

    @classmethod
    def hash(cls, 
             data: Union[str, bytes], 
             **kwargs) -> bytes:
        if not hasattr(cls, '_hash_module'):
            cls._hash_module = c.module('crypto.hash')()
        return cls._hash_module(data, **kwargs)
    

    @classmethod
    def readme_paths(cls):

        readme_paths =  [f for f in c.ls(cls.dirpath()) if f.endswith('md')]
        return readme_paths

    @classmethod
    def has_readme(cls):
        return len(cls.readme_paths()) > 0
    
    @classmethod
    def readme(cls) -> str:
        readme_paths = cls.readme_paths()
        if len(readme_paths) == 0:
            return ''
        return c.get_text(readme_paths[0])

    @classmethod
    def encrypt(cls, 
                data: Union[str, bytes],
                key: str = None, 
                password: str = None,
                **kwargs
                ) -> bytes:
        """
        encrypt data with key
        """
        key = c.get_key(key)
        return key.encrypt(data, password=password,**kwargs)
    

    def test_encrypt(self):
        data = 'hello world'
        password = 'bitconnect'
        encrypted = self.encrypt(data, password=password)
        decrypted = self.decrypt(encrypted, password=password)
        assert data == decrypted, f'Encryption failed. {data} != {decrypted}'
        return {'success': True, 'msg': 'Encryption successful'}
    

    @classmethod
    def decrypt(cls, 
                data: Union[str, bytes],
                key: str = None, 
                password : str = None,
                **kwargs) -> bytes:
        key = c.get_key(key)
        return key.decrypt(data, password=password, **kwargs)
    
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
        futures = []
        for m in modules:
            job_kwargs = {'module':  m, 'fn': fn, 'network': network, **kwargs}
            future = c.submit(c.call, kwargs=job_kwargs, args=[*args] , timeout=timeout)
            futures.append(future)
        responses = c.wait(futures, timeout=timeout)
        return responses
    

    
    def resolve_key(self, key: str = None) -> str:
        if key == None:
            if hasattr(self, 'key'):
                key = self.key
            key = self.resolve_keypath(key)
        key = self.get_key(key)
        return key  
    
    
    @classmethod
    def type_str(cls, x):
        return type(x).__name__
                
    @classmethod  
    def keys(cls, search = None, ss58=False,*args, **kwargs):
        if search == None:
            search = cls.module_path()
            if search == 'module':
                search = None
        keys = c.module('key').keys(search, *args, **kwargs)
        if ss58:
            keys = [c.get_key_address(k) for k in keys]
        return keys

    @classmethod  
    def get_mem(cls, *args, **kwargs):
        return c.module('key').get_mem(*args, **kwargs)
    
    mem = get_mem
    
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
    def new_key(cls,  *args,  **kwargs):
        return c.module('key').new_key( *args, **kwargs)
    

    @classmethod
    def save_keys(cls, *args,  **kwargs):
        return c.module('key').save_keys(*args, **kwargs)
    savemems = savekeys = save_keys

    @classmethod
    def load_keys(cls, *args,  **kwargs):
        return c.module('key').load_keys(*args, **kwargs)
    loadmems = loadkeys = load_keys

    @classmethod
    def load_key(cls, *args,  **kwargs):
        return c.module('key').load_key(*args, **kwargs)


    def sign(self, data:dict  = None, key: str = None, **kwargs) -> bool:
        key = self.resolve_key(key)
        signature =  key.sign(data, **kwargs)
        return signature
    @classmethod
    def verify(cls, auth, key=None, **kwargs ) -> bool:  
        key = c.get_key(key)
        return key.verify(auth, **kwargs)
    
    @classmethod
    def get_signer(cls, data:dict ) -> bool:        
        return c.module('key').get_signer(data)
    
    @classmethod
    def start(cls, *args, **kwargs):
        return cls(*args, **kwargs)
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
    def is_encrypted(cls, data, prefix=encrypted_prefix):
        if isinstance(data, str):
            if data.startswith(prefix):
                return True
        elif isinstance(data, dict):
            return bool(data.get('encrypted', False) == True)
        else:
            return False
    
    @classmethod
    def fleet(cls,
            module = None, 
            n=2, 
            tag=None, 
            max_workers=10, 
            parallel=True, 
            timeout=20, 
            remote=False,  
            **kwargs):

        if module == None:
            module = cls.module_path()

        if tag == None:
            tag = ''

        futures = []
        for i in range(n):
            f = c.submit(c.serve,  
                            kwargs={'module': module, 'tag':tag + str(i), **kwargs}, 
                            timeout=timeout)
            futures += [f]
        results = []
        for future in  c.as_completed(futures, timeout=timeout):
            result = future.result()
            c.print(result)
            results += [result]

        return results
        
    executor_cache = {}
    @classmethod
    def executor(cls, max_workers:int=None, mode:str="thread", cache:bool = True, maxsize=200, **kwargs):
        if cache:
            if mode in cls.executor_cache:
                return cls.executor_cache[mode]
        executor =  c.module(f'executor.{mode}')(max_workers=max_workers, maxsize=maxsize , **kwargs)
        if cache:
            cls.executor_cache[mode] = executor
        return executor
    

    @classmethod
    def submit(cls, 
                fn, 
                params = None,
                kwargs: dict = None, 
                args:list = None, 
                timeout:int = 20, 
                return_future:bool=True,
                init_args : list = [],
                init_kwargs:dict= {},
                executor = None,
                module: str = None,
                mode:str='thread',
                max_workers : int = 100,
                ):
        kwargs = {} if kwargs == None else kwargs
        args = [] if args == None else args
        if params != None:
            if isinstance(params, dict):
                kwargs = {**kwargs, **params}
            elif isinstance(params, list):
                args = [*args, *params]
            else:
                raise ValueError('params must be a list or a dictionary')
        
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
            method_type = c.classify_fn(getattr(module, fn))
        elif callable(fn):
            method_type = c.classify_fn(fn)
        else:
            raise ValueError('fn must be a string or a callable')
        
        if method_type == 'self':
            module = module(*init_args, **init_kwargs)

        future = executor.submit(fn=fn, args=args, kwargs=kwargs, timeout=timeout)

        if not hasattr(cls, 'futures'):
            cls.futures = []
        
        cls.futures.append(future)
            
        
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
    def client(cls, *args, **kwargs) -> 'Client':
        return c.module('client')(*args, **kwargs)
    
    @classmethod
    def serialize(cls, x, **kwargs):
        return c.serializer().serialize(x, **kwargs)

    @classmethod
    def serializer(cls, *args, **kwargs):
        return  c.module('serializer')(*args, **kwargs)
    
    @classmethod
    def deserialize(cls, x, **kwargs):
        return c.serializer().deserialize(x, **kwargs)
    
    @classmethod
    def process(cls, *args, **kwargs):
        return c.module('process').process(*args, **kwargs)

    @classmethod
    def copy(cls, data: Any) -> Any:
        import copy
        return copy.deepcopy(data)

    @classmethod
    def mv_key(cls, key:str, new_key:str):
        return c.module('key').mv_key(key, new_key)
    
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
    def pwdtree(cls):
        tree2path   =  c.tree2path()
        pwd = c.pwd()
        return {v:k for k,v in tree2path.items()}.get(pwd, None)
    which_tree = pwdtree
    
    @classmethod
    def istree(cls):
        return cls.pwdtree() != None

    @classmethod
    def is_pwd(cls, module:str = None):
        if module != None:
            module = c.module(module)
        else:
            module = cls
        return module.dirpath() == c.pwd()


    
    @classmethod
    def currnet_module(cls):
        return c.module(cls.module_path())
    
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
    def is_error(cls, x:Any):
        if isinstance(x, dict):
            if 'error' in x:
                return True
            if 'success' in x and x['success'] == True:
                return True
        return False
    
    @classmethod
    def is_int(cls, value) -> bool:
        o = False
        try :
            int(value)
            if '.' not in str(value):
                o =  True
        except:
            pass
        return o
    
        
    @classmethod
    def is_float(cls, value) -> bool:
        o =  False
        try :
            float(value)
            if '.' in str(value):
                o = True
        except:
            pass

        return o 


    def update_config(self, k, v):
        self.config[k] = v
        return self.config
    # local update  
    @classmethod
    def update(cls, 
               module = None,
               tree:bool = True,
               namespace: bool = False,
               subspace: bool = False,
               network: str = 'local',
               **kwargs
               ):
        responses = []
        if tree:
            r = c.tree()
            responses.append(r)
        if module != None:
            return c.module(module).update()
        # update local namespace
        c.ip(update=True)
        if namespace:
            responses.append(c.namespace(network=network, update=True))
        if subspace:
            responses.append(c.module('subspace').sync())
        
        c.ip(update=1)

        return {'success': True, 'responses': responses}

    @classmethod
    def sync(cls, *args, **kwargs):
            
        return c.module('subspace')().sync(*args, **kwargs)
        

    @classmethod
    def filter(cls, text_list: List[str], filter_text: str) -> List[str]:
        return [text for text in text_list if filter_text in text]

    @classmethod
    def put_text(cls, path:str, text:str, key=None, bits_per_character=8) -> None:
        # Get the absolute path of the file
        path = cls.resolve_path(path)
        if not isinstance(text, str):
            text = c.python2str(text)
        if key != None:
            text = c.get_key(key).encrypt(text)
        # Write the text to the file
        with open(path, 'w') as file:
            file.write(text)
        # get size
        text_size = len(text)*bits_per_character
    
        return {'success': True, 'msg': f'Wrote text to {path}', 'size': text_size}
    
    def rm_lines(self, path:str, start_line:int, end_line:int) -> None:
        # Get the absolute path of the file
        text = c.get_text(path)
        text = text.split('\n')
        text = text[:start_line-1] + text[end_line:]
        text = '\n'.join(text)
        c.put_text(path, text)
        return {'success': True, 'msg': f'Removed lines {start_line} to {end_line} from {path}'}
    
    def rm_line(self, path:str, line:int, text=None) -> None:
        # Get the absolute path of the file
        text =  c.get_text(path)
        text = text.split('\n')
        text = text[:line-1] + text[line:]
        text = '\n'.join(text)
        c.put_text(path, text)
        return {'success': True, 'msg': f'Removed line {line} from {path}'}
        # Write the text to the file
            
    @classmethod
    def add_line(cls, path:str, text:str, line=None) -> None:
        # Get the absolute path of the file
        path = cls.resolve_path(path)
        text = str(text)
        # Write the text to the file
        if line != None:
            line=int(line)
            lines = c.get_text(path).split('\n')
            lines = lines[:line] + [text] + lines[line:]
            c.print(lines)

            text = '\n'.join(lines)
        with open(path, 'w') as file:
            file.write(text)


        return {'success': True, 'msg': f'Added line to {path}'}

    add_text = add_line
    
           
           
    @classmethod
    def readlines(self, path:str,
                  start_line:int = 0,
                  end_line:int = 0, 
                  resolve:bool = True) -> List[str]:
        # Get the absolute path of the file
        if resolve:
            path = self.resolve_path(path)
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
    def free_gpu_memory(cls, *args, **kwargs) -> Dict[int, float]:
        return c.module('os').free_gpu_memory(*args, **kwargs)
    
    
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
    def find_code_lines(cls,  search:str = None , module=None) -> List[str]:
        module_code = c.module(module).code()
        return c.find_lines(search=search, text=module_code)



    @classmethod
    def find_lines(self, text:str, search:str) -> List[str]:
        """
        Finds the lines in text with search
        """
        found_lines = []
        lines = text.split('\n')
        for line in lines:
            if search in line:
                found_lines += [line]
        
        return found_lines

    
            

    # @classmethod
    # def code2module(cls, code:str='print x'):
    #      new_module = 


    @classmethod
    def new_module( cls,
                   module : str ,
                   repo : str = None,
                   base_module : str = 'demo',
                   tree : bool = 'commune',
                   overwrite : bool  = True,
                   **kwargs):
        
        """ Makes directories for path.
        """
        if module == None: 
            assert repo != None, 'repo must be specified if module is not specified'
            module = os.path.basename(repo).replace('.git','').replace(' ','_').replace('-','_').lower()
        tree_path = c.tree2path().get(tree)
        
        class_name = ''
        for m in module.split('.'):
            class_name += m[0].upper() + m[1:] # capitalize first letter

        if c.module_exists(module): 
            if overwrite:
                module_path = c.module(module).dirpath() if c.is_file_module(module) else c.module(module).filepath()
                c.rm(module_path)
            else:
                return {'success': False,
                        'path': module_path,
                         'msg': f' module {module} already exists, set overwrite=True to overwrite'}

        # get the code ready from the base module
        c.print(f'Getting {base_module}')
        base_module = c.module(base_module)
        is_folder_module = base_module.is_folder_module()

        base_module_class = base_module.class_name()
        module_class_name = ''.join([m[0].upper() + m[1:] for m in module.split('.')])

        # build the path2text dictionary 
        if is_folder_module:
            dirpath = tree_path + '/'+ module.replace('.','/') + '/'
            base_dirpath = base_module.dirpath()
            path2text = c.path2text( base_module.dirpath())
            path2text = {k.replace(base_dirpath +'/',dirpath ):v for k,v in path2text.items()}         
        else:
            module_path = tree_path + '/'+ module.replace('.','/') + '.py'
            code = base_module.code()
            path2text = {module_path: code}

        og_path2text = c.copy(path2text)
        for path, text in og_path2text.items():
            file_type = path.split('.')[-1]
            is_module_python_file = (file_type == 'py' and 'class ' + base_module_class in text)

            if is_folder_module:
                if file_type ==  'yaml' or is_module_python_file:
                    path_filename = path.split('/')[-1]
                    new_filename = module.replace('.', '_') + '.'+ file_type
                    path = path[:-len(path_filename)] + new_filename


            if is_module_python_file:
                text = text.replace(base_module_class, module_class_name)

            path2text[path] = text
            c.put_text(path, text)
            c.print(f'Created {path} :: {module}')
       
        assert c.module_exists(module), f'Failed to create module {module}'
        

        return {'success': True, 'msg': f'Created module {module}', 'path': path, 'paths': list(c.path2text(c.module(module).dirpath()).keys())}
    
    add_module = new_module
    
    make_dir= mkdir

    @classmethod
    def path2text(cls, path:str, relative=False):

        path = cls.resolve_path(path)
        assert os.path.exists(path), f'path {path} does not exist'
        if os.path.isdir(path):
            filepath_list = c.glob(path + '/**')
        else:
            assert os.path.exists(path), f'path {path} does not exist'
            filepath_list = [path] 
        path2text = {}
        for filepath in filepath_list:
            try:
                path2text[filepath] = c.get_text(filepath)
            except Exception as e:
                pass
        if relative:
            pwd = c.pwd()
            path2text = {os.path.relpath(k, pwd):v for k,v in path2text.items()}
        return path2text

    @classmethod
    def resolve_module(cls, module=None):
        if module == None:
            module = cls
        if isinstance(module, str):
            module = c.module(module)
        return module


    thread_map = {}

    @classmethod
    def get_fn(cls, fn:str, 
               seperator='/',ignore_module_pattern:bool = False):
        
        if isinstance(fn, str):
            if seperator in fn and (not ignore_module_pattern):
                # module{sperator}fn
                fn_splits = fn.split(seperator)
                # incase you have multiple seperators in the  name
                module = seperator.join(fn_splits[:-1])
                fn = fn_splits[-1]
                # get the model
                module = c.module(module)
            else:
                module = cls
            fn = getattr(module, fn)
        elif callable(fn):
            pass
        elif isinstance(fn, property):
            pass
        else:
            raise ValueError(f'fn must be a string or callable, got {type(fn)}')
        # assert callable(fn), 'Is not callable'
        return fn
    


    resolve_fn = get_fn
       
            
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
            

    def link_cmd(cls, old, new):
        
        link_cmd = cls.get('link_cmd', {})
        assert isinstance(old, str), old
        assert isinstance(new, str), new
        link_cmd[new] = old 
        
        cls.put('link_cmd', link_cmd)


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
        
        kwargs = c.locals2kwargs(kwargs)
        if 'remote' in kwargs:
            kwargs['remote'] = False
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
        
        return cls.launch(fn=fn, 
                   module = module,
                    kwargs=kwargs,
                    refresh=refresh,
                    name=name, 
                    cwd = cwd or cls.dirpath(),
                    **extra_launch_kwargs)
        
        return {'success': True, 'msg': f'Launched {name}', 'timestamp': c.timestamp()}

    @classmethod
    def choice(cls, options:Union[list, dict])->list:
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
        return random.choice(cls.colors())
    randcolor = randcolour = colour = color = random_colour = random_color

    @classmethod
    def random_float(cls, min=0, max=1):
        return random.uniform(min, max)


    @classmethod
    def random_ratio_selection(cls, x:list, ratio:float = 0.5)->list:
        if type(x) in [float, int]:
            x = list(range(int(x)))
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
    def obj2typestr(cls, obj):
        return str(type(obj)).split("'")[1]

    @classmethod
    def is_coroutine(cls, future):
        """
        returns True if future is a coroutine
        """
        return cls.obj2typestr(future) == 'coroutine'

    @classmethod
    def as_completed(cls , futures:list, timeout:int=10, **kwargs):
        return concurrent.futures.as_completed(futures, timeout=timeout)
    @classmethod
    def wait(cls, futures:list, timeout:int = None, generator:bool=False, return_dict:bool = True) -> list:
        import concurrent.futures


        is_singleton = bool(not isinstance(futures, list))

        futures = [futures] if is_singleton else futures
        # if type(futures[0]) in [asyncio.Task, asyncio.Future]:
        #     return c.gather(futures, timeout=timeout)
            
        if len(futures) == 0:
            return []
        if c.is_coroutine(futures[0]):
            return c.gather(futures, timeout=timeout)
        
        future2idx = {future:i for i,future in enumerate(futures)}

        if timeout == None:
            if hasattr(futures[0], 'timeout'):
                timeout = futures[0].timeout
            else:
                timeout = 30
    
        if generator:
            def get_results(futures):
                try: 
                    for future in concurrent.futures.as_completed(futures, timeout=timeout):
                        if return_dict:
                            idx = future2idx[future]
                            yield {'idx': idx, 'result': future.result()}
                        else:
                            yield future.result()
                except Exception as e:
                    c.print(f'Error: {e}')
                    yield None
                
        else:
            def get_results(futures):
                results = [None]*len(futures)
                try:
                    for future in concurrent.futures.as_completed(futures, timeout=timeout):
                        idx = future2idx[future]
                        results[idx] = future.result()
                        del future2idx[future]
                    if is_singleton: 
                        results = results[0]
                except Exception as e:
                    unfinished_futures = [future for future in futures if future in future2idx]
                    c.print(f'Error: {e}, {len(unfinished_futures)} unfinished futures with timeout {timeout} seconds')
                return results

                
            
        return get_results(futures)
    
    @staticmethod
    def address2ip(address:str) -> str:
        return str('.'.join(address.split(':')[:-1]))

    @staticmethod
    def as_completed( futures, timeout=10, **kwargs):
        import concurrent.futures
        return concurrent.futures.as_completed(futures, timeout=timeout, **kwargs)

    @classmethod
    def gather(cls,jobs:list, timeout:int = 20, loop=None)-> list:

        if loop == None:
            loop = c.get_event_loop()

        if not isinstance(jobs, list):
            singleton = True
            jobs = [jobs]
        else:
            singleton = False

        assert isinstance(jobs, list) and len(jobs) > 0, f'Invalid jobs: {jobs}'
        # determine if we are using asyncio or multiprocessing

        # wait until they finish, and if they dont, give them none

        # return the futures that done timeout or not
        async def wait_for(future, timeout):
            try:
                result = await asyncio.wait_for(future, timeout=timeout)
            except asyncio.TimeoutError:
                result = {'error': f'TimeoutError: {timeout} seconds'}

            return result
        
        jobs = [wait_for(job, timeout=timeout) for job in jobs]
        future = asyncio.gather(*jobs)
        results = loop.run_until_complete(future)

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
        path1 = cls.resolve_path(path1)
        path2 = cls.resolve_path(path2)
        return c.module('os').mv(path1, path2)

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
        return {'success': True, 'msg': f'Copied {path1} to {path2}'}
    

    # def cp_module(self, module:str, new_module:str = None, refresh:bool = False):
    #     if refresh:c
    #         self.rm_module(new_module)
    
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
    
    def is_fn_self(self, fn):
        fn = self.resolve_fn(fn)
        c.print(dir(fn))
        return hasattr(fn, '__self__') and fn.__self__ == self

    @staticmethod
    def retry(fn, trials:int = 3, verbose:bool = True):
        # if fn is a self method, then it will be a bound method, and we need to get the function
        if hasattr(fn, '__self__'):
            fn = fn.__func__
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
        fns = cls.fns(search=search)
        fn2str = {}
        for fn in fns:
            fn2str[fn] = cls.fn_code(fn)
            
        return fn2str
    @classmethod
    def fn2hash(cls, fn=None , mode='sha256', **kwargs):
        fn2hash = {}
        for k,v in cls.fn2str(**kwargs).items():
            fn2hash[k] = c.hash(v,mode=mode)
        if fn:
            return fn2hash[fn]
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

    @classmethod
    def python2types(cls, d:dict)-> dict:
        return {k:str(type(v)).split("'")[1] for k,v in d.items()}
    
    @staticmethod
    def echo(x):
        return x
    

    
    @classmethod
    def pool(cls , n=5, **kwargs):
        for i in range(n):
            cls.serve(tag=str(i), **kwargs)
        
    @classmethod
    def self_functions(cls):
        return c.classify_fns(cls)['self']
    
    self_functions = self_functions

    @classmethod
    def classify_fns(cls, obj= None, mode=None):
        method_type_map = {}
        obj = obj or c.module(obj)
        if isinstance(obj, str):
            obj = c.module(obj)
        for attr_name in dir(obj):
            method_type = None
            try:
                method_type = cls.classify_fn(getattr(obj, attr_name))
            except Exception as e:
                continue
        
            if method_type not in method_type_map:
                method_type_map[method_type] = []
            method_type_map[method_type].append(attr_name)
        if mode != None:
            method_type_map = method_type_map[mode]
        return method_type_map


    @classmethod
    def get_function_args(cls, fn):
        if not callable(fn):
            fn = cls.get_fn(fn)
        args = inspect.getfullargspec(fn).args
        return args

    
    
    @classmethod
    def has_function_arg(cls, fn, arg:str):
        args = cls.get_function_args(fn)
        return arg in args

    
    fn_args = get_fn_args =  get_function_args
    
    @classmethod
    def classify_fn(cls, fn):
        
        if not callable(fn):
            fn = cls.get_fn(fn)
        if not callable(fn):
            return None
        args = cls.get_function_args(fn)
        if len(args) == 0:
            return 'static'
        elif args[0] == 'self':
            return 'self'
        else:
            return 'class'
        
    def fn2type(self):
        fn2type = {}
        fns = self.fns()
        for f in fns:
            if callable(getattr(self, f)):
                fn2type[f] = self.classify_fn(getattr(self, f))
        return fn2type

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
        cls = c.resolve_class(obj)
        return list(cls.__mro__[1:-1])

    @staticmethod
    def get_parent_functions(cls) -> List[str]:
        parent_classes = c.get_parents(cls)
        function_list = []
        for parent in parent_classes:
            function_list += c.get_functions(parent)

        return list(set(function_list))

    @classmethod
    def is_property(cls, fn: 'Callable') -> bool:
        '''
        is the function a property
        '''
        try:
            fn = cls.get_fn(fn, ignore_module_pattern=True)
        except :
            return False

        return isinstance(fn, property)


    @classmethod
    def property_fns(cls) -> bool:
        '''
        Get a list of property functions in a class
        '''
        return [fn for fn in dir(cls) if cls.is_property(fn)]

    @classmethod
    def get_functions(cls, obj: Any = None,
                      search = None,
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
            if search != None and search not in fn_name:
                continue
            
            fn_obj = getattr(obj, fn_name)
            
            if not callable(fn_obj):
                continue
            # skip hidden functions if include_hidden is False
            if (not include_hidden) and ((fn_name.startswith('__') or fn_name.endswith('_'))):
                if fn_name != '__init__':
                    continue

            # if the function is in the parent class, skip it
            if  (fn_name in parent_functions) and (not include_parents):
                continue

            # if the function is a property, skip it
            if hasattr(type(obj), fn_name) and \
                isinstance(getattr(type(obj), fn_name), property):
                continue
            
            # if the function is callable, include it
            if callable(getattr(obj, fn_name)):
                functions.append(fn_name)


        functions = list(set(functions))     
            
        return functions

    
    
    @classmethod
    def self_fns(cls: Union[str, type], obj=None):
        '''
        Gets the self methods in a class
        '''
        obj = obj or cls
        functions =  c.get_functions(obj)
        signature_map = {f:cls.get_function_args(getattr(obj, f)) for f in functions}
        return [k for k, v in signature_map.items() if 'self' in v]
    
    self_functions = get_self_functions = self_fns 

    @classmethod
    def class_functions(cls: Union[str, type], obj=None):
        '''
        Gets the self methods in a class
        '''
        obj = obj or cls
        functions =  c.get_functions(obj)
        signature_map = {f:cls.get_function_args(getattr(obj, f)) for f in functions}
        return [k for k, v in signature_map.items() if 'cls' in v]
    
    class_methods = get_class_methods =  class_fns = class_functions

    @classmethod
    def static_functions(cls: Union[str, type], obj=None):
        '''
        Gets the self methods in a class
        '''
        obj = obj or cls
        functions =  c.get_functions(obj)
        signature_map = {f:cls.get_function_args(getattr(obj, f)) for f in functions}
        return [k for k, v in signature_map.items() if not ('self' in v or 'cls' in v)]
    
    static_methods = static_fns =  static_functions
    
    @classmethod
    def fn_signature(cls, fn) -> dict: 
        '''
        get the signature of a function
        '''
        if isinstance(fn, str):
            fn = getattr(cls, fn)
        return dict(inspect.signature(fn)._parameters)
    
    get_function_signature = fn_signature
    @classmethod
    def is_arg_key_valid(cls, key='config', fn='__init__'):
        fn_signature = cls.fn_signature(fn)
        if key in fn_signature: 
            return True
        else:
            for param_info in fn_signature.values():
                if param_info.kind._name_ == 'VAR_KEYWORD':
                    return True
        
        return False

    @classmethod
    def has_var_keyword(cls, fn='__init__', fn_signature=None):
        if fn_signature == None:
            fn_signature = cls.resolve_fn(fn)
        for param_info in fn_signature.values():
            if param_info.kind._name_ == 'VAR_KEYWORD':
                return True
        return False

    @staticmethod
    def get_function_input_variables(fn)-> dict:
        return list(c.resolve_fn(fn).keys())

    @classmethod
    def fn_defaults(cls, fn):
        """
        Gets the function defaults
        """
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
    
    
    @classmethod
    def jload(cls, json_string):
        import json
        return json.loads(json_string.replace("'", '"'))
    
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
    def my_value(cls, *args, **kwargs):
        return c.module('subspace')().my_value(*args, **kwargs)
    
    @classmethod
    def get_value(cls, *args, **kwargs):
        return c.module('subspace')().get_value(*args, **kwargs)
    
    @classmethod
    def get_stake_to(cls, *args, **kwargs):
        return c.module('subspace')().get_stake_to(*args, **kwargs)
    
    @classmethod
    def get_stake_from(cls, *args, **kwargs):
        return c.module('subspace')().get_stake_from(*args, **kwargs)
    

    @classmethod
    def partial(cls, fn, *args, **kwargs):
        from functools import partial
        return partial(fn, *args, **kwargs)
        
        
    @staticmethod
    def sizeof( obj):
        import sys
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
    def code(cls, module = None, search=None, *args, **kwargs):
        if '/' in str(module) or module in cls.fns():
            return c.fn_code(module)
            
        module = cls.resolve_module(module)
        text =  c.get_text( module.pypath(), *args, **kwargs)
        if search != None:
            find_lines = c.find_lines(text=text, search=search)
            return find_lines
        return text
        

    @classmethod
    def get_text_line(cls, module = None, *args, **kwargs):
        module = cls.resolve_module(module)
        return c.get_text_line( module.pypath(), *args, **kwargs)
    

    pycode = code

    @classmethod
    def chash(cls,  *args, **kwargs):
        """
        The hash of the code, where the code is the code of the class (cls)
        """
        code = cls.code(*args, **kwargs)
        return c.hash(code)
    
    @classmethod
    def match_module_hash(cls, hash:str, module:str=None, *args, **kwargs):
        '''
        match the hash of a module
        '''
        
        if module != None:
            module = c.module(module)
        else:
            module = cls
        return module.chash(*args, **kwargs) == hash
    
    @classmethod
    def find_code_line(cls, search:str, code:str = None):
        if code == None:
            code = cls.code() # get the code
        found_lines = [] # list of found lines
        for i, line in enumerate(code.split('\n')):
            if search in line:
                found_lines.append({'idx': i+1, 'text': line})
        if len(found_lines) == 0:
            return None
        elif len(found_lines) == 1:
            return found_lines[0]['idx']
        return found_lines
    
    @classmethod
    def fn_info(cls, fn:str='test_fn') -> dict:
        r = {}
        code = cls.fn_code(fn)
        lines = code.split('\n')
        mode = 'self'
        if '@classmethod' in lines[0]:
            mode = 'class'
        elif '@staticmethod' in lines[0]:
            mode = 'static'
    
        start_line_text = None
        lines_before_fn_def = 0
        for l in lines:
            
            if f'def {fn}('.replace(' ', '') in l.replace(' ', ''):
                start_line_text = l
                break
            else:
                lines_before_fn_def += 1
            
        assert start_line_text != None, f'Could not find function {fn} in {cls.pypath()}'
        module_code = cls.code()
        start_line = cls.find_code_line(start_line_text, code=module_code) - lines_before_fn_def - 1
        end_line = start_line + len(lines)   # find the endline
        has_docs = bool('"""' in code or "'''" in code)
        filepath = cls.filepath()

        # start code line
        for i, line in enumerate(lines):
            
            is_end = bool(')' in line and ':' in line)
            if is_end:
                start_code_line = i
                break 

        
        return {
            'start_line': start_line,
            'end_line': end_line,
            'has_docs': has_docs,
            'code': code,
            'n_lines': len(lines),
            'hash': c.hash(code),
            'path': filepath,
            'start_code_line': start_code_line + start_line ,
            'mode': mode
            
        }
    

    @classmethod
    def set_line(cls, idx:int, text:str):
        code = cls.code()
        lines = code.split('\n')
        if '\n' in text:
            front_lines = lines[:idx]
            back_lines = lines[idx:]
            new_lines = text.split('\n')
            c.print(new_lines)
            lines = front_lines + new_lines + back_lines
        else:
            lines[idx-1] = text
        new_code = '\n'.join(lines)
        cls.put_text(cls.filepath(), new_code)
        return {'success': True, 'msg': f'Set line {idx} to {text}'}

    @classmethod
    def add_line(cls, idx=0, text:str = '',  module=None  ):
        '''
        ### Documentation
        
        #### `add_line` Method
        
        **Description:**
        
        The `add_line` method is a class method that allows you to insert one or multiple lines of text at a specified index in the code of a file or module. If no module is provided, it defaults to modifying the code of the class itself.
        
        **Parameters:**
        
        - `idx` (optional): The index (line number) at which the new text should be inserted. Default is `0`.
        - `text` (optional): A string representing the new line(s) of text to be added. If multiple lines are provided, they should be separated by '\n'. Default is an empty string `''`.
        - `module` (optional): The module whose code should be modified. If `None`, the class's own code is modified.
        
        **Returns:**
        
        A dictionary with two key-value pairs:
        - `'success'`: A boolean value indicating the success of the operation.
        - `'msg'`: A formatted string message indicating the line number and text that was added.
        
        **Usage:**
        
        ```python
        result = ClassName.add_line(idx=5, text="New line of code", module='some_module')
        print(result)
        # Output: {'success': True, 'msg': 'Added line 5 to New line of code'}
        ```
        
        **Notes:**
        
        - The method accesses and modifies the code by converting it into a list of lines.
        - After inserting the new lines of text, the modified code is joined back into a single string and updated within the file or module.
        - The method assumes that the class contains `code`, `put_text`, and `filepath` methods which are responsible for retrieving the current code, updating the text in the file, and providing the file path respectively.
        
        ---
        
        Developers should ensure that the index provided is within the bounds of the code line count to avoid any errors. The method does not perform any syntax or error-checking on the new lines of text to be added, so developers should ensure the text is valid code before insertion.
        '''

        code = cls.code() if module == None else c.module(module).code()
        lines = code.split('\n')
        new_lines = text.split('\n') if '\n' in text else [text]
        lines = lines[:idx] + new_lines + lines[idx:]
        new_code = '\n'.join(lines)
        cls.put_text(cls.filepath(), new_code)
        return {'success': True, 'msg': f'Added line {idx} to {text}'}

    @classmethod
    def get_line(cls, idx):


        code = cls.code()
        lines = code.split('\n')
        assert idx < len(lines), f'idx {idx} is out of range for {len(lines)}'  
        line =  lines[max(idx, 0)]
        c.print(len(line))
        return line



    
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
    @staticmethod
    def num_words( text):
        return len(text.split(' '))
    

    def generate_completions(self, past_tokens = 10, future_tokens = 10, tokenizer:str='gpt2', mode:str='lines', **kwargs):
        code = self.code()
        code_lines = code.split('\n')
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

    @classmethod
    def is_repo(cls, path='.'):
        return c.module('repo')().is_repo(path)
    
    @classmethod
    def staked(cls, *args, **kwargs):
        return c.module('subspace')().staked(*args, **kwargs)

    @classmethod
    def stake_transfer(cls, *args, **kwargs):
        return c.module('subspace')().stake_transfer(*args, **kwargs)

    @classmethod
    def add_profit_shares(cls, *args, **kwargs):
        return c.module('subspace')().add_profit_shares(*args, **kwargs)

    @classmethod
    def profit_shares(cls, *args, **kwargs):
        return c.module('subspace')().profit_shares(*args, **kwargs)


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

    @classmethod
    def update_modules(cls, *args, **kwargs):
        return c.module('subspace')().update_modules(*args, **kwargs)
    
    @classmethod
    def set_weights(cls, *args, **kwargs):
        return c.module('subspace')().vote(*args, **kwargs)
    vote = set_weights
    @classmethod
    def set_weights(cls, *args, **kwargs):
        return c.module('subspace')().vote(*args, **kwargs)

    @classmethod
    def vote_loop(cls, *args, **kwargs):
        return c.module('vali.parity').vote_loop(*args, **kwargs)
    
    voteloop = vote_loop 

    

    @classmethod
    def self_vote(cls, *args, **kwargs):
        return c.module('subspace')().self_vote(*args, **kwargs)
    @classmethod
    def self_vote_pool(cls, *args, **kwargs):
        return c.module('subspace')().self_vote_pool(*args, **kwargs)
    
    @classmethod
    def stake(cls, *args, **kwargs):
        return c.module('subspace')().stake(*args, **kwargs)
    
    @classmethod
    def my_stake_from(cls, *args, **kwargs):
        return c.module('subspace')().my_stake_from(*args, **kwargs)
    
    @classmethod
    def my_stake_to(cls, *args, **kwargs):
        return c.module('subspace')().my_stake_to(*args, **kwargs)
    

    @classmethod
    def stake_many(cls, *args, **kwargs):
        return c.module('subspace')().stake_many(*args, **kwargs)
    @classmethod
    def transfer_many(cls, *args, **kwargs):
        return c.module('subspace')().transfer_many(*args, **kwargs)

    @classmethod
    def random_word(cls, *args, n=1, seperator='_', **kwargs):
        random_words = c.module('key').generate_mnemonic(*args, **kwargs).split(' ')[0]
        random_words = random_words.split(' ')[:n]
        if n == 1:
            return random_words[0]
        else:
            return seperator.join(random_words.split(' ')[:n])
    @classmethod
    def random_words(cls, n=2, **kwargs):
        return c.module('key').generate_mnemonic(n=n, **kwargs)
    @classmethod
    def unstake_many(cls, *args, **kwargs):
        return c.module('subspace')().unstake_many(*args, **kwargs)


    unstake_all = unstake_many
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
    @classmethod
    def balances(cls, *args, **kwargs):
        return c.module('subspace')().balance(*args, **kwargs)


    @classmethod
    def get_balance(cls, *args, **kwargs):
        return c.module('subspace')().get_balance(*args, **kwargs)
    @classmethod
    def get_balances(cls, *args, **kwargs):
        return c.module('subspace')().get_balances(*args, **kwargs)

    @classmethod
    def key2balances(cls, *args, **kwargs):
        return c.module('subspace')().key2balances(*args, **kwargs)

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
    


    @staticmethod
    def valid_ss58_address(address:str):
        return c.module('key').valid_ss58_address(str(address))
    is_valid_ss58_address = valid_ss58_address

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
        return c.module('subspace')().key2stake( *args, **kwargs)

    def live_keys(self,  *args, **kwargs):
        return c.module('subspace')().live_keys( *args, **kwargs)
    def dead_keys(self,  *args, **kwargs):
        return c.module('subspace')().dead_keys( *args, **kwargs)
    @classmethod
    def key2balance(cls, *args, **kwargs):
        return c.module('subspace')().key2balance(*args, **kwargs)
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
    def n(self, *args, **kwargs):
        return c.module('subspace')().n(*args, **kwargs)

    @classmethod
    def query_map(self, *args, **kwargs):
        return c.module('subspace')().query_map(*args, **kwargs)
    
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
    def subnet2netuid(cls, *args, **kwargs):
        return c.module('subspace')().subnet2netuid(*args, **kwargs)
    
    @classmethod
    def netuid2subnet(cls, *args, **kwargs):
        return c.module('subspace')().netuid2subnet(*args, **kwargs)
    

    @classmethod
    def subnet(cls, *args, **kwargs):
        return c.module('subspace')().subnet(*args, **kwargs)

    @classmethod
    def netuids(cls, *args, **kwargs):
        return c.module('subspace')().netuids(*args, **kwargs)
    
    def glob_hash(self, path=None, **kwargs):
        
        glob_dict = c.glob(path, **kwargs)
        glob_hash = c.hash(glob_dict)
        c.put('glob_hash', glob_hash)
    
    @classmethod
    def my_subnets(cls, *args, **kwargs):
        return c.module('subspace')().my_subnets(*args, **kwargs)

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
        return c.module('subspace')().key2stake(*args, **kwargs)
        
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
        return c.module('subspace')().n(*args, **kwargs)
    @classmethod
    def stats(cls, *args, **kwargs):
        return c.module('subspace')().stats(*args, **kwargs)

    @classmethod
    def vstats(cls, *args, **kwargs):
        return c.module('vali').all_stats(*args, **kwargs)
    @classmethod
    def valis(cls, network=None):
        return c.servers('vali', network=network)
    
    @classmethod
    def check_valis(cls, *args, **kwargs):
        return c.module('subspace')().check_valis(*args, **kwargs)
    
    @classmethod
    def check_servers(cls, *args, **kwargs):
        return c.module('subspace')().check_servers(*args, **kwargs)

    @classmethod
    def scan(cls, 
                 search=None, 
                 max_futures:int=100, 
                 network='local', 
                 update=False, 
                 schema=True, 
                 namespace=True, 
                 hardware=True, 
                 **kwargs):

        infos = {} 
        namespace = c.namespace(search=search, network=network, update=update)
        futures = []
        name2future = {}
        for name, address in namespace.items():
            future = [c.submit(c.call, kwargs={'fn': 'info', 'hardware': hardware, 'schema': schema }, module='subspace')]
            name2future[name] = future
            futures = list(name2future.values())
            if len(name2future) >= max_futures:
                for f in c.as_completed(futures):
                    name2future.pop(f)
                    result = f.result()
                    c.print(result)
                    if 'error' not in result:
                        infos[name] = result
        







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
        cls.putc('shortcuts', shortcuts)
        return {'success': True, 'msg': f'added shortcut ({shortcut} -> {name})'}

    @classmethod
    def resolve_shortcut(cls, name:str) -> str:
        return c.getc('shortcuts').get(name, name)
    


    @classmethod
    def talk(cls, *args, **kwargs):
        return c.module('model.openrouter')().talk(*args, **kwargs)

    @classmethod
    def yesno(self, prompt:str):
        return c.module('model.openrouter')().talk(f"{prompt} give a yes or no response ONLY IN ONE WORD", max_tokens=10)
    ask = a = talk

    @classmethod
    def containers(cls):
        return c.module('docker').containers()

    @staticmethod
    def chunk(sequence:list = [0,2,3,4,5,6,6,7],
            chunk_size:int=4,
            num_chunks:int= None):
        assert chunk_size != None or num_chunks != None, 'must specify chunk_size or num_chunks'
        if chunk_size == None:
            chunk_size = len(sequence) / num_chunks
        if chunk_size > len(sequence):
            return [sequence]
        if num_chunks == None:
            num_chunks = int(len(sequence) / chunk_size)
        if num_chunks == 0:
            num_chunks = 1
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
        c.cmd(f'{c.libpath}/scripts/nvidia_docker_setup.sh', cwd=self.libpath, verbose=True, bash=True)

    def install_docker(self):
        self.chmod_scripts()
        c.cmd(f'{c.libpath}/scripts/install_docker.sh', cwd=self.libpath, verbose=True, bash=True)

    @classmethod
    def install_rust(cls, sudo=True) :
        cls.chmod_scripts()
        c.cmd(f'{c.libpath}/scripts/install_rust_env.sh', cwd=cls.libpath, verbose=True, bash=True, sudo=sudo)

    @classmethod
    def install_npm(cls, sudo=False) :
        c.cmd('apt install npm', sudo=sudo)

    @classmethod
    def install_pm2(cls, sudo=True) :
        c.cmd('npm install pm2 -g', sudo=sudo)

    @classmethod
    def install_python(cls, sudo=True) :
        c.cmd('apt install -y python3-dev python3-pip', verbose=True, bash=True, sudo=sudo)

    def cancel(self, futures):
        for f in futures:
            f.cancel()
        return {'success': True, 'msg': 'cancelled futures'}
       
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
                result = cls.get(fn_name, **cache_params)
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
    def name2compose(self, **kwargs):
        return c.module('docker').name2compose(**kwargs)

    @classmethod
    def generator(cls, n=10):
        for i in range(n):
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
        if isinstance(obj, str):
            if not hasattr(cls, obj):
                return False
            obj = getattr(cls, obj)
        if not callable(obj):
            result = inspect.isgenerator(obj)
        else:
            result =  inspect.isgeneratorfunction(obj)
        return result
    
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
                    name = None,
                    tag = None,
                    start:bool = True,
                    tag_seperator:str='::', 
                    **extra_kwargs):
        
        if isinstance(fn, str):
            fn = c.get_fn(fn)
        if args == None:
            args = []
        if kwargs == None:
            kwargs = {}

        assert callable(fn), f'target must be callable, got {fn}'
        assert  isinstance(args, list), f'args must be a list, got {args}'
        assert  isinstance(kwargs, dict), f'kwargs must be a dict, got {kwargs}'
        
        # unique thread name
        if name == None:
            name = fn.__name__
            cnt = 0
            while name in cls.thread_map:
                cnt += 1
                if tag == None:
                    tag = ''
                name = name + tag_seperator + tag + str(cnt)
        
        if name in cls.thread_map:
            cls.thread_map[name].cancel()

        t = threading.Thread(target=fn, args=args, kwargs=kwargs, **extra_kwargs)
        # set the time it starts
        setattr(t, 'start_time', c.time())
        t.daemon = daemon
        if start:
            t.start()
        cls.thread_map[name] = t
        return t

    @classmethod
    def join_threads(cls, threads:[str, list]):

        threads = cls.thread_map
        for t in threads.values():
            # throw error if thread is not in thread_map
            t.join()
        return {'success': True, 'msg': 'all threads joined', 'threads': threads}

    @classmethod
    def threads(cls, search:str=None, **kwargs):
        threads = list(cls.thread_map.keys())
        if search != None:
            threads = [t for t in threads if search in t]
        return threads

    @classmethod
    def thread_count(cls):
        return threading.active_count()
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
    def role2users(cls, *args, **kwargs):
        return c.module('user')().role2users(*args, **kwargs)
    @classmethod
    def is_user(cls, address):
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
    def is_admin(cls, address:str):
        return c.module('user').is_admin(address=address)
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
    def getcwd(cls):
        return os.getcwd()

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
        return c.module('emoji').emoji(name)

    @classmethod
    def emojis(cls, search = None):
        
        emojis =  c.module('emoji').emojis
        if search != None:
            emojis = {k:v for k,v in emojis.items() if search in k}
        return 

    @staticmethod
    def tqdm(*args, **kwargs):
        from tqdm import tqdm
        return tqdm(*args, **kwargs)
    progress = tqdm
    
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
    def type(cls,x ):
        return type(x).__name_
        
    ## API MANAGEMENT ##
    
    def set_api_key(self, api_key:str, cache:bool = True):
        api_key = os.getenv(str(api_key), None)
        if api_key == None:
            api_key = self.get_api_key()

        
        self.api_key = api_key
        if cache:
            self.add_api_key(api_key)

        assert isinstance(api_key, str)


    @classmethod
    def add_api_key(cls, api_key:str):
        assert isinstance(api_key, str)
        api_keys = cls.get('api_keys', [])
        api_keys.append(api_key)
        api_keys = list(set(api_keys))
        cls.put('api_keys', api_keys)
        return {'api_keys': api_keys}


    @classmethod
    def add_api_keys(cls, *api_keys:str):
        if len(api_keys) == 1 and isinstance(api_keys[0], list):
            api_keys = api_keys[0]
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
        api_keys = c.get(cls.resolve_path('api_keys'), [])
        for i in range(len(api_keys)):
            if api_key == api_keys[i]:
                api_keys.pop(i)
                break   
        path = cls.resolve_path('api_keys')
        c.put(path, api_keys)
        return {'api_keys': api_keys}

    @classmethod
    def get_api_key(cls, module=None):
        if module != None:
            cls = c.module(module)
        api_keys = cls.api_keys()
        if len(api_keys) == 0:
            return None
        else:
            return c.choice(api_keys)

    @classmethod
    def api_keys(cls):
        return c.get(cls.resolve_path('api_keys'), [])
    

    @classmethod
    def rm_api_keys(self):
        c.put(self.resolve_path('api_keys'), [])
        return {'api_keys': []}

    @classmethod
    def send_api_keys(cls, module:str, network='local'):
        api_keys = cls.api_keys()
        assert len(api_keys) > 0, 'no api keys to send'
        module = c.connect(module, network=network)
        return module.add_api_keys(api_keys)

    @classmethod
    def loop(cls, interval=30, network=None, remote:bool=True, local:bool=True, save:bool=True):
        while True:
            current_time = c.timestamp()
            elapsed = current_time - start_time
            if elapsed > interval:
                c.print('SYNCING AND UPDATING THE SERVERS_INFO')
                # subspace.sync(network=network, remote=remote, local=local, save=save)
                start_time = current_time
            c.sleep(interval)


    @staticmethod
    def get_pid():
        return os.getpid()
        
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
                elif c.is_int(v):
                    v = eval(v)
                else:
                    v = v
            
            kwargs[k] = v

        return kwargs
    
    @classmethod
    def memory_info(cls, fmt:str='gb'):
        return c.module('os').memory_info(fmt=fmt)
    
    @classmethod
    def remove_number_from_word(cls, word:str) -> str:
        while word[-1].isdigit():
            word = word[:-1]
        return word

    @classmethod
    def users(cls):
        users = c.get(cls.resolve_path('users'), {})
        root_key_address  = c.root_key().ss58_address
        if root_key_address not in users:
            cls.add_admin(root_key_address)
        return cls.get('users', {})
    

    @classmethod
    def lag(cls,  *args, **kwargs):
        return c.module('subspace').lag(*args, **kwargs)

    @classmethod
    def loops(cls, **kwargs):
        return c.pm2ls('loop', **kwargs)


    def loop_fleet(self, n=2, **kwargs):
        responses = []
        for i in range(n):
            kwargs['remote'] = False
            responses += [self.remote_fn('loop', kwargs=kwargs, tag=i)]
        return responses
    
    @classmethod
    def remote_fn_fleet(cls, fn:str, n=2, **kwargs):
        responses = []
        for i in range(n):
            responses += [cls.remote_fn(fn, kwargs=kwargs, tag=i)]
        return responses
    

    def generate(self, *args, **kwargs):
        return 'hey'

    @classmethod
    def add_peers(cls, *args, **kwargs):
        return c.module('remote').add_peers(*args,**kwargs)

    @classmethod
    def sid(cls):
        return c.module('subspace.chain')().id()

    @classmethod
    def ticket(self, *args, **kwargs):
        return c.module('ticket')().create(*args, **kwargs)
    
    def save_ticket(self, key=None, **kwargs):
        
        key = c.get_key(key)
        return key.save_ticket(**kwargs)

    def load_ticket(self, key=None, **kwargs):
        key = c.get_key(key)
        return key.load_ticket(**kwargs)

    @classmethod
    def verify_ticket(cls, *args, **kwargs):

        return c.get_key().verify_ticket(*args, **kwargs)
    
    @classmethod
    def load_style(cls):
        return c.module('streamlit').load_style()

    @classmethod
    def active_thread_count(cls): 
        return threading.active_count()
    
    @classmethod
    def init_args(cls):
        return list(cls.config().keys())

    @classmethod
    def soup(cls, *args, **kwargs):
        from bs4 import BeautifulSoup
        return BeautifulSoup(*args, **kwargs)
    
    ########
   
    @classmethod
    def document(cls, fn):
        '''
        ## Documentation
        
        ### `docu` method
        
        ```python
        @classmethod
        def docu(cls, fn):
            return c.module('agent.coder')().document_fn(fn)
        ```
        
        #### Description:
        This class method is responsible for documenting a given function `fn`.
        
        #### Parameters:
        - `fn`: A function object that needs to be documented.
        
        #### Returns:
        - Returns the documentation of the provided function `fn` as generated by the `document_fn` method of the `agent.coder
        '''
        return c.module('coder')().document_fn(fn)

    comment = document

    def set_page_config(self,*args, **kwargs):
        return c.module('streamlit').set_page_config(*args, **kwargs)
        
    @classmethod
    def get_state(cls, network='main', netuid='all', update=True, path='state'):
        t1 = c.time()
        if not update:
            state = cls.get(path, default=None)
            if state != None:
                return state
            
        subspace = c.module('subspace')(network=network)

        state = {
            'subnets': subspace.subnet_params(netuid=netuid),
            'modules': subspace.modules(netuid=netuid),
            'balances': subspace.balances(),
            'stake_to': subspace.stake_to(netuid=netuid),
        }

        state['total_balance'] = sum(state['balances'].values())/1e9
        state['key2address'] = c.key2address()
        state['lag'] = c.lag()
        state['block_time'] = 8
        c.print(f'get_state took {c.time() - t1:.2f} seconds')
        cls.put(path, state)
        return state
        
    @classmethod
    def eval(cls, module, vali=None,  **kwargs):
        vali = c.module('vali')() if vali == None else c.module(vali)
        return c.eval(module, **kwargs)
    
    @classmethod
    def run_epoch(self, *args, vali=None, **kwargs):
        vali = c.module('vali')() if vali == None else c.module(vali)
        return vali.run_epoch(*args, **kwargs)

    @classmethod
    def comment(self,fn:str='module/ls'):
        return c.module('code')().comment(fn)

    @classmethod
    def host2ssh(cls, *args, **kwarg):
        return c.module('remote').host2ssh(*args, **kwarg)

    @classmethod
    def imported_modules(self, module:str = None):
        return c.module('code').imported_modules(module=module)
    
    def server2fn(self, *args, **kwargs ):
        servers = c.servers(*args, **kwargs)
        futures = []
        server2fn = {}
        for s in servers:
            server2fn[s] = c.submit(f'{s}/schema', kwargs=dict(code=True))
        futures = list(server2fn.values())
        fns = c.wait(futures,timeout=10)
        for s, f in zip(servers, fns):
            server2fn[s] = f
        return server2fn

    def docker_compose_file(self, *args, **kwargs):
        x = c.load_yaml(f'{c.libpath}/docker-compose.yml', *args, **kwargs)
        port_range = c.port_range()
        x['services']["commune"][f'ports'] = [f"{port_range[0]}-{port_range[1]}:{port_range[0]}-{port_range[1]}"]
        return x

    @classmethod
    def launcher_keys(cls, netuid=0, min_stake=500, **kwargs):
        keys = c.keys()
        key2balance =  c.key2balance(**kwargs)
        key2balance = {k: v for k,v in key2balance.items() if v > min_stake}
        return [k for k in keys]
    
    @classmethod
    def top_launchers(cls, amount=600, **kwargs):
        launcher_keys = cls.launcher_keys()
        key2address = c.key2address()
        destinations = []
        amounts = []
        launcher2balance = c.get_balances(launcher_keys)
        for k in launcher_keys:
            k_address = key2address[k]
            amount_needed = amount - launcher2balance.get(k_address, 0)
            if amount_needed > 0:
                destinations.append(k_address)
                amounts.append(amount_needed)
            else:
                c.print(f'{k} has enough balance --> {launcher2balance.get(k, 0)}')

        return c.transfer_many(amounts=amounts, destinations=destinations, **kwargs)
    
    load_launcher_keys = top_launchers
    @classmethod
    def launcher2balance(cls):
        keys = cls.launcher_keys()
        return c.get_balances(keys)
    
Module = c

Module.run(__name__)
    



