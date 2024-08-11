import os
import inspect
from copy import deepcopy
from typing import *
import json
from glob import glob
import argparse
import nest_asyncio
import time
t1 = time.time()

nest_asyncio.apply()

# YOU CAN DEFINE YOUR CORE MODULES BY JUST adding a class to the core folder
# for instance if you have a class called 'os_fam' the file would be ./commune/module/_os_fam.py
def get_core_modules(prefix = 'commune.module', core_prefix = '_'):
    """
    find the core modules that construct the commune block module
    """
    core_dirpath = os.path.dirname(__file__)
    core_modules = []
    for f in os.listdir(core_dirpath):
        f = f.split('/')[-1].split('.')[0]
        if f.startswith(core_prefix) and not f.startswith('__') :
            core_modules.append(f[1:])
    results = []
    for cm in core_modules:
        obj_name = cm.upper() if cm.lower() == 'os' else cm.capitalize()
        exec(f'from {prefix}.{core_prefix}{cm} import {obj_name}')
        results.append(eval(obj_name))
    return results

# AGI BEGINS 
CORE_MODULES = get_core_modules()

class c(*CORE_MODULES):
    core_modules = ['module', 'key', 'client', 'server', 'serializer']
    libname = lib_name = lib = 'commune' # the name of the library
    helper_functions  = ['info',
                'metadata',
                'schema',
                'server_name',
                'is_admin',
                'namespace',
                'whitelist', 
                'forward',
                'fns'] # whitelist of helper functions to load
    cost = 1
    description = """This is a module"""
    base_module = 'module' # the base module
    giturl = git_url = 'https://github.com/commune-ai/commune.git' # tge gutg
    root_module_class = 'c' # WE REPLACE THIS THIS Module at the end, kindof odd, i know, ill fix it fam, chill out dawg, i didnt sleep with your girl
    default_port_range = [50050, 50150] # the port range between 50050 and 50150
    default_ip = local_ip = loopback = '0.0.0.0'
    address = '0.0.0.0:8888' # the address of the server (default)
    rootpath = root_path  = root   = '/'.join(__file__.split('/')[:-2])  # the path to the root of the library
    homepath = home_path = os.path.expanduser('~') # the home path
    libpath = lib_path = os.path.dirname(root_path) # the path to the library
    datapath =  data_path = os.path.join(root_path, 'data') # the path to the data folder
    modules_path = os.path.join(libpath, 'modules') # the path to the modules folder
    repo_path  = os.path.dirname(root_path) # the path to the repo
    blacklist = [] # blacklist of functions to not to access for outside use
    server_mode = 'http' # http, grpc, ws (websocket)
    default_network = 'local' # local, subnet
    cache = {} # cache for module objects
    home = os.path.expanduser('~') # the home directory
    __ss58_format__ = 42 # the ss58 format for the substrate address
    cache_path = os.path.expanduser(f'~/.{libname}')
    default_tag = 'base'

    def __init__(self, *args, **kwargs):
        pass

    @property
    def key(self):
        if not hasattr(self, '_key'):
            if not hasattr(self, 'server_name') or self.server_name == None:
                self.server_name = self.module_name()
            self._key = c.get_key(self.server_name, create_if_not_exists=True)
        return self._key
    
    @key.setter
    def key(self, key: 'Key'):
        if key == None:
            key = self.server_name
        self._key = key if hasattr(key, 'ss58_address') else c.get_key(key, create_if_not_exists=True)
        return self._key

    @classmethod
    async def async_call(cls, *args,**kwargs):
        return c.call(*args, **kwargs)
    
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
    def filepath(cls, obj=None) -> str:
        '''
        removes the PWD with respect to where module.py is located
        '''
        obj = cls.resolve_object(obj)
        try:
            module_path =  inspect.getfile(obj)
        except Exception as e:
            c.print(f'Error: {e} {cls}', color='red')
            module_path =  inspect.getfile(cls)
        return module_path

    pythonpath = pypath =  filepath

    @classmethod
    def dirpath(cls) -> str:
        '''
        removes the PWD with respect to where module.py is located
        '''
        return os.path.dirname(cls.filepath())
    folderpath = dirname = dirpath

    @classmethod
    def module_name(cls, obj=None):
        if hasattr(cls, 'name') and isinstance(cls.name, str):
            return cls.name
        obj = cls.resolve_object(obj)
        module_file =  inspect.getfile(obj)
        return cls.path2simple(module_file)
    
    path  = name = module_name 
    
    @classmethod
    def module_class(cls) -> str:
        return cls.__name__
    @classmethod
    def class_name(cls, obj= None) -> str:
        obj = obj if obj != None else cls
        return obj.__name__

    classname = class_name

    @classmethod
    def config_path(cls) -> str:
        return cls.filepath().replace('.py', '.yaml')

    @classmethod
    def sandbox(cls):
        c.cmd(f'python3 {c.root_path}/sandbox.py', verbose=True)
        return 
    sand = sandbox

    module_cache = {}
    _obj = None
    @classmethod
    def get_module(cls, 
                   path:str = 'module',  
                   cache=True,
                   trials = 5,
                   verbose = False,
                   update_tree_if_fail = True,
                   init_kwargs = None,
                   ) -> str:
        if path in ['module', 'c']:
            return c
        if trials > 0:
            try:
                module = cls.get_module(path=path, cache=cache, trials=trials-1, verbose=verbose, update_tree_if_fail=update_tree_if_fail, init_kwargs=init_kwargs)
                return module
            except Exception as e:
                if verbose:
                    c.print(f'Error: {e}', color='red')
        """
        params: 
            path: the path to the module
            cache: whether to cache the module
            tree: the tree to search for the module
            update_if_fail: whether to update the tree if the module is not found
        """

        # if the module is a valid import path 
        path = path or 'module'
        shortcuts = c.shortcuts()
        if path in shortcuts:
            path = shortcuts[path]
        module = None
        cache_key = path
        t0 = c.time()
        if cache and cache_key in c.module_cache:
            module = c.module_cache[cache_key]
            return module

        module = c.simple2object(path)
        # ensure module
        if verbose:
            c.print(f'Loaded {path} in {c.time() - t0} seconds', color='green')
        
        if init_kwargs != None:
            module = module(**init_kwargs)
        is_module = c.is_module(module)
        # if not is_module:
        #     module = cls.obj2module(module)
        if cache:
            c.module_cache[cache_key] = module            
        return module
    
    @classmethod
    def obj2module(cls,obj):
        import commune as c
        class WrapperModule(c.Module):
            _obj = obj
            def __name__(self):
                return self._obj.__name__
            def __class__(self):
                return self._obj.__class__
            
        m = WrapperModule
                
        for fn in dir(obj):
            try:
                setattr(m, fn, getattr(obj, fn))
            except:
                pass
            
        return m()
        


    
    @classmethod
    def storage_dir(cls):
        return f'{c.cache_path}/{cls.module_name()}'
        
    @classmethod
    def refresh_storage(cls):
        cls.rm(cls.storage_dir())

    @classmethod
    def refresh_storage_dir(cls):
        c.rm(cls.storage_dir())
        c.makedirs(cls.storage_dir())
        
    ############ JSON LAND ###############

    @classmethod
    def __str__(cls):
        return cls.__name__

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

    @classmethod
    def is_module(cls, obj=None) -> bool:
        
        if obj is None:
            obj = cls
        if all([hasattr(obj, k) for k in ['info', 'schema', 'set_config', 'config']]):
            return True
        return False
    
    @classmethod
    def root_functions(cls):
        return c.fns()
    
    @classmethod
    def is_root(cls, obj=None) -> bool:
        required_features = ['module_class','root_module_class', 'module_name']
        if obj is None:
            obj = cls
        if all([hasattr(obj, k) for k in required_features]):
            module_class = obj.module_class()
            if module_class == cls.root_module_class:
                return True
        return False
    is_module_root = is_root_module = is_root
    
    @classmethod
    def serialize(cls, *args, **kwargs):
        return c.module('serializer')().serialize(*args, **kwargs)


    @property
    def server_name(self):
        if not hasattr(self, '_server_name'): 
            self._server_name = self.module_name()
        return self._server_name
            
    @server_name.setter
    def server_name(self, name):
        self._server_name = name

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
            module = cls.module_name() if module == None else module

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
    def resolve_object(cls, obj:str = None, **kwargs):
        if cls._obj != None:
            return cls._obj
        else:
            return obj or cls
    
    def self_destruct(self):
        c.kill(self.server_name)    
        
    def self_restart(self):
        c.restart(self.server_name)

    @classmethod
    def pm2_start(cls, *args, **kwargs):
        return c.module('pm2').start(*args, **kwargs)
    
    @classmethod
    def pm2_launch(cls, *args, **kwargs):
        return c.module('pm2').launch(*args, **kwargs)
                              
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
    def argparse(cls, verbose: bool = False, **kwargs):
        parser = argparse.ArgumentParser(description='Argparse for the module')
        parser.add_argument('-m', '--m', '--module', '-module', dest='function', help='The function', type=str, default=cls.module_name())
        parser.add_argument('-fn', '--fn', dest='function', help='The function', type=str, default="__init__")
        parser.add_argument('-kw',  '-kwargs', '--kwargs', dest='kwargs', help='key word arguments to the function', type=str, default="{}") 
        parser.add_argument('-p', '-params', '--params', dest='params', help='key word arguments to the function', type=str, default="{}") 
        parser.add_argument('-i','-input', '--input', dest='input', help='key word arguments to the function', type=str, default="{}") 
        parser.add_argument('-args', '--args', dest='args', help='arguments to the function', type=str, default="[]")  
        args = parser.parse_args()
        if verbose:
            c.print('Argparse Args: ',args, color='cyan')
        args.kwargs = json.loads(args.kwargs.replace("'",'"'))
        args.params = json.loads(args.params.replace("'",'"'))
        args.inputs = json.loads(args.input.replace("'",'"'))
        args.args = json.loads(args.args.replace("'",'"'))
        args.fn = args.function
        # if you pass in the params, it will override the kwargs
        if len(args.params) > 0:
            if isinstance(args.params, dict):
                args.kwargs = args.params
            elif isinstance(args.params, list):
                args.args = args.params
            else:
                raise Exception('Invalid params', args.params)
            
        return args
        
    @classmethod
    def run(cls, name:str = None) -> Any: 
        is_main =  name == '__main__' or name == None or name == cls.__name__
        if not is_main:
            return {'success':False, 'message':f'Not main module {name}'}
        args = cls.argparse()
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
    def module_fn(cls, module:str, fn:str , args:list = None, kwargs:dict= None):
        module = c.module(module)
        is_self_method = bool(fn in module.self_functions())
        if is_self_method:
            module = module()
            fn = getattr(module, fn)
        else:
            fn =  getattr(module, fn)
        args = args or []
        kwargs = kwargs or {}
        return fn(*args, **kwargs)
    
    fn = module_fn
    
    @classmethod
    def module(cls,module: Any = 'module' , verbose=False, **kwargs):
        '''
        Wraps a python class as a module
        '''
        t0 = c.time()
        module_class =  c.get_module(module,**kwargs)
        latency = c.time() - t0
        c.print(f'Loaded {module} in {latency} seconds', color='green', verbose=verbose)
        return module_class
    _module = m = mod = module

    # UNDER CONSTRUCTION (USE WITH CAUTION)
    
    def setattr(self, k, v):
        setattr(self, k, v)
        
    def setattributes(self, new_attributes:Dict[str, Any]) -> None:
        '''
        Set a dictionary to the slf attributes 
        '''
        assert isinstance(new_attributes, dict), f'locals must be a dictionary but is a {type(locals)}'
        self.__dict__.update(new_attributes)

    @classmethod
    def pip_install(cls, 
                    lib:str= None,
                    upgrade:bool=True ,
                    verbose:str=True,
                    ):
        

        if lib in c.modules():
            c.print(f'Installing {lib} Module from local directory')
            lib = c.resolve_object(lib).dirpath()
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
        lines = [l for l in cls.cmd(f'pip3 list', verbose=False).split('\n') if l.startswith(lib)]
        if len(lines)>0:
            return lines[0].split(' ')[-1].strip()
        else:
            return f'No Library Found {lib}'
    


    

    @classmethod
    def resolve_ip(cls, ip=None, external:bool=True) -> str:
        if ip == None:
            if external:
                ip = cls.external_ip()
            else:
                ip = '0.0.0.0'
        assert isinstance(ip, str)
        return ip


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

    ### DICT LAND ###

    def to_dict(self)-> Dict:
        return self.__dict__
    
    @classmethod
    def from_dict(cls, input_dict:Dict[str, Any]) -> 'Module':
        return cls(**input_dict)
        
    def to_json(self) -> str:
        state_dict = self.to_dict()
        assert isinstance(state_dict, dict), 'State dict must be a dictionary'
        assert self.jsonable(state_dict), 'State dict must be jsonable'
        return json.dumps(state_dict)
    
    @classmethod
    def from_json(cls, json_str:str) -> 'Module':
        import json
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def test_fns(cls, *args, **kwargs):
        return [f for f in cls.functions(*args, **kwargs) if f.startswith('test_')]
    
    @classmethod
    def argv(cls, include_script:bool = False):
        import sys
        args = sys.argv
        if include_script:
            return args
        else:
            return args[1:]

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
    
    is_module_folder = is_folder_module

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
    

    @classmethod
    def decrypt(cls, 
                data: Union[str, bytes],
                key: str = None, 
                password : str = None,
                **kwargs) -> bytes:
        key = c.get_key(key)
        return key.decrypt(data, password=password, **kwargs)
    

    
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
            search = cls.module_name()
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
    def set_key(self, key:str, **kwargs) -> None:
        key = self.get_key(key)
        self.key = key
        return key
    
    @classmethod
    def resolve_keypath(cls, key = None):
        if key == None:
            key = cls.module_name()
        return key
    

    def sign(self, data:dict  = None, key: str = None, **kwargs) -> bool:
        key = self.resolve_key(key)
        signature =  key.sign(data, **kwargs)
        return signature
    
    
    @classmethod
    def verify(cls, auth, key=None, **kwargs ) -> bool:  
        key = c.get_key(key)
        return key.verify(auth, **kwargs)

    @classmethod
    def verify_ticket(cls, auth, key=None, **kwargs ) -> bool:  
        key = c.get_key(key)
        return key.verify_ticket(auth, **kwargs)

    

    @classmethod
    def start(cls, *args, **kwargs):
        return cls(*args, **kwargs)
    
    def remove_user(self, key: str) -> None:
        if not hasattr(self, 'users'):
            self.users = []
        self.users.pop(key, None)
    
    @classmethod
    def check_module(cls, module:str):
        return c.connect(module)

    @classmethod
    def is_pwd(cls, module:str = None):
        if module != None:
            module = c.module(module)
        else:
            module = cls
        return module.dirpath() == c.pwd()

    
    def new_modules(self, *modules, **kwargs):
        for module in modules:
            self.new_module(module=module, **kwargs)



    @classmethod
    def new_module( cls,
                   module : str ,
                   base_module : str = 'demo', 
                   folder_module : bool = False,
                   update=1
                   ):
        
        base_module = c.module(base_module)
        module_class_name = ''.join([m[0].capitalize() + m[1:] for m in module.split('.')])
        base_module_class_name = base_module.class_name()
        base_module_code = base_module.code().replace(base_module_class_name, module_class_name)
        pwd = c.pwd()
        path = os.path.join(pwd, module.replace('.', '/'))
        if folder_module:
            dirpath = path
            filename = module.replace('.', '_')
            path = os.path.join(path, filename)
        
        path = path + '.py'
        dirpath = os.path.dirname(path)
        if os.path.exists(path) and not update:
            return {'success': True, 'msg': f'Module {module} already exists', 'path': path}
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)

        c.put_text(path, base_module_code)
        
        return {'success': True, 'msg': f'Created module {module}', 'path': path}
    
    add_module = new_module


    thread_map = {}


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
        
        kwargs = c.locals2kwargs(kwargs)
        if 'remote' in kwargs:
            kwargs['remote'] = False
        if len(fn.split('.'))>1:
            module = '.'.join(fn.split('.')[:-1])
            fn = fn.split('.')[-1]
            
        kwargs = kwargs if kwargs else {}
        args = args if args else []
        if name == None:
            module_path = cls.resolve_object(module).module_name()
            name = f"{module_path}{tag_seperator}{fn}"
            if tag != None:
                name = f'{name}{tag_seperator}{tag}'

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
            # resolve the tag
            if tag != None:
                name = f'{name}{tag_seperator}{tag}'
 
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


    @classmethod
    def shortcuts(cls, cache=True) -> Dict[str, str]:
        return cls.get_yaml(os.path.dirname(__file__)+ '/shortcuts.yaml')

    def __repr__(self) -> str:
        return f'<{self.class_name()}'
    def __str__(self) -> str:
        return f'<{self.class_name()}'
    

    def pytest(self):
        test_path = c.root_path + '/tests'
        return c.cmd(f'pytest {test_path}', verbose=True)
    


    def check_word(self, word:str)-> str:
        files = c.glob('./')
        progress = c.tqdm(len(files))
        for f in files:
            try:
                text = c.get_text(f)
            except Exception as e:
                continue
            if word in text:
                return True
            progress.update(1)
        return False
    
    def file2text(self, path:str='./')-> List[str]:
        files = c.glob(path)
        file2lines = {}
        progress = c.tqdm(len(files))
        for f in files:
            try:
                file2lines[f] = c.get_text(f)
            except Exception as e:
                pass
            progress.update(1)
        return file2lines
    
    def file2lines(self, path:str='./')-> List[str]:
        file2text = self.file2text(path)
        file2lines = {f: text.split('\n') for f, text in file2text.items()}
        return file2lines
    

    
    def num_files(self, path:str='./')-> int:
        return len(c.glob(path))
    
    def hidden_files(self, path:str='./')-> List[str]:
        path = self.resolve_path(path)
        files = [f[len(path)+1:] for f in  c.glob(path)]
        print(files)
        hidden_files = [f for f in files if f.startswith('.')]
        return hidden_files
    
    def find_word(self, word:str, path='./')-> str:
        path = c.resolve_path(path)
        files = c.glob(path)
        progress = c.tqdm(len(files))
        found_files = {}
        for f in files:
            try:
                text = c.get_text(f)
                if word not in text:
                    continue
                lines = text.split('\n')
            except Exception as e:
                continue
            
            line2text = {i:line for i, line in enumerate(lines) if word in line}
            found_files[f[len(path)+1:]]  = line2text
            progress.update(1)
        return found_files


    # def update(self):
    #     c.ip(update=1)
    #     return c.namespace(update=1, public=1)
    

    # def update_loop(self, interval:int = 1):
    #     while True:
    #         self.update()
    #         c.namespace(public=1)
    #         time.sleep(interval)

c.enable_routes()
Module = c # Module is alias of c
Module.run(__name__)



