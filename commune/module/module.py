import os
import inspect
from typing import *
import json

import argparse
import nest_asyncio
nest_asyncio.apply()

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
    core_modules = ['module', 'key', 'subspace', 'web3', 'serializer', 'pm2',  
                    'executor', 'client', 'server', 
                    'namespace' ]
    libname = lib_name = lib = 'commune' # the name of the library
    cost = 1
    description = """This is a module"""
    base_module = 'module' # the base module
    giturl = 'https://github.com/commune-ai/commune.git' # tge gutg
    root_module_class = 'c' # WE REPLACE THIS THIS Module at the end, kindof odd, i know, ill fix it fam, chill out dawg, i didnt sleep with your girl
    default_port_range = [50050, 50150] # the port range between 50050 and 50150
    default_ip = local_ip = loopback = '0.0.0.0'
    address = '0.0.0.0:8888' # the address of the server (default)
    rootpath = root_path  = root  = '/'.join(__file__.split('/')[:-2])  # the path to the root of the library
    homepath = home_path = os.path.expanduser('~') # the home path
    libpath = lib_path = os.path.dirname(root_path) # the path to the library
    repopath = repo_path  = os.path.dirname(root_path) # the path to the repo
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

    pythonpath = pypath =  file_path =  filepath

    @classmethod
    def dirpath(cls) -> str:
        '''
        removes the PWD with respect to where module.py is located
        '''
        return os.path.dirname(cls.filepath())
    folderpath = dirname = dir_path =  dirpath

    @classmethod
    def module_name(cls, obj=None):
        if hasattr(cls, 'name') and isinstance(cls.name, str):
            return cls.name
        obj = cls.resolve_object(obj)
        module_file =  inspect.getfile(obj)
        return c.path2simple(module_file)
    
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
    def obj2module(cls,obj):
        import commune as c
        class WrapperModule(c.Module):
            _obj = obj
            def __name__(self):
                return obj.__name__
            def __class__(self):
                return obj.__class__
            @classmethod
            def filepath(cls) -> str:
                return super().filepath(cls._obj)  

        for fn in dir(WrapperModule):
            try:
                setattr(obj, fn, getattr(WrapperModule, fn))
            except:
                pass 
 
        return obj
    
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
    @classmethod
    def deserialize(cls, *args, **kwargs):
        return c.module('serializer')().deserialize(*args, **kwargs)
    
    @property
    def server_name(self):
        if not hasattr(self, '_server_name'): 
            self._server_name = self.module_name()
        return self._server_name
            
    @server_name.setter
    def server_name(self, name):
        self._server_name = name

    @classmethod
    def resolve_object(cls, obj:str = None, **kwargs):
        if isinstance(obj, str):
            obj = c.module(obj, **kwargs)
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
    def argparse(cls):
        parser = argparse.ArgumentParser(description='Argparse for the module')
        parser.add_argument('-m', '--m', '--module', '-module', dest='function', help='The function', type=str, default=cls.module_name())
        parser.add_argument('-fn', '--fn', dest='function', help='The function', type=str, default="__init__")
        parser.add_argument('-kw',  '-kwargs', '--kwargs', dest='kwargs', help='key word arguments to the function', type=str, default="{}") 
        parser.add_argument('-p', '-params', '--params', dest='params', help='key word arguments to the function', type=str, default="{}") 
        parser.add_argument('-i','-input', '--input', dest='input', help='key word arguments to the function', type=str, default="{}") 
        parser.add_argument('-args', '--args', dest='args', help='arguments to the function', type=str, default="[]")  
        args = parser.parse_args()
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
    def commit_hash(cls, libpath:str = None):
        if libpath == None:
            libpath = c.libpath
        return c.cmd('git rev-parse HEAD', cwd=libpath, verbose=False).split('\n')[0].strip()

    @classmethod
    def commit_ticket(cls, **kwargs):
        commit_hash = cls.commit_hash()
        ticket = c.ticket(commit_hash, **kwargs)
        assert c.verify(ticket)
        return ticket

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
    def info_hash(self):
        return c.commit_hash()

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

    @classmethod
    def pip_exists(cls, lib:str, verbose:str=True):
        return bool(lib in cls.pip_libs())
    
    @classmethod
    def version(cls, lib:str=libname):
        lines = [l for l in cls.cmd(f'pip3 list', verbose=False).split('\n') if l.startswith(lib)]
        if len(lines)>0:
            return lines[0].split(' ')[-1].strip()
        else:
            return f'No Library Found {lib}'

    def forward(self, a=1, b=2):
        return a+b
    
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

    def resolve_key(self, key: str = None) -> str:
        if key == None:
            if hasattr(self, 'key'):
                key = self.key
            key = self.resolve_keypath(key)
        key = self.get_key(key)
        return key  
    
    def sign(self, data:dict  = None, key: str = None, **kwargs) -> bool:
        return self.resolve_key(key).sign(data, **kwargs)
    
    @classmethod
    def verify(cls, auth, key=None, **kwargs ) -> bool:  
        return c.get_key(key).verify(auth, **kwargs)

    @classmethod
    def verify_ticket(cls, auth, key=None, **kwargs ) -> bool:  
        return c.get_key(key).verify_ticket(auth, **kwargs)

    @classmethod
    def start(cls, *args, **kwargs):
        return cls(*args, **kwargs)
    
    def remove_user(self, key: str) -> None:
        if not hasattr(self, 'users'):
            self.users = []
        self.users.pop(key, None)
    
    @classmethod
    def is_pwd(cls, module:str = None):
        if module != None:
            module = c.module(module)
        else:
            module = cls
        return module.dirpath() == c.pwd()
    

    @classmethod
    def shortcuts(cls, cache=True) -> Dict[str, str]:
        return cls.get_yaml(os.path.dirname(__file__)+ '/module.yaml' ).get('shortcuts')
    
    def __repr__(self) -> str:
        return f'<{self.class_name()}'
    def __str__(self) -> str:
        return f'<{self.class_name()}'


    @classmethod
    def get_commune(cls): 
        from commune import c
        return c
    
    def pull(self):
        return c.cmd('git pull', verbose=True, cwd=c.libpath)
    
    def push(self, msg:str = 'update'):
        c.cmd('git add .', verbose=True, cwd=c.libpath)
        c.cmd(f'git commit -m "{msg}"', verbose=True, cwd=c.libpath)
        return c.cmd('git push', verbose=True, cwd=c.libpath)
    @classmethod
    def base_config(cls, cache=True):
        if cache and hasattr(cls, '_base_config'):
            return cls._base_config
        cls._base_config = cls.get_yaml(cls.config_path())
        return cls._base_config

    @classmethod
    def local_config(cls, filename_options = ['module', 'commune', 'config', 'cfg'], cache=True):
        if cache and hasattr(cls, '_local_config'):
            return cls._local_config
        local_config = {}
        for filename in filename_options:
            if os.path.exists(f'./{filename}.yaml'):
                local_config = cls.get_yaml(f'./{filename}.yaml')
            if local_config != None:
                break
        cls._local_config = local_config
        return cls._local_config
    
    @classmethod
    def local_module(cls, filename_options = ['module', 'agent', 'block'], cache=True):
        for filename in filename_options:
            path = os.path.dirname(f'./{filename}.py')
            for filename in filename_options:
                if os.path.exists(path):
                    classes = cls.find_classes(path)
                    if len(classes) > 0:
                        return classes[-1]
        return None
    
    # local update  
    @classmethod
    def update(cls, 
               module = None,
               namespace: bool = False,
               subspace: bool = False,
               network: str = 'local',
               **kwargs
               ):
        responses = []
        if module != None:
            return c.module(module).update()
        # update local namespace
        if namespace:
            responses.append(c.namespace(network=network, update=True))
        return {'success': True, 'responses': responses}

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
    
    def logs(self, name:str = None, verbose: bool = False):
        return c.pm2_logs(name, verbose=verbose)
    
    def hardware(self, *args, **kwargs):
        return c.obj('commune.utils.os.hardware')(*args, **kwargs)

    def set_params(self,*args, **kwargs):
        return self.set_config(*args, **kwargs)
    
    def init_module(self,*args, **kwargs):
        return self.set_config(*args, **kwargs)
  



    helper_functions  = ['info',
                'metadata',
                'schema',
                'server_name',
                'is_admin',
                'namespace',
                'whitelist', 
                'endpoints',
                'forward',
                'module_name', 
                'class_name',
                'name',
                'address',
                'fns'] # whitelist of helper functions to load
    
    def add_endpoint(self, name, fn):
        setattr(self, name, fn)
        self.endpoints.append(name)
        assert hasattr(self, name), f'{name} not added to {self.__class__.__name__}'
        return {'success':True, 'message':f'Added {fn} to {self.__class__.__name__}'}

    def is_endpoint(self, fn) -> bool:
        if isinstance(fn, str):
            fn = getattr(self, fn)
        return hasattr(fn, '__metadata__')

    def get_endpoints(self, search: str =None , helper_fn_attributes = ['helper_functions', 
                                                                        'whitelist', 
                                                                        '_endpoints',
                                                                        '__endpoints___']):
        endpoints = []
        for k in helper_fn_attributes:
            if hasattr(self, k):
                fn_obj = getattr(self, k)
                if callable(fn_obj):
                    endpoints += fn_obj()
                else:
                    endpoints += fn_obj
        for f in dir(self):
            try:
                if not callable(getattr(self, f)) or  (search != None and search not in f):
                    continue
                fn_obj = getattr(self, f) # you need to watchout for properties
                is_endpoint = hasattr(fn_obj, '__metadata__')
                if is_endpoint:
                    endpoints.append(f)
            except Exception as e:
                print(f'Error in get_endpoints: {e} for {f}')
        return sorted(list(set(endpoints)))
    
    endpoints = get_endpoints
    

    def cost_fn(self, fn:str, args:list, kwargs:dict):
        return 1

    @classmethod
    def endpoint(cls, 
                 cost=1, # cost per call 
                 user2rate : dict = None, 
                 rate_limit : int = 100, # calls per minute
                 timestale : int = 60,
                 public:bool = False,
                 cost_keys = ['cost', 'w', 'weight'],
                 **kwargs):
        
        for k in cost_keys:
            if k in kwargs:
                cost = kwargs[k]
                break

        def decorator_fn(fn):
            metadata = {
                **cls.fn_schema(fn),
                'cost': cost,
                'rate_limit': rate_limit,
                'user2rate': user2rate,   
                'timestale': timestale,
                'public': public,            
            }
            import commune as c
            fn.__dict__['__metadata__'] = metadata

            return fn

        return decorator_fn
    


    def metadata(self, to_string=False):
        if hasattr(self, '_metadata'):
            return self._metadata
        metadata = {}
        metadata['schema'] = self.schema()
        metadata['description'] = self.description
        metadata['urls'] = {k: v for k,v in self.urls.items() if v != None}
        if to_string:
            return self.python2str(metadata)
        self._metadata =  metadata
        return metadata

    def info(self , 
             module = None,
             lite_features = ['name', 'address', 'schema', 'key', 'description'],
             lite = True,
             cost = False,
             **kwargs
             ) -> Dict[str, Any]:
        '''
        hey, whadup hey how is it going
        '''
        info = self.metadata()
        info['name'] = self.server_name or self.module_name()
        info['address'] = self.address
        info['key'] = self.key.ss58_address
        return info
    
    @classmethod
    def is_public(cls, fn):
        if not cls.is_endpoint(fn):
            return False
        return getattr(fn, '__metadata__')['public']


    urls = {'github': None,
             'website': None,
             'docs': None, 
             'twitter': None,
             'discord': None,
             'telegram': None,
             'linkedin': None,
             'email': None}
    

    
    def schema(self,
                search = None,
                docs: bool = True,
                defaults:bool = True, 
                cache=True) -> 'Schema':
        if self.is_str_fn(search):
            return self.fn_schema(search, docs=docs, defaults=defaults)
        schema = {}
        if cache and self._schema != None:
            return self._schema
        fns = self.get_endpoints()
        for fn in fns:
            if search != None and search not in fn:
                continue
            if callable(getattr(self, fn )):
                schema[fn] = self.fn_schema(fn, defaults=defaults,docs=docs)        
        # sort by keys
        schema = dict(sorted(schema.items()))
        if cache:
            self._schema = schema

        return schema
    

    @classmethod
    def has_routes(cls):
        return cls.config().get('routes') is not None
    
    route_cache = None
    @classmethod
    def routes(cls, cache=True):
        if cls.route_cache is not None and cache:
            return cls.route_cache 
        routes =  cls.get_yaml(os.path.dirname(__file__)+ '/module.yaml').get('routes')
        cls.route_cache = routes
        return routes

    #### THE FINAL TOUCH , ROUTE ALL OF THE MODULES TO THE CURRENT MODULE BASED ON THE routes CONFIG


    @classmethod
    def route_fns(cls):
        routes = cls.routes()
        route_fns = []
        for module, fns in routes.items():
            for fn in fns:
                if isinstance(fn, dict):
                    fn = fn['to']
                elif isinstance(fn, list):
                    fn = fn[1]
                elif isinstance(fn, str):
                    fn
                else:
                    raise ValueError(f'Invalid route {fn}')
                route_fns.append(fn)
        return route_fns
            

    @staticmethod
    def resolve_to_from_fn_routes(fn):
        '''
        resolve the from and to function names from the routes
        option 1: 
        {fn: 'fn_name', name: 'name_in_current_module'}
        option 2:
        {from: 'fn_name', to: 'name_in_current_module'}
        '''
        
        if type(fn) in [list, set, tuple] and len(fn) == 2:
            # option 1: ['fn_name', 'name_in_current_module']
            from_fn = fn[0]
            to_fn = fn[1]
        elif isinstance(fn, dict) and all([k in fn for k in ['fn', 'name']]):
            if 'fn' in fn and 'name' in fn:
                to_fn = fn['name']
                from_fn = fn['fn']
            elif 'from' in fn and 'to' in fn:
                from_fn = fn['from']
                to_fn = fn['to']
        else:
            from_fn = fn
            to_fn = fn
        
        return from_fn, to_fn
    

    @classmethod
    def enable_routes(cls, routes:dict=None, verbose=False):
        from functools import partial
        """
        This ties other modules into the current module.
        The way it works is that it takes the module name and the function name and creates a partial function that is bound to the module.
        This allows you to call the function as if it were a method of the current module.
        for example
        """
        my_path = cls.class_name()
        if not hasattr(cls, 'routes_enabled'): 
            cls.routes_enabled = False

        t0 = cls.time()

        # WARNING : THE PLACE HOLDERS MUST NOT INTERFERE WITH THE KWARGS OTHERWISE IT WILL CAUSE A BUG IF THE KWARGS ARE THE SAME AS THE PLACEHOLDERS
        # THE PLACEHOLDERS ARE NAMED AS module_ph and fn_ph AND WILL UNLIKELY INTERFERE WITH THE KWARGS
        def fn_generator( *args, module_ph, fn_ph, **kwargs):
            module_ph = cls.module(module_ph)
            fn_type = module_ph.classify_fn(fn_ph)
            module_ph = module_ph() if fn_type == 'self' else module_ph
            return getattr(module_ph, fn_ph)(*args, **kwargs)

        if routes == None:
            if not hasattr(cls, 'routes'):
                return {'success': False, 'msg': 'routes not found'}
            routes = cls.routes() if callable(cls.routes) else cls.routes
        for m, fns in routes.items():
            if fns in ['all', '*']:
                fns = c.functions(m)

            for fn in fns: 
                # resolve the from and to function names
                from_fn, to_fn = cls.resolve_to_from_fn_routes(fn)
                # create a partial function that is bound to the module
                fn_obj = partial(fn_generator, fn_ph=from_fn, module_ph=m )
                # make sure the funciton is as close to the original function as possible
                fn_obj.__name__ = to_fn
                # set the function to the current module
                setattr(cls, to_fn, fn_obj)
                cls.print(f'ROUTE({m}.{fn} -> {my_path}:{fn})', verbose=verbose)

        t1 = cls.time()
        cls.print(f'enabled routes in {t1-t0} seconds', verbose=verbose)
        cls.routes_enabled = True
        return {'success': True, 'msg': 'enabled routes'}
    
    @classmethod
    def fn2module(cls):
        '''
        get the module of a function
        '''
        routes = cls.routes()
        fn2module = {}
        for module, fn_routes in routes.items():
            for fn_route in fn_routes:
                if isinstance(fn_route, dict):
                    fn_route = fn_route['to']
                elif isinstance(fn_route, list):
                    fn_route = fn_route[1]
                fn2module[fn_route] = module    
        return fn2module

    def is_route(cls, fn):
        '''
        check if a function is a route
        '''
        return fn in cls.fn2module()
    

    
    @classmethod
    def has_test_module(cls, module=None):
        module = module or cls.module_name()
        return cls.module_exists(cls.module_name() + '.test')
    
    @classmethod
    def test(cls,
              module=None,
              timeout=42, 
              trials=3, 
              parallel=False,
              ):
        module = module or cls.module_name()

        if cls.has_test_module(module):
            cls.print('FOUND TEST MODULE', color='yellow')
            module = module + '.test'
        self = cls.module(module)()
        test_fns = self.test_fns()
        print(f'testing {module} {test_fns}')

        def trial_wrapper(fn, trials=trials):
            def trial_fn(trials=trials):

                for i in range(trials):
                    try:
                        return fn()
                    except Exception as e:
                        print(f'Error: {e}, Retrying {i}/{trials}')
                        cls.sleep(1)
                return False
            return trial_fn
        fn2result = {}
        if parallel:
            future2fn = {}
            for fn in self.test_fns():
                cls.print(f'testing {fn}')
                f = cls.submit(trial_wrapper(getattr(self, fn)), timeout=timeout)
                future2fn[f] = fn
            for f in cls.as_completed(future2fn, timeout=timeout):
                fn = future2fn.pop(f)
                fn2result[fn] = f.result()
        else:
            for fn in self.test_fns():
                fn2result[fn] = trial_wrapper(getattr(self, fn))()       
        return fn2result


c.enable_routes()
Module = c # Module is alias of c
Module.run(__name__)


