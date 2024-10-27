import os
import inspect
import json
import shutil
import time
import sys
import argparse
from functools import partial
import os
from copy import deepcopy
from typing import *
import nest_asyncio
import asyncio
nest_asyncio.apply()

class c:

    endpoints = ['ask', 'generate', 'forward']

    core_features = ['module_name', 'module_class',  'filepath', 'dirpath', 'tree']

    # these are shortcuts for the module finder c.module('openai') --> c.module('modle.openai') 
    # if openai : model.openai
    shortcuts =  {
        'openai' : 'model.openai',
        'openrouter':  'model.openrouter',
        'or' : ' model.openrouter',
        'r' :  'remote',
        's' :  'network.subspace',
        'subspace': 'network.subspace', 
        'namespace': 'network', 
        'local': 'network',
        'network.local': 'network',
        }

    libname = lib_name = lib = __file__.split('/')[-3]# the name of the library
    organization = org = orgname = 'commune-ai' # the organization
    git_host  = 'https://github.com'
    cost = 1
    description = """This is a module"""
    base_module = 'module' # the base module
    giturl = f'{git_host}/{org}/{libname}.git' # tge gutg
    root_module_class = 'c' # WE REPLACE THIS THIS Module at the end, kindof odd, i know, ill fix it fam, chill out dawg, i didnt sleep with your girl
    default_port_range = [50050, 50150] # the port range between 50050 and 50150
    default_ip = local_ip = loopback = '0.0.0.0'
    address = '0.0.0.0:8888' # the address of the server (default)
     # the path to the root of the library
    src_path = source_path = rootpath = root_path  = root  = '/'.join(__file__.split('/')[:-1]) 
    homepath = home_path = os.path.expanduser('~') # the home path
    libpath = lib_path = os.path.dirname(root_path) # the path to the library
    repopath = repo_path  = os.path.dirname(root_path) # the path to the repo
    cache = {} # cache for module objects
    home = os.path.expanduser('~') # the home directory
    __ss58_format__ = 42 # the ss58 format for the substrate address
    storage_path = os.path.expanduser(f'~/.{libname}')
    default_tag = 'base'

    def __init__(self, *args, **kwargs):
        pass
    

    @classmethod
    def filepath(cls, obj=None) -> str:
        obj = cls.resolve_object(obj)
        try:
            module_path =  inspect.getfile(obj)
        except Exception as e:
            c.print(f'Error: {e} {cls}', color='red')
            module_path =  inspect.getfile(cls)
        return module_path

    path = file_path =  filepath
    @classmethod
    def dirpath(cls, obj=None) -> str:
        return os.path.dirname(cls.filepath(obj))
    dir_path =  dirpath

    @classmethod
    def module_name(cls, obj=None):
        obj = obj or cls
        module_file =  inspect.getfile(obj)
        return c.path2name(module_file)
    

    def vs(self, path = None):
        path = path or c.libpath
        if c.module_exists(path):
            path = c.filepath(path)
        path = c.abspath(path)
        
        return c.cmd(f'code {path}')

    @classmethod
    def get_module_name(cls, obj=None):
        obj = cls.resolve_object(obj)
        if hasattr(obj, 'module_name'):
            return obj.module_name
        else:
            return cls.__name__
    
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
        return cls.filepath()[:-3] + '.yaml'

    @classmethod
    def sandbox(cls):
        return c.cmd(f'python ./sandbox.py', verbose=True)
    
    sand = sandbox

    module_cache = {}
    _obj = None


    
    def syspath(self):
        return sys.path
    
    @classmethod
    def storage_dir(cls):
        return f'{c.storage_path}/{cls.module_name()}'

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

    @classmethod
    def is_module(cls, obj=None) -> bool:
        
        if obj is None:
            obj = cls
        if all([hasattr(obj, k) for k in c.core_features]):
            return True
        return False
    
    @classmethod
    def is_root(cls, obj=None) -> bool:
        required_features = c.core_features
        obj = obj or cls
        return bool(c.is_module(obj) and obj.module_class() == cls.root_module_class)


    def print( *text:str,  **kwargs):
        return c.obj('commune.utils.misc.print')(*text, **kwargs)

    def is_error( *text:str,  **kwargs):
        return c.obj('commune.utils.misc.is_error')(*text, **kwargs)

    
    is_module_root = is_root_module = is_root

    @classmethod
    def resolve_object(cls, obj:str = None, **kwargs):
        if isinstance(obj, str):
            if c.object_exists(obj):
                return c.obj(obj)
            if c.module_exists(obj):
                return c.module(obj)
        if obj == None:
            if cls._obj != None:
                return cls._obj
            else:
                obj = cls
        return obj
    
    @classmethod
    def pwd(cls):
        pwd = os.getcwd() # the current wor king directory from the process starts 
        return pwd
                            
    @classmethod
    def argparse(cls):
        parser = argparse.ArgumentParser(description='Argparse for the module')
        parser.add_argument('-m', '--m', '--module', '-module', dest='module', help='The function', type=str, default=cls.module_name())
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
    
    # UNDER CONSTRUCTION (USE WITH CAUTION)
    
    def setattr(self, k, v):
        setattr(self, k, v)

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
    def pytest(cls, *args, **kwargs):
        return c.cmd(f'pytest {c.libpath}/tests',  stream=1, *args, **kwargs)
    
    @classmethod
    def argv(cls, include_script:bool = False):
        import sys
        args = sys.argv
        if include_script:
            return args
        else:
            return args[1:]

    @classmethod
    def is_module_file(cls, module = None) -> bool:
        if module != None:
            cls = c.module(module)
        dirpath = cls.dirpath()
        filepath = cls.filepath()
        return bool(dirpath.split('/')[-1] != filepath.split('/')[-1].split('.')[0])
    is_file_module = is_module_file
    @classmethod
    def is_module_folder(cls,  module = None) -> bool:
        if module != None:
            cls = c.module(module)
        return not cls.is_file_module()
    
    is_folder_module = is_module_folder 

    @classmethod
    def get_key(cls,key:str = None , **kwargs) -> None:
        from commune.key import Key
        return Key.get_key(key, **kwargs)
    key = get_key

    @classmethod
    def files(cls, path='./', search:str = None,  **kwargs) -> List[str]:
        files =c.glob(path, **kwargs)
        if search != None:
            files = [f for f in files if search in f]
        return files
    
    @classmethod
    def encrypt(cls,data: Union[str, bytes], password: str = None, key: str = None,  **kwargs ) -> bytes:
        return c.get_key(key).encrypt(data, password=password,**kwargs)

    @classmethod
    def decrypt(cls, data: Any,  password : str = None, key: str = None, **kwargs) -> bytes:
        return c.get_key(key).decrypt(data, password=password)
    
    
    @classmethod
    def sign(cls, data:dict  = None, key: str = None, **kwargs) -> bool:
        return c.get_key(key).sign(data, **kwargs)
    
    @classmethod
    def verify(cls, auth, key=None, **kwargs ) -> bool:  
        return c.get_key(key).verify(auth, **kwargs)

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

    def set_key(self, key:str, **kwargs) -> None:
        self.key = self.resolve_key(key)
        return self.key

    
    def resolve_key(self, key: str = None) -> str:
        if key != None:
            if isinstance(key, str):
                key =  c.get_key(key)
        else:
            if hasattr(self, 'key'):
                key = self.key
            key = c.key(self.module_name())
        assert hasattr(key, 'ss58_address'), f'Key {key} does not have a sign method'
        return key

    @classmethod
    def is_pwd(cls, module:str = None):
        module = c.module(module) if module != None else cls
        return module.dirpath() == c.pwd()
    
    def __repr__(self) -> str:
        return f'<{self.class_name()}'
    def __str__(self) -> str:
        return f'<{self.class_name()}'
    
    def pull(self):
        return c.cmd('git pull', verbose=True, cwd=c.libpath)
    
    def push(self, msg:str = 'update'):
        c.cmd('git add .', verbose=True, cwd=c.libpath)
        c.cmd(f'git commit -m "{msg}"', verbose=True, cwd=c.libpath)
        return c.cmd('git push', verbose=True, cwd=c.libpath)
    
    # local update  
    @classmethod
    def update(cls,  ):
        c.namespace(update=True)
        c.ip(update=1)
        return {'ip': c.ip(), 'namespace': c.namespace()}
    
    def set_params(self,*args, **kwargs):
        return self.set_config(*args, **kwargs)
    
    def init_module(self,*args, **kwargs):
        return self.set_config(*args, **kwargs)

    def ensure_attribute(self, k, v=None):
        if not hasattr(self, k):
            setattr(self, k, v)

    def schema(self,
                obj = None,
                docs: bool = True,
                defaults:bool = True, **kwargs) -> 'Schema':
        if c.is_fn(obj):
            return c.fn_schema(obj, docs=docs, defaults=defaults)

        fns = self.get_functions(obj)
        schema = {}
        for fn in fns:
            try:
                schema[fn] = self.fn_schema(fn, defaults=defaults,docs=docs)    
            except Exception as e:
                print(f'Error: {e}')    
        # sort by keys
        schema = dict(sorted(schema.items()))
        return schema

    @classmethod
    def utils_paths(cls, search=None):
        utils = c.find_functions(c.root_path + '/utils')
        if search != None:
            utils = [u for u in utils if search in u]
        return sorted(utils)
    
    @classmethod
    def util2code(cls, search=None):
        utils = cls.utils()
        util2code = {}
        for f in utils:
            if search != None:
                if search in f:
                    util2code[f] = c.fn_code(f)
        return util2code
    
    def util_modules(self, search=None):
        return sorted(list(set([f.split('.')[-2] for f in self.utils_paths(search)])))

    utils = utils_paths
    @classmethod
    def util2path(cls, search=None):
        utils_paths = cls.utils_paths(search=search)
        util2path = {}
        for f in utils_paths:
            util2path[f.split('.')[-1]] = f
        return util2path

    @classmethod
    def add_utils(cls, obj=None):
        obj = obj or cls
        from functools import partial
        utils = obj.util2path()
        def wrapper_fn2(fn, *args, **kwargs):
            try:
                fn = c.import_object(fn)
                return fn(*args, **kwargs)
            except : 
                fn = fn.split('.')[-1]
                return getattr(c, fn)(*args, **kwargs)
        for k, fn in utils.items():
            setattr(obj, k, partial(wrapper_fn2, fn))
        return {'success': True, 'message': 'added utils'}
    route_cache = None

    def get_yaml( path:str=None, default={}, **kwargs) -> Dict:
        '''fLoads a yaml file'''
        import yaml
        path = os.path.abspath(path)
        with open(path, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        return data

    @classmethod
    def get_routes(cls, cache=True):
        if not hasattr(cls, 'routes'):
            if cls.route_cache is not None and cache:
                return cls.route_cache 
            routes_path = os.path.dirname(__file__)+ '/routes.json'
            routes =  cls.get_yaml(routes_path)
        else:
            routes = getattr(cls, 'routes')
            if callable(routes):
                routes = routes()
        cls.route_cache = routes
        return routes
    #### THE FINAL TOUCH , ROUTE ALL OF THE MODULES TO THE CURRENT MODULE BASED ON THE routes CONFIG

    @classmethod
    def fn2route(cls):
        routes = cls.get_routes()
        fn2route = {}
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
                fn2route[fn] = module
        return fn2route
            
    @classmethod
    def fn2routepath(cls):
        fn2route = {}
        for fn, module in cls.fn2route().items():
            fn2route[fn] = module + '.' + fn
        return fn2route
            
    @classmethod
    def add_routes(cls, routes:dict=None, verbose=False, add_utils=True):
        from functools import partial
        """
        This ties other modules into the current module.
        The way it works is that it takes the module name and the function name and creates a partial function that is bound to the module.
        This allows you to call the function as if it were a method of the current module.
        for example
        """
        t0 = time.time()
        # WARNING : THE PLACE HOLDERS MUST NOT INTERFERE WITH THE KWARGS OTHERWISE IT WILL CAUSE A BUG IF THE KWARGS ARE THE SAME AS THE PLACEHOLDERS
        # THE PLACEHOLDERS ARE NAMED AS module_ph and fn_ph AND WILL UNLIKELY INTERFERE WITH THE KWARGS
        def fn_generator(*args, fn_ph, **kwargs):
            def fn(*args, **kwargs):
                try:
                    fn_obj = c.import_object(fn_ph)
                except: 
                    module = '.'.join(fn_ph.split('.')[:-1])
                    fn = fn_ph.split('.')[-1]

                    module = c.get_module(module)
                    fn_obj = getattr(module, fn)
                    if c.classify_fn(fn_obj) == 'self':
                        fn_obj = getattr(module(), fn)
                if callable(fn_obj):
                    return fn_obj(*args, **kwargs)
                else:
                    return fn_obj
        
            return fn(*args, **kwargs)
        
        routes = cls.get_routes()
        for module, fns in routes.items():
            if c.module_exists(module):
                if fns in ['all', '*']:
                    continue
                for fn in fns: 
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

                    fn_ph = module + '.' + from_fn
                    fn_obj = partial(fn_generator, fn_ph=fn_ph) 
                    fn_obj.__name__ = to_fn
                    if not hasattr(cls, to_fn):
                        setattr(cls, to_fn, fn_obj)
                    else: 
                        c.print(f'WARNING ROUTERS: {to_fn} already exists in {cls.module_name()}', color='yellow')
        latency = time.time() - t0
        return {'success': True, 'msg': 'enabled routes', 'latency': latency}
    
    @classmethod
    def has_test_module(cls, module=None):
        module = module or cls.module_name()
        return cls.module_exists(cls.module_name() + '.test')
    
    @classmethod
    def test(cls,
              module=None,
              timeout=42, 
              trials=3, 
              parallel=True,
              ):
        module = module or cls.module_name()

        if c.module_exists( module + '.test'):
            module =  module + '.test'
        module = c.module(module)()
        test_fns = module.test_fns()

        def trial_wrapper(fn, trials=trials):
            def trial_fn(trials=trials):

                for i in range(trials):
                    try:
                        return fn()
                    except Exception as e:
                        print(f'Error: {e}, Retrying {i}/{trials}')
                        cls.c.sleep(1)
                return False
            return trial_fn
        fn2result = {}
        if parallel:
            future2fn = {}
            for fn in test_fns:
                f = cls.submit(trial_wrapper(getattr(module, fn)), timeout=timeout)
                future2fn[f] = fn
            for f in cls.as_completed(future2fn, timeout=timeout):
                fn = future2fn.pop(f)
                fn2result[fn] = f.result()
        else:
            for fn in cls.test_fns():
                print(f'testing {fn}')
                fn2result[fn] = trial_wrapper(getattr(cls, fn))()       
        return fn2result
    
    @classmethod
    def add_to_globals(cls, globals_input:dict = None):
        from functools import partial
        globals_input = globals_input or {}
        for k,v in c.__dict__.items():
            globals_input[k] = v     
        for f in c.class_functions() + c.static_functions():
            globals_input[f] = getattr(c, f)

        for f in c.self_functions():
            def wrapper_fn(f, *args, **kwargs):
                try:
                    fn = getattr(Module(), f)
                except:
                    fn = getattr(Module, f)
                return fn(*args, **kwargs)
        
            globals_input[f] = partial(wrapper_fn, f)

        return globals_input


    def set_config(self, config:Optional[Union[str, dict]]=None ) -> 'Munch':
        '''
        Set the config as well as its local params
        '''
        # in case they passed in a locals() dict, we want to resolve the kwargs and avoid ambiguous args
        config = config or {}
        config = {**self.config(), **config}
        if isinstance(config, dict):
            config = c.dict2munch(config)
        self.config = config 
        return self.config

    def config_exists(self, path:str=None) -> bool:
        '''
        Returns true if the config exists
        '''
        path = path if path else self.config_path()
        return self.path_exists(path)

    @classmethod
    def config(cls) -> 'Munch':
        '''
        Returns the config
        '''
        config = cls.load_config()
        if not config:
            if hasattr(cls, 'init_kwargs'):
                config = cls.init_kwargs() # from _schema.py
            else:
                config = {}
        return config

    @classmethod
    def load_config(cls, path:str=None, 
                    default=None,
                    to_munch:bool = True  
                    ) -> Union['Munch', Dict]:
        '''
        Args:
            path: The path to the config file
            to_munch: If true, then convert the config to a munch
        '''

        default = default or {}
        path = path if path else cls.config_path()

        if os.path.exists(path):
            config = cls.load_yaml(path)
        else:
            config = default
        config = config or {} 
        if to_munch:
            config =  cls.dict2munch(config)
        return config
    
    @classmethod
    def save_config(cls, config:Union['Munch', Dict]= None, path:str=None) -> 'Munch':
        from copy import deepcopy
        from munch import Munch
        '''
        Saves the config to a yaml file
        '''
        if config == None:
            config = cls.config()
        
        if isinstance(config, Munch):
            config = cls.munch2dict(deepcopy(config))
        elif isinstance(config, dict):
            config = deepcopy(config)
        else:
            raise ValueError(f'config must be a dict or munch, not {type(config)}')
        
        assert isinstance(config, dict), f'config must be a dict, not {config}'

        config = cls.save_yaml(data=config , path=path)

        return config
      
    @classmethod
    def has_config(cls) -> bool:
        try:
            return os.path.exists(cls.config_path())
        except:
            return False
    
    @classmethod
    def config_path(cls) -> str:
        return os.path.abspath('./config.yaml')

    def update_config(self, config):
        self.config.update(config)
        return self.config

    @classmethod
    def put_json(cls, 
                 path:str, 
                 data:Dict, 
                 meta = None,
                 verbose: bool = False,
                 **kwargs) -> str:
        if meta != None:
            data = {'data':data, 'meta':meta}
        if not path.endswith('.json'):
            path = path + '.json'
        path = cls.resolve_path(path=path)
        # cls.lock_file(path)
        if isinstance(data, dict):
            data = json.dumps(data)
        cls.put_text(path, data)
        return path
    

    save_json = put_json

    @classmethod
    def rm(cls, path,possible_extensions = ['json'], avoid_paths = ['~', '/']):
        path = cls.resolve_path(path=path)
        avoid_paths = [cls.resolve_path(p) for p in avoid_paths]
        assert path not in avoid_paths, f'Cannot remove {path}'
        if not os.path.exists(path):
            for pe in possible_extensions:
                if path.endswith(pe) and os.path.exists(path + f'.{pe}'):
                    path = path + f'.{pe}'
                    break
        if not os.path.exists(path): 
            return {'success':False, 'message':f'{path} does not exist'}
        if os.path.isdir(path):
            return shutil.rmtree(path)
        if os.path.isfile(path):
            os.remove(path)
        assert not os.path.exists(path), f'{path} was not removed'

        return {'success':True, 'message':f'{path} removed'}
    
    @classmethod
    def glob(cls,  path =None, files_only:bool = True, recursive:bool=True):
        import glob
        path = cls.resolve_path(path)
        if os.path.isdir(path):
            path = os.path.join(path, '**')
        paths = glob.glob(path, recursive=recursive)
        if files_only:
            paths =  list(filter(lambda f:os.path.isfile(f), paths))
        return paths
    
    @classmethod
    def get_json(cls, 
                path:str,
                default:Any=None,
                **kwargs):
        path = cls.resolve_path(path=path, extension='json')
        try:
            data = cls.get_text(path, **kwargs)
        except Exception as e:
            return default
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception as e:
                return default
        if isinstance(data, dict):
            if 'data' in data and 'meta' in data:
                data = data['data']
        return data
    @classmethod
    async def async_get_json(cls,*args, **kwargs):
        return  cls.get_json(*args, **kwargs)
    load_json = get_json

    @classmethod
    def path_exists(cls, path:str)-> bool:
        if os.path.exists(path):
            return True
        path = cls.resolve_path(path)
        exists =  os.path.exists(path)
        return exists
    
    file_exists = path_exists

    @classmethod
    def mv(cls, path1, path2):
        path1 = cls.resolve_path(path1)
        path2 = cls.resolve_path(path2)
        assert os.path.exists(path1), path1
        if not os.path.isdir(path2):
            path2_dirpath = os.path.dirname(path2)
            if not os.path.isdir(path2_dirpath):
                os.makedirs(path2_dirpath, exist_ok=True)
        shutil.move(path1, path2)
        assert os.path.exists(path2), path2
        assert not os.path.exists(path1), path1
        return path2

    @classmethod
    def resolve_path(cls, path:str = None, extension:Optional[str]=None):
        '''
        Abspath except for when the path does not have a
        leading / or ~ or . in which case it is appended to the storage dir
        '''
        if path == None:
            return cls.storage_dir()
        if path.startswith('/'):
            path = path
        elif path.startswith('~') :
            path = os.path.expanduser(path)
        elif path.startswith('.'):
            path = os.path.abspath(path)
        else:
            storage_dir = cls.storage_dir()
            if storage_dir not in path:
                path = os.path.join(storage_dir, path)
        if extension != None and not path.endswith(extension):
            path = path + '.' + extension
        return path
    
    @classmethod
    def abspath(cls, path:str):
        return os.path.abspath(path)
     
    def file2size(self, path='./', fmt='mb') -> int:
        files = c.glob(path)
        file2size = {}
        pwd = c.pwd()
        for file in files:
            file2size[file.replace(pwd+'/','')] = self.format_data_size(self.filesize(file), fmt)

        # sort by size
        file2size = dict(sorted(file2size.items(), key=lambda item: item[1]))
        return file2size

    @classmethod
    def put_text(cls, path:str, text:str, key=None, bits_per_character=8) -> None:
        # Get the absolute path of the file
        path = cls.resolve_path(path)
        dirpath = os.path.dirname(path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        if not isinstance(text, str):
            text = cls.python2str(text)
        if key != None:
            text = cls.get_key(key).encrypt(text)
        # Write the text to the file
        with open(path, 'w') as file:
            file.write(text)
        # get size
        text_size = len(text)*bits_per_character
    
        return {'success': True, 'msg': f'Wrote text to {path}', 'size': text_size}
    
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
        path = cls.resolve_path(path)
        try:
            ls_files = os.listdir(path)
        except Exception as e:
            return []
        if return_full_path:
            ls_files = [os.path.abspath(os.path.join(path,f)) for f in ls_files]
        ls_files = sorted(ls_files)
        if search != None:
            ls_files = list(filter(lambda x: search in x, ls_files))
        return ls_files
    
    @classmethod
    def put(cls, 
            k: str, 
            v: Any,  
            mode: bool = 'json',
            encrypt: bool = False, 
            password: str = None, **kwargs) -> Any:
        '''
        Puts a value in the config
        '''
        encrypt = encrypt or password != None
        
        if encrypt or password != None:
            v = cls.encrypt(v, password=password)

        if not c.jsonable(v):
            v = c.serialize(v)    
        
        data = {'data': v, 'encrypted': encrypt, 'timestamp': cls.timestamp()}            
        
        # default json 
        getattr(cls,f'put_{mode}')(k, data)

        data_size = cls.sizeof(v)
    
        return {'k': k, 'data_size': data_size, 'encrypted': encrypt, 'timestamp': cls.timestamp()}
    
    @classmethod
    def get(cls,
            k:str, 
            default: Any=None, 
            mode:str = 'json',
            max_age:str = None,
            cache :bool = False,
            full :bool = False,
            update :bool = False,
            password : str = None,
            verbose = False,
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
            data['data'] = c.decrypt(data['data'], password=password)

        data = data or default
        
        if isinstance(data, dict):
            if update:
                max_age = 0
            if max_age != None:
                timestamp = data.get('timestamp', None)
                if timestamp != None:
                    age = int(time.time() - timestamp)
                    if age > max_age: # if the age is greater than the max age
                        c.print(f'{k} is too old ({age} > {max_age})', verbose=verbose)
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
    
    def get_age(self, k:str) -> int:
        data = self.get_json(k)
        timestamp = data.get('timestamp', None)
        if timestamp != None:
            age = int(time.time() - timestamp)
            return age
        return -1
    
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

        if not os.path.exists(path):
            if os.path.exists(path + '.json'):
                path = path + '.json'

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


    def is_encrypted(self, path:str) -> bool:
        try:
            return self.get_json(path).get('encrypted', False)
        except:
            return False

    @classmethod
    def storage_dir(cls):
        return f'{c.storage_path}/{cls.module_name()}'
    
    tmp_dir = cache_dir   = storage_dir

    def is_dir_empty(self, path:str):
        return len(self.ls(path)) == 0
    
    @staticmethod
    def sleep(period):
        time.sleep(period) 
    

    def num_files(self, path:str='./')-> int:
        import commune as c
        return len(c.glob(path))
            
    @classmethod
    def fn2code(cls, search=None, module=None)-> Dict[str, str]:
        module = module if module else cls
        functions = module.fns(search)
        fn_code_map = {}
        for fn in functions:
            try:
                fn_code_map[fn] = module.fn_code(fn)
            except Exception as e:
                print(f'Error: {e}')
        return fn_code_map
    
    @classmethod
    def fn_code(cls,fn:str, **kwargs) -> str:
        '''
        Returns the code of a function
        '''
        try:
            fn = cls.get_fn(fn)
            code_text = inspect.getsource(fn)
        except Exception as e:
            code_text = None
            raise e
            print(f'Error in getting fn_code: {e}')                    
        return code_text
    
    @classmethod
    def fn_hash(cls,fn:str = 'subspace/ls', detail:bool=False,  seperator: str = '/') -> str:
        fn_code = cls.fn_code(fn, detail=detail, seperator=seperator)
        return cls.hash(fn_code)

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
    def get_parents(cls, obj = None,recursive=True, avoid_classes=['object']) -> List[str]:
        obj = cls.resolve_object(obj)
        parents =  list(obj.__bases__)
        if recursive:
            for parent in parents:
                parent_parents = cls.get_parents(parent, recursive=recursive)
                if len(parent_parents) > 0:
                    for pp in parent_parents: 
                        if pp.__name__ not in avoid_classes:
                        
                            parents += [pp]
        return parents


    @classmethod
    def fn_schema(cls, fn:str,
                            defaults:bool=True,
                            code:bool = False,
                            docs:bool = True, **kwargs)->dict:
        '''
        Get function schema of function in cls
        '''
        fn_schema = {}
        fn = cls.get_fn(fn)
        input_schema  = cls.fn_signature(fn)
        for k,v in input_schema.items():
            v = str(v)
            if v.startswith('<class'):
                input_schema[k] = v.split("'")[1]
            elif v.startswith('typing.'):
                input_schema[k] = v.split(".")[1].lower()
            else:
                input_schema[k] = v

        fn_schema['input'] = input_schema
        fn_schema['output'] = input_schema.pop('return', {})

        if docs:         
            fn_schema['docs'] =  fn.__doc__ 
        if code:
            fn_schema['code'] = cls.fn_code(fn)

        fn_args = cls.get_function_args(fn)
        fn_schema['type'] = 'static'
        for arg in fn_args:
            if arg not in fn_schema['input']:
                fn_schema['input'][arg] = 'NA'
            if arg in ['self', 'cls']:
                fn_schema['type'] = arg
                fn_schema['input'].pop(arg)

        if defaults:
            fn_defaults = c.fn_defaults(fn=fn) 
            for k,v in fn_defaults.items(): 
                if k not in fn_schema['input'] and v != None:
                    fn_schema['input'][k] = type(v).__name__ if v != None else None

        fn_schema['input'] = {k: {'type':v, 'default':fn_defaults.get(k)} for k,v in fn_schema['input'].items()}

        return fn_schema

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
    
    init_params = init_kwargs
    
    @classmethod
    def lines_of_code(cls, code:str=None):
        if code == None:
            code = cls.code()
        return len(code.split('\n'))


    @classmethod
    def is_fn(cls, fn:str):
        return '/' in str(fn) or hasattr(cls, str(fn)) or (c.object_exists(fn) and callable(c.obj(fn)))
    @classmethod
    def code(cls, module = None, search=None, *args, **kwargs):
        if cls.is_fn(module):
            return cls.fn_code(module)
        module = cls.resolve_object(module)
        text =  c.get_text( c.filepath(module), *args, **kwargs)
        if search != None:
            return cls.find_lines(text=text, search=search)
        return text
    pycode = code
    @classmethod
    def chash(cls,  *args, **kwargs):
        import commune as c
        """
        The hash of the code, where the code is the code of the class (cls)
        """
        code = cls.code(*args, **kwargs)
        return c.hash(code)
    
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
        module_code = cls.code()
        in_fn = False
        start_line = 0
        end_line = 0
        fn_code_lines = []
        for i, line in enumerate(module_code.split('\n')):
            if f'def {fn}('.replace(' ', '') in line.replace(' ', ''):
                in_fn = True
                start_line = i + 1
            if in_fn:
                fn_code_lines.append(line)
                if ('def ' in line or '' == line) and len(fn_code_lines) > 1:
                    end_line = i - 1
                    break

        if not in_fn:
            end_line = start_line + len(fn_code_lines)   # find the endline

        for i, line in enumerate(lines):
            is_end = bool(')' in line and ':' in line)
            if is_end:
                start_code_line = i
                break 
        return {
            'start_line': start_line,
            'end_line': end_line,
            'code': code,
            'n_lines': len(lines),
            'hash': cls.hash(code),
            'start_code_line': start_code_line + start_line ,
            'mode': mode
        }

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

    @classmethod
    def resolve_class(cls, obj):
        '''
        resolve class of object or return class if it is a class
        '''
        if cls.is_class(obj):
            return obj
        else:
            return obj.__class__

    @classmethod
    def has_var_keyword(cls, fn='__init__', fn_signature=None):
        if fn_signature == None:
            fn_signature = cls.resolve_fn(fn)
        for param_info in fn_signature.values():
            if param_info.kind._name_ == 'VAR_KEYWORD':
                return True
        return False
    
    @classmethod
    def fn_signature(cls, fn) -> dict: 
        '''
        get the signature of a function
        '''
        if isinstance(fn, str):
            fn = getattr(cls, fn)
        return dict(inspect.signature(fn)._parameters)
    
    get_function_signature = function_signature = fn_signature
    @classmethod
    def is_arg_key_valid(cls, key='config', fn='__init__'):
        fn_signature = cls.function_signature(fn)
        if key in fn_signature: 
            return True
        else:
            for param_info in fn_signature.values():
                if param_info.kind._name_ == 'VAR_KEYWORD':
                    return True
        return False
    
    @classmethod
    def self_functions(cls: Union[str, type], obj=None, search=None):
        '''
        Gets the self methods in a class
        '''
        obj = cls.resolve_object(obj)
        functions =  cls.get_functions(obj)
        signature_map = {f:cls.get_function_args(getattr(obj, f)) for f in functions}
        if search != None:
            functions = [f for f in functions if search in f]
        return [k for k, v in signature_map.items() if 'self' in v]
    
    @classmethod
    def class_functions(cls: Union[str, type], obj=None):
        '''
        Gets the self methods in a class
        '''
        obj = cls.resolve_object(obj)
        functions =  cls.get_functions(obj)
        signature_map = {f:cls.get_function_args(getattr(obj, f)) for f in functions}
        return [k for k, v in signature_map.items() if 'cls' in v]
    
    class_methods = get_class_methods =  class_fns = class_functions

    @classmethod
    def static_functions(cls: Union[str, type], obj=None):
        '''
        Gets the self methods in a class
        '''
        obj = obj or cls
        functions =  cls.get_functions(obj)
        signature_map = {f:cls.get_function_args(getattr(obj, f)) for f in functions}
        return [k for k, v in signature_map.items() if not ('self' in v or 'cls' in v)]
    
    static_methods = static_fns =  static_functions

    @classmethod
    def property_fns(cls) -> bool:
        '''
        Get a list of property functions in a class
        '''
        return [fn for fn in dir(cls) if cls.is_property(fn)]
    
    parents = get_parents
    
    @classmethod
    def parent2functions(cls, obj=None):
        '''
        Get the parent classes of a class
        '''
        obj = cls.resolve_object(obj)
        parent_functions = {}
        for parent in cls.parents(obj):
            parent_functions[parent.__name__] = cls.get_functions(parent)
        return parent_functions
    
    parent2fns = parent2functions

    @classmethod
    def get_functions(cls, 
                      obj: Any = None,
                      search = None,
                      splitter_options = ["   def " , "    def "] ,
                      include_parents:bool=True, 
                      include_hidden:bool = False) -> List[str]:
        '''
        Get a list of functions in a class
        
        Args;
            obj: the class to get the functions from
            include_parents: whether to include the parent functions
            include_hidden:  whether to include hidden functions (starts and begins with "__")
        '''

        obj = cls.resolve_object(obj)
        functions = []
        text = inspect.getsource(obj)
        functions = []
        # just
        for splitter in splitter_options:
            for line in text.split('\n'):
                if f'"{splitter}"' in line:
                    continue
                if line.startswith(splitter):
                    functions += [line.split(splitter)[1].split('(')[0]]

        functions = sorted(list(set(functions)))
        if search != None:
            functions = [f for f in functions if search in f]
        return functions
    
    @classmethod
    def functions(cls, search = None, include_parents = True):
        return cls.get_functions(search=search, include_parents=include_parents)

    @classmethod
    def get_conflict_functions(cls, obj = None):
        '''
        Does the object conflict with the current object
        '''
        if isinstance(obj, str):
            obj = cls.get_module(obj)
        root_fns = cls.root_functions()
        conflict_functions = []
        for fn in obj.functions():
            if fn in root_fns:
                print(f'Conflict: {fn}')
                conflict_functions.append(fn)
        return conflict_functions
    
    @classmethod
    def does_module_conflict(cls, obj):
        return len(cls.get_conflict_functions(obj)) > 0
    
 
    def n_fns(self, search = None):
        return len(self.fns(search=search))
    
    fn_n = n_fns
    @classmethod
    def fns(self, search = None, include_parents = True):
        return self.get_functions(search=search, include_parents=include_parents)
    @classmethod
    def is_property(cls, fn: 'Callable') -> bool:
        '''
        is the function a property
        '''
        fn = c.get_fn(fn)
        return isinstance(fn, property)

    def is_fn_self(self, fn):
        fn = self.resolve_fn(fn)
        return hasattr(fn, '__self__') and fn.__self__ == self
    
    @classmethod
    def exists(cls, path:str):
        return os.path.exists(path) or os.path.exists(cls.resolve_path(path))
    @classmethod
    def is_fn(cls, fn, splitters = [':', '/', '.']):
        try:
            if hasattr(cls, fn):
                fn = getattr(cls, fn)
            elif c.object_exists(fn):

                fn = c.obj(fn)
            elif any([s in fn for s in splitters]):
                splitter = [s for s in splitters if s in fn][0]
                module = splitter.join(fn.split(splitter)[:-1])
                fn = fn.split(splitter)[-1]
                fn = getattr(c.get_module(module), fn)
        except Exception as e:
            print('Error in is_fn:', e, fn)
            return False
        return callable(fn)

    @classmethod
    def get_fn(cls, fn:str, module=None, init_kwargs = None):
        """
        Gets the function from a string or if its an attribute 
        """
        module = module or cls
        if isinstance(fn, str):
            if c.object_exists(fn):
                return c.obj(fn)
            if hasattr(module, fn):
                # step 3, if the function is routed
                fn2routepath = cls.fn2routepath()
                if fn in fn2routepath:
                    fn = fn2routepath[fn]
                    if c.module_exists(module):
                        module = c.get_module(module)
                        fn 
                        fn = getattr(c.get_module(module), fn.split('.')[-1])
                        return fn
                return getattr(module, fn)
            
            for splitter in ['.', '/']:
                if splitter in fn:
                    module_name= splitter.join(fn.split(splitter)[:-1])
                    fn_name = fn.split(splitter)[-1]
                    if c.module_exists(module_name):
                        module = c.get_module(module_name)
                        fn = getattr(module, fn_name)
                        break

        return fn
        
    @classmethod
    def self_functions(cls, search = None):
        fns =  cls.classify_fns(cls)['self']
        if search != None:
            fns = [f for f in fns if search in f]
        return fns

    @classmethod
    def classify_fns(cls, obj= None, mode=None):
        method_type_map = {}
        obj = cls.resolve_object(obj)
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
    def get_args(cls, fn) -> List[str]:
        """
        get the arguments of a function
        params:
            fn: the function
        """
        # if fn is an object get the __
        
        if not callable(fn):
            fn = cls.get_fn(fn)
        try:
            args = inspect.getfullargspec(fn).args
        except Exception as e:
            args = []
        return args
    
    get_function_args = get_args 

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
            return 'cls'
        args = cls.get_function_args(fn)
        if len(args) == 0:
            return 'property'
        if args[0] == 'self':
            return 'self'
        elif args[0] == 'cls':
            return 'class'

        return 'static'
        
    @classmethod
    def python2types(cls, d:dict)-> dict:
        return {k:str(type(v)).split("'")[1] for k,v in d.items()}
    @classmethod
    def fn2hash(cls, fn=None , mode='sha256', **kwargs):
        fn2hash = {}
        for k,v in cls.fn2code(**kwargs).items():
            fn2hash[k] = c.hash(v,mode=mode)
        if fn:
            return fn2hash[fn]
        return fn2hash
    # TAG CITY     
    @classmethod
    def parent_functions(cls, obj = None, include_root = True):
        functions = []
        obj = obj or cls
        parents = cls.get_parents(obj)
        for parent in parents:
            is_parent_root = cls.is_root_module(parent)
            if is_parent_root:
                continue
            
            for name, member in parent.__dict__.items():
                if not name.startswith('__'):
                    functions.append(name)
        return functions

    @classmethod
    def child_functions(cls, obj=None):
        obj = cls.resolve_object(obj)
        methods = []
        for name, member in obj.__dict__.items():
            if inspect.isfunction(member) and not name.startswith('__'):
                methods.append(name)
        return methods

    def kwargs2attributes(self, kwargs:dict, ignore_error:bool = False):
        for k,v in kwargs.items():
            if k != 'self': # skip the self
                # we dont want to overwrite existing variables from 
                if not ignore_error: 
                    assert not hasattr(self, k)
                setattr(self, k)

    def num_fns(self):
        return len(self.fns())

    def fn2type(self):
        fn2type = {}
        fns = self.fns()
        for f in fns:
            if callable(getattr(self, f)):
                fn2type[f] = self.classify_fn(getattr(self, f))
        return fn2type
    
    @classmethod
    def is_dir_module(cls, path:str) -> bool:
        """
        determine if the path is a module
        """
        filepath = cls.name2path(path)
        if path.replace('.', '/') + '/' in filepath:
            return True
        if ('modules/' + path.replace('.', '/')) in filepath:
            return True
        return False
    
    @staticmethod
    def is_imported(package:str) :
        return  bool(package in sys.modules)
    
    @classmethod
    def is_parent(cls, obj=None):
        obj = obj or cls 
        return bool(obj in cls.get_parents())

    @classmethod
    def find_code_lines(cls,  search:str = None , module=None) -> List[str]:
        module_code = cls.get_module(module).code()
        return cls.find_lines(search=search, text=module_code)

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

    @classmethod
    def params(cls, module=None, fn='__init__'):
        module = c.module(module) if  module else cls
        params =  c.fn_defaults(getattr(module, fn))
        params.pop('self', None)
        return params

    @classmethod
    def name2path(cls, 
                    simple:str,
                    extension = '.py',
                    ignore_prefixes = ['', 
                                       'src', 
                                      'commune', 
                                      'commune/module', 
                                      'commune/modules', 
                                      'modules', 
                                      'module',
                                      'blocks', 
                                      'agents', 
                                      'commune/agents'],
                    **kwargs) -> bool:
        """
        converts the module path to a file path

        for example 

        model.openai.gpt3 -> model/openai/gpt3.py, model/openai/gpt3_module.py, model/openai/__init__.py 
        model.openai -> model/openai.py or model/openai_module.py or model/__init__.py

        Parameters:
            path (str): The module path
        """
        # if cls.libname in simple and '/' not in simple and cls.can_import_module(simple):
        #     return simple
        shortcuts = c.shortcuts
        simple = shortcuts.get(simple, simple)

        if simple.endswith(extension):
            simple = simple[:-len(extension)]

        path = None
        pwd = c.pwd()
        path_options = []
        simple = simple.replace('/', '.')

        # create all of the possible paths by combining the ignore_prefixes with the simple path
        dir_paths = list([pwd+ '/' + x for x in ignore_prefixes]) # local first
        dir_paths += list([c.libpath + '/' + x for x in ignore_prefixes]) # add libpath stuff

        for dir_path in dir_paths:
            if dir_path.endswith('/'):
                dir_path = dir_path[:-1]
            # '/' count how many times the path has been split
            module_dirpath = dir_path + '/' + simple.replace('.', '/')
            if os.path.isdir(module_dirpath):
                simple_filename = simple.replace('.', '_')
                filename_options = [simple_filename, simple_filename + '_module', 'module_'+ simple_filename] + ['module'] + simple.split('.') + ['__init__']
                path_options +=  [module_dirpath + '/' + f  for f in filename_options]  
            else:
                module_filepath = dir_path + '/' + simple.replace('.', '/') 
                path_options += [module_filepath]

            for p in path_options:
                p = cls.resolve_extension(p)
                if os.path.exists(p):
                    p_text = cls.get_text(p)
                    path =  p
                    if c.libname in p_text and 'class ' in p_text or '  def ' in p_text:
                        return p   
            if path != None:
                break
        return path


    @classmethod
    def path2name(cls,  
                    path:str, 
                    ignore_prefixes = ['src', 'commune', 'modules', 'commune.modules', 'module'],
                    module_folder_filnames = ['__init__', 'main', 'module'],
                    module_extension = 'py',
                    ignore_suffixes = ['module'],
                    name_map = {'commune': 'module'},
                    compress_path = True,
                    verbose = False,
                    **kwargs
                    ) -> str:
        
        path  = os.path.abspath(path)
        path_filename_with_extension = path.split('/')[-1] # get the filename with extension     
        path_extension = path_filename_with_extension.split('.')[-1] # get the extension
        assert path_extension == module_extension, f'Invalid extension {path_extension} for path {path}'
        path_filename = path_filename_with_extension[:-len(path_extension)-1] # remove the extension
        path_filename_chunks = path_filename.split('_')
        path_chunks = path.split('/')

        if path.startswith(c.libpath):
            path = path[len(c.libpath):]
        else:
            pwd = c.pwd()
            if path.startswith(pwd):
                path = path[len(pwd):]
            else:
                raise ValueError(f'Path {path} is not in libpath {c.libpath} or pwd {pwd}') 
        dir_chunks = path.split('/')[:-1] if '/' in path else []
        is_module_folder = all([bool(chunk in dir_chunks) for chunk in path_filename_chunks])
        is_module_folder = is_module_folder or (path_filename in module_folder_filnames)
        if is_module_folder:
            path = '/'.join(path.split('/')[:-1])
        path = path[1:] if path.startswith('/') else path
        path = path.replace('/', '.')
        module_extension = '.'+module_extension
        if path.endswith(module_extension):
            path = path[:-len(module_extension)]
        if compress_path:
            path_chunks = path.split('.')
            simple_path = []
            for chunk in path_chunks:
                if chunk not in simple_path:
                    simple_path += [chunk]
            simple_path = '.'.join(simple_path)
        else:
            simple_path = path
        for prefix in ignore_prefixes:
            prefix += '.'
            if simple_path.startswith(prefix) and simple_path != prefix:
                simple_path = simple_path[len(prefix):]
                c.print(f'Prefix {prefix} in path {simple_path}', color='yellow', verbose=verbose)
        # FILTER SUFFIXES
        for suffix in ignore_suffixes:
            suffix = '.' + suffix
            if simple_path.endswith(suffix) and simple_path != suffix:
                simple_path = simple_path[:-len(suffix)]
                c.print(f'Suffix {suffix} in path {simple_path}', color='yellow', verbose=verbose)
        # remove leading and trailing dots
        if simple_path.startswith('.'):
            simple_path = simple_path[1:]
        if simple_path.endswith('.'):
            simple_path = simple_path[:-1]
        simple_path = name_map.get(simple_path, simple_path)
        return simple_path

    @classmethod
    def find_classes(cls, path='./', depth=8, **kwargs):
        path = os.path.abspath(path)
        if os.path.isdir(path):
            classes = []
            if depth == 0:
                return []
            for p in c.ls(path):
                if os.path.isdir(p):
                    classes += cls.find_classes(p, depth=depth-1)
                elif p.endswith('.py'):
                    p_classes =  cls.find_classes(p)
                    classes += p_classes
            return classes
        code = cls.get_text(path)
        classes = []
        file_path = cls.path2objectpath(path)
        
        for line in code.split('\n'):
            if line.startswith('class ') and line.strip().endswith(':'):
                new_class = line.split('class ')[-1].split('(')[0].strip()
                if new_class.endswith(':'):
                    new_class = new_class[:-1]
                if ' ' in new_class:
                    continue
                classes += [new_class]
        classes = [file_path + '.' + c for c in classes]
        return classes

    @classmethod
    def find_class2functions(cls, path):

        path = os.path.abspath(path)
        if os.path.isdir(path):
            class2functions = {}
            for p in cls.glob(path+'/**/**.py', recursive=True):
                if p.endswith('.py'):
                    object_path = cls.path2objectpath(p)
                    response =  cls.find_class2functions(p )
                    for k,v in response.items():
                        class2functions[object_path+ '.' +k] = v
            return class2functions

        code = cls.get_text(path)
        classes = []
        class2functions = {}
        class_functions = []
        new_class = None
        for line in code.split('\n'):
            if all([s in line for s in ['class ', ':']]):
                new_class = line.split('class ')[-1].split('(')[0].strip()
                if new_class.endswith(':'):
                    new_class = new_class[:-1]
                if ' ' in new_class:
                    continue
                classes += [new_class]
                if len(class_functions) > 0:
                    class2functions[new_class] = cls.copy(class_functions)
                class_functions = []
            if all([s in line for s in ['   def', '(']]):
                fn = line.split(' def')[-1].split('(')[0].strip()
                class_functions += [fn]
        if new_class != None:
            class2functions[new_class] = class_functions

        return class2functions
    
    @classmethod
    def path2objectpath(cls, path:str, **kwargs) -> str:
        
        path = os.path.abspath(path)
        dir_prefixes  = [c.libpath , c.pwd()]
        for dir_prefix in dir_prefixes:
            if path.startswith(dir_prefix):
                path =   path[len(dir_prefix) + 1:].replace('/', '.')
                break
        if path.endswith('.py'):
            path = path[:-3]
        return path.replace('__init__.', '.')
    
    @classmethod
    def objectpath2path(cls, objectpath:str, **kwargs) -> str:
        options  = [c.libpath, c.pwd()]
        for option in options:
            path = option + '/' + objectpath.replace('.', '/') + '.py'
            if os.path.exists(path):
                return path
        raise ValueError(f'Path not found for objectpath {objectpath}')

    @classmethod
    def find_functions(cls, path = './', **kwargs):
        fns = []
        if os.path.isdir(path):
            path = os.path.abspath(path)
            for p in cls.glob(path+'/**/**.py', recursive=True):
                p_fns = c.find_functions(p)
                file_object_path = cls.path2objectpath(p)
                p_fns = [file_object_path + '.' + f for f in p_fns]
                for fn in p_fns:
                    fns += [fn]

        else:
            code = cls.get_text(path)
            for line in code.split('\n'):
                if line.startswith('def ') or line.startswith('async def '):
                    fn = line.split('def ')[-1].split('(')[0].strip()
                    fns += [fn]
        return fns
    
    @classmethod
    def find_async_functions(cls, path):
        if os.path.isdir(path):
            path2classes = {}
            for p in cls.glob(path+'/**/**.py', recursive=True):
                path2classes[p] = cls.find_functions(p)
            return path2classes
        code = cls.get_text(path)
        fns = []
        for line in code.split('\n'):
            if line.startswith('async def '):
                fn = line.split('def ')[-1].split('(')[0].strip()
                fns += [fn]
        return [c for c in fns]
    
    @classmethod
    def find_objects(cls, path:str = './', depth=10, search=None, **kwargs):
        classes = cls.find_classes(path,depth=depth)
        functions = cls.find_functions(path)

        if search != None:
            classes = [c for c in classes if search in c]
            functions = [f for f in functions if search in f]
        object_paths = functions + classes
        return object_paths
    objs = search =  find_objects
    @classmethod
    def name2objectpath(cls, 
                          simple_path:str,
                           cactch_exception = False, 
                           **kwargs) -> str:

        object_path = cls.name2path(simple_path, **kwargs)
        classes =  cls.find_classes(object_path)
        return classes[-1]

    @classmethod
    def name2object(cls, path:str = None, **kwargs) -> str:
        path = path or 'module'
        path =  c.name2objectpath(path, **kwargs)
        try:
            return cls.import_object(path)
        except:
            path = cls.tree().get(path)
            return cls.import_object(path)
            
    included_pwd_in_path = False
    @classmethod
    def import_module(cls, 
                      import_path:str, 
                      included_pwd_in_path=True, 
                      try_prefixes = ['commune','commune.modules', 'modules', 'commune.network.substrate', 'subspace']
                      ) -> 'Object':
        from importlib import import_module
        pwd = os.getenv('PWD', c.libpath)
        if included_pwd_in_path and not cls.included_pwd_in_path:
            import sys            
            sys.path.append(pwd)
            sys.path = list(set(sys.path))
            cls.included_pwd_in_path = True
        # if commune is in the path more than once, we want to remove the duplicates
        if cls.libname in import_path:
            import_path = cls.libname + import_path.split(cls.libname)[-1]

        try:
            return import_module(import_path)
        except Exception as _e:
            for prefix in try_prefixes:
                try:
                    return import_module(f'{prefix}.{import_path}')
                except Exception as e:
                    pass
            raise _e
    
    @classmethod
    def can_import_module(cls, module:str) -> bool:
        '''
        Returns true if the module is valid
        '''
        try:
            cls.import_module(module)
            return True
        except:
            return False
        
    @classmethod
    def can_import_object(cls, module:str) -> bool:
        '''
        Returns true if the module is valid
        '''
        try:
            cls.import_object(module)
            return True
        except:
            return False

    @classmethod
    def ensure_syspath(cls, path:str, **kwargs):
        if path not in sys.path:
            sys.path.append(path)
            sys.path = list(set(sys.path))
        return {'path': path, 'sys.path': sys.path}

    @classmethod
    def import_object(cls, key:str, **kwargs)-> Any:
        ''' Import an object from a string with the format of {module_path}.{object}'''

        key = key.replace('/', '.')
        module_obj = c.import_module('.'.join(key.split('.')[:-1]))
        return  getattr(module_obj, key.split('.')[-1])
    
    o = obj = get_obj = import_object

    @classmethod
    def object_exists(cls, path:str, verbose=False)-> Any:
        try:
            c.import_object(path, verbose=verbose)
            return True
        except Exception as e:
            return False
    
    imp = get_object = importobj = import_object

    @classmethod
    def module_exists(cls, module:str, **kwargs) -> bool:
        '''
        Returns true if the module exists
        '''
        try:
            module = c.shortcuts.get(module, module)
            return os.path.exists(c.name2path(module))
        except Exception as e:
            return False
    
    @classmethod
    def has_app(cls, module:str, **kwargs) -> bool:
        return cls.module_exists(module + '.app', **kwargs)
    
    
    @classmethod
    def get_path(cls, module:str, **kwargs) -> bool:
        return c.module(module).filepath()
    
    @classmethod
    def objectpaths2names(cls,  paths):
        paths = [cls.objectpath2name(p) for p in paths]
        paths = [p for p in paths if p]
        return paths

    @classmethod
    def objectpath2name(cls, p, 
                        avoid_terms=['modules', 'agents', 'module']):
        chunks = p.split('.')
        if len(chunks) < 2:
            return None
        file_name = chunks[-2]
        chunks = chunks[:-1]
        path = ''
        for chunk in chunks:
            if chunk in path:
                continue
            path += chunk + '.'
        if file_name.endswith('_module'):
            path = '.'.join(path.split('.')[:-1])
        
        if path.startswith(cls.libname + '.'):
            path = path[len(cls.libname)+1:]

        if path.endswith('.'):
            path = path[:-1]

        if '_' in file_name:
            file_chunks =  file_name.split('_')
            if all([c in path for c in file_chunks]):
                path = '.'.join(path.split('.')[:-1])
        for avoid in avoid_terms:
            avoid = f'{avoid}.' 
            if avoid in path:
                path = path.replace(avoid, '')
        for avoid_suffix in ['module', 'agent']:
            if path.endswith('.' + avoid_suffix):
                path = path[:-len(avoid_suffix)-1]
        return path

    @classmethod
    def local_modules(cls, search=None, depth=2, **kwargs):
        object_paths = cls.find_classes(cls.pwd(), depth=depth)
        object_paths = cls.objectpaths2names(object_paths) 
        if search != None:
            object_paths = [p for p in object_paths if search in p]
        return sorted(list(set(object_paths)))
    @classmethod
    def lib_tree(cls, depth=10, **kwargs):
        return c.get_tree(c.libpath, depth=depth, **kwargs)
    @classmethod
    def core_tree(cls, depth=10, **kwargs):
        tree =  c.get_tree(c.libpath, depth=depth, **kwargs)
        return {k:v for k,v in tree.items() if '.modules.' not in v}
    @classmethod
    def local_tree(cls , depth=4, **kwargs):
        return c.get_tree(c.pwd(), depth=depth, **kwargs)
    
    @classmethod
    def get_tree(cls, path, depth = 10, max_age=60, update=False):
        tree_cache_path = 'tree/'+path.replace('/', '_')
        tree = c.get(tree_cache_path, None, max_age=max_age, update=update)
        if tree == None:
            c.print(f'BUIDLING TREE --> {path}', color='green')
            class_paths = cls.find_classes(path, depth=depth)
            simple_paths = cls.objectpaths2names(class_paths) 
            tree = dict(zip(simple_paths, class_paths))
            c.put(tree_cache_path, tree)
        return tree
    
    @staticmethod
    def round(x:Union[float, int], sig: int=6, small_value: float=1.0e-9):
        from commune.utils.math import round_sig
        return round_sig(x, sig=sig, small_value=small_value)

    @classmethod
    def module(cls, path:str = 'module',  cache=True,verbose = False, trials=1, **_kwargs ) -> str:
        
        og_path = path
        path = path or 'module'
        t0 = time.time()
        og_path = path
        if path in c.module_cache and cache:
            module = c.module_cache[path]
        else:
            if path in ['module', 'c']:
                module =  c
            else:

                tree = c.tree()
                path = c.shortcuts.get(path, path)
                path = tree.get(path, path)
                try:
                    module = c.import_object(path)
                except Exception as e:
                    if trials > 0:
                        trials -= 1
                        tree = c.tree(update=True)
                        return c.module(path, cache=cache, verbose=verbose, trials=trials)
                    else:
                        raise e
            if cache:
                c.module_cache[path] = module    
        latency = c.round(time.time() - t0, 3)
        # if 
        if not hasattr(module, 'module_name'):
            
            module.module_name = module.name = lambda *args, **kwargs : c.module_name(module)
            module.module_class = lambda *args, **kwargs : c.module_class(module)
            module.resolve_object = lambda *args, **kwargs : c.resolve_object(module)
            module.filepath = lambda *args, **kwargs : c.filepath(module)
            module.dirpath = lambda *args, **kwargs : c.dirpath(module)
            module.code = lambda *args, **kwargs : c.code(module)
            module.schema = lambda *args, **kwargs : c.schema(module)
            module.functions = module.fns = lambda *args, **kwargs : c.get_functions(module)
            module.params = lambda *args, **kwargs : c.params(module)
            module.key = c.get_key(module.module_name(), create_if_not_exists=True)
            
        c.print(f'Module({og_path}->{path})({latency}s)', verbose=verbose)     
        return module

    get_module = module
    
    _tree = None
    @classmethod
    def tree(cls, search=None, cache=True, update_lib=False, update_local=True,**kwargs):
        if cls._tree != None and cache:
            return cls._tree
        local_tree = c.local_tree(update=update_local)
        lib_tree = c.lib_tree(update=update_lib)
        tree = {**lib_tree, **local_tree}
        if cache:
            cls._tree = tree
        if search != None:
            tree = {k:v for k,v in tree.items() if search in k}
        return tree
    
    def overlapping_modules(self, search:str=None, **kwargs):
        local_modules = self.local_modules(search=search)
        lib_modules = self.lib_modules(search=search)
        return [m for m in local_modules if m in lib_modules]
    
    @classmethod
    def lib_modules(cls, search=None, depth=10000, **kwargs):
        object_paths = cls.find_classes(cls.libpath, depth=depth )
        object_paths = cls.objectpaths2names(object_paths) 
        if search != None:
            object_paths = [p for p in object_paths if search in p]
        return sorted(list(set(object_paths)))
    
    @classmethod
    def find_modules(cls, search=None, **kwargs):
        lib_modules = cls.lib_modules(search=search)
        local_modules = cls.local_modules(search=search, depth=4)
        return sorted(list(set(local_modules + lib_modules)))

    _modules = None
    @classmethod
    def modules(cls, search=None, cache=True,   **kwargs)-> List[str]:
        modules = cls._modules
        if not cache or modules == None:
            modules =  cls.find_modules(search=None, **kwargs)
        if search != None:
            modules = [m for m in modules if search in m]            
        return modules
    get_modules = modules

    @classmethod
    def has_module(cls, module, path=None):
        path = path or c.libpath
        return module in c.modules()
    
    def new_modules(self, *modules, **kwargs):
        for module in modules:
            self.new_module(module=module, **kwargs)

    @classmethod
    def new_module( cls,
                   path : str ,
                   name= None, 
                   base_module : str = 'base', 
                   update=1
                   ):
        path = os.path.abspath(path)
        path = path + '.py' if not path.endswith('.py') else path
        name = name or c.path2name(path)
        base_module = c.module(base_module)
        module_class_name = ''.join([m[0].capitalize() + m[1:] for m in name.split('.')])
        code = base_module.code()
        code = code.replace(base_module.__name__,module_class_name)
        dirpath = os.path.dirname(path)
        assert os.path.exists(path) or update
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        c.put_text(path, code)
        return {'name': name, 'path': path, 'msg': 'Module Created'}
    
    add_module = new_module

    @classmethod
    def has_local_module(cls, path=None):
        import commune as c 
        path = '.' if path == None else path
        if os.path.exists(f'{path}/module.py'):
            text = c.get_text(f'{path}/module.py')
            if 'class ' in text:
                return True
        return False
    
    @classmethod
    def filter(cls, text_list: List[str], filter_text: str) -> List[str]:
        return [text for text in text_list if filter_text in text]

    @staticmethod
    def tqdm(*args, **kwargs):
        from tqdm import tqdm
        return tqdm(*args, **kwargs)
    progress = tqdm

    @classmethod
    def jload(cls, json_string):
        import json
        return json.loads(json_string.replace("'", '"'))

    @classmethod
    def partial(cls, fn, *args, **kwargs):
        return partial(fn, *args, **kwargs)
    
    def init_nn(self):
        import torch
        torch.nn.Module.__init__(self)

    @classmethod
    def repo_url(cls, *args, **kwargs):
        return cls.module('git').repo_url(*args, **kwargs)    

    @classmethod
    def ps(cls, *args, **kwargs):
        return cls.get_module('docker').ps(*args, **kwargs)
 
    @classmethod
    def chown(cls, path:str = None, sudo:bool =True):
        path = cls.resolve_path(path)
        user = os.getenv('USER')
        cmd = f'chown -R {user}:{user} {path}'
        cls.cmd(cmd , sudo=sudo, verbose=True)
        return {'success':True, 'message':f'chown cache {path}'}

    @classmethod
    def chown_cache(cls, sudo:bool = True):
        return cls.chown(c.storage_path, sudo=sudo)
    
    @classmethod
    def get_util(cls, util:str, prefix='commune.utils'):
        path = prefix+'.' + util
        if c.object_exists(path):
            return c.import_object(path)
        else:
            return c.util2path().get(path)

    @classmethod
    def root_key(cls):
        return cls.get_key()

    @classmethod
    def root_key_address(cls) -> str:
        return cls.root_key().ss58_address
    
    @classmethod
    def is_root_key(cls, address:str)-> str:
        return address == cls.root_key().ss58_address

    @classmethod
    def folder_structure(cls, path:str='./', search='py', max_depth:int=5, depth:int=0)-> dict:
        import glob
        files = cls.glob(path + '/**')
        results = []
        for file in files:
            if os.path.isdir(file):
                cls.folder_structure(file, search=search, max_depth=max_depth, depth=depth+1)
            else:
                if search in file:
                    results.append(file)
        return results

    str2hash = hash

    def set_api_key(self, api_key:str, cache:bool = True):
        api_key = os.getenv(str(api_key), None)
        if api_key == None:
            api_key = self.get_api_key()
        self.api_key = api_key
        if cache:
            self.add_api_key(api_key)
        assert isinstance(api_key, str)

    def add_api_key(self, api_key:str, path=None):
        assert isinstance(api_key, str)
        path = self.resolve_path(path or 'api_keys')
        api_keys = self.get(path, [])
        api_keys.append(api_key)
        api_keys = list(set(api_keys))
        self.put(path, api_keys)
        return {'api_keys': api_keys}
    
    def set_api_keys(self, api_keys:str):
        api_keys = list(set(api_keys))
        self.put('api_keys', api_keys)
        return {'api_keys': api_keys}
    
    def rm_api_key(self, api_key:str):
        assert isinstance(api_key, str)
        api_keys = self.get(self.resolve_path('api_keys'), [])
        for i in range(len(api_keys)):
            if api_key == api_keys[i]:
                api_keys.pop(i)
                break   
        path = self.resolve_path('api_keys')
        self.put(path, api_keys)
        return {'api_keys': api_keys}

    def get_api_key(self, module=None):
        if module != None:
            self = self.module(module)
        api_keys = self.api_keys()
        if len(api_keys) == 0:
            raise 
        else:
            return self.choice(api_keys)

    def api_keys(self):
        return self.get(self.resolve_path('api_keys'), [])
    
    def rm_api_keys(self):
        self.put(self.resolve_path('api_keys'), [])
        return {'api_keys': []}
    
    @classmethod
    def executor(cls, max_workers:int=None, mode:str="thread", maxsize=200, **kwargs):
        return c.module(f'executor')(max_workers=max_workers, maxsize=maxsize ,mode=mode, **kwargs)

    def explain_myself(self):
        context = c.file2text(self.root_path)
        return c.ask(f'{context} write full multipage docuemntation aobut this, be as simple as possible with examples \n')

    @classmethod
    def remote_fn(cls, 
                    fn: str='train', 
                    module: str = None,
                    args : list = None,
                    kwargs : dict = None, 
                    name : str =None,
                    refresh : bool =True,
                    interpreter = 'python3',
                    autorestart : bool = True,
                    force : bool = False,
                    cwd = None,
                    **extra_launch_kwargs
                    ):

        kwargs = c.locals2kwargs(kwargs)
        kwargs = kwargs if kwargs else {}
        args = args if args else []
        if 'remote' in kwargs:
            kwargs['remote'] = False
        assert fn != None, 'fn must be specified for pm2 launch'
        kwargs = {
            'module': module, 
            'fn': fn,
            'args': args,
            'kwargs': kwargs
        }
        name = name or module
        if refresh:
            c.kill(name)
        module = c.module(module)
        kwargs_str = json.dumps(kwargs).replace('"', "'")
        filepath = module.filepath()
        cwd = os.path.dirname(filepath)
        root_filepath = c.module('module').filepath()
        command = f"pm2 start {root_filepath} --name {name} --interpreter {interpreter}"
        if not autorestart:
            command += ' --no-autorestart'
        if force:
            command += ' -f '
        command = command +  f' -- --fn module_fn --kwargs "{kwargs_str}"'
        return c.cmd(command, cwd=cwd)
    
    def explain(self, module, prompt='explain this fam', **kwargs):
        return c.ask(c.code(module) + prompt, **kwargs)

    @staticmethod
    def resolve_extension( filename:str, extension = '.py') -> str:
        if filename.endswith(extension):
                return filename
        return filename + extension
    
    def help(self, *text, module=None, global_context=f'{rootpath}/docs', **kwargs):
        text = ' '.join(map(str, text))
        if global_context != None:
            print(c.file2text(global_context))
            text = text + str(c.file2text(global_context))
        module = module or self.module_name()
        context = c.code(module)
        return c.ask(f'{context} write full multipage docuemntation aobut this, be as simple as possible with examples \n')
    
    def time(self):
        return time.time()
    
    def has_module(self, path:str):
        for path in c.files(path): 
            if path.endswith('.py'):
                return True

c.routes = c.get_routes()
c.add_routes()
Module = c # Module is alias of c
Module.run(__name__)


