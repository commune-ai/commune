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
nest_asyncio.apply()
config = json.load(open(__file__.replace(__file__.split('/')[-1], 'config.json')))

class c:
    # config attributes
    code_url = config["code_url"]
    repo_name  = config['name']# the name of the library
    description = config['description'] # the description of the library
    free = config["free"] # if the server is free 
    endpoints = config['endpoints']
    core_features = ['module_name', 'module_class',  'filepath', 'dirpath']
    port_range = config['port_range'] # the port range between 50050 and 50150
    shortcuts = config["shortcuts"]
    routes = config['routes']

    # path attributes
    root_path  = os.path.dirname(__file__)
    repo_path  = os.path.dirname(os.path.dirname(__file__)) # the path to the repo
    lib_path = os.path.dirname(os.path.dirname(__file__)) # the path to the library
    home_path  = os.path.expanduser('~') # the home path

     # the path to the modules
    modules_path =  lib_path + '/modules'  if os.path.exists(lib_path + '/modules') else root_path + '/module'
    home_modules_path = home_path + '/modules'
    temp_modules_path = lib_path + '/modules/temp'
    storage_path = os.path.expanduser(f'~/.{repo_name}')
    cache = {} # cache for module objects
    @classmethod
    def module(cls, 
               path:str = 'module', 
               params : dict = None, 
               cache=True, 
               trials=1, 
               verbose=False,
               tree:dict=None ) -> str:
        path = path or 'module'
        if path in ['module', c.repo_name[0]]:
            return c
        t0 = time.time()        
        path = c.shortcuts.get(path, path)
        tree = tree or c.tree()
        simp_path = path
        module = tree.get(path, path)
        cache_id = c.hash(module)
        if (not cache_id in c.module_cache) or not cache:
            try:
                module = c.obj2module(c.obj(module)) # if the model
            except Exception as e:
                c.tree(update=1)
                try:
                    module = c.obj2module(c.obj(module))
                except Exception as e: 
                    repo2path = c.repo2path()
                    if module in repo2path:
                        sys.path.append(repo2path[module])
                        module = c.obj2module(c.obj(module))
                    else:
                        raise Exception(f'Error loading module {module} {e}')
                        
            c.module_cache[cache_id] = module
        module = c.module_cache[cache_id]
        if params != None:
            module = module(**params)
        loadtime = c.time() - t0
        c.print(f'Module(name={obj_path} objpath={obj_path} loadtime={loadtime:.2f}') if verbose else ''
        return module
    
    get_agent = block =  get_block = get_module =  mod =  module
    
    @classmethod
    def obj2module(cls, obj:'Object', verbose=False, core_features=core_features):
        module = obj
        if c.is_object_module(obj):
            return module
        c.print(f'obj2module({obj})', verbose=verbose)
        module.resolve_module = lambda *args, **kwargs : c.resolve_module(module)
        module.module_name = lambda *args, **kwargs : c.module_name(module)
        module.storage_dir = lambda *args, **kwargs : c.storage_dir(module)
        module.filepath = lambda *args, **kwargs : c.filepath(module)
        module.dirpath = lambda *args, **kwargs : c.dirpath(module)
        module.code = lambda *args, **kwargs : c.code(module)
        module.info = lambda *args, **kwargs: c.info(module)
        module.code_hash = lambda *args, **kwargs : c.code_hash(module)
        module.schema = lambda *args, **kwargs : c.schema(module)
        module.fns = lambda *args, **kwargs : c.fns(module)

        if not hasattr(module, 'code'):
            module.code = lambda *args, **kwargs : c.code(module)
        if not hasattr(module, 'fn2code'):
            module.fn2code = lambda *args, **kwargs : c.fn2code(module)
        if not hasattr(module, 'fn2hash'): 
            module.fn2hash = lambda *args, **kwargs : c.fn2hash(module)
        if not hasattr(module, 'config_path'):
            module.config_path = lambda *args, **kwargs : c.config_path(module)
        if not hasattr(module, 'config'):
            module.config = lambda *args, **kwargs : c.config(module)
        return module

    @classmethod
    def filepath(cls, obj=None) -> str:
        return inspect.getfile(cls.resolve_module(obj))
    
    @classmethod
    def objectpath(cls, obj=None) -> str:
        return c.classes(cls.filepath(obj))[-1]

    @classmethod 
    def obj2code(self, path='./', search=None):
        obj2code = {}
        for obj in c.objs(path):
            if search != None and str(search) not in obj:
                continue
                
            try:
                obj2code[obj] = c.code(obj)
            except:
                pass
        return obj2code

    @classmethod 
    def obj2hash(self, path='./', search=None):
        obj2hash = {}
        for obj in c.objs(path):
            if search != None and str(search) not in obj:
                continue
                
            try:
                obj2hash[obj] = c.hash(c.code(obj))
            except:
                pass
        return obj2hash



    def filehash(self, path:str='./', reverse=True) -> int:
        path = c.abspath(path)
        return c.hash({k[len(path)+1:]:v for k,v in c.file2text(path).items()})
    
    @classmethod
    def dirpath(cls, obj=None) -> str:
        return os.path.dirname(cls.filepath(obj))
    
    @classmethod
    def module_name(cls, obj=None):
        obj = obj or cls
        if  isinstance(obj, str):
            obj = c.module(obj)
        module_file =  inspect.getfile(obj)
        return c.path2name(module_file)
    path  = name = module_name 
    def vs(self, path = None):
        path = path or c.lib_path
        path = os.path.abspath(path)
        return c.cmd(f'code {path}')
    
    @classmethod
    def module_class(cls, obj=None) -> str:
        return (obj or cls).__name__

    @classmethod
    def class_name(cls, obj= None) -> str:
        obj = obj if obj else cls
        return obj.__name__

    @classmethod
    def config_path(cls, obj = None) -> str:
        return c.dirpath(obj) + '/config.json'

    @classmethod
    def sandbox(cls, path='./', filename='sandbox.py'):
        for file in  c.files(path):
            if file.endswith(filename):
                return c.cmd(f'python3 {file}', verbose=True)
        return {'success': False, 'message': 'sandbox not found'}
    
    sand = sandbox
    cache_path = f'{storage_path}/cache'
    module_cache = {}
    _obj = None

    def syspath(self):
        return sys.path
    
    @classmethod
    def storage_dir(cls, module=None):
        module = cls.resolve_module(module)
        return f'{c.storage_path}/{module.module_name()}'

    @classmethod
    def is_object_module(cls, obj) -> bool:
        return all([hasattr(obj, k) for k in c.core_features])

    @staticmethod
    def config_paths(path='./'):
        return c.files(path, search='config.json')

    @classmethod
    def config_keys(cls, path='./'):
        return list(cls.config().keys())

    @staticmethod
    def json_paths(path='./'):
        return c.files(path, endswith='.json')

    @classmethod
    def is_admin(cls, key:str) -> bool:
        return c.get_key().key_address == key

    def print( *text:str,  **kwargs):
        return c.obj('commune.utils.log.print_console')(*text, **kwargs)

    @classmethod
    def resolve_module(cls, obj:str = None, default=None, fn_splitter='/', **kwargs):
        if obj == None:
            obj = cls
        elif isinstance(obj, str):
            obj = c.module(obj)
        return obj

    @classmethod
    def pwd(cls):
        pwd = os.getcwd() # the current wor king directory from the process starts 
        return pwd
    
    @classmethod
    def run(cls, fn=None, params=None, **_kwargs) -> Any: 
        if fn != None:
            return c.get_fn(fn)(**(params or {}))
        parser = argparse.ArgumentParser(description='Argparse for the module')
        parser.add_argument('-m', '--m', '--module', '-module', dest='module', help='The function', type=str, default=cls.module_name())
        parser.add_argument('-fn', '--fn', dest='fn', help='The function', type=str, default="__init__")
        parser.add_argument('-kw',  '-kwargs', '--kwargs', dest='kwargs', help='key word arguments to the function', type=str, default="{}") 
        parser.add_argument('-p', '-params', '--params', dest='params', help='key word arguments to the function', type=str, default="{}") 
        parser.add_argument('-args', '--args', dest='args', help='arguments to the function', type=str, default="[]")  
        argv = parser.parse_args()
        argv.kwargs = json.loads(argv.kwargs.replace("'",'"'))
        argv.params = params or json.loads(argv.params.replace("'",'"'))
        argv.args = json.loads(argv.args.replace("'",'"'))
        argv.fn = fn or argv.fn

        if len(argv.params) > 0:
            if isinstance(argv.params, dict):
                argv.kwargs = argv.params
            elif isinstance(argv.params, list):
                argv.args = argv.params
            else:
                raise Exception('Invalid params', argv.params)
        if argv.fn == '__init__' or c.classify_fn(getattr(cls, argv.fn)) == 'self':
            module =  cls(*argv.args, **argv.kwargs)   
        else:
            module = cls  
        print(argv)
        return getattr(module, argv.fn)(*argv.args, **argv.kwargs)     
        
    @classmethod
    def commit_hash(cls, lib_path:str = None):
        if lib_path == None:
            lib_path = c.lib_path
        return c.cmd('git rev-parse HEAD', cwd=lib_path, verbose=False).split('\n')[0].strip()

    @classmethod
    def fn(cls,fn:str, params=None, args=None, kwargs=None, module:str = None) -> Any:
        if '/' in fn:
            module, fn = fn.split('/')
        module = c.module(module)
        fn_obj = getattr(module, fn)
        is_self_method = 'self' in c.get_args(fn_obj)
        if is_self_method:
            module = module()
        fn_obj =  getattr(module, fn)
        params = params or {}
        if isinstance(params, list):
            args = params
        elif isinstance(params, dict):
            kwargs = params
        args = args or []
        kwargs = kwargs or {}
        return fn_obj(*args, **kwargs)
    
    # UNDER CONSTRUCTION (USE WITH CAUTION)
    run_fn = fn
    def forward(self, *args, **kwargs):
        return c.ask(*args, **kwargs)
    
    tests_path = f'{lib_path}/tests'
        # return c.cmd(f'pytest {c.tests_path}',  stream=1, *args, **kwargs)

    @classmethod
    def is_module_file(cls, module = None, exts=['py', 'rs', 'ts'], folder_filenames=['module', 'agent']) -> bool:
        dirpath = c.dirpath(module)
        filepath = c.filepath(module)
        for ext in exts:
            for fn in folder_filenames:
                if filepath.endswith(f'/{fn}.{ext}'):
                    return False
        return bool(dirpath.split('/')[-1] != filepath.split('/')[-1].split('.')[0])

    @classmethod
    def module2isfolder(cls): 
        module2isfolder = {}
        for m in c.modules():
            try:
                module2isfolder[m] = c.is_module_folder(m)
            except Exception as e:
                pass
        return module2isfolder    

    @classmethod
    def is_module_folder(cls,  module = None) -> bool:
        return not c.is_module_file(module)
    
    is_folder_module = is_module_folder 

    @classmethod
    def get_key(cls,key:str = None , **kwargs) -> None:
        from commune.key import Key
        if not isinstance(key, str) and hasattr(key,"module_name" ):
            key = key.module_name()
        return Key.get_key(key, **kwargs)
        

    @classmethod
    def key(cls,key:str = None , **kwargs) -> None:
        from commune.key import Key
        if not isinstance(key, str) and hasattr(key,"module_name" ):
            key = key.module_name()
        return Key.get_key(key, **kwargs)
    
    @classmethod
    def files(cls, 
              path='./', 
              search:str = None, 
              avoid_terms = ['__pycache__', '.git', '.ipynb_checkpoints', 'node_modules', 'artifacts', 'egg-info'], 
              endswith:str = None,
              startswith:str = None,
              **kwargs) -> List[str]:
        """
        Lists all files in the path
        params:
            path: the path to search
            search: the term to search for
            avoid_terms: the terms to avoid
        return :
            a list of files in the path
        """
        files =c.glob(path, **kwargs)
        files = [f for f in files if not any([at in f for at in avoid_terms])]
        if search != None:
            files = [f for f in files if search in f]
        return files


    def file2objs(self, path:str = './', **kwargs) -> Dict[str, Any]:
        obj2file = {}
        for file in c.files(path):
            try:                
                if not file.endswith('.py'):
                    continue  
                objs =  c.objs(file)
                if len(objs) > 0:
                    obj2file[file] = objs
            except Exception as e:
                print(f'Error loading {file} {e}')
                pass
        return obj2file


    @classmethod
    def num_files(cls, path='./',  **kwargs) -> List[str]: 
        return len(cls.files(path))
    
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
    def verify(cls, data, key=None, **kwargs ) -> bool:  
        return c.get_key(key).verify(data, **kwargs)

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
        
    # local update  
    @classmethod
    def update(cls,  ):
        c.namespace(update=True)
        c.ip(update=1)
        return {'ip': c.ip(), 'namespace': c.namespace()}

    @classmethod
    def utils(cls, search=None):
        utils = c.path2fns(c.root_path + '/utils', tolist=True)
        if search != None:
            utils = [u for u in utils if search in u]
        return sorted(utils)

    @classmethod
    def util2code(cls, search=None):
        utils = cls.utils()
        util2code = {}
        for f in utils:
            if search != None and search not in f:
                continue
            try:
                util2code[f] = c.code(f)
            except:
                pass
        return util2code

    @classmethod
    def util2hash(cls, search=None):
        return {k:c.hash(v) for k,v in c.util2code(search=search).items()}

    @classmethod
    def get_utils(cls, search=None):
        utils = c.path2fns(c.root_path + '/utils', tolist=True)
        if search != None:
            utils = [u for u in utils if search in u]
        return sorted(utils)
        
    @classmethod
    def util2path(cls, search=None):
        utils_paths = cls.utils(search=search)
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
                fn = c.obj(fn)
                return fn(*args, **kwargs)
            except : 
                fn = fn.split('.')[-1]
                return getattr(c, fn)(*args, **kwargs)
        for k, fn in utils.items():
            setattr(obj, k, partial(wrapper_fn2, fn))
        return {'success': True, 'message': 'added utils'}

    @classmethod
    def get_routes(cls):
        routes = cls.routes
        for util in  c.utils():
            k = '.'.join(util.split('.')[:-1])
            v = util.split('.')[-1]
            routes[k] = routes.get(k , [])
            routes[k].append(v)
        return routes

    @classmethod 
    def fn2route(cls):
        routes = cls.get_routes()
        fn2route = {}
        tree = c.tree()
        for module, fns in routes.items():
            is_module = bool( module in tree)
            splitter = '/' if  is_module else '/'
            for fn in fns:
                fn2route[fn] =  module + splitter + fn
        fn2route = dict(sorted({k: v for k, v in fn2route.items() if v != ''}.items(), key=lambda x: x[0]))
        return fn2route

    @classmethod
    def add_routes(cls, routes:dict=None):

        """
        This ties other modules into the current module.
        The way it works is that it takes the module name and the function name and creates a partial function that is bound to the module.
        This allows you to call the function as if it were a method of the current module.
        for example
        """
        routes = cls.get_routes()
        t0 = time.time()
        # WARNING : THE PLACE HOLDERS MUST NOT INTERFERE WITH THE KWARGS OTHERWISE IT WILL CAUSE A BUG IF THE KWARGS ARE THE SAME AS THE PLACEHOLDERS
        # THE PLACEHOLDERS ARE NAMED AS module_ph and fn_ph AND WILL UNLIKELY INTERFERE WITH THE KWARGS
        def fn_generator(*args, route, **kwargs):
            def fn(*args, **kwargs):
                try:
                    fn_obj = c.obj(route)
                except: 
                    if '/' in route:
                        module = '/'.join(route.split('/')[:-1])
                        fn = route.split('/')[-1]
                    module = c.module(module)
                    fn_obj = getattr(module, fn)
                    if c.classify_fn(fn_obj) == 'self':
                        fn_obj = getattr(module(), fn)
                if callable(fn_obj):
                    return fn_obj(*args, **kwargs)
                else:
                    return fn_obj
            return fn(*args, **kwargs)

        for module, fns in routes.items():
            for fn in fns: 
                if isinstance(fn, list):
                    to_fn = fn[1]
                    fn = fn[0]
                if isinstance(fn, dict):
                    to_fn = fn['to']
                    fn = fn['from']
                if isinstance(fn, str):
                    to_fn = fn
        
                if hasattr(cls, to_fn):
                    print(f'Warning: {to_fn} already exists')
                else:
                    fn_obj = partial(fn_generator, route=module + '/' + fn) 
                    fn_obj.__name__ = fn
                    setattr(cls, to_fn, fn_obj)
        latency = time.time() - t0
        return {'success': True, 'msg': 'enabled routes', 'latency': latency}

    @classmethod
    def add_to_globals(cls, globals_input:dict = None):
        from functools import partial
        globals_input = globals_input or {}
        for k,v in c.__dict__.items():
            globals_input[k] = v     
        for f in c.class_functions(c) + c.static_functions(c):
            globals_input[f] = getattr(c, f)
        for f in c.classify_fns(c, mode='self'):
            def wrapper_fn(f, *args, **kwargs):
                fn = getattr(Module(), f)
                return fn(*args, **kwargs)
            globals_input[f] = partial(wrapper_fn, f)
        return globals_input

    def set_config(self, config:Optional[Union[str, dict]]=None ) -> 'Munch':
        '''
        Set the config as well as its local params
        '''
        config = config or {}
        config = {**self.config(), **config}
        if isinstance(config, dict):
            config = c.dict2munch(config)
        self.config = config 
        return self.config

    def search(self, search:str = None, **kwargs):
        return c.objects(search=search, **kwargs)

    @classmethod
    def config(cls, module=None, to_munch=False, fn='__init__') -> 'Munch':
        module = module or cls
        path = module.config_path()
        if os.path.exists(path):
            config = c.get_json(path)
        else:
            config =  c.get_params(getattr(module, fn)) if hasattr(module, fn) else {}
        return c.dict2munch(config) if to_munch else config

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
    def rm(cls, path:str,
           possible_extensions = ['json'], 
           avoid_paths = ['~', '/', './', storage_path]):
        avoid_paths.append(c.storage_path)
        path = cls.resolve_path(path)
        avoid_paths = [cls.resolve_path(p) for p in avoid_paths] 
        assert path not in avoid_paths, f'Cannot remove {path}'
        path_exists = lambda p: os.path.exists(p)
        if not path_exists(path): 
            for pe in possible_extensions:
                if path.endswith(pe) and os.path.exists(path + f'.{pe}'):
                    path = path + f'.{pe}'
                    break
            if not path_exists(path):
                return {'success':False, 'message':f'{path} does not exist'}
        if os.path.isdir(path):
            shutil.rmtree(path)
        if os.path.isfile(path):
            os.remove(path)
        assert not os.path.exists(path), f'{path} was not removed'
        return {'success':True, 'message':f'{path} removed'}
    
    @classmethod
    def glob(cls,  path =None, files_only:bool = True, depth=None, recursive:bool=True):
        import glob
        path = cls.resolve_path(path)
        if os.path.isdir(path) and not path.endswith('**'):
            path = os.path.join(path, '**')
        if depth != None:
            paths = glob.glob(path, recursive=False)
        else:
            paths = glob.glob(path, recursive=recursive)
        if files_only:
            paths =  list(filter(lambda f:os.path.isfile(f), paths))
        
        return paths
    
    @classmethod
    def get_json(cls, path:str,default:Any=None, **kwargs):
        path = cls.resolve_path(path)

        # return c.util('get_json')(path, default=default, **kwargs)
        if not path.endswith('.json'):
            path = path + '.json'
        if not os.path.exists(path):
            return default
        try:
            with open(path, 'r') as file:
                data = json.load(file)
        except Exception as e:
            return default
        return data

    
    @classmethod
    def resolve_path(cls, 
                     path:str = None, 
                     extension:Optional[str]=None, 
                     storage_dir=None) -> str:
        '''
        Abspath except for when the path does not have a
        leading / or ~ or . in which case it is appended to the storage dir
        '''
    
        storage_dir = storage_dir or cls.storage_dir()
        if path == None :
            return storage_dir
        if path.startswith('/'):
            path = path
        elif path.startswith('~/') :
            path = os.path.expanduser(path)
        elif path.startswith('./'):
            path = os.path.abspath(path)
        else:
            if storage_dir not in path:
                path = os.path.join(storage_dir, path)
        if extension != None and not path.endswith(extension):
            path = path + '.' + extension
        return path

    @classmethod
    def put_text(cls, path:str, text:str, key=None) -> None:
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
        return {'success': True, 'path': f'{path}', 'size': len(text)*8}
    
    @classmethod
    def ls(cls, path:str = '', 
           search = None,
           include_hidden = False, 
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
            encrypt: bool = False, 
            password: str = None, **kwargs) -> Any:
        '''
        Puts a value in the config
        '''
        encrypt = encrypt or password != None
        if encrypt or password != None:
            v = c.encrypt(v, password=password)
        data = {'data': v, 'encrypted': encrypt, 'timestamp': time.time()}    
        c.put_json(k, data)
        return {'k': k, 'encrypted': encrypt, 'timestamp': time.time()}
    
    @classmethod
    def get(cls,
            k:str, 
            default: Any=None, 
            max_age:str = None,
            full :bool = False, 
            update :bool = False,
            password : str = None,
            verbose = False,
            **kwargs) -> Any:
        
        '''
        Puts a value in sthe config, with the option to encrypt it
        Return the value
        '''
        k = cls.resolve_path(k)
        data = c.get_json(k,default=default, **kwargs)
        if password != None:
            assert data['encrypted'] , f'{k} is not encrypted'
            data['data'] = c.decrypt(data['data'], password=password)
        data = data or default
        if not isinstance(data, dict):
            return default
        if update:
            max_age = 0

        if max_age != None:
            timestamp = 0
            for k in ['timestamp', 'time']:
                if k in data:
                    timestamp = data[k]
                    break
            age = int(time.time() - timestamp)
            if age > max_age: # if the age is greater than the max age
                c.print(f'{k} is too old ({age} > {max_age})', verbose=verbose)
                return default
        
        if not full:
            if isinstance(data, dict):
                if 'data' in data:
                    data = data['data']
            if isinstance(data, str) and data.startswith('{') and data.endswith('}'):
                data = data.replace("'",'"')
                data = json.loads(data)
        return data
    
    @classmethod
    def get_text(cls, path: str, **kwargs ) -> str:
        # Get the absolute path of the file
        path = cls.resolve_path(path)
        with open(path, 'r') as file:
            content = file.read()
        return content

    @classmethod
    def text(cls, path: str = './', **kwargs ) -> str:
        # Get the absolute path of the file
        path = cls.resolve_path(path)
        assert not c.home_path == path, f'Cannot read {path}'
        if os.path.isdir(path):
            return c.file2text(path)
        with open(path, 'r') as file:
            content = file.read()
        return content

    @staticmethod
    def sleep(period):
        time.sleep(period) 

    @classmethod
    def fn2code(cls, module=None)-> Dict[str, str]:
        module = cls.resolve_module(module)
        functions = c.fns(module)
        fn_code_map = {}
        for fn in functions:
            try:
                fn_code_map[fn] = c.code(getattr(module, fn))
            except Exception as e:
                c.print(f'Error {e} {fn}', color='red')
        return fn_code_map
    
    @classmethod
    def fn2hash(cls, module=None)-> Dict[str, str]:
        module = cls.resolve_module(module)   
        return {k:c.hash(v) for k,v in c.fn2code(module).items()}

    @classmethod
    def fn_code(cls,fn:str, module=None,**kwargs) -> str:
        '''
        Returns the code of a function
        '''
        fn = cls.get_fn(fn)      
        return inspect.getsource(fn)       
    
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
        obj = cls.resolve_module(obj)
        parents =  list(obj.__bases__)
        if recursive:
            for parent in parents:
                parent_parents = cls.get_parents(parent, recursive=recursive)
                if len(parent_parents) > 0:
                    for pp in parent_parents: 
                        if pp.__name__ not in avoid_classes:
                        
                            parents += [pp]
        return parents
    fn2cost = {}

    @classmethod
    def fn_schema(cls, fn:str = '__init__', **kwargs)->dict:
        '''
        Get function schema of function in cls
        '''     
        schema = {}

        fn = c.get_fn(fn)
        if not callable(fn):
            return {'fn_type': 'property', 'type': type(fn).__name__}
        inout_schema = {}
        for k,v in dict(inspect.signature(fn)._parameters).items():
            inout_schema[k] = {
                    'value': "_empty"  if v.default == inspect._empty else v.default, 
                    'type': '_empty' if v.default == inspect._empty else str(type(v.default)).split("'")[1] 
            }
        output_schema = {
            'value': None,
            'type': str(fn.__annotations__.get('return', None) if hasattr(fn, '__annotations__') else None)
        }
        schema['input'] = inout_schema
        schema['output'] = output_schema
        schema['docs'] = fn.__doc__
        schema['path'] = inspect.getfile(fn)
        schema['filebounds'] = inspect.getsourcelines(fn)
        schema['filebounds'] = {'start': schema['filebounds'][1], 'end': len(schema['filebounds'][0]) + schema['filebounds'][1], 'length': len(schema['filebounds'][0])}
        return schema


    @classmethod
    def schema(cls, module = None, **kwargs)->dict:
        '''
        Get function schema of function in cls
        '''     
        schema = {}
        if '/' in str(module):
            return c.fn_schema(module, **kwargs)
        module = c.resolve_module(module)
        fns = c.fns(module)
        for fn in fns:
            schema[fn] = c.fn_schema(getattr(module, fn))
        return schema

        
    @classmethod
    def resolve_obj(cls, obj = None, search=None, *args, **kwargs) -> Union[str, Dict[str, str]]:
        if obj == None:
            obj = cls
        if isinstance(obj, str) and '/' in obj:
            obj = c.get_fn(obj)
        else:
            obj = cls.resolve_module(obj)
        return obj

    @classmethod
    def code(cls, obj = None, search=None, *args, **kwargs) -> Union[str, Dict[str, str]]:
        return inspect.getsource(c.resolve_obj(obj))
    @classmethod
    def codemap(cls, module = None , search=None, *args, **kwargs) ->  Dict[str, str]:
        module = module or cls.module_name()
        dirpath = c.dirpath(module)
        path = dirpath if c.is_module_folder(module) else c.filepath(module)
        code_map = c.file2text(path)
        code_map = {k[len(dirpath+'/'): ]:v for k,v in code_map.items()}
        return code_map

    @classmethod
    def code_map(cls, module , search=None, *args, **kwargs) ->  Dict[str, str]:
        return c.codemap(module=module, search=search,**kwargs)

    @classmethod
    def codehash(cls, module , search=None, *args, **kwargs) -> Union[str, Dict[str, str]]:
        return c.hash(c.code_map(module=module, search=search,**kwargs))

    code_hash = codehash
    @classmethod
    def getsource(cls, module = None, search=None, *args, **kwargs) -> Union[str, Dict[str, str]]:
        if module != None:
            if isinstance(module, str) and '/' in module:
                fn = module.split('/')[-1]
                module = '/'.join(module.split('/')[:-1])
                module = getattr(c.module(module), fn)
            else:
                module = cls.resolve_module(module)
        else: 
            module = cls
        return inspect.getsource(module)

    @classmethod
    def module_hash(cls, module=None,  *args, **kwargs):
        return c.hash(c.code(module or cls.module_name(), **kwargs))
    @classmethod
    def code_hash(cls, module=None,  *args, **kwargs):
        return c.hash(c.code(module or cls.module_name(), **kwargs))

    @classmethod
    def get_params(cls, fn):
        """
        Gets the function defaults
        """
        fn = cls.get_fn(fn)
        params = dict(inspect.signature(fn)._parameters)
        for k,v in params.items():
            if v._default != inspect._empty and  v._default != None:
                params[k] = v._default
            else:
                params[k] = None
        return params
    
    @classmethod
    def class_functions(cls: Union[str, type], obj=None):
        '''
        Gets the self methods in a class
        '''
        obj = cls.resolve_module(obj)
        functions =  c.fns(obj)
        signature_map = {f:c.get_args(getattr(obj, f)) for f in functions}
        return [k for k, v in signature_map.items() if 'cls' in v]

    @classmethod
    def static_functions(cls: Union[str, type], obj=None):
        '''
        Gets the self methods in a class
        '''
        obj = obj or cls
        functions =  c.fns(obj)
        signature_map = {f:c.get_args(getattr(obj, f)) for f in functions}
        return [k for k, v in signature_map.items() if not ('self' in v or 'cls' in v)]
    
    static_methods = static_fns =  static_functions

    @classmethod
    def property_fns(cls) -> bool:
        '''
        Get a list of property functions in a class
        '''
        return [fn for fn in dir(cls) if cls.is_property(fn)]
    
    parents = get_parents


    def dir(self, obj=None, search=None, *args, **kwargs):
        obj = c.resolve_obj(obj)
        if search != None:
            return [f for f in dir(obj) if search in f]
        return dir(obj)
    
    @classmethod
    def fns(cls, 
                      obj: Any = None,
                      search = None,
                      splitter_options = ["   def " , "    def "] ,
                      include_hidden = False,
                      include_children = False,
                      **kwargs) -> List[str]:
        '''
        Get a list of functions in a class (in text parsing)
        Args;
            obj: the class to get the functions from
            include_parents: whether to include the parent functions
            include_hidden:  whether to include hidden functions (starts and begins with "__")
        '''
        obj = cls.resolve_module(obj)
        text = inspect.getsource(obj)
        functions = []
        # just
        for splitter in splitter_options:
            for line in text.split('\n'):
                if f'"{splitter}"' in line: # removing edge case
                    continue
                if line.startswith(splitter):
                    functions += [line.split(splitter)[1].split('(')[0].strip()]
        functions = sorted(list(set(functions)))
        if search != None:
            functions = [f for f in functions if search in f]
        if not include_hidden: 
            functions = [f for f in functions if not f.startswith('__') and not f.startswith('_')]
        return functions

    @classmethod
    def resolve_name(cls, module):
        module = c.shortcuts.get(module, module)
        if isinstance(module, str):
            return module
        return module.__name__

    @classmethod
    def clear_info_history(cls):
        return c.rm('info')

    @classmethod
    def resolve_info_path(self, name):
        if not isinstance(name, str):
            name = str(name)
        return c.resolve_path('info/' + name)



    @classmethod 
    def info(cls, module:str='module',  # fam
            lite: bool =True, 
            max_age : Optional[int]=None, 
            lite_features : List[str] = ['schema', 'name', 'key', 'founder', 'hash', 'time'],
            keep_last_n : int = 10,
            relative=True,
            update: bool =False, 
            **kwargs):
            
        path = c.resolve_info_path(module)
        info = c.get(path, None, max_age=max_age, update=update)
        if info == None:
            code = c.code_map(module)
            schema = c.schema(module)
            founder = c.founder().address
            key = c.get_key(module).address
            info =  {
                    'code': code, 
                    'schema': schema, 
                    'name': module, 
                    'key': key,  
                    'founder': founder, 
                    'hash': c.hash(code), 
                    'time': time.time()
                    }
            c.put(path, info)
        if lite:
            info = {k: v for k,v in info.items() if k in lite_features}
        return  info

    def get_tags(self,module=None, search=None, **kwargs):
        tags = []
        module =  c.resolve_module(module)
        if hasattr(module, 'tags'):
            tags = module.tags
        assert isinstance(tags, list), f'{module} does not have tags'
        if search != None:
            tags = [t for t in tags if search in t]
        return tags


    module2info_path = 'info/module2info'
    module2error_path = 'info/module2error'
    def module2info(self, search=None, max_age = 1000, update=False):
        module2info = {}
        path = c.resolve_path(self.module2info_path)
        error_path = c.resolve_path(self.module2error_path)
        module2error = c.get(error_path, {})
        module2info = c.get(path, module2info, max_age=max_age, update=update)
        if len(module2info) == 0:
            for m in c.modules():
                try:
                    module2info[m] = c.info(m)
                except Exception as e:
                    module2error[m] = c.detailed_error(e)
                    pass
            c.put(path, module2info)
            c.put(error_path, module2error)
        return module2info

    def module2error(self ):
        return  c.get(self.module2error_path, {})

    def module2hash(self, search = None, max_age = None, **kwargs):
        infos = self.infos(search=search, max_age=max_age, **kwargs)
        return {i['name']: i['hash'] for i in infos if 'name' in i}

    @classmethod
    def is_property(cls, fn: 'Callable') -> bool:
        '''
        is the function a property
        '''
        fn = c.get_fn(fn)
        return isinstance(fn, property)
    
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
            return False
        return callable(fn)


    @classmethod
    def get_fn(cls, fn:str, splitter='/', default_fn='forward') -> 'Callable':
        """
        Gets the function from a string or if its an attribute 
        """
        if isinstance(fn, str):
            if fn.startswith(splitter):
                return getattr(c.module()(), fn.split('/')[-1])
            elif fn.endswith(splitter):
                module = c.module(fn[:-1])
                return getattr(module, default_fn)
            fn_obj = None
            module = cls
            if splitter in fn:
                module_name= splitter.join(fn.split(splitter)[:-1])
                fn_name = fn.split(splitter)[-1]
                if c.module_exists(module_name):
                    module = c.get_module(module_name)
                    fn_obj =  getattr(module, fn_name)
            elif c.object_exists(fn):
                fn_obj =  c.obj(fn)
            args = c.get_args(fn_obj)
            if 'self' in args:
                fn_obj = getattr(module(), fn.split('/')[-1])

            
        else:
            fn_obj = fn
        # assert fn_obj != None, f'{fn} is not a function or object'
        return fn_obj

    @classmethod
    def classify_fns(cls, obj= None, mode=None):
        assert mode in ['cls', 'self']
        method_type_map = {}
        obj = cls.resolve_module(obj)
        for attr_name in dir(obj):
            method_type = None
            try:
                method_type = c.classify_fn(getattr(obj, attr_name))
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
        if not callable(fn):
            return []
        try:
            args = inspect.getfullargspec(fn).args
        except Exception as e:
            args = []
        return args
    
    @classmethod
    def classify_fn(cls, fn):
        if not callable(fn):
            fn = cls.get_fn(fn)
        if not callable(fn):
            return 'cls'
        args = c.get_args(fn)
        if len(args) == 0:
            return 'property'
        if args[0] == 'self':
            return 'self'
        elif args[0] == 'cls':
            return 'class'

        return 'static'

    @classmethod
    def class2fns(cls, path:str = './', tolist=False, **kwargs):
        path = os.path.abspath(path)
        class2fns = {}
        for path, classes in c.path2classes(path=path).items():
            try:
                for cl in classes:
                    class2fns[cl] = c.fns(cl)
            except Exception as e:
                pass
        if tolist:
            class2fns_list = []
            for k,v in class2fns.items():
                for class_name,fn_list in v.items():
                    class2fns_list += [f'{class_name}/{fn}' for fn in fn_list]
            return class2fns_list
           
        return class2fns


    def has_pwd_module(self):
        module_name = ['module', 'mod', 'agent', 'block']
        pwd = c.pwd()
        return os.path.exists(pwd + '/{}.py')

        
    @classmethod
    def classes(cls, path='./',  **kwargs):
        return  cls.path2classes(path=path, tolist=True, **kwargs)
      
    @classmethod
    def path2classes(cls, path='./',
                     class_prefix = 'class ', 
                     file_extension = '.py',
                     tolist = False,
                     depth=4,
                     relative=False,
                     class_suffix = ':', **kwargs) :
        path = os.path.abspath(path)
        path2classes = {}

        if os.path.isdir(path) and depth > 0:
            for p in c.ls(path):
                try:
                    for k,v in cls.path2classes(p, depth=depth-1).items():
                        if len(v) > 0:
                            path2classes[k] = v
                except Exception as e:
                    pass
                
        elif os.path.isfile(path) and path.endswith('.py'):
            code = cls.get_text(path)
            classes = []
            file_path = cls.path2objectpath(path)
            for line in code.split('\n'):
                if line.startswith(class_prefix) and line.strip().endswith(class_suffix):
                    new_class = line.split(class_prefix)[-1].split('(')[0].strip()
                    if new_class.endswith(class_suffix):
                        new_class = new_class[:-1]
                    if ' ' in new_class:
                        continue
                    classes += [new_class]
            if file_path.startswith(c.home_path):
                file_path = file_path[len(c.home_path)+1:]
            if '/' in file_path:
                file_path = file_path.replace('/', '.')
            if relative:
                path = self.path2relative(path)
            path2classes =  {path:  [file_path + '.' + c for c in classes]}


        if tolist: 
            classes = []
            for k,v in path2classes.items():
                classes.extend(v)
            return classes
   
        return path2classes


    @staticmethod
    def path2relative(path='./'):
        path = c.resolve_path(path)
        pwd = c.pwd()
        home_path = c.home_path
        prefixe2replacement = {pwd: './', home_path: '~/'}
        for pre, rep in prefixe2replacement.items():
            if path.startswith(pre):
                path = path[len(pre):]
                path = rep + path[len(pre):]
        return path
            
            

    
    @staticmethod
    def round(x, sig=6, small_value=1.0e-9):
        import math
        """
        rounds a number to a certain number of significant figures
        """
        return round(x, sig - int(math.floor(math.log10(max(abs(x), abs(small_value))))) - 1)

    @classmethod
    def path2objectpath(cls, path:str, **kwargs) -> str:
        path = os.path.abspath(path)
        dir_prefixes  = [c.lib_path , c.pwd()]
        for dir_prefix in dir_prefixes:
            if path.startswith(dir_prefix):
                path =   path[len(dir_prefix) + 1:].replace('/', '.')
                break
        if path.endswith('.py'):
            path = path[:-3]
        return path.replace('__init__.', '.')
        

    @classmethod
    def path2name(cls, path):
        name = cls.path2objectpath(path)
        name_chunks = []
        for chunk in name.split('.'):
            if chunk in ['modules', 'agents']:
                continue
            if chunk not in name_chunks:
                name_chunks += [chunk]
        if name_chunks[0] == c.repo_name:
            name_chunks = name_chunks[1:]
        return '.'.join(name_chunks)

    @classmethod
    def objectpath2path(cls, objectpath:str, **kwargs) -> str:
        options  = [c.lib_path, c.pwd()]
        for option in options:
            path = option + '/' + objectpath.replace('.', '/') + '.py'
            if os.path.exists(path):
                return path
        raise ValueError(f'Path not found for objectpath {objectpath}')

    @classmethod
    def path2fns(cls, path = './', tolist=False, **kwargs):
        fns = []
        path = os.path.abspath(path)
        if os.path.isdir(path):
            path2fns = {}
            for p in c.glob(path+'/**/**.py', recursive=True):
                for k,v in c.path2fns(p, tolist=False).items():
                    if len(v) > 0:
                        path2fns[k] = v
        else:
            code = c.get_text(path)
            path_prefix = c.path2objectpath(path)
            for line in code.split('\n'):
                if line.startswith('def ') or line.startswith('async def '):
                    fn = line.split('def ')[-1].split('(')[0].strip()
                    fns += [path_prefix + '.'+ fn]
            path2fns =  {path: fns}
        if tolist:
            fns = []
            for k,v in path2fns.items():
                fns += v
            return fns
        return path2fns

    @classmethod
    def objs(cls, path:str = './', depth=10, search=None, **kwargs):
        classes = c.classes(path,depth=depth)
        functions = c.path2fns(path, tolist=True)
        objs = functions + classes

        if search != None:
            objs = [f for f in objs if search in f]
        return objs

    @classmethod
    def objects(cls, path:str = './', depth=10, search=None, **kwargs):
        return c.objs(path=path, depth=depth, search=search, **kwargs)

    @classmethod
    def ensure_sys_path(cls, paths:List[str] = None):
        paths = paths or [ c.pwd()]
        if not hasattr(c, 'included_pwd_in_path'):
            c.included_pwd_in_path = False
        if  not c.included_pwd_in_path:
            for p in paths:
                if not p in sys.path:
                    sys.path = [p] + sys.path
            sys.path = list(set(sys.path))
            c.included_pwd_in_path = True
        return sys.path

    @classmethod
    def import_module(cls, import_path:str ) -> 'Object':
        from importlib import import_module
        c.ensure_sys_path()
        return import_module(import_path)

    @classmethod
    def is_python_module(cls, module):
        try:
            c.import_module(module)
            return True
        except Exception as e:
            return False

    @classmethod
    def obj(cls, key:str, splitters=['/', '::', '.'], **kwargs)-> Any:
        ''' Import an object from a string with the format of {module_path}.{object}'''
        module_path = None
        object_name = None
        for splitter in splitters:
            key = key.replace(splitter, '.')
        
        module_path = '.'.join(key.split('.')[:-1])
        object_name = key.split('.')[-1]
        
        if isinstance(key, str) and key.endswith('.py') and c.path_exists(key):
            key = c.path2objectpath(key)
        assert module_path != None and object_name != None, f'Invalid key {key}'
        module_obj = c.import_module(module_path)
        try:
            return  getattr(module_obj, object_name)
        except Exception as e:
            return c.import_module(key)
    
    @classmethod
    def import_object(cls, key:str, **kwargs)-> Any:
        return c.obj(key, **kwargs)

    @classmethod
    def object_exists(cls, path:str, verbose=False)-> Any:

        # better way to check if an object exists?

        try:
            c.obj(path, verbose=verbose)
            return True
        except Exception as e:
            return False

    @classmethod
    def module_exists(cls, module:str, **kwargs) -> bool:
        '''
        Returns true if the module exists
        '''
        try:
            tree = c.tree()
            module_exists =  module in tree
        except Exception as e:
            module_exists =  False
        return module_exists
    
    @classmethod
    def objectpath2name(cls, 
                        p:str,
                        avoid_terms=['modules', 'agents', 'module', '_modules', '_agents', ],
                        avoid_suffixes = ['module', 'mod']):
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
        if path.startswith(c.repo_name + '.'):
            path = path[len(c.repo_name)+1:]
        if path.endswith('.'):
            path = path[:-1]
        for avoid in avoid_terms:
            avoid = f'{avoid}.' 
            if avoid in path:
                path = path.replace(avoid, '')
        if len(path) == 0:
            return 'module'
        for avoid in avoid_suffixes:
            if path.endswith(avoid) and path != avoid:
                path = path[:-(1+len(avoid))]
        return path
    @classmethod
    def local_modules(cls, search=None, **kwargs):
        return list(c.local_tree(c.pwd(), search=search, **kwargs).keys())

    @classmethod
    def home_modules(cls, search=None, **kwargs):
        return list(c.get_tree(c.home_modules_path, search=search, **kwargs).keys())

    @classmethod
    def lib_tree(cls, depth=10, **kwargs):
        return c.get_tree(c.lib_path, depth=depth, **kwargs)
        
    @classmethod
    def core_tree(cls, **kwargs):
        tree =  c.get_tree(c.lib_path, **kwargs)
        return {k:v for k,v in tree.items() if 'modules.' not in v}
    @classmethod
    def local_tree(cls , depth=4, **kwargs):
        return c.get_tree(c.pwd(), depth=depth, **kwargs)
    
    @classmethod
    def get_tree(cls, path, depth = 10, max_age=60, update=False, **kwargs):
        tree_cache_path = 'tree/'+os.path.abspath(path).replace('/', '_')
        tree = c.get(tree_cache_path, None, max_age=max_age, update=update)
        if tree == None:
            class_paths = cls.classes(path, depth=depth)
            simple_paths = [cls.objectpath2name(p) for p in class_paths]
            tree = dict(zip(simple_paths, class_paths))
            c.put(tree_cache_path, tree)
        return tree

    def test(self, module=None):
        path = c.test_path
        if module != None:
            path = path + '/' + module  + '_test.py'
        assert os.path.exists(path), f'Path {path} does not exist'        
        return c.cmd(f"pytest {path}")
    
    _tree = None
    @classmethod
    def tree(cls, search=None,  max_age=60,update=False, **kwargs):
        local_tree = c.local_tree(update=update, max_age=max_age)
        lib_tree = c.lib_tree(update=update, max_age=max_age)
        modules_tree = c.modules_tree(update=update, max_age=max_age)
        # overlap the local tree over the lib tree
        tree = {**lib_tree, **local_tree}
        if search != None:
            tree = {k:v for k,v in tree.items() if search in k}
        return tree
        
    @classmethod
    def modules_tree(cls, search=None, **kwargs):
        return c.get_tree(c.modules_path, search=search, **kwargs)
    @classmethod
    def core_modules(cls, search=None, depth=10000, avoid_folder_terms = ['modules.'], **kwargs):
        object_paths = cls.classes(c.lib_path, depth=depth )
        object_paths = [cls.objectpath2name(p) for p in object_paths if all([avoid not in p for avoid in avoid_folder_terms])]
        if search != None:
            object_paths = [p for p in object_paths if search in p]
        return sorted(list(set(object_paths)))

    def module2size(self, search=None, depth=10000, **kwargs):
        module2size = {}
        module2code = c.module2code(search=search, depth=depth, **kwargs)
        for k,v in module2code.items():
            module2size[k] = len(v)
        module2size = dict(sorted(module2size.items(), key=lambda x: x[1], reverse=True))
        return module2size

    @classmethod
    def get_modules(cls, search=None, **kwargs):
        return list(cls.tree(search=search, **kwargs).keys())

    @classmethod
    def modules(cls, search=None, cache=True, max_age=60, update=False, **extra_kwargs)-> List[str]:
        modules = c.get('modules', max_age=max_age, update=update)
        if not cache or modules == None:
            modules =  cls.get_modules(search=None, **extra_kwargs)
            c.put('modules', modules)
        if search != None:
            modules = [m for m in modules if search in m]     
        return modules
    blocks = mods = modules

    @classmethod
    def check_info(cls,info, features=['key', 'hash', 'time', 'founder', 'name', 'schema']):
        try:
            assert isinstance(info, dict), 'info is not a dictionary'
            for feature in features:
                assert feature in info, f'{feature} not in info'
        except Exception as e:
            return False
        return True

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
    def filter(cls, text_list: List[str], filter_text: str) -> List[str]:
        return [text for text in text_list if filter_text in text]

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
    def util(cls, util:str, prefix='commune.utils'):
        return c.obj(c.util2path().get(util))

    @classmethod
    def run_util(cls, util:str, *args, **kwargs):
        return c.util(util)(*args, **kwargs)

    @classmethod
    def root_key(cls):
        return cls.get_key()

    @classmethod
    def founder(cls):
        return c.get_key()

    @classmethod
    def repo2path(cls, search=None):
        repo2path = {}
        for p in c.ls('~/'): 
            if os.path.exists(p+'/.git'):
                r = p.split('/')[-1]
                if search == None or search in r:
                    repo2path[r] = p
        return dict(sorted(repo2path.items(), key=lambda x: x[0]))
    
    def repos(self, search=None):
        return list(self.repo2path(search=search).keys())

    @classmethod
    def date(cls):
        import datetime
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @classmethod
    def datetime(cls):
        import datetime
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @classmethod
    def ask(cls, *args, **kwargs):
        return c.module("agent")().ask(*args, **kwargs) 

    def clone(self, repo:str = 'commune-ai/commune', path:str=None, **kwargs):
        gitprefix = 'https://github.com/'
        if not repo.startswith(gitprefix):
            repo = gitprefix + repo
        path = os.path.abspath(os.path.expanduser(path or  '~/'+repo.split('/')[-1]))
        cmd =  f'git clone {repo} {path}'
        c.cmd(cmd, verbose=True)
        return c.cmd(f'code {path}')

    def has_module(self, path:str):
        for path in c.files(path): 
            if path.endswith('.py'):
                return True  

    def help(self, module:str, search=None):
        module = c.module(module)
        return module.help(search=search)
                
    @classmethod
    def module2fns(cls, path=None):
        tree = c.get_tree(path or cls.lib_path)
        module2fns = {}
        for m,m_path in tree.items():
            try:
                module2fns[m] = c.module(m).fns()
            except Exception as e:
                pass
        return module2fns

    @classmethod
    def module2schema(cls, path=None):
        tree = c.get_tree(path or cls.lib_path)
        module2fns = {}
        for m,m_path in tree.items():
            try:
                module2fns[m] = c.schema(m)
            except Exception as e:
                print(e)
                pass
        return module2fns

    def module2code(self, search=None, update=False, max_age=60, **kwargs):
        module2code = {}
        module2code = c.get('module2code', None, max_age=max_age, update=update)
        if module2code != None:
            return module2code
        module2code = {}
        for m in c.modules(search=search, **kwargs):
            try:
                module2code[m] = c.code(m)
            except Exception as e:
                pass
        c.put('module2code', module2code)
        return module2code
    
    @classmethod
    def fn2module(cls, path=None):
        module2fns = cls.module2fns(path)
        fn2module = {}
        for m in module2fns:
            for f in moducle2fns[m]:
                fn2module[f] = m
        return fn2module

    def routes_from_to(self):
        routes = c.routes
        from_to_map = {}
        for m, fns in routes.items():
            for fn in fns:
                route = m + '/' + fn
                assert  fn not in from_to_map, f'Function {route}  already exists in {from_to_map[fn]}'
                from_to_map[fn] = m + '/' + fn
        return from_to_map

    @staticmethod
    def readmes( path='./', search=None):
        files =  c.files(path)
        readmes = [f for f in files if f.endswith('.md')]
        return readmes

    @classmethod
    def readme2text(path='./', search=None):
        readmes = c.readmes(path=path, search=search)
        return {r:c.get_text(r) for r in readmes}

    def app(self,
           module:str = 'agent', 
           name : Optional[str] = None,
           port:int=None):
        module = c.shortcuts.get(module, module)
        name = name or module
        port = port or c.free_port()
        if c.module_exists(module + '.app'):
            module = module + '.app'
        module_class = c.module(module)
        return c.cmd(f'streamlit run {module_class.filepath()} --server.port {port}')

    def code_url(self, path:str='./') -> str:
        path = self.resolve_path(path or self.lib_path)
        return c.cmd(f'git remote get-url origin', cwd=path)

    @classmethod
    def getsourcelines(cls, module = None, search=None, *args, **kwargs) -> Union[str, Dict[str, str]]:
        if module != None:
            if isinstance(module, str) and '/' in module:
                fn = module.split('/')[-1]
                module = '/'.join(module.split('/')[:-1])
                module = getattr(c.module(module), fn)
            else:
                module = cls.resolve_module(module)
        else: 
            module = cls
        return inspect.getsourcelines(module)

    def getdocs(self, module:str = 'module' , search=None, **kwargs):
        code = c.code(module)
        return c.ask(code + 'make a document thats better', search=search, process_text=True, **kwargs)

    @classmethod
    def filebounds(cls, obj=None) -> List[int]:
        """
        Gets the starting and ending line numbers of an object in its source code.
        
        Args:
            obj: The object to get bounds for. Defaults to cls if None.
            
        Returns:
            Tuple of (start_line, end_line) line numbers
        """
        import inspect
        
        obj = c.resolve_obj(obj)
            
        try:
            # Get source lines and line number
            source_lines, start_line = c.getsourcelines(obj)
            end_line = start_line + len(source_lines) - 1
            return [start_line, end_line]
        except Exception as e:
            print(e, 'Error getting object bounds')
            return [None, None]

    def test_fn(self, fn:str, *args, **kwargs):
        fn = c.get_fn(fn)
        return fn(*args, **kwargs)
        

    def _rm_fn(self, fn:str = 'test_fn' , module=None):
        """
        rm the function from the module from the code
        you can specify it either as a function or a string
        """
        if '/' in fn:
            fn = fn.split('/')[-1]
            module = '/'.join(fn.split('/')[:-1])
        else:
            module = 'module'
        filepath  = c.filepath(module)
        text = c.get_text(filepath)
        filebounds = c.filebounds(module + '/' + fn)
        start_line, end_line = filebounds

        text = '\n'.join(text.split('\n')[:start_line-1] + text.split('\n')[end_line:])
        c.put_text(filepath, text)
        return 

    def _add_fn(self, fn:str = 'test_fn' , module=None):
        """
        rm the function from the module from the code
        you can specify it either as a function or a string
        """
        if '/' in fn:
            fn = fn.split('/')[-1]
            module = '/'.join(fn.split('/')[:-1])
        else:
            module = 'module'
        filepath  = c.filepath(module)
        text = c.get_text(filepath)
        filebounds = c.filebounds(module + '/' + fn)
        start_line, end_line = filebounds

        text = '\n'.join(text.split('\n')[:start_line-1] + text.split('\n')[end_line:])
        c.put_text(filepath, text)

c.add_routes()
Module = c # Module is alias of c
if __name__ == "__main__":
    Module.run()


