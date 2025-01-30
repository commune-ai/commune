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

class c:


    org = 'commune-ai' # the organization
    reponame  = lib = __file__.split('/')[-2]# the name of the library
    cost = 1 
    description = """This is a module"""
    base_module = 'module' # the base module
    giturl = f'https://github.com/{org}/{reponame}.git' # tge gutg
    default_fn = 'forward' # default function
    free = False # if the server is free 
    endpoints = ['ask', 'generate', 'forward']
    core_features = ['module_name', 'module_class',  'filepath', 'dirpath']
    rootpath = root_path  = root  = '/'.join(__file__.split('/')[:-1]) 
    libpath = lib_path = os.path.dirname(root_path) # the path to the library
    testpath = test_path = libpath + '/tests'
    default_port_range = [50050, 50150] # the port range between 50050 and 50150
    default_ip = local_ip = loopback = '0.0.0.0'   
    home_path  = os.path.expanduser('~') # the home path
    repo_path  = os.path.dirname(root_path) # the path to the repo
    docs_path = reponame + '/docs'
    storage_path = os.path.expanduser(f'~/.{reponame}')
    modules_path = os.path.dirname(__file__) + '/modules'

    cache = {} # cache for module objects
    shortcuts =  {
        'openai' : 'model.openai',
        'openrouter':  'model.openrouter',
        'or' : 'model.openrouter',
        'r' :  'remote',
        's' :  'subspace',
        'subspace': 'subspace', 
        'namespace': 'network', 
        "client": 'server.client',
        'network': 'server.network',
        'local': 'server.network',
        }
    splitters = [':', '/', '.']
    route_cache = None
    _obj = None


    def giturl(self, path:str='./') -> str:
        path = self.resolve_path(path or self.libpath)
        return c.cmd(f'git remote get-url origin', cwd=path)
    @classmethod
    def module(cls, 
               path:str = 'module', 
               params : dict = None, 
               cache=True, 
               trials=1, 
               verbose=False,
               tree:dict=None ) -> str:
        path = path or 'module'
        if path in ['module', c.reponame[0]]:
            return c
        t0 = c.time()
        path = path.replace('/','.')
        path = c.shortcuts.get(path, path)
        tree = tree or c.tree()
        simp_path = path
        path = tree.get(path, path)
        obj_path = path
        if (obj_path in c.module_cache):
            module = c.module_cache[obj_path]
        else:
            module = c.import_object(obj_path)
            module = c.obj2module(module) # if the model
            c.module_cache[obj_path] = module
        if params != None:
            module = module(**params)
        loadtime = c.time() - t0
        c.print(f'Module(simp_path={obj_path}, obj_path={obj_path}, t={loadtime:.2f}s') if verbose else ''
        return module
    
    get_agent = block =  get_block = get_module =  mod =  module
    
    @classmethod
    def obj2module(cls, obj:'Object', verbose=False):
        module = obj
        if c.is_object_module(obj):
            return module
        c.print(f'ConvertingObj2Module({obj})', verbose=verbose)
        module.module_name = module.name = lambda *args, **kwargs : c.module_name(module)
        module.resolve_module = lambda *args, **kwargs : c.resolve_module(module)
        module.key = c.get_key(module.module_name(), create_if_not_exists=True)
        module.storage_dir = lambda *args, **kwargs : c.storage_dir(module)
        module.filepath = lambda *args, **kwargs : c.filepath(module)
        module.dirpath = lambda *args, **kwargs : c.dirpath(module)
        module.code = lambda *args, **kwargs : c.code(module)
        module.code_hash = lambda *args, **kwargs : c.code_hash(module)
        module.schema = lambda *args, **kwargs : c.schema(module)
        module.functions = module.fns = lambda *args, **kwargs : c.fns(module)
        module.fn2code = lambda *args, **kwargs : c.fn2code(module)
        module.fn2hash = lambda *args, **kwargs : c.fn2hash(module)
        module.config = lambda *args, **kwargs : c.config(module)
        if not hasattr(module, 'ask'):
            def ask(*args, **kwargs):
                args = [module.code()] + list(args)
                return c.ask(*args, **kwargs)
            module.ask = ask
        return module

    @classmethod
    def filepath(cls, obj=None) -> str:
        obj = cls.resolve_module(obj)
        return inspect.getfile(obj)
    
    @classmethod
    def objectpath(cls, obj=None) -> str:
        return c.classes(cls.filepath(obj))[-1]

    def file2size(self, path:str='./', reverse=True) -> int:
        file2size =  {k:len(str(v)) for k,v in c.file2text(path).items()}
        file2size = dict(sorted(file2size.items(), key=lambda x: x[1], reverse=reverse))
        return file2size
    
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
        path = path or c.libpath
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
        obj = obj or cls
        return obj.filepath()[:-3] + '.yaml'

    @classmethod
    def sandbox(cls, path='./', filename='sandbox.py'):
        for file in  c.files(path):
            if file.endswith(filename):
                return c.cmd(f'python3 {file}', verbose=True)
        return {'success': False, 'message': 'sandbox not found'}
    
    sand = sandbox

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

    def print( *text:str,  **kwargs):
        return c.obj('commune.utils.log.print_console')(*text, **kwargs)

    def is_error( *text:str,  **kwargs):
        return c.obj('commune.utils.log.is_error')(*text, **kwargs)
    @classmethod
    def resolve_module(cls, obj:str = None, default=None, fn_splitter='/', **kwargs):
        obj = obj or cls._obj or default or cls
        if isinstance(obj, str):
            if fn_splitter in obj:
                fn = obj.split(fn_splitter)[-1]
                obj = fn_splitter.join(obj.split(fn_splitter)[:-1])
            else:
                fn = None
            obj =  c.module(obj)
            if fn != None:
                return getattr(obj(), fn)
        assert obj != None, f'Object {obj} does not exist'
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
    def commit_hash(cls, libpath:str = None):
        if libpath == None:
            libpath = c.libpath
        return c.cmd('git rev-parse HEAD', cwd=libpath, verbose=False).split('\n')[0].strip()

    @classmethod
    def run_fn(cls,fn:str, args:list = None, kwargs:dict= None, module:str = None) -> Any:
        if '/' in fn:
            module, fn = fn.split('/')
        module = c.module(module)
        fn_obj = getattr(module, fn)
        is_self_method = 'self' in c.get_args(fn_obj)
        if is_self_method:
            module = module()
        fn_obj =  getattr(module, fn)
        args = args or []
        kwargs = kwargs or {}
        return fn_obj(*args, **kwargs)
    
    # UNDER CONSTRUCTION (USE WITH CAUTION)
    
    def forward(self, *args, **kwargs):
        return c.ask(*args, **kwargs)
    
    tests_path = f'{libpath}/tests'
        # return c.cmd(f'pytest {c.tests_path}',  stream=1, *args, **kwargs)
    
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


    @classmethod
    def is_module_folder(cls,  module = None) -> bool:
        if module != None:
            cls = c.module(module)
        return not cls.is_file_module()
    
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
              path=None, 
              search:str = None, 
              avoid_terms = ['__pycache__', '.git', '.ipynb_checkpoints', 'node_modules', 'artifacts', 'egg-info'], 
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
        if cls.module_name == 'module':
            path = path or './'
        else:
            path = path or cls.storage_dir()
        files =c.glob(path, **kwargs)
        files = [f for f in files if not any([at in f for at in avoid_terms])]
        if search != None:
            files = [f for f in files if search in files]
        return files

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
        return '0x'+c.get_key(key).sign(data, **kwargs).hex()
    
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

    # local update  
    @classmethod
    def update(cls,  ):
        c.namespace(update=True)
        c.ip(update=1)
        return {'ip': c.ip(), 'namespace': c.namespace()}

    @classmethod
    def utils(cls, search=None):
        utils = c.path2functions(c.rootpath + '/utils')
        if search != None:
            utils = [u for u in utils if search in u]
        return sorted(utils)

    @classmethod
    def util2code(cls, search=None):
        utils = cls.utils()
        util2code = {}
        for f in utils:
            util2code[f] = c.code(f)
        return util2code

    @classmethod
    def util2hash(cls, search=None):
        return {k:c.hash(v) for k,v in c.util2code(search=search).items()}

    @classmethod
    def get_utils(cls, search=None):
        utils = c.path2functions(c.rootpath + '/utils')
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
                fn = c.import_object(fn)
                return fn(*args, **kwargs)
            except : 
                fn = fn.split('.')[-1]
                return getattr(c, fn)(*args, **kwargs)
        for k, fn in utils.items():
            setattr(obj, k, partial(wrapper_fn2, fn))
        return {'success': True, 'message': 'added utils'}

    @staticmethod
    def get_yaml( path:str=None, default={}, **kwargs) -> Dict:
        from .utils.os import get_yaml
        return get_yaml(path=path, default=default, **kwargs)
    
    @classmethod
    def get_routes(cls):
        if not hasattr(cls, 'routes'):
            routes_path = os.path.dirname(__file__)+ '/routes.json'
            routes =  cls.get_yaml(routes_path)
        else:
            routes = getattr(cls, 'routes')
            if callable(routes):
                routes = routes()
        for util in  c.utils():
            k = '.'.join(util.split('.')[:-1])
            v = util.split('.')[-1]
            routes[k] = routes.get(k , [])
            routes[k].append(v)
        for k , v in routes.items():
            routes[k] = list(set(v))
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
        return fn2route

    @classmethod
    def add_routes(cls, routes:dict=None):

        cls.routes = c.get_json(__file__.replace(__file__.split('/')[-1], 'routes.json'))

        """
        This ties other modules into the current module.
        The way it works is that it takes the module name and the function name and creates a partial function that is bound to the module.
        This allows you to call the function as if it were a method of the current module.
        for example
        """
        routes = routes or cls.get_routes()
        t0 = time.time()
        # WARNING : THE PLACE HOLDERS MUST NOT INTERFERE WITH THE KWARGS OTHERWISE IT WILL CAUSE A BUG IF THE KWARGS ARE THE SAME AS THE PLACEHOLDERS
        # THE PLACEHOLDERS ARE NAMED AS module_ph and fn_ph AND WILL UNLIKELY INTERFERE WITH THE KWARGS
        def fn_generator(*args, route, **kwargs):
            def fn(*args, **kwargs):
                try:
                    fn_obj = c.import_object(route)
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
                if not hasattr(cls, fn):
                    fn_obj = partial(fn_generator, route=module + '/' + fn) 
                    fn_obj.__name__ = fn
                    setattr(cls, fn, fn_obj)
        latency = time.time() - t0
        return {'success': True, 'msg': 'enabled routes', 'latency': latency}
    
    @classmethod
    def is_class(cls, obj):
        if isinstance(obj, str):
            try:
                obj = c.obj(obj)
            except Exception as e:
                return False
        return inspect.isclass(obj)



    @classmethod
    def add_to_globals(cls, globals_input:dict = None):
        from functools import partial
        globals_input = globals_input or {}
        for k,v in c.__dict__.items():
            globals_input[k] = v     
        for f in c.class_functions(c) + c.static_functions(c):
            globals_input[f] = getattr(c, f)
        for f in c.self_functions(c):
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

    @classmethod
    def config(cls, module=None, to_munch=True, fn='__init__') -> 'Munch':
        module = module or cls
        path = module.config_path()
        if os.path.exists(path):
            config = c.load_yaml(path)
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
    def get_json(cls, path:str,default:Any=None, **kwargs):
        path = cls.resolve_path(path)
        if not os.path.exists(path):
            if not path.endswith('.json'):
                path = path + '.json'

        try:
            with open(path, 'r') as file:
                data = json.load(file)
        except Exception as e:
            return default
        return data
    
    load_json = get_json
    
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
    def abspath(cls, path:str):
        return os.path.abspath(os.path.expanduser(path))
    
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
        cls.put_json(k, data)
        return {'k': k, 'data_size': cls.sizeof(v), 'encrypted': encrypt, 'timestamp': time.time()}
    
    @classmethod
    def get(cls,
            k:str, 
            default: Any=None, 
            max_age:str = None,
            full :bool = False, 
            update :bool = False,
            password : str = None,
            time_features = ['timestamp', 'time'],
            verbose = False,
            **kwargs) -> Any:
        
        '''
        Puts a value in sthe config, with the option to encrypt it
        Return the value
        '''
        data = cls.get_json(k,default=default, **kwargs)

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
            for k in time_features:
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
        return data
    
    def get_age(self, k:str) -> int:
        data = self.get_json(k)
        timestamp = data.get('timestamp', None)
        if timestamp != None:
            age = int(time.time() - timestamp)
            return age
        return -1
    
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
    
    @classmethod
    def module_schema(cls, module = None):
        module = cls.resolve_module(module)
        module_schema = {}
        for fn in c.functions(module):
            schema = c.schema(getattr(module, fn))
            module_schema[fn] = schema
        return module_schema

    fn2cost = {}

    @classmethod
    def schema(cls, fn:str = '__init__', **kwargs)->dict:
        '''
        Get function schema of function in cls
        '''     
        schema = {}

        fn = cls.get_fn(fn)
        if not callable(fn):
            return {'fn_type': 'property', 'type': type(fn).__name__}

        for k,v in dict(inspect.signature(fn)._parameters).items():
            schema[k] = {
                    'default': "_empty"  if v.default == inspect._empty else v.default, 
                    'type': str(type(v.default)).split("'")[1]  if v.default == inspect._empty and v.default != None else v.annotation.__name__
            }
        return schema

    @classmethod
    def code(cls, module = None, search=None, *args, **kwargs):
        if module != None:
            util2path = cls.util2path()
            if module in util2path:
                module = util2path[module]
        obj = cls.resolve_module(module)
        
        return inspect.getsource(obj)

    pycode = code
    @classmethod
    def module_hash(cls, module=None,  *args, **kwargs):
        return c.hash(c.code(module or cls.module_name(), **kwargs))
    @classmethod
    def code_hash(cls, module=None,  *args, **kwargs):
        return c.hash(c.code(module or cls.module_name(), **kwargs))

    @classmethod
    def dir(cls, module=None, search=None, **kwargs):
        module = cls.resolve_module(module)
        if search != None:
            return [f for f in dir(module) if search in f]
            
        return dir(module)

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
        functions =  c.functions(obj)
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
    def functions(cls, obj=None, search = None, include_parents = True):
        obj = cls.resolve_module(obj)
        return c.fns(obj=obj, search=search, include_parents=include_parents)
 
    def n_fns(self, search = None):
        return len(self.fns(search=search))
    
    fn_n = n_fns
    @classmethod
    def is_property(cls, fn: 'Callable') -> bool:
        '''
        is the function a property
        '''
        fn = c.get_fn(fn)
        return isinstance(fn, property)
    
    @classmethod
    def path_exists(cls, path:str):
        return os.path.exists(path)
   
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
    def get_fn(cls, fn:str, splitters=[":", "/"]) -> 'Callable':
        """
        Gets the function from a string or if its an attribute 
        """
        if isinstance(fn, str):
            fn_obj = None
            module = cls
            for splitter in splitters:
                if splitter in fn:
                    module_name= splitter.join(fn.split(splitter)[:-1])
                    fn_name = fn.split(splitter)[-1]
                    if c.module_exists(module_name):
                        module = c.get_module(module_name)
                        fn_obj =  getattr(module, fn_name)
            if hasattr(cls, fn):
                fn2route = cls.fn2route() 
                if fn in fn2route:
                    return c.obj(fn2route[fn])
                fn_obj =  getattr(cls, fn)
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
    def self_functions(cls, obj=None, search = None):
        obj = obj or cls
        fns =  c.classify_fns(obj)['self']
        if search != None:
            fns = [f for f in fns if search in f]
        return fns

    @classmethod
    def classify_fns(cls, obj= None, mode=None):
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
    def name2path(cls, 
                    name:str,
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
        name = c.shortcuts.get(name, nsme)
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
                p = p if p.endswith(extension) else p + extension
                if os.path.exists(p):
                    p_text = c.get_text(p)
                    path =  p
                    if c.reponame in p_text and 'class ' in p_text or '  def ' in p_text:
                        break
            if path != None:
                break
        return path

    @classmethod
    def path2name(cls,  
                    path:str, 
                    ignore_prefixes = ['src', 
                                       'commune', 
                                       'modules', 
                                       'commune.modules',
                                       'module'],
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
    def classes(cls, path='./', depth=8, 
                     class_prefix = 'class ', 
                     file_extension = '.py',
                     class_suffix = ':', **kwargs):
        path = os.path.abspath(path)
        if os.path.isdir(path):
            classes = []
            if depth == 0:
                return []
            for p in c.ls(path):
                if os.path.isdir(p):
                    classes += cls.classes(p, depth=depth-1)
                elif p.endswith(file_extension):
                    p_classes =  cls.classes(p)
                    classes += p_classes
            return classes
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
        classes = [file_path + '.' + c for c in classes]
        return classes
    

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
    def path2functions(cls, path = './', **kwargs):
        fns = []
        if os.path.isdir(path):
            path = os.path.abspath(path)
            for p in c.glob(path+'/**/**.py', recursive=True):
                p_fns = c.path2functions(p)
                file_object_path = c.path2objectpath(p)
                p_fns = [file_object_path + '.' + f for f in p_fns]
                for fn in p_fns:
                    fns += [fn]
        else:
            code = c.get_text(path)
            for line in code.split('\n'):
                if line.startswith('def ') or line.startswith('async def '):
                    fn = line.split('def ')[-1].split('(')[0].strip()
                    fns += [fn]
        return fns
    
    @classmethod
    def objects(cls, path:str = './', depth=10, search=None, **kwargs):
        classes = c.classes(path,depth=depth)
        functions = c.path2functions(path)
        if search != None:
            functions = [f for f in functions if search in f]
        object_paths = functions + classes
        return object_paths

    objs = objects
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
    def get_object(cls, key:str, splitters=['/', '::',  '.'], **kwargs)-> Any:
        ''' Import an object from a string with the format of {module_path}.{object}'''
        module_path = None
        object_name = None
        for splitter in splitters:
            if splitter in key:
                module_path = '.'.join(key.split(splitter)[:-1])
                object_name = key.split(splitter)[-1]
                break
        if isinstance(key, str) and key.endswith('.py') and c.path_exists(key):
            key = c.path2objectpath(key)
        assert module_path != None and object_name != None, f'Invalid key {key}'
        module_obj = c.import_module(module_path)
        return  getattr(module_obj, object_name)
    
    @classmethod
    def obj(cls, key:str, **kwargs)-> Any:
        return c.get_object(key, **kwargs)
    
    @classmethod
    def import_object(cls, key:str, **kwargs)-> Any:
        return c.get_object(key, **kwargs)

    @classmethod
    def object_exists(cls, path:str, verbose=False)-> Any:

        # better way to check if an object exists?

        try:
            c.import_object(path, verbose=verbose)
            return True
        except Exception as e:
            return False
    

    @classmethod
    def module_exists(cls, module:str, **kwargs) -> bool:
        '''
        Returns true if the module exists
        '''
        try:
            module = c.shortcuts.get(module, module)
            module_exists = os.path.exists(c.name2path(module))
            if not module_exists:
                module_exists = bool(c.object_exists(path))
        except Exception as e:
            module_exists =  False
        return module_exists
    
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
        
        if path.startswith(c.lib + '.'):
            path = path[len(c.lib)+1:]

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
        for avoid_suffix in ['module']:
            if path.endswith('.' + avoid_suffix):
                path = path[:-len(avoid_suffix)-1]
        return path

    @classmethod
    def local_modules(cls, search=None, **kwargs):
        return list(c.local_tree(search=search, **kwargs).keys())
    
    @classmethod
    def lib_tree(cls, depth=10, **kwargs):
        return c.get_tree(c.libpath, depth=depth, **kwargs)
        
    @classmethod
    def core_tree(cls, **kwargs):
        tree =  c.get_tree(c.libpath, **kwargs)
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
        object_paths = cls.classes(cls.libpath, depth=depth )
        object_paths = [cls.objectpath2name(p) for p in object_paths if all([avoid not in p for avoid in avoid_folder_terms])]
        if search != None:
            object_paths = [p for p in object_paths if search in p]
        return sorted(list(set(object_paths)))

    @classmethod
    def core_code(cls, search=None, depth=10000, **kwargs):
        return {k:c.code(k) for k in cls.core_modules(search=search, depth=depth, **kwargs)}

    def core_size(self, search=None, depth=10000, **kwargs):
        return {k:len(c.code(k)) for k in self.core_modules(search=search, depth=depth, **kwargs)}
    
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
    
    def n(self, search=None):
        return len(c.modules(search=search))

    @classmethod
    def modules(cls, search=None, cache=True, max_age=60, update=False, **extra_kwargs)-> List[str]:
        modules = cls.get('modules', max_age=max_age, update=update)
        if not cache or modules == None:
            modules =  cls.get_modules(search=None, **extra_kwargs)
            cls.put('modules', modules)
        if search != None:
            modules = [m for m in modules if search in m]     
        return modules
    blocks = mods = modules

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

    def build(self, *args, **kwargs):
        return c.module('builder')().build(*args, **kwargs)
    
    @classmethod
    def filter(cls, text_list: List[str], filter_text: str) -> List[str]:
        return [text for text in text_list if filter_text in text]

    @staticmethod
    def tqdm(*args, **kwargs):
        from tqdm import tqdm
        return tqdm(*args, **kwargs)
    progress = tqdm
    
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

    def repo2path(self, search=None):
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
    def time(cls):
        return time.time()

    def ask(self, *args, **kwargs):
        return c.module("agent")().ask(*args, **kwargs) 

    def clone(self, repo:str = 'commune-ai/commune', path:str=None, **kwargs):
        gitprefix = 'https://github.com/'
        if not repo.startswith(gitprefix):
            repo = gitprefix + repo
        path = os.path.abspath(path or  '~/'+repo.split('/')[-1])
        cmd =  f'git clone {repo} {path}'
        return c.cmd(cmd, verbose=True)
    
    def copy_module(self,module:str, path:str):
        code = c.code(module)
        path = os.path.abspath(path)
        import time 
        # put text one char at a time to the file
        # append the char to the code
        c.rm(path)
        for char in code:
            time.sleep(0.000001)
            # append the char to the code one at a time so i can see the progress
            with open(path, 'a') as f:
                f.write(char)
        return {'path': path, 'module': module}
    
    def has_module(self, path:str):
        for path in c.files(path): 
            if path.endswith('.py'):
                return True  
    
    @classmethod
    def module2fns(cls, path=None):
        path = path or cls.dirpath()
        tree = c.get_tree(path)
        module2fns = {}
        for m,m_path in tree.items():
            if '.modules.' in m_path:
                continue
            try:
                module2fns[m] = c.module(m).fns()
            except Exception as e:
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
            for f in module2fns[m]:
                fn2module[f] = m
        return fn2module

    def epoch(self, *args, **kwargs):
        return c.mod('vali')().epoch(*args, **kwargs)

    def e(self, *args, **kwargs):
        return c.mod('vali')().epoch(*args, **kwargs)

    def routes_from_to(self):
        routes = c.routes
        from_to_map = {}
        for m, fns in routes.items():
            for fn in fns:
                route = m + '/' + fn
                assert  fn not in from_to_map, f'Function {route}  already exists in {from_to_map[fn]}'
                from_to_map[fn] = m + '/' + fn
        return from_to_map
    
    def run_test(self, module=None, parallel=True):
        module = module or self
        fns = [f for f in dir(module) if 'test_' in f]
        fn2result = {}
        fn2error = {}
        if parallel:
            future2fn = {}
            for fn in fns:
                future = c.submit(getattr(module, fn))
                future2fn[future] = fn
            for future in c.as_completed(future2fn):
                fn = future2fn[future]
                try:
                    result = future.result()
                    fn2result[fn] = result
                except Exception as e:
                    fn2error[fn] = {'success': False, 'msg': str(e)}
        else:
            for fn in fns:
                try:
                    result = getattr(module, fn)()
                    fn2result[fn] = result
                except Exception as e:
                    fn2error[fn] = {'success': False, 'msg': str(e)}
        if len(fn2error) > 0:
            raise Exception(f'Errors: {fn2error}')
        
        return fn2result

    def readmes(self, path='./', search=None):
        files =  c.files(path)
        readmes = [f for f in files if f.endswith('.md')]
        return readmes
    def docs(self, path='./', search=None):
        files =  c.files(path)
        readmes = [f for f in files if f.endswith('.md')]
        return {k.replace(c.abspath('~') +'/', '~/'):c.get_text(k) for k in readmes}
c.add_routes()
Module = c # Module is alias of c
if __name__ == "__main__":
    Module.run()


