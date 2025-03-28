import os
import inspect
import json
import yaml
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

    @staticmethod
    def is_module(obj):
        """
        does the obj have al of the core features
        """
        return  all([hasattr(obj, k) for k in c.core_features]) # if the object is a module

    @classmethod
    def module(cls, path: str = 'module', params: dict = None, verbose=False, cache=True, **kwargs) -> str:
        # Initialize cache if needed
        if not hasattr(c, 'module_cache'):
            c.module_cache = {}
    
        # Return path if it's already a module object
        if not isinstance(path, str) and path is not None:
            return path
            
        t0 = time.time()
        
        # Handle self-references
        if path in ['module', c.repo_name[0], c.repo_name] or c.repo_name.startswith(path):
            return c
        
        # Normalize path
        path = path.replace('/', '.')
        path = c.shortcuts.get(path, path)
        tree = c.tree()
        module = tree.get(path, path)
        if (c.repo_name + '.' + c.repo_name) == module:
            return c
        # Try to load the module
        try:
            obj = c.obj(module)
        except Exception as e:
            if verbose:
                print(f'Error loading {module} {e}')
            # Try to find the module in the tree
            tree = c.tree(update=1)
            if module not in tree:
                tree_keys = [k for k in tree.keys() if module in k]
                if tree_keys:
                    module = tree.get(tree_keys[0])
            obj = c.obj(module)
        
        # Add utility methods to non-module objects
        if not c.is_module(obj):
            methods = {
                'resolve_module': lambda *args, **kwargs: c.resolve_module(obj),
                'module_name': lambda *args, **kwargs: c.module_name(obj),
                'storage_dir': lambda *args, **kwargs: c.storage_dir(obj),
                'filepath': lambda *args, **kwargs: c.filepath(obj),
                'dirpath': lambda *args, **kwargs: c.dirpath(obj),
                'code': lambda *args, **kwargs: c.code(obj),
                'info': lambda *args, **kwargs: c.info(obj),
                'cid': lambda *args, **kwargs: c.cid(obj),
                'schema': lambda *args, **kwargs: c.schema(obj),
                'fns': lambda *args, **kwargs: c.fns(obj),
                'fn2code': lambda *args, **kwargs: c.fn2code(obj),
                'fn2hash': lambda *args, **kwargs: c.fn2hash(obj),
                'dir': lambda *args, **kwargs: c.dir(obj),
                'chat': lambda *args, **kwargs: c.chat(obj),
                'ask': lambda *args, **kwargs: c.ask(obj),
                'config_path': lambda *args, **kwargs: c.config_path(obj),
                'config': lambda *args, **kwargs: c.config(obj),
            }
        
            # Set all methods at once
            for name, method in methods.items():
                if not hasattr(obj, name):                
                    setattr(obj, name, method)
            # Add optional methods if they don't exist

        
        # Apply parameters if provided
        if isinstance(params, dict):
            print(f'Params({obj})')
            obj = obj(**params)
        elif isinstance(params, list):
            obj = obj(*params)
            
        if verbose:
            c.print(f'Module({module} t={(c.time() - t0):.2f})')
            
        return obj

    get_agent = block =  get_block = get_module =  mod =  module

    def go(self, module=None, **kwargs):
        try:
            path = c.filepath(module)
        except:
            path = c.modules_path + '/' + module
        if path.split('/')[-1] == path.split('/')[-2]:
            path = '/'.join(path.split('/')[:-1])
        assert os.path.exists(path), f'{path} does not exist'
        return c.cmd(f'code {path}', **kwargs)

    @classmethod
    def filepath(cls, obj=None) -> str:
        return inspect.getfile(cls.resolve_module(obj))

    @classmethod
    def getfile(cls, obj=None) -> str:
        return inspect.getfile(cls.resolve_module(obj))
    
    @classmethod
    def pytest(cls, path:str = None):
        path = path or (c.core_path + '/test.py')
        return c.cmd(f'pytest {path}')

    @classmethod
    def path(cls, obj=None) -> str:
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

    @staticmethod
    def abspath(path:str):
        return os.path.abspath(os.path.expanduser(path))
    @staticmethod
    def path2hash(path:str='./') -> int:
        path = c.abspath(path)
        if os.path.isfile(path):
            return c.hash(c.text(path))
        else:
            return {k[len(path)+1:]:c.hash(c.text(k)) for k in c.files(path)}
    
    @staticmethod
    def filehash(path:str):
        return c.hash(c.path2hash(path))
        
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

    module_path = module_name 
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
        global config_path
        if obj in [None, 'module']:
            return config_path
        json_path =  c.dirpath(obj) + '/config.json'
        yaml_path =  c.dirpath(obj) + '/config.yaml'
        if os.path.exists(json_path):
            return json_path
        elif ocore_featuress.path.exists(yaml_path):
            return yaml_path

    @classmethod
    def sandbox(cls, path='./', filename='sandbox.py'):
        for file in  c.files(path):
            if file.endswith(filename):
                return c.cmd(f'python3 {file}', verbose=True)
        return {'success': False, 'message': 'sandbox not found'}
    
    sand = sandbox
    
    @classmethod
    def storage_dir(cls, module=None):
        module = cls.resolve_module(module)
        return os.path.abspath(os.path.expanduser(f'~/.commune/{module.module_name()}'))

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
        return c.obj('commune.utils.print_console')(*text, **kwargs)

    @classmethod
    def resolve_module(cls, obj:str = None, default=None, fn_splitter='/', **kwargs):
        if obj == None:
            obj = cls
        else:
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
        return getattr(module, argv.fn)(*argv.args, **argv.kwargs)     
        
    @classmethod
    def commit_hash(cls, lib_path:str = None):
        if lib_path == None:
            lib_path = c.lib_path
        return c.cmd('git rev-parse HEAD', cwd=lib_path, verbose=False).split('\n')[0].strip()

    @classmethod
    def run_fn(cls,fn:str, params:Optional[dict]=None, args=None, kwargs=None, module='module') -> Any:
        """
        get a fucntion from a strings
        """

        if '/' in fn:
            module, fn = fn.split('/')
        else:
            assert hasattr(module, fn), f'{fn} not in {module}'
        module = c.module(module)
        params = params or {}
        fn_obj = getattr(module, fn)
        if 'self' in c.get_args(fn_obj):
            module = module()
        fn_obj =  getattr(module, fn)
        if isinstance(params, list):
            args = params
        elif isinstance(params, dict):
            kwargs = params
        args = args or []
        kwargs = kwargs or {}
        return fn_obj(*args, **kwargs)
    
    # UNDER CONSTRUCTION (USE WITH CAUTION)


        dirpath = c.dirpath(module)
    def gitpath(self, module:str = 'module', **kwargs) -> str:
        """
        find the github url of a module
        """
        dirpath = c.dirpath(module)
        while len(dirpath.split('/')) > 1:
            dirpath = '/'.join(dirpath.split('/')[:-1])
            git_path = dirpath + '/.git'
            if os.path.exists(git_path):
                return git_path
        return None


    def get_args_kwargs(self,params={},  args:List = [], kwargs:dict = {}, ) -> Tuple:
        """
        resolve params as args 
        """
        params = params or {}
        args = args or []
        kwargs = kwargs or {}
        if isinstance(params, list):
            args = params
        elif isinstance(params, dict):
            if 'args' in params and 'kwargs' in params and len(params) == 2:
                args = params['args']
                kwargs = params['kwargs']
            else:
                kwargs = params
        return args, kwargs

    def forward(self, fn:str='info', params:dict=None, signature=None) -> Any:
        params = params or {}
        assert fn in self.endpoints, f'{fn} not in {self.endpoints}'
        if hasattr(self, fn):
            fn_obj = getattr(self, fn)
        else:
            fn_obj = c.fn(fn)
        return fn_obj(**params)

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
        return Key().get_key(key, **kwargs)
        

    @classmethod
    def key(cls,key:str = None , **kwargs) -> None:
        from commune.key import Key
        return Key().get_key(key, **kwargs)

    @classmethod
    def keys(cls,key:str = None , **kwargs) -> None:
        from commune.key import Key
        return Key().keys(key, **kwargs)

    @classmethod
    def key2address(cls,key:str = None , **kwargs) -> None:
        from commune.key import Key
        return Key().key2address(key, **kwargs)
    
    @classmethod
    def files(cls, 
              path='./', 
              search:str = None, 
              avoid_terms = ['__pycache__', '.git', '.ipynb_checkpoints', 'node_modules', 'artifacts', 'egg-info'], 
              endswith:str = None,
              hidden:bool = False,
              startswith:str = None,
              **kwargs) -> List[str]:
        """
        Lists all files in the path
        """
        files =c.glob(path, **kwargs)
        if not hidden:
            files = [f for f in files if not '/.' in f]
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
    def encrypt(cls,data: Union[str, bytes], key: str = None, password: str = None, **kwargs ) -> bytes:
        key = c.get_key(key) 
        return key.encrypt(data, password=password)

    @classmethod
    def decrypt(cls, data: Any,  password : str = None, key: str = None, **kwargs) -> bytes:
        key = c.get_key(key)
        return key.decrypt(data, password=password)
    
    @classmethod
    def sign(cls, data:dict  = None, key: str = None,  crypto_type='sr25519', mode='str', **kwargs) -> bool:
        key = c.get_key(key, crypto_type=crypto_type)
        return key.sign(data, mode=mode, **kwargs)

    def signtest(self, data:dict  = 'hey', key: str = None,  crypto_type='sr25519', mode='str', **kwargs) -> bool:
        signature = c.sign(data, key, crypto_type=crypto_type, mode=mode, **kwargs)
        return c.verify(data, signature, key=key, crypto_type=crypto_type, **kwargs)
    @classmethod
    def size(cls, module) -> int:
        return len(str(c.code_map(module)))

    @classmethod
    def verify(cls, data, signature=None, address=None,  crypto_type='sr25519',  key=None, **kwargs ) -> bool:  
        key = c.get_key(key, crypto_type=crypto_type)
        return key.verify(data=data, signature=signature, address=address, **kwargs)

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
        utils = c.path2fns(c.core_path + '/utils.py', tolist=True)
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
        utils = c.path2fns(c.core_path + '/utils', tolist=True)
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
    def routes(cls):
        routes = c.config()['routes']
        for util in  c.utils():
            k = '.'.join(util.split('.')[:-1])
            v = util.split('.')[-1]
            routes[k] = routes.get(k , [])
            routes[k].append(v)
        return routes

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
    def name2config(cls,**kwargs):
        return {p.split('/')[-2]:p for p in c.configs(c.modules_path)}




    @classmethod
    def cfg(cls, module=None, mode='dict', fn='__init__') -> 'Munch':
        return cls.config(module=module, mode=mode, fn=fn)

    @classmethod
    def config(cls, module=None, mode='dict', fn='__init__', modes=['json', 'yaml']) -> 'Munch':
        # if os.path.exists(c.modules_path + '/' in module):
        path = None
        if module == None:
            dirpath = c.lib_path
        else:
            module = c.module(module)
            dirpath = c.dirpath(module)
        paths = [os.path.join(dirpath, f'config.{m}') for m in modes if os.path.exists(os.path.join(dirpath, f'config.{m}'))]
        if len(paths) > 0:
            path = paths[0]
        else:
            raise Exception(f'No config file found in {dirpath} for {module}')

        filetype = path.split('.')[-1] if path != None else mode
        if os.path.exists(path):
            if filetype == 'json':
                config = json.load(open(path, 'r'))
            elif filetype in ['yaml', 'yml']:
                config = yaml.load(open(path, 'r'), Loader=yaml.FullLoader)
            else:
                raise Exception(f'Invalid config file {path}')
        else:
            module = c.module(module)
            config =  c.get_params(getattr(module, fn)) if hasattr(module, fn) else {}
        if mode == 'dict':
            pass
        elif mode == 'munch':
            from munch import Munch
            config =  Munch(config)
        else:
            raise Exception(f'Invalid mode {mode}')
        return config

    @classmethod
    def dict2munch(cls, d:Dict) -> 'Munch':
        from munch import Munch
        return Munch(d)

    @classmethod
    def put_json(cls, 
                 path:str, 
                 data:Dict, 
                 meta = None,
                 verbose: bool = False,
                 **kwargs) -> str:
        if not path.endswith('.json'):
            path = path + '.json'
        path = cls.resolve_path(path=path)
        if isinstance(data, dict):
            data = json.dumps(data)
        cls.put_text(path, data)
        return path

    save_json = put_json

    @classmethod
    def rm(cls, path:str, possible_extensions = ['json'], avoid_paths = ['~', '/', './']):
        avoid_paths = list(set((avoid_paths)))
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
    def glob(cls, path:str='./', depth:Optional[int]=None, recursive:bool=True, files_only:bool = True,):
        import glob
        path = cls.resolve_path(path)
        if depth != None:
            if isinstance(depth, int) and depth > 0:
                paths = []
                for path in c.ls(path):
                    if os.path.isdir(path):
                        paths += c.glob(path, depth=depth-1)
                    else:
                        paths.append(path)
            else:
                paths = []
            return paths

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

        if you specify "abc" it will be resolved to the storage dir
        {storage_dir}/abc, in this case its ~/.commune
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
    def ls(cls, path:str = './', 
           search = None,
           include_hidden = False, 
           depth=None,
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
        data = c.get_json(k, default=default, **kwargs)
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
        from commune.utils import get_text
        return get_text(path, **kwargs)

    @classmethod
    def text(cls, path: str = './', **kwargs ) -> str:
        # Get the absolute path of the file
        path = c.abspath(path)
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
    def fnschema(cls, fn:str = '__init__', include_code=False, **kwargs)->dict:
        '''
        Get function schema of function in cls
        '''     
        schema = {}
        fn_obj = c.get_fn(fn)
        if not callable(fn_obj):
            return {'fn_type': 'property', 'type': type(fn_obj).__name__}
        fn_signature = inspect.signature(fn_obj)
        schema['input'] = {}
        for k,v in dict(fn_signature._parameters).items():
            schema['input'][k] = {
                    'value': "_empty"  if v.default == inspect._empty else v.default, 
                    'type': '_empty' if v.default == inspect._empty else str(type(v.default)).split("'")[1] 
            }
        schema['output'] = {
            'value': None,
            'type': str(fn_obj.__annotations__.get('return', None) if hasattr(fn_obj, '__annotations__') else None)
        }
        schema['docs'] = fn_obj.__doc__
        schema['cost'] = 1 if not hasattr(fn_obj, '__cost__') else fn_obj.__cost__
        schema['name'] = fn_obj.__name__
        schema['source'] = c.source(fn_obj, include_code=include_code)
        return schema



    
    @classmethod
    def source(cls, obj, include_code=True):
        """
        Get the source code of a function
        """
        if isinstance(obj, str):
            obj = c.fn(obj)
        sourcelines = inspect.getsourcelines(obj)
        source = ''.join(sourcelines[0])
        return {
                             'start': sourcelines[1], 
                             'length': len(sourcelines[0]),
                             'path': inspect.getfile(obj).replace(c.home_path, '~'),
                             'code': source if include_code else None,
                             'hash': c.hash(source),
                             'end': len(sourcelines[0]) + sourcelines[1]
                             }

    @classmethod
    def schema(cls, obj = None, **kwargs)->dict:
        '''
        Get function schema of function in cls
        '''   
        if '/' in str(obj) or callable(obj):
            schema = c.fnschema(obj, **kwargs)
        else:
            module = c.resolve_module(obj)
            fns = c.fns(module)
            schema = {fn: c.fnschema(getattr(module, fn)) for fn in fns}
        return schema
 
    @classmethod
    def resolve_obj(cls, obj = None, search=None, *args, **kwargs) -> Union[str, Dict[str, str]]:
        obj = obj or cls
        if isinstance(obj, str) and '/' in obj:
            obj = c.fn(obj)
        else:
            obj = cls.resolve_module(obj)
        return obj

    @classmethod
    def code(cls, obj = None, search=None, *args, **kwargs) -> Union[str, Dict[str, str]]:
        return inspect.getsource(cls.resolve_obj(obj))
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
    def code_hash_map(cls, module , search=None, *args, **kwargs) ->  Dict[str, str]:
        return {k:c.hash(str(v)) for k,v in c.code_map(module=module, search=search,**kwargs).items()}

    @classmethod
    def cid(cls, module , search=None, *args, **kwargs) -> Union[str, Dict[str, str]]:
        return c.hash(cls.code_hash_map(module=module, search=search,**kwargs))

    cid = cid
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
        return c.hash(c.code_map(module or cls.module_name(), **kwargs))
    @classmethod
    def cid(cls, module=None,  *args, **kwargs):
        return c.hash(c.code(module, **kwargs))

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
    def fns(cls, obj: Any = None,
                      search = None,
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
        obj = c.resolve_module(obj)
        text = inspect.getsource(obj)
        functions = []
        for splitter in ["   def " , "    def "]:
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
            max_age : Optional[int]=1000, 
            lite_features : List[str] = ['schema', 'name', 'key', 'founder', 'hash', 'time'],
            keep_last_n : int = 10,
            relative=True,
            update: bool =False, 
            key=None,
            **kwargs):
            
        path = c.resolve_info_path(module)
        info = c.get(path, None, max_age=max_age, update=update)
        if info == None:
            code = c.code_map(module)
            schema = c.schema(module)
            founder = c.founder().address
            key = c.get_key(key or module).address
            info =  {
                    'code': code, 
                    'schema': schema, 
                    'name': module, 
                    'key': key,  
                    'founder': founder, 
                    'cid': c.cid(module),
                    'time': time.time()
                    }
           
            info['signature'] = c.sign(module)

            c.put(path, info)
        if lite:
            info = {k: v for k,v in info.items() if k in lite_features}
        return  info


    def epoch(self, *args, **kwargs):
        return c.run_epoch(*args, **kwargs)

    def get_tags(self,module=None, search=None, **kwargs):
        tags = []
        module =  c.resolve_module(module)
        if hasattr(module, 'tags'):
            tags = module.tags
        assert isinstance(tags, list), f'{module} does not have tags'
        if search != None:
            tags = [t for t in tags if search in t]
        return tags

    def pwd2key(self, pwd):
        return c.module('key').str2key(pwd)

        
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
    def submit(cls, 
                fn, 
                params = None,
                kwargs: dict = None, 
                args:list = None, 
                timeout:int = 40, 
                module: str = None,
                mode:str='thread',
                max_workers : int = 100,
                ):
        fn = c.get_fn(fn)
        executor = c.module('executor')(max_workers=max_workers, mode=mode) 
        return executor.submit(fn=fn, params=params, args=args, kwargs=kwargs, timeout=timeout)

    @classmethod
    def get_fn(cls, fn:str, splitter='/', params=None, default_fn='forward') -> 'Callable':
        """
        Gets the function from a string or if its an attribute 
        """
        if isinstance(fn, str):
            if fn.startswith('/'):
                fn = 'module' + fn
            elif fn.endswith('/'):
                module = c.module(fn[:-1])
                fn = default_fn
                return getattr(module, default_fn)
            fn_obj = None
            module = cls
            if '/' in fn:
                module = c.module('/'.join(fn.split('/')[:-1]))
                fn = fn.split('/')[-1]
            elif c.object_exists(fn):
                fn_obj =  c.obj(fn)
            else:
                raise Exception(f'{fn} is not a function or object')
            fn_obj = getattr(module, fn)
            args = c.get_args(fn_obj)
            if 'self' in args:
                module = module()
                fn_obj = getattr(module, fn)
  
        elif callable(fn):
            fn_obj = fn
        else:
            raise Exception(f'{fn} is not a function or object')
        if params != None:
            return fn_obj(**params)
        return fn_obj
    @classmethod
    def fn(cls, fn:str, splitter='/', params=None, default_fn='forward') -> 'Callable':
        return c.get_fn(fn, splitter=splitter, params=params, default_fn=default_fn)

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
    def password(cls, max_age=None, update=False, **kwargs):
        path = c.resolve_path('password')
        pwd = c.get(path, None, max_age=max_age, update=update,**kwargs)
        if pwd == None:
            pwd = c.hash(c.mnemonic())
            c.put(path, pwd)
            c.print('Generating new password', color='blue')
        return pwd

    @classmethod
    def temp_password(cls, max_age=10000, update=False, **kwargs):
        path = c.resolve_path('temp_password')
        pwd = c.get(path, None, max_age=max_age, update=update,**kwargs)
        if pwd == None:
            pwd = c.hash(c.mnemonic())
            c.put(path, pwd)
            c.print('Generating new password', color='blue')
        return pwd

    @classmethod
    def mnemonic(cls, words=24):
        return c.mod('key')().generate_mnemonic(words=words)
      


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
        dir_prefixes  = [c.pwd(), c.lib_path, c.home_path]
        for dir_prefix in dir_prefixes:
            if path.startswith(dir_prefix):
                path =   path[len(dir_prefix) + 1:].replace('/', '.')
                break
        if path.endswith('.py'):
            path = path[:-3]
        return path.replace('__init__.', '.')
        
    @classmethod
    def path2name(cls, path, ignore_folder_names = ['modules', 'agents', 'src', 'mods']):
        name = cls.path2objectpath(path)
        name_chunks = []
        for chunk in name.split('.'):
            if chunk in ignore_folder_names:
                continue
            if chunk not in name_chunks:
                name_chunks += [chunk]
        if name_chunks[0] == c.repo_name:
            name_chunks = name_chunks[1:]
        return '.'.join(name_chunks)

    @staticmethod
    def path2classes(path='./',
                     class_prefix = 'class ', 
                     file_extension = '.py',
                     tolist = False,
                     depth=4,
                     relative=False,
                     class_suffix = ':', **kwargs) :
        path = c.abspath(path)
        path2classes = {}
        if os.path.isdir(path) and depth > 0:
            for p in c.ls(path):
                try:
                    for k,v in c.path2classes(p, depth=depth-1).items():
                        if len(v) > 0:
                            path2classes[k] = v
                except Exception as e:
                    pass
        elif os.path.isfile(path) and path.endswith('.py'):
            code = c.get_text(path)
            classes = []
            file_path = c.path2objectpath(path)
            for line in code.split('\n'):
                if line.startswith(class_prefix) and line.strip().endswith(class_suffix):
                    new_class = line.split(class_prefix)[-1].split('(')[0].strip()
                    if new_class.endswith(class_suffix):
                        new_class = new_class[:-1]
                    if ' ' in new_class:
                        continue
                    classes += [new_class]
            if file_path.startswith(path):
                file_path = file_path[len(path)+1:]
            if '/' in file_path:
                file_path = file_path.replace('/', '.')
            if relative:
                path = c.path2relative(path)
            path2classes =  {path:  [file_path + '.' + cl for cl in classes]}
        if tolist: 
            classes = []
            for k,v in path2classes.items():
                classes.extend(v)
            return classes
   
        return path2classes

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
    def objs(cls, path:str = './', depth=10, search=None, **kwargs) -> List[str]:
        classes = c.classes(path,depth=depth)
        functions = c.path2fns(path, tolist=True)
        objs = functions + classes
        if search != None:
            objs = [f for f in objs if search in f]
        return objs

    @classmethod
    def lsfns(cls, path:str = './', depth=10, search=None, **kwargs) -> List[str]:
        functions = c.path2fns(path, tolist=True)
        return functions

    @classmethod
    def objects(cls, path:str = './', depth=10, search=None, **kwargs):
        return c.objs(path=path, depth=depth, search=search, **kwargs)

    @classmethod
    def import_module(cls, import_path:str ) -> 'Object':
        from importlib import import_module
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
        if (c.repo_name + '.' + c.repo_name) in key:
            key = key.replace((c.repo_name + '.' + c.repo_name) ,c.repo_name)

        if key.split('.')[0] == c.repo_name[0]:
            key = key.replace(c.repo_name[0] + '.', c.repo_name + '.')
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

    def m(self):
        """enter modules path in vscode"""
        return c.cmd(f'code {c.modules_path}')
    def a(self):
        """enter modules path in vscode"""
        return c.cmd(f'code {c.app_path}')

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
    lmods = local_modules
    @classmethod
    def lib_tree(cls, depth=10, **kwargs):
        return c.get_tree(c.lib_path, depth=depth, **kwargs)


    

    @classmethod
    def core_tree(cls, **kwargs):
        tree =  c.get_tree(c.core_path, **kwargs)
        return {k:v for k,v in tree.items() if 'modules.' not in v}

    @classmethod
    def core_readmes(cls, **kwargs):
        return cls.readmes(c.lib_path, **kwargs)

    @classmethod
    def core_readme2text(cls, **kwargs):
        return cls.readme2text(c.lib_path, **kwargs)
    

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

    _tree = None
    @classmethod
    def tree(cls, search=None,  max_age=60,update=False, **kwargs):
        core_tree = c.core_tree(update=update, max_age=max_age)
        local_tree = c.local_tree(update=update, max_age=max_age)
        lib_tree = c.lib_tree(update=update, max_age=max_age)
        modules_tree = c.modules_tree(update=update, max_age=max_age)
        tree = {**modules_tree, **local_tree, **core_tree }
        if search != None:
            tree = {k:v for k,v in tree.items() if search in k}
        return tree
    @classmethod
    def modules_tree(cls, search=None, **kwargs):
        tree =  c.get_tree(c.modules_path, search=search, **kwargs)
        # tree = {k.replace('modules.',''):v for k,v in tree.items() }
        return tree

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

    @classmethod
    def mods(cls, search=None, cache=True, max_age=60, update=False, **extra_kwargs)-> List[str]:   
        return cls.modules(search=search, cache=cache, max_age=max_age, update=update, **extra_kwargs)

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
    def new( cls,
                   path : str ,
                   name= None, 
                   base_module : str = 'base', 
                   update=0
                   ):
        name = name or path
        if name.endswith('.git'):
            git_path = c.giturl(path)
            name =  path.split('/')[-1].replace('.git', '')
        dirpath = os.path.abspath(c.modules_path +'/'+ name.replace('.', '/'))
        filename = name.replace('.', '_') + '.py'
        path = f'{dirpath}/{filename}'
        # path = dirpath + '/' + module_name + '.py'
        base_module_obj = c.module(base_module)
        code = c.code(base_module)
        code = code.replace(base_module_obj.__name__, ''.join([m[0].capitalize() + m[1:] for m in name.split('.')]))
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        c.put_text(path, 'import commune as c \n'+code)
        return {'name': name, 'path': path, 'msg': 'Module Created'}
    
    add_module = new_module = new

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
    def chat(cls, *args, module=None, **kwargs):
        if module != None:
            args = [c.code(module)] + list(args)
        return c.module("agent")().ask(*args, **kwargs) 

    @classmethod
    def ask(cls, *args, module=None, **kwargs):
        if module != None:
            args = [c.code(module)] + list(args)
        return c.module("agent")().ask(*args, **kwargs) 

    @classmethod
    def ai(cls, *args, **kwargs):
        return c.module("agent")().ask(*args, **kwargs) 

    @classmethod
    def clone(cls, repo:str = 'commune-ai/commune', path:str=None, **kwargs):
        gitprefix = 'https://github.com/'
        repo = gitprefix + repo if not repo.startswith(gitprefix) else repo
        path = os.path.abspath(os.path.expanduser(path or  '~/'+repo.split('/')[-1]))
        cmd =  f'git clone {repo} {path}'
        c.cmd(cmd, verbose=True)
        return {'path': path, 'repo': repo, 'msg': 'Repo Cloned'}

    def has_module(self, path:str):
        for path in c.files(path): 
            if path.endswith('.py'):
                return True  

    def has_modules(self, path:str):
        return path in c.get_tree(c.modules_path)
        
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

    @classmethod 
    def test_fns(cls, module=None):
        return [f for f in dir(c.module(module)) if f.startswith('test_') or f == 'test']


    @classmethod
    def test(cls, module=None, timeout=50, modules=[ 'server', 'vali','key', 'chain']):
        
        if module == None:
            test_results ={}
            for m in modules:
                test_results[m] = cls.test(m, timeout=timeout)
            return test_results
        elif c.module_exists(module + '.test'):
            module = module + '.test'

        module_obj = c.module(module)()
        fn2result = {}
        for i, fn in enumerate(cls.test_fns(module)):
            fn_path = f'{module}/{fn}'
            buffer = 5 * '*-' 
            emoji = '⏳'
            title = 'TEST'
            c.print(f'{buffer}{emoji}\{title}({fn_path})\t{emoji}{buffer}', color='yellow')
            try:
                fn2result[fn] = getattr(module_obj, fn)()
            except Exception as e:
                e = c.detailed_error(e)
                return {'fn': fn, 'error': e}
            title = 'PASS'
            emoji = '✅'
            c.print(f'{buffer}{emoji}\{title}({fn_path})\t{emoji}{buffer}', color='green')
        return fn2result


    @classmethod
    def test_module(cls, module='module', timeout=50):
        """
        Test the module
        """

        if c.module_exists(module + '.test'):
            module = module + '.test'

        if module == 'module':
            module = 'test'
        Test = c.module(module)
        test_fns = [f for f in dir(Test) if f.startswith('test_') or f == 'test']
        test = Test()
        futures = []
        for fn in test_fns:
            print(f'Testing({fn})')
            future = c.submit(getattr(test, fn), timeout=timeout)
            futures += [future]
        results = []
        for future in c.as_completed(futures, timeout=timeout):
            print(future.result())
            results += [future.result()]
        return results

    testmod = test_module

    @staticmethod
    def readmes( path='./', search=None, avoid_terms=['/modules/']):
        files =  c.files(path)
        files = [f for f in files if f.endswith('.md')]
        files = [f for f in files if all([avoid not in f for avoid in avoid_terms])]
        if search != None:
            files = [f for f in files if search in f]
        return files
    config_name_options = ['config', 'cfg', 'module', 'block',  'agent', 'mod', 'bloc']

    def is_config_module(self, module:str):
        dirpath = c.dirpath(module)
        # if has a config.yaml, config.json, cfg.yaml, cfg.json, module.yaml, module.json
        name_options = c.config_name_options
        ext_options = ['yaml', 'json']
        for name in name_options:
            for ext in ext_options:
                if os.path.exists(f'{dirpath}/{name}.{ext}') :
                    return True
        return False


    @staticmethod
    def configs( path='./', modes=['yaml', 'json'], search=None, names=['config', 'cfg', 'module', 'block',  'agent', 'mod', 'bloc']):
        """
        Returns a list of config files in the path
        """
        def is_config(f):
            name_options = c.config_name_options
            return any(f.endswith(f'{name}.{m}') for name in names for m in modes)
        configs =  [f for f in  c.files(path) if is_config(f)]
        if search != None:
            configs = [f for f in configs if search in f]
        return configs

    @staticmethod
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

    @classmethod
    def sys_paths(cls):
        return sys.path

    @classmethod
    def sync_routes(cls, routes:dict=None, verbose=False):

        """
        This ties other modules into the current module.
        The way it works is that it takes the module name and the function name and creates a partial function that is bound to the module.
        This allows you to call the function as if it were a method of the current module.
        for example
        """
        routes = cls.routes()
        t0 = time.time()
        # WARNING : THE PLACE HOLDERS MUST NOT INTERFERE WITH THE KWARGS OTHERWISE IT WILL CAUSE A BUG IF THE KWARGS ARE THE SAME AS THE PLACEHOLDERS
        # THE PLACEHOLDERS ARE NAMED AS module_ph and fn_ph AND WILL UNLIKELY INTERFERE WITH THE KWARGS
        def fn_generator(*args, route, **kwargs):
            
            def fn_wrapper(*args, **kwargs):
                try:
                    fn_obj = c.obj(route)
                except Exception as e:
                    if '/' in route:
                        module = '/'.join(route.split('/')[:-1])
                        fn = route.split('/')[-1]
                    module = c.module(module)
                    fn_obj = getattr(module, fn)
                    fn_args = c.get_args(fn_obj)
                    if 'self' in fn_args:
                        fn_obj = getattr(module(), fn)
                if callable(fn_obj):
                    return fn_obj(*args, **kwargs)
                else:
                    return fn_obj
            return fn_wrapper(*args, **kwargs)

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
                    if verbose:
                        print(f'Warning: {to_fn} already exists')
                else:
                    fn_obj = partial(fn_generator, route=f'{module}/{fn}') 
                    fn_obj.__name__ = to_fn
                    setattr(cls, to_fn, fn_obj)
        duration = time.time() - t0
        return {'success': True, 'msg': 'enabled routes', 'duration': duration}

    @classmethod
    def add_globals(cls, globals_input:dict = None):
        """
        add the functions and classes of the module to the global namespace
        """
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

    @staticmethod
    def giturl(url:str='commune-ai/commune'):
        gitprefix = 'https://github.com/'
        gitsuffix = '.git'
        if not url.startswith(gitprefix):
            url = gitprefix + url
        if not url.endswith(gitsuffix):
            url = url + gitsuffix
        return url

    @staticmethod
    def sync_module(url, max_age=10000, update=False):
        """ 
        Syncs a module from a git repository

        params:
            url:
                - can be a string
                - a dictionary with the keys 'name' and 'url'
                -  a list with the first element being the url and the second element being the name
                
        returns:
            - True if the module was synced
            - False if the module was not synced

        """
        if isinstance(url, str):
            name = url.split('/')[-1].replace('.git', '')
        elif isinstance(url, dict):
            name = url['name']
            url = url['url']
        elif isinstance(url, list):
            url = url[0]
            name = url[1]
        modules_flag_path = 'sync_modules/' + name
        modules_flag = c.get(modules_flag_path, max_age=max_age, update=update)
        if modules_flag != None:
            return True
        url = c.giturl(url)
        if name in ['modules', 'mods']:
            module_path = c.modules_path
        else:
            module_path = c.modules_path + '/' + name.replace('.','/')
        if not os.path.exists(module_path):
            os.makedirs(module_path, exist_ok=True)
        if os.path.exists(module_path+'/.git'):
            cmd = f'git pull {url} {module_path}' 
        else:
            cmd = f'git clone {url} {module_path}'
        c.cmd(cmd, cwd=module_path)
        c.put(modules_flag_path, True)
        return True

    @classmethod
    def live(self):
        return c.servers()

    @staticmethod
    def sync_modules(max_age=10, update=False):
        results = []
        synced_modules = c.get('synced_modules', max_age=max_age, update=update)
        if synced_modules != None:
            return synced_modules
        
        futures = []
        for url in c.config()['modules']:
            params = {'url': url, 'max_age': max_age, 'update': update}
            futures += [c.submit(c.sync_module, params)]
        
        results = []
        progress = c.tqdm(len(futures))
        for future in c.as_completed(results):
            results.append(c.sync_module(url, max_age=max_age, update=update))
            progress.update(1)

        return results

    def add_tags(self, module='openrouter', goal='RETURN TAGS AS A LIST AS THE CONTENT'):
        text = f'''
        --CONTENT--
        {c.code_map(module)}
        --GOAL--
        {goal}
        --FORMAT--
        <START_OUTPUT>JSON(data=['tag1', 'tag2'])<END_OUTPUT>
        '''
        model = c.module('openrouter')()
        output = ''
        for ch in  model.forward(text,process_text=False, stream=1):
            print(ch)
            output += ch
        
        output = output.split('START_OUTPUT>')[-1].split('<END_OUTPUT')[0]
        return json.loads(output)


    @classmethod
    def core_context(cls):
        return c.readme2text(c.core_path)


    def requirements(self, module='model.openai'):
        return c.ask(f'make a dope module for the followin return outputs as a json', module=module)


    @classmethod
    def help(cls, *question, mod:str='module', model='google/gemini-2.0-flash-001',  search=None):
        x = {
            'code_map': c.code(mod),
            'core_context': c.core_context(),
            'question' : ' '.join(question), 
            'goal': 'respond in the output field',
            'output': None
        }
        output = ''
        for ch in  c.ask(str(x), process_text=False, model=model):
            print(ch, end='')
            output += ch
        return output

    @classmethod
    def sync(cls, max_age=10, update=True, **kwargs):
        """
        Initialize the module by sycing with the config
        """

        # assume the name of this module is the name of .../
        c.repo_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
        c.storage_path = os.path.expanduser(f'~/.{c.repo_name}')
        c.core_path  = c.corepath = os.path.dirname(__file__)
        c.repo_path  = c.repopath = os.path.dirname(os.path.dirname(__file__)) # the path to the repo
        c.lib_path  = c.libpath = os.path.dirname(os.path.dirname(__file__)) # the path to the library
        c.home_path = c.homepath  = os.path.expanduser('~') # the home path
        c.modules_path = c.modspath = c.core_path + '/modules'
        c.app_path = c.core_path + '/app'
        c.tests_path = f'{c.lib_path}/tests'
        if not hasattr(c, 'included_pwd_in_path'):
            c.included_pwd_in_path = False
        if  not c.included_pwd_in_path:
            paths = [c.modules_path, c.pwd()]
            for p in paths:
                if not p in sys.path:
                    sys.path.append(p)
            c.included_pwd_in_path = True


        # config attributes
        config = c.config()
        c.core_modules = config['core_modules'] # the core modules
        c.test_modules = config['test_modules']
        c.repo_name  = config['name'] # the name of the library
        c.endpoints = config['endpoints']
        c.core_features = config['core_features']
        c.port_range = config['port_range'] # the port range between 50050 and 50150
        c.shortcuts =  c.shortys = config["shortcuts"]


        c.sync_routes()
        c.sync_modules(max_age=max_age, update=update)

        return {'success': True, 'msg': 'synced config'}

c.sync()
Module = c # Module is alias of c
if __name__ == "__main__":
    Module.run()


