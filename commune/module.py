
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

class Module:

    def __init__(self, globals_input = None, **kwargs): 
        self.sync(globals_input=globals_input, **kwargs)

    def module(self, 
                path: str = 'module', 
                params: dict = None,  
                cache=True, 
                verbose=False, 
                **kwargs) -> str:
        t0 = time.time()
        # Initialize cache if needed

        path = path or 'module'
        # Normalize path
        path = path.replace('/', '.')
        path = self.shortcuts.get(path, path)
        tree = self.tree(update=1)
        module = self.tree().get(path, path)
        # Try to load the module
        if module in ['module']:
            return Module
        if module in tree:
            obj = self.obj(module)
        else:
            if module not in tree:
                tree_keys = [k for k in tree.keys() if module.endswith(k) or k.endswith(module)]
                if tree_keys:
                    module = tree.get(tree_keys[0])
            obj = self.obj(module)
        # Apply parameters if provided
        if isinstance(params, dict):
            obj = obj(**params)
        elif isinstance(params, list):
            obj = obj(*params)
        else: 
            # no params set
            pass
        latency = time.time() - t0
        return obj

    def mod(self, path:str = 'module', params:dict = None, cache=True, verbose=False, **kwargs) -> str:
        return self.module(path=path, params=params, cache=cache, verbose=verbose, **kwargs)
    
    def forward(self, fn:str='info', params:dict=None, signature=None) -> Any:
        params = params or {}
        assert fn in self.endpoints, f'{fn} not in {self.endpoints}'
        if hasattr(self, fn):
            fn_obj = getattr(self, fn)
        else:
            fn_obj = self.fn(fn)
        return fn_obj(**params)

    def go(self, module=None, **kwargs):
        try:
            path = self.dirpath(module)
        except:
            path = self.modules_path + '/' + module
        if path.split('/')[-1] == path.split('/')[-2]:
            path = '/'.join(path.split('/')[:-1])
        assert os.path.exists(path), f'{path} does not exist'
        return self.cmd(f'code {path}', **kwargs)

    def filepath(self, obj=None) -> str:
        return inspect.getfile(self.resolve_module(obj))

    def getfile(self, obj=None) -> str:
        return inspect.getfile(self.resolve_module(obj))

    def path(self, obj=None) -> str:
        return inspect.getfile(self.resolve_module(obj))

    def abspath(self,path:str):
        return os.path.abspath(os.path.expanduser(path))
        
    def dirpath(self, obj=None) -> str:
        dirpath =  os.path.dirname(self.filepath(obj))
        if dirpath.split('/')[-1] == dirpath.split('/')[-2]:
            dirpath = '/'.join(dirpath.split('/')[:-1])
        return dirpath
    
    def module_name(self, obj=None):
        obj = obj or Module
        if  isinstance(obj, str):
            obj = self.module(obj)
        module_file =  inspect.getfile(obj)
        return self.path2name(module_file)

    def vs(self, path = None):
        path = path or self.lib_path
        path = os.path.abspath(path)
        return self.cmd(f'code {path}')
    
    def module_class(self, obj=None) -> str:
        return (obj or self).__name__

    def class_name(self, obj= None) -> str:
        obj = obj if obj else self
        return obj.__name__
    
    def config_path(self, obj = None) -> str:
        global config_path
        if obj in [None, 'module']:
            return config_path
        json_path =  self.dirpath(obj) + '/config.json'
        yaml_path =  self.dirpath(obj) + '/config.yaml'
        if os.path.exists(json_path):
            return json_path
        elif os.path.exists(yaml_path):
            return yaml_path
    
    def storage_dir(self, module=None):
        module = self.resolve_module(module)
        return os.path.abspath(os.path.expanduser(f'~/.commune/{self.module_name(module)}'))
    
    def is_admin(self, key:str) -> bool:
        return self.get_key().key_address == key

    def print(self,  *text:str,  **kwargs):
        return self.obj('commune.utils.print_console')(*text, **kwargs)

    def time(self, t=None) -> float:
        import time
        return time.time()
        
    def resolve_module(self, obj:str = None, default=None, fn_splitter='/', **kwargs):
        if obj == None:
            obj = Module
        if isinstance(obj, str):
            obj = self.module(obj)
        return obj

    def pwd(self):
        pwd = os.getcwd() # the current wor king directory from the process starts 
        return pwd

    def token(self, data, key=None, module='auth.jwt',  **kwargs) -> str:
        token = self.module(module)().get_token(data=data, key=key, **kwargs)
        assert self.verify_token(token), f'Token {token} is not valid'
        return token
    def verify_token(self, token:str = None,  module='auth.jwt',  *args, **kwargs) -> str:
        return self.module(module)().verify_token(token=token, *args, **kwargs)

    def run(self, fn=None, params=None, **_kwargs) -> Any: 
        if fn != None:
            return self.get_fn(fn)(**(params or {}))
        parser = argparse.ArgumentParser(description='Argparse for the module')
        parser.add_argument('--module', dest='module', help='The function', type=str, default=self.module_name())
        parser.add_argument('--fn', dest='fn', help='The function', type=str, default="__init__")
        parser.add_argument('--kwargs', dest='kwargs', help='key word arguments to the function', type=str, default="{}") 
        parser.add_argument('--params', dest='params', help='key word arguments to the function', type=str, default="{}") 
        parser.add_argument( '--args', dest='args', help='arguments to the function', type=str, default="[]")  
        argv = parser.parse_args()
        argv.kwargs = json.loads(argv.kwargs.replace("'",'"'))
        argv.params = params or json.loads(argv.params.replace("'",'"'))
        argv.args = json.loads(argv.args.replace("'",'"'))
        argv.module = argv.module.replace('/', '.')
        argv.fn = fn or argv.fn
        if len(argv.params) > 0:
            if isinstance(argv.params, dict):
                argv.kwargs = argv.params
            elif isinstance(argv.params, list):
                argv.args = argv.params
            else:
                raise Exception('Invalid params', argv.params)
        module = self.module(argv.module)()
        return getattr(module, argv.fn)(*argv.args, **argv.kwargs)     
        
    def commit_hash(self, lib_path:str = None):
        if lib_path == None:
            lib_path = self.lib_path
        return self.cmd('git rev-parse HEAD', cwd=lib_path, verbose=False).split('\n')[0].strip()
    
    def run_fn(self,fn:str, params:Optional[dict]=None, args=None, kwargs=None, module='module') -> Any:
        """
        get a fucntion from a strings
        """

        if '/' in fn:
            module, fn = fn.split('/')
        else:
            assert hasattr(module, fn), f'{fn} not in {module}'
        module = self.module(module)
        params = params or {}
        fn_obj = getattr(module, fn)
        if 'self' in self.get_args(fn_obj):
            module = module()
        fn_obj =  getattr(module, fn)
        if isinstance(params, list):
            args = params
        elif isinstance(params, dict):
            kwargs = params
        args = args or []
        kwargs = kwargs or {}
        return fn_obj(*args, **kwargs)
    
    def gitpath(self, module:str = 'module', **kwargs) -> str:
        """
        find the github url of a module
        """
        dirpath = self.dirpath(module)
        while len(dirpath.split('/')) > 1:
            dirpath = '/'.join(dirpath.split('/')[:-1])
            git_path = dirpath + '/.git'
            if os.path.exists(git_path):
                return git_path
        return None

    def is_module_file(self, module = None, exts=['py', 'rs', 'ts'], folder_filenames=['module', 'agent']) -> bool:
        dirpath = self.dirpath(module)
        filepath = self.filepath(module)
        for ext in exts:
            for fn in folder_filenames:
                if filepath.endswith(f'/{fn}.{ext}'):
                    return False
        return bool(dirpath.split('/')[-1] != filepath.split('/')[-1].split('.')[0])
    
    def is_module_folder(self,  module = None) -> bool:
        return not self.is_module_file(module)
    
    is_folder_module = is_module_folder 

    def get_key(self,key:str = None , **kwargs) -> None:
        return self.module('key')().get_key(key, **kwargs)
        
    def key(self,key:str = None , **kwargs) -> None:
        return self.get_key(key, **kwargs)

    def keys(self,key:str = None , **kwargs) -> None:
        return self.get_key().keys(key, **kwargs)

    def key2address(self,key:str = None , **kwargs) -> None:
        return self.get_key().key2address(key, **kwargs)
    
    def files(self, 
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
        files =self.glob(path, **kwargs)
        if not hidden:
            files = [f for f in files if not '/.' in f]
        files = [f for f in files if not any([at in f for at in avoid_terms])]
        if search != None:
            files = [f for f in files if search in f]
        return files

    def encrypt(self,data: Union[str, bytes], key: str = None, password: str = None, **kwargs ) -> bytes:
        key = self.get_key(key) 
        return key.encrypt(data, password=password)

    def decrypt(self, data: Any,  password : str = None, key: str = None, **kwargs) -> bytes:
        key = self.get_key(key)
        return key.decrypt(data, password=password)
    
    def sign(self, data:dict  = None, key: str = None,  crypto_type='sr25519', mode='str', **kwargs) -> bool:
        key = self.get_key(key, crypto_type=crypto_type)
        return key.sign(data, mode=mode, **kwargs)

    def signtest(self, data:dict  = 'hey', key: str = None,  crypto_type='sr25519', mode='str', **kwargs) -> bool:
        signature = self.sign(data, key, crypto_type=crypto_type, mode=mode, **kwargs)
        return self.verify(data, signature, key=key, crypto_type=crypto_type, **kwargs)
    
    def size(self, module) -> int:
        return len(str(self.code_map(module)))

    def verify(self, data, signature=None, address=None,  crypto_type='sr25519',  key=None, **kwargs ) -> bool:  
        key = self.get_key(key, crypto_type=crypto_type)
        return key.verify(data=data, signature=signature, address=address, **kwargs)

    def utils(self, search=None):
        utils = self.path2fns(self.core_path + '/utils.py', tolist=True)
        if search != None:
            utils = [u for u in utils if search in u]
        return sorted(utils)
        
    def util2path(self, search=None):
        utils_paths = self.utils(search=search)
        util2path = {}
        for f in utils_paths:
            util2path[f.split('.')[-1]] = f
        return util2path

    def routes(self):
        routes = self.config()['routes']
        for util in  self.utils():
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
            config = self.dict2munch(config)
        self.config = config 
        return self.config

    def search(self, search:str = None, **kwargs):
        return self.objs(search=search, **kwargs)

    def config(self, module=None, mode='dict', fn='__init__', file_types=['json', 'yaml', 'yml']) -> 'Munch':
        # if os.path.exists(self.modules_path + '/' in module):
        path = None
        for file_type in file_types:
            if os.path.exists(f'./config.{file_type}'):
                dirpath = f'./'
            if module == None:
                dirpath = self.lib_path 
            else:
                dirpath = self.dirpath(module)
            path = os.path.join(dirpath, f'config.{file_type}')
            if os.path.exists(path):
                break
        assert path != None, f'Config file not found in {self.modules_path} or {self.lib_path}'
        filetype = path.split('.')[-1] if path != None else mode
        if os.path.exists(path):
            if filetype == 'json':
                config = json.load(open(path, 'r'))
            elif filetype in ['yaml', 'yml']:
                config = yaml.load(open(path, 'r'), Loader=yaml.FullLoader)
            else:
                raise Exception(f'Invalid config file {path}')
        else:
            module = self.module(module)
            config =  self.get_params(getattr(module, fn)) if hasattr(module, fn) else {}
        if mode == 'dict':
            pass
        elif mode == 'munch':
            from munch import Munch
            config =  Munch(config)
        else:
            raise Exception(f'Invalid mode {mode}')
        return config

    
    def dict2munch(self, d:Dict) -> 'Munch':
        from munch import Munch
        return Munch(d)
    
    def put_json(self, 
                 path:str, 
                 data:Dict, 
                 meta = None,
                 verbose: bool = False,
                 **kwargs) -> str:
        if not path.endswith('.json'):
            path = path + '.json'
        path = self.get_path(path=path)
        if isinstance(data, dict):
            data = json.dumps(data)
        self.put_text(path, data)
        return path

    def rm(self, path:str, possible_extensions = ['json'], avoid_paths = ['~', '/', './']):
        avoid_paths = list(set((avoid_paths)))
        path = self.get_path(path)
        avoid_paths = [self.get_path(p) for p in avoid_paths] 
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
    
    def glob(self, path:str='./', depth:Optional[int]=None, recursive:bool=True, files_only:bool = True,):
        import glob
        path = self.get_path(path)
        if depth != None:
            if isinstance(depth, int) and depth > 0:
                paths = []
                for path in self.ls(path):
                    if os.path.isdir(path):
                        paths += self.glob(path, depth=depth-1)
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
    
    def get_json(self, path:str,default:Any=None, **kwargs):
        path = self.get_path(path)

        # return self.util('get_json')(path, default=default, **kwargs)
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
    
    def get_path(self, 
                     path:str = None, 
                     extension:Optional[str]=None) -> str:
        '''
        Abspath except for when the path does not have a

        if you specify "abc" it will be resolved to the storage dir
        {storage_dir}/abc, in this case its ~/.commune
        leading / or ~ or . in which case it is appended to the storage dir
        '''
        storage_dir = self.storage_dir()
        if path == None :
            return storage_dir
        if path.startswith('/'):
            path = path
        elif path.startswith('~/') :
            path = os.path.expanduser(path)
        elif path.startswith('.'):
            path = os.path.abspath(path)
        else:
            if storage_dir not in path:
                path = os.path.join(storage_dir, path)
        if extension != None and not path.endswith(extension):
            path = path + '.' + extension
        return path

    def put_text(self, path:str, text:str, key=None) -> None:
        # Get the absolute path of the file
        path = self.get_path(path)
        dirpath = os.path.dirname(path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        if not isinstance(text, str):
            text = self.python2str(text)
        if key != None:
            text = self.get_key(key).encrypt(text)
        # Write the text to the file
        with open(path, 'w') as file:
            file.write(text)
        # get size
        return {'success': True, 'path': f'{path}', 'size': len(text)*8}
    
    def ls(self, path:str = './', 
           search = None,
           include_hidden = False, 
           depth=None,
           return_full_path:bool = True):
        """
        provides a list of files in the path 
        this path is relative to the module path if you dont specifcy ./ or ~/ or /
        which means its based on the module path
        """
        path = self.get_path(path)
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
 
    def put(self, 
            k: str, 
            v: Any,  
            encrypt: bool = False, 
            password: str = None, **kwargs) -> Any:
        '''
        Puts a value in the config
        '''
        encrypt = encrypt or password != None
        if encrypt or password != None:
            v = self.encrypt(v, password=password)
        data = {'data': v, 'encrypted': encrypt, 'timestamp': time.time()}    
        self.put_json(k, data)
        return {'k': k, 'encrypted': encrypt, 'timestamp': time.time()}
    
    def get(self,
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
        k = self.get_path(k)
        data = self.get_json(k, default=default, **kwargs)
        if password != None:
            assert data['encrypted'] , f'{k} is not encrypted'
            data['data'] = self.decrypt(data['data'], password=password)
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
                self.print(f'{k} is too old ({age} > {max_age})', verbose=verbose)
                return default
        
        if not full:
            if isinstance(data, dict):
                if 'data' in data:
                    data = data['data']
            if isinstance(data, str) and data.startswith('{') and data.endswith('}'):
                data = data.replace("'",'"')
                data = json.loads(data)
        return data
    
    def get_text(self, path: str, **kwargs ) -> str:
        # Get the absolute path of the file
        path = self.get_path(path)
        from commune.utils import get_text
        return get_text(path, **kwargs)

    def text(self, path: str = './', **kwargs ) -> str:
        # Get the absolute path of the file
        path = self.abspath(path)
        assert not self.home_path == path, f'Cannot read {path}'
        if os.path.isdir(path):
            return self.file2text(path)
        with open(path, 'r') as file:
            content = file.read()
        return content

    def sleep(self, period):
        time.sleep(period) 

    def fn2code(self, module=None)-> Dict[str, str]:
        module = self.resolve_module(module)
        functions = self.fns(module)
        fn_code_map = {}
        for fn in functions:
            try:
                fn_code_map[fn] = self.code(getattr(module, fn))
            except Exception as e:
                self.print(f'Error {e} {fn}', color='red')
        return fn_code_map
    
    def fn2hash(self, module=None)-> Dict[str, str]:
        module = self.resolve_module(module)   
        return {k:self.hash(v) for k,v in self.fn2code(module).items()}

    def fn_code(self,fn:str, module=None,**kwargs) -> str:
        '''
        Returns the code of a function
        '''
        fn = self.get_fn(fn)      
        return inspect.getsource(fn)       
    
    def is_generator(self, obj):
        """
        Is this shiz a generator dawg?
        """
        if isinstance(obj, str):
            if not hasattr(self, obj):
                return False
            obj = getattr(self, obj)
        if not callable(obj):
            result = inspect.isgenerator(obj)
        else:
            result =  inspect.isgeneratorfunction(obj)
        return result

    fn2cost = {}

    def fnschema(self, fn:str = '__init__', include_code=False, **kwargs)->dict:
        '''
        Get function schema of function in self
        '''     
        schema = {}
        fn_obj = self.get_fn(fn)
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
        schema['source'] = self.source(fn_obj, include_code=include_code)
        return schema
    
    def source(self, obj, include_code=True):
        """
        Get the source code of a function
        """
        if isinstance(obj, str):
            obj = self.fn(obj)
        sourcelines = inspect.getsourcelines(obj)
        source = ''.join(sourcelines[0])
        return {
                             'start': sourcelines[1], 
                             'length': len(sourcelines[0]),
                             'path': inspect.getfile(obj).replace(self.home_path, '~'),
                             'code': source if include_code else None,
                             'hash': self.hash(source),
                             'end': len(sourcelines[0]) + sourcelines[1]
                             }
    
    def schema(self, obj = None, **kwargs)->dict:
        '''
        Get function schema of function in self
        '''   
        if '/' in str(obj) or callable(obj) :
            schema = self.fnschema(obj, **kwargs)
        else:
            module = self.resolve_module(obj)
            fns = self.fns(module)
            schema = {fn: self.fnschema(getattr(module, fn)) for fn in fns}
        return schema
 
    def resolve_obj(self, obj = None, search=None, *args, **kwargs) -> Union[str, Dict[str, str]]:
        if isinstance(obj, str):
            if '/' in obj:
                obj = self.fn(obj)
            if self.module_exists(obj):
                obj = self.module(obj)
            else:
                util2path = self.util2path()
                if  obj in util2path:
                    obj = self.obj(util2path[obj])
        else:
            obj = self.resolve_module(obj)
        return obj

    def code(self, obj = None, search=None, *args, **kwargs) -> Union[str, Dict[str, str]]:
        return inspect.getsource(self.resolve_obj(obj))
    
    def codemap(self, module = None , search=None, *args, **kwargs) ->  Dict[str, str]:
        dirpath = self.dirpath(module)
        path = dirpath if self.is_module_folder(module) else self.filepath(module)
        code_map = self.file2text(path)
        code_map = {k[len(dirpath+'/'): ]:v for k,v in code_map.items()}
        return code_map

    def code_map(self, module , search=None, *args, **kwargs) ->  Dict[str, str]:
        return self.codemap(module=module, search=search,**kwargs)
    
    def code_hash_map(self, module , search=None, *args, **kwargs) ->  Dict[str, str]:
        return {k:self.hash(str(v)) for k,v in self.code_map(module=module, search=search,**kwargs).items()}

    
    def cid(self, module , search=None, *args, **kwargs) -> Union[str, Dict[str, str]]:
        return self.hash(self.code_hash_map(module=module, search=search,**kwargs))

    
    def getsource(self, module = None, search=None, *args, **kwargs) -> Union[str, Dict[str, str]]:
        if module != None:
            if isinstance(module, str) and '/' in module:
                fn = module.split('/')[-1]
                module = '/'.join(module.split('/')[:-1])
                module = getattr(self.module(module), fn)
            else:
                module = self.resolve_module(module)
        else: 
            module = self
        return inspect.getsource(module)

    def get_params(self, fn):
        """
        Gets the function defaults
        """
        fn = self.get_fn(fn)
        params = dict(inspect.signature(fn)._parameters)
        for k,v in params.items():
            if v._default != inspect._empty and  v._default != None:
                params[k] = v._default
            else:
                params[k] = None
        return params

    def dir(self, obj=None, search=None, *args, **kwargs):
        obj = self.resolve_obj(obj)
        if search != None:
            return [f for f in dir(obj) if search in f]
        return dir(obj)
    
    def fns(self, obj: Any = None,
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
        obj = self.resolve_module(obj)
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
        
    
    def clear_info_history(self):
        return self.rm('info')

    
    def resolve_info_path(self, name):
        if not isinstance(name, str):
            name = str(name)
        return self.get_path('info/' + name)

     
    def info(self, module:str='module',  # fam
            lite: bool =True, 
            max_age : Optional[int]=1000, 
            lite_features : List[str] = ['schema', 'name', 'key', 'founder', 'hash', 'time'],
            keep_last_n : int = 10,
            relative=True,
            update: bool =False, 
            key=None,
            **kwargs):
            
        path = self.resolve_info_path(module)
        info = self.get(path, None, max_age=max_age, update=update)
        if info == None:
            code = self.code_map(module)
            schema = self.schema(module)
            founder = self.founder().address
            key = self.get_key(key or module).address
            info =  {
                    'code': code, 
                    'schema': schema, 
                    'name': module, 
                    'key': key,  
                    'founder': founder, 
                    'cid': self.cid(module),
                    'time': time.time()
                    }
           
            info['signature'] = self.sign(module)

            self.put(path, info)
        if lite:
            info = {k: v for k,v in info.items() if k in lite_features}
        return  info

    def epoch(self, *args, **kwargs):
        return self.run_epoch(*args, **kwargs)

    def pwd2key(self, pwd):
        return self.module('key').str2key(pwd)

    def is_property(self, fn: 'Callable') -> bool:
        '''
        is the function a property
        '''
        fn = self.get_fn(fn)
        return isinstance(fn, property)
    
    def submit(self, 
                fn, 
                params = None,
                kwargs: dict = None, 
                args:list = None, 
                timeout:int = 40, 
                module: str = None,
                mode:str='thread',
                max_workers : int = 100,
                ):
        fn = self.get_fn(fn)
        executor = self.module('executor')(max_workers=max_workers, mode=mode) 
        return executor.submit(fn=fn, params=params, args=args, kwargs=kwargs, timeout=timeout)

    def get_fn(self, fn:str, params=None, splitter='/', default_fn='forward') -> 'Callable':
        """
        Gets the function from a string or if its an attribute 
        """
        if isinstance(fn, str):
            if fn.startswith('/'):
                fn = 'module' + fn
            elif fn.endswith('/'):
                module = self.module(fn[:-1])
                fn = default_fn
                return getattr(module, default_fn)
            fn_obj = None
            module = Module()
            if '/' in fn:
                module = self.module('/'.join(fn.split('/')[:-1]))()
                fn = fn.split('/')[-1]
            if hasattr(module, fn):
                fn_obj = getattr(module, fn)
            elif self.object_exists(fn):
                fn_obj =  self.obj(fn)
            else:
                raise Exception(f'{fn} is not a function or object')
        elif callable(fn):
            fn_obj = fn
        else:
            raise Exception(f'{fn} is not a function or object')
        if params != None:
            return fn_obj(**params)
        return fn_obj
    
    def fn(self, fn:str,  params=None, splitter='/', default_fn='forward') -> 'Callable':
        return self.get_fn(fn, params=params, splitter=splitter, default_fn=default_fn)

    def get_args(self, fn) -> List[str]:
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

    def client(self, *args, **kwargs) -> 'Client':
        return self.module('client')().client(*args, **kwargs)
    
    def classes(self, path='./',  **kwargs):
        return  self.path2classes(path=path, tolist=True, **kwargs)

    def password(self, max_age=None, update=False, **kwargs):
        path = self.get_path('password')
        pwd = self.get(path, None, max_age=max_age, update=update,**kwargs)
        if pwd == None:
            pwd = self.hash(self.mnemonic() + str(time.time()))
            self.put(path, pwd)
            self.print('Generating new password', color='blue')
        return pwd

    def mnemonic(self, words=24):

        if words not in [12, 15, 18, 21, 24]:
            if words > 24 : 
                # tile to over 24
                tiles = words // 24 + 1
                mnemonic_tiles = [self.mnemonic(24) for _ in range(tiles)]
                mnemonic = ' '.join(mnemonic_tiles)
            if words < 24:
                # tile to under 12
                mnemonic = self.mnemonic(24)
            return ' '.join(mnemonic.split()[:words])

        mnemonic =  self.mod('key')().generate_mnemonic(words=words)
        return mnemonic

    def path2relative(self, path='./'):
        path = self.get_path(path)
        pwd = os.getcwd()
        home_path = self.home_path
        prefixe2replacement = {pwd: './', home_path: '~/'}
        for pre, rep in prefixe2replacement.items():
            if path.startswith(pre):
                path = path[len(pre):]
                path = rep + path[len(pre):]
        return path
    
    def path2objectpath(self, path:str, **kwargs) -> str:
        path = os.path.abspath(path)
        dir_prefixes  = [os.getcwd(), self.lib_path, self.home_path]
        for dir_prefix in dir_prefixes:
            if path.startswith(dir_prefix):
                path =   path[len(dir_prefix) + 1:].replace('/', '.')
                break
        if path.endswith('.py'):
            path = path[:-3]
        return path.replace('__init__.', '.')
        
    def path2name(self, path, ignore_folder_names = ['modules', 'agents', 'src', 'mods']):
        name = self.path2objectpath(path)
        name_chunks = []
        for chunk in name.split('.'):
            if chunk in ignore_folder_names:
                continue
            if chunk not in name_chunks:
                name_chunks += [chunk]
        if name_chunks[0] == self.repo_name:
            name_chunks = name_chunks[1:]
        return '.'.join(name_chunks)
    
    def path2classes(self, path='./',
                     class_prefix = 'class ', 
                     file_extension = '.py',
                     tolist = False,
                     depth=4,
                     relative=False,
                     class_suffix = ':', **kwargs) :

        """

        Get the classes for each path inside the path variable

        Args:
        """
        path = self.abspath(path)
        path2classes = {}
        if os.path.isdir(path) and depth > 0:
            for p in self.ls(path):
                try:
                    for k,v in self.path2classes(p, depth=depth-1).items():
                        if len(v) > 0:
                            path2classes[k] = v
                except Exception as e:
                    pass
        elif os.path.isfile(path) and path.endswith('.py'):
            code = self.get_text(path)
            classes = []
            file_path = self.path2objectpath(path)
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
                path = self.path2relative(path)
            path2classes =  {path:  [file_path + '.' + cl for cl in classes]}
        if tolist: 
            classes = []
            for k,v in path2classes.items():
                classes.extend(v)
            return classes
   
        return path2classes

    def path2fns(self, path = './', tolist=False, **kwargs):
        fns = []
        path = os.path.abspath(path)
        if os.path.isdir(path):
            path2fns = {}
            for p in self.glob(path+'/**/**.py', recursive=True):
                for k,v in self.path2fns(p, tolist=False).items():
                    if len(v) > 0:
                        path2fns[k] = v
        else:
            code = self.get_text(path)
            path_prefix = self.path2objectpath(path)
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

    def objs(self, path:str = './', depth=10, search=None, **kwargs) -> List[str]:
        classes = self.classes(path,depth=depth)
        functions = self.path2fns(path, tolist=True)
        objs = functions + classes
        if search != None:
            objs = [f for f in objs if search in f]
        return objs


    def obj(self, key:str, splitters=['/', '::', '.'], **kwargs)-> Any:
        if not hasattr(self, 'obj_cache'): 
            self.obj_cache = {}
        if key in self.obj_cache:
            return self.obj_cache[key]
        from commune.utils import import_object
        if (self.repo_name + '.' + self.repo_name) in key:
            key = key.replace((self.repo_name + '.' + self.repo_name) ,self.repo_name)
        obj =  import_object(key, splitters=splitters, **kwargs)
        self.obj_cache[key] = obj
        return obj
    
    def object_exists(self, path:str, verbose=False)-> Any:

        # better way to check if an object exists?

        try:
            self.obj(path, verbose=verbose)
            return True
        except Exception as e:
            return False

    def m(self):
        """enter modules path in vscode"""
        return self.cmd(f'code {self.modules_path}')
    
    def module_exists(self, module:str, **kwargs) -> bool:
        '''
        Returns true if the module exists
        '''
        try:
            tree = self.tree()
            module_exists =  module in tree
        except Exception as e:
            module_exists =  False
        return module_exists
    
    def objectpath2name(self, 
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
        if path.startswith(self.repo_name + '.'):
            path = path[len(self.repo_name)+1:]
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
    
    def local_tree(self , **kwargs):
        return self.get_tree(os.getcwd(), **kwargs)

    def lib_tree(self,**kwargs):
        return self.get_tree(self.lib_path,  **kwargs)
    
    def get_tree(self, path='./', depth = 10, max_age=60, update=False, **kwargs):
        """
        Get the tree of the modules in the path
        a tree is a dictionary of the form {module_name: module_path}
        the module_name is based on the directory path 
        """
        path = self.abspath(path)
        path_hash = self.hash(path)
        tree_cache_path = 'tree/'+self.hash(os.path.abspath(path))
        tree = self.get(tree_cache_path, None, max_age=max_age, update=update)
        if tree == None:
            class_paths = self.classes(path, depth=depth)
            simple_paths = [self.objectpath2name(p) for p in class_paths]
            tree = dict(zip(simple_paths, class_paths))
            self.put(tree_cache_path, tree)
        return tree
    
    def tree(self, search=None,  max_age=60,update=False, **kwargs):
        local_tree = self.local_tree(update=update, max_age=max_age)
        lib_tree = self.lib_tree(update=update, max_age=max_age)
        tree = {**local_tree, **lib_tree }
        if search != None:
            tree = {k:v for k,v in tree.items() if search in k}
        return tree
    
    def get_modules(self, search=None, **kwargs):
        return list(self.tree(search=search, **kwargs).keys())

    def modules(self, search=None, cache=True, max_age=60, update=False, **extra_kwargs)-> List[str]:  
        return self.get_modules(search=search, cache=cache, max_age=max_age, update=update, **extra_kwargs)
    
    def mods(self, search=None, cache=True, max_age=60, update=False, **extra_kwargs)-> List[str]:   
        return self.modules(search=search, cache=cache, max_age=max_age, update=update, **extra_kwargs)

    def check_info(self,info, features=['key', 'hash', 'time', 'founder', 'name', 'schema']):
        try:
            assert isinstance(info, dict), 'info is not a dictionary'
            for feature in features:
                assert feature in info, f'{feature} not in info'
        except Exception as e:
            return False
        return True
    
    def new( self,
                   path : str ,
                   name= None, 
                   base_module : str = 'base', 
                   update=0
                   ):
        name = name or path
        if name.endswith('.git'):
            git_path = self.giturl(path)
            name =  path.split('/')[-1].replace('.git', '')
        dirpath = os.path.abspath(self.modules_path +'/'+ name.replace('.', '/'))
        filename = name.replace('.', '_') + '.py'
        path = f'{dirpath}/{filename}'
        # path = dirpath + '/' + module_name + '.py'
        base_module_obj = self.module(base_module)
        code = self.code(base_module)
        code = code.replace(base_module_obj.__name__, ''.join([m[0].capitalize() + m[1:] for m in name.split('.')]))
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        self.put_text(path, 'import commune as c \n'+code)
        return {'name': name, 'path': path, 'msg': 'Module Created'}
    
    add_module = new_module = new
    
    def util(self, util:str, prefix='commune.utils'):
        return self.obj(self.util2path().get(util))

    def founder(self):
        return self.get_key()

    def repo2path(self, search=None):
        repo2path = {}
        for p in self.ls('~/'): 
            if os.path.exists(p+'/.git'):
                r = p.split('/')[-1]
                if search == None or search in r:
                    repo2path[r] = p
        return dict(sorted(repo2path.items(), key=lambda x: x[0]))

    def repos(self, search=None):
        return list(self.repo2path(search=search).keys())

    def chat(self, *args, module=None, **kwargs):
        if module != None:
            args = [self.code(module)] + list(args)
        return self.module("agent")().ask(*args, **kwargs) 
    
    def ask(self, *args, module=None, path='./' , **kwargs):
        if module != None:
            args = [self.code(module)] + list(args)
        return self.module("agent")().ask(*args, **kwargs) 

    def ai(self, *args, **kwargs):
        return self.module("agent")().ask(*args, **kwargs) 
    
    def clone(self, repo:str = 'commune-ai/commune', path:str=None, **kwargs):
        gitprefix = 'https://github.com/'
        repo = gitprefix + repo if not repo.startswith(gitprefix) else repo
        path = os.path.abspath(os.path.expanduser(path or  '~/'+repo.split('/')[-1]))
        cmd =  f'git clone {repo} {path}'
        self.cmd(cmd, verbose=True)
        return {'path': path, 'repo': repo, 'msg': 'Repo Cloned'}

    def test_fns(self, module=None):
        return [f for f in dir(self.module(module)) if f.startswith('test_') or f == 'test']

    def test(self, module=None, timeout=50, modules=[ 'server', 'vali','key', 'chain']):
        
        if module == None:
            test_results ={}
            for m in modules:
                test_results[m] = self.test(m, timeout=timeout)
            return test_results
        elif self.module_exists(module + '.test'):
            module = module + '.test'

        module_obj = self.module(module)()
        fn2result = {}
        for i, fn in enumerate(self.test_fns(module)):
            fn_path = f'{module}/{fn}'
            buffer = 5 * '*-' 
            emoji = '⏳'
            title = 'TEST'
            self.print(f'{buffer}{emoji}\{title}({fn_path})\t{emoji}{buffer}', color='yellow')
            try:
                fn2result[fn] = getattr(module_obj, fn)()
            except Exception as e:
                e = self.detailed_error(e)
                return {'fn': fn, 'error': e}
            title = 'PASS'
            emoji = '✅'
            self.print(f'{buffer}{emoji}\{title}({fn_path})\t{emoji}{buffer}', color='green')
        return fn2result

    def test_module(self, module='module', timeout=50):
        """
        Test the module
        """

        if self.module_exists(module + '.test'):
            module = module + '.test'

        if module == 'module':
            module = 'test'
        Test = self.module(module)
        test_fns = [f for f in dir(Test) if f.startswith('test_') or f == 'test']
        test = Test()
        futures = []
        for fn in test_fns:
            print(f'Testing({fn})')
            future = self.submit(getattr(test, fn), timeout=timeout)
            futures += [future]
        results = []
        for future in self.as_completed(futures, timeout=timeout):
            print(future.result())
            results += [future.result()]
        return results

    testmod = test_module
    
    def readmes(self,  path='./', search=None, avoid_terms=['/modules/']):
        files =  self.files(path)
        files = [f for f in files if f.endswith('.md')]
        files = [f for f in files if all([avoid not in f for avoid in avoid_terms])]
        if search != None:
            files = [f for f in files if search in f]
        return files
    config_name_options = ['config', 'cfg', 'module', 'block',  'agent', 'mod', 'bloc']

    def configs( path='./', modes=['yaml', 'json'], search=None, names=['config', 'cfg', 'module', 'block',  'agent', 'mod', 'bloc']):
        """
        Returns a list of config files in the path
        """
        def is_config(f):
            name_options = self.config_name_options
            return any(f.endswith(f'{name}.{m}') for name in names for m in modes)
        configs =  [f for f in  self.files(path) if is_config(f)]
        if search != None:
            configs = [f for f in configs if search in f]
        return configs

    def app(self,
           module:str = 'agent', 
           name : Optional[str] = None,
           port:int=None):
        module = self.shortcuts.get(module, module)
        name = name or module
        port = port or self.free_port()
        if self.module_exists(module + '.app'):
            module = module + '.app'
        module_class = self.module(module)
        return self.cmd(f'streamlit run {module_class.filepath()} --server.port {port}')
    
    def sync_routes(self, routes:dict=None, verbose=False):

        """
        This ties other modules into the current module.
        The way it works is that it takes the module name and the function name and creates a partial function that is bound to the module.
        This allows you to call the function as if it were a method of the current module.
        for example
        """
        routes = self.routes()
        t0 = time.time()
        # WARNING : THE PLACE HOLDERS MUST NOT INTERFERE WITH THE KWARGS OTHERWISE IT WILL CAUSE A BUG IF THE KWARGS ARE THE SAME AS THE PLACEHOLDERS
        # THE PLACEHOLDERS ARE NAMED AS module_ph and fn_ph AND WILL UNLIKELY INTERFERE WITH THE KWARGS
        def fn_generator(*args, route, **kwargs):
            
            def fn_wrapper(*args, **kwargs):
                try:
                    fn_obj = self.obj(route)
                except Exception as e:
                    if '/' in route:
                        module = '/'.join(route.split('/')[:-1])
                        fn = route.split('/')[-1]
                    module = self.module(module)
                    fn_obj = getattr(module, fn)
                    fn_args = self.get_args(fn_obj)
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
                if hasattr(self, to_fn):
                    if verbose:
                        print(f'Warning: {to_fn} already exists')
                else:
                    fn_obj = partial(fn_generator, route=f'{module}/{fn}') 
                    fn_obj.__name__ = to_fn
                    setattr(self, to_fn, fn_obj)
        duration = time.time() - t0
        return {'success': True, 'msg': 'enabled routes', 'duration': duration}

    def giturl(self, url:str='commune-ai/commune'):
        gitprefix = 'https://github.com/'
        gitsuffix = '.git'
        if not url.startswith(gitprefix):
            url = gitprefix + url
        if not url.endswith(gitsuffix):
            url = url + gitsuffix
        return url

    def sync_module(self, url, max_age=10000, update=False):
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
        modules_flag = self.get(modules_flag_path, max_age=max_age, update=update)
        if modules_flag != None:
            return True
        url = self.giturl(url)
        if name in ['modules', 'mods']:
            module_path = self.modules_path
        else:
            module_path = self.modules_path + '/' + name.replace('.','/')
        if not os.path.exists(module_path):
            os.makedirs(module_path, exist_ok=True)
        if os.path.exists(module_path+'/.git'):
            cmd = f'git pull {url} {module_path}' 
        else:
            cmd = f'git clone {url} {module_path}'
        self.cmd(cmd, cwd=module_path)
        self.put(modules_flag_path, True)
        return True

    def isref(self, module='datura', expected_features = ['api', 'app', 'code'], suffix_options = ['_url', 'url']):
        try:
            module = self.module(module)
            filtered_features = []
            for feature in dir(module):
                feature_options = [f'{feature}{suffix}' for suffix in suffix_options]
                for feature_option in feature_options:
                    if hasattr(module, feature_option):
                        feature_obj = getattr(module, feature_option)
                        if feature.startswith('_'):
                            continue
                        if callable(feature_obj):
                            continue
                        if feature in expected_features:
                            filtered_features += [feature]

                feature_obj = getattr(module, feature)
                if feature.startswith('_'):
                    continue
                if callable(feature_obj):
                    continue
                if feature in expected_features:
                    filtered_features += [feature]
        except Exception as e:
            self.print(e)
            return False
        return len(filtered_features) > 0


    def exref(self, module:str = 'datura', expected_features = ['api', 'app', 'code']):
        dirpath = self.dirpath(module)
        module = self.module(module)
        code_link = module.code
        if not code_link.startswith('https://'):
            code_link = self.giturl(code_link)
        code_link = code_link.replace('.git', '')
        cmd = f'git clone {code_link} {dirpath}'
        cmds = [f'rm -rf {dirpath}', f'git clone {code_link} {dirpath}']
        if input(f'Are you sure you want to run {cmds}') == 'y':
            for cmd in cmds:
                self.cmd(cmd, cwd=dirpath)
        else:
            self.print('Aborting')
            return False
    
    def refs(self, module:str = 'datura', expected_features = ['api', 'app', 'code']):
        modules = self.modules()
        filtered_modules = []
        for module in modules:
            isref = self.isref(module, expected_features=expected_features)
            if isref:
                filtered_modules += [module]
        return filtered_modules

    def push(self, module):
        modules_path = self.modules_path
        module_path = modules_path + '/' + module.replace('.', '/')
        if not os.path.exists(module_path):
            raise Exception(f'Module {module} does not exist')
        

    def git_info(self, path:str = None, name:str = None, n=10):
        return c.fn('git/git_info', {'path': path, 'name': name, 'n': n})
    
    def sync_modules(self, max_age=10, update=False):
        results = []
        synced_modules = self.get('synced_modules', max_age=max_age, update=update)
        if synced_modules != None:
            return synced_modules
        
        futures = []
        for url in self.config()['modules']:
            params = {'url': url, 'max_age': max_age, 'update': update}
            futures += [self.submit(self.sync_module, params)]
        
        results = []
        for future in self.as_completed(results):
            results.append(self.sync_module(url, max_age=max_age, update=update))

        return results

    def isrepo(self, module:str = None):
        path = self.dirpath(module)
        return os.path.exists(path + '/.git')

    def diff(self, module:str = 'model.openai', name:str = None):
        dirpath = self.dirpath(module)
        cmd = 'git diff'
        cwd = dirpath
        return self.cmd(cmd, cwd=cwd)

    def push(self, module:str = 'model.openai', name:str = None):
        """
        Push the module to the git repository
        """
        name = name or module
        if not os.path.exists(module):
            self.cmd(f'git clone {module} {name}')
        else:
            self.cmd(f'git pull {module} {name}')
        return {'success': True, 'msg': 'pushed module'}

    def add_globals(self, globals_input:dict = None):
        """
        add the functions and classes of the module to the global namespace
        """
        from functools import partial
        globals_input = globals_input or {}
        for k,v in self.__dict__.items():
            globals_input[k] = v     
        for f in self.fns(Module, mode='self'):
            def wrapper_fn(f, *args, **kwargs):
                fn = getattr(Module(), f)
                return fn(*args, **kwargs)
            globals_input[f] = partial(wrapper_fn, f)
        return globals_input

    def sync(self,  globals_input=None, max_age=10, update=True, **kwargs):
        """
        Initialize the module by sycing with the config
        """
        # assume the name of this module is the name of .../
        self.repo_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
        self.storage_path = os.path.expanduser(f'~/.{self.repo_name}')
        self.core_path  = self.corepath = os.path.dirname(__file__)
        self.repo_path  = self.repopath = os.path.dirname(os.path.dirname(__file__)) # the path to the repo
        self.lib_path  = self.libpath = os.path.dirname(os.path.dirname(__file__)) # the path to the library
        self.home_path = self.homepath  = os.path.expanduser('~') # the home path
        self.modules_path = self.modspath = self.core_path + '/modules'
        self.app_path = self.core_path + '/app'
        self.tests_path = f'{self.lib_path}/tests'
        if not hasattr(Module, 'included_pwd_in_path'):
            self.included_pwd_in_path = False
        if  not self.included_pwd_in_path:
            paths = [self.modules_path, os.getcwd()]
            for p in paths:
                if not p in sys.path:
                    sys.path.append(p)
            self.included_pwd_in_path = True

        # config attributes
        config = self.config()
        self.core = config['core'] # the core modules
        self.repo_name  = config['name'] # the name of the library
        self.endpoints = config['endpoints']
        self.core_features = config['core_features']
        self.port_range = config['port_range'] # the port range between 50050 and 50150
        self.shortcuts =  self.shortys = config["shortcuts"]
        self.sync_routes()
        self.sync_modules(max_age=max_age, update=update)

        if globals_input != None:
            globals_input = self.add_globals(globals_input)

        return {'success': True, 'msg': 'synced config'}
        

    def main(self,
                fn='vs',  
                module='module', 
                default_fn = 'forward'):
        t0 = time.time()
        argv = sys.argv[1:]
        # ---- FUNCTION
        module = self.module(module)()
        if len(argv) == 0:
            argv += [fn]

        fn = argv.pop(0)

        if hasattr(module, fn):
            fn_obj = getattr(module, fn)
        elif '/' in fn:
            if fn.startswith('/'):
                fn = fn[1:]
            if fn.endswith('/'):
                fn = fn + default_fn
            new_module = '/'.join(fn.split('/')[:-1]).replace('/', '.')
            module =  self.module(new_module)()
            fn = fn.split('/')[-1]
            fn_obj = getattr(module, fn)

        else:
            raise Exception(f'Function {fn} not found in module {module}')
        # ---- PARAMS ----
        params = {'args': [], 'kwargs': {}} 
        parsing_kwargs = False
        if len(argv) > 0:
            for arg in argv:
                if '=' in arg:
                    parsing_kwargs = True
                    key, value = arg.split('=')
                    params['kwargs'][key] = self.str2python(value)
                else:
                    assert parsing_kwargs is False, 'Cannot mix positional and keyword arguments'
                    params['args'].append(self.str2python(arg))        
        # run thefunction
        result = fn_obj(*params['args'], **params['kwargs']) if callable(fn_obj) else fn_obj
        speed = time.time() - t0
        module_name = module.__class__.__name__
        self.print(f'Call({module_name}/{fn}, speed={speed:.2f}s)')
        duration = time.time() - t0
        is_generator = self.is_generator(result)
        if is_generator:
            for item in result:
                if isinstance(item, dict):
                    self.print(item)
                else:
                    self.print(item, end='')
        else:
            self.print(result)

    

c = Module()
if __name__ == "__main__":
    Module().run()


