
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
                module: str = 'module', 
                params: dict = None,  
                cache=True, 
                verbose=False, 
                trials = 1,
                **kwargs) -> str:
        # Load the module
        module = module or 'module'
        # Try to load the module
        if module in ['module']:
            return Module
        # Normalize path
        if not isinstance(module, str):
            return module
        module = module.replace('/', '.')
        module = self.shortcuts.get(module, module)
        if not hasattr(self, 'module_cache'):
            self.module_cache = {}
        if module in self.module_cache:
            return self.module_cache[module]
        else:
            tree = self.tree()
            obj_path = tree.get(module, module)
            obj = self.obj(obj_path)
            self.module_cache[module] = obj
        # Apply parameters if provided
        if params != None:
            if isinstance(params, dict):
                obj = obj(**params)
            elif isinstance(params, list):
                obj = obj(*params)
        return obj

    mod = get_module = module
    def forward(self, fn:str='info', params:dict=None, signature=None) -> Any:
        params = params or {}
        assert fn in self.endpoints, f'{fn} not in {self.endpoints}'
        if hasattr(self, fn):
            fn_obj = getattr(self, fn)
        else:
            fn_obj = self.fn(fn)
        return fn_obj(**params)

    def go(self, module=None, **kwargs):
        module = module or 'module'

        path = f"{self.modules_path}/{module}"
        repo2path = self.repo2path()
        if os.path.exists(path):
            path = self.abspath(path)
        if self.module_exists(module):
            path = self.dirpath(module)
        elif module in repo2path:
            path = repo2path[module]
        elif os.path.exists(module):
            path = module
        else:
            raise Exception(f'{module} not found')
        assert os.path.exists(path), f'{path} does not exist'
        return self.cmd(f'code {path}', **kwargs)

    def g(self, module=None, **kwargs):
        """
        go the file
        """
        return self.go(module=module, **kwargs)


    def gof(self, module=None, **kwargs):
        """
        go the file
        """
        try:
            path = self.filepath(module)
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
        if dirpath.endswith('src'):
            dirpath = '/'.join(dirpath.split('/')[:-1])
        if '/src' in dirpath:
            dirpath = dirpath.split('/src')[0]
        return dirpath
    
    def module_name(self, obj=None):
        obj = obj or Module
        if  isinstance(obj, str):
            obj = self.module(obj)
        module_file =  inspect.getfile(obj)
        return self.path2name(module_file)

    def vs(self, path = None):
        path = path or __file__
        path = os.path.abspath(path)
        return self.cmd(f'code {path}')
        
    def co(self, path = None):
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

        return os.getcwd()

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
    
    is_file_module = is_module_file

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
        if self.module_exists(path):
            path = self.dirpath(path)
        files =self.glob(path, **kwargs)
        if not hidden:
            files = [f for f in files if not '/.' in f]
        files = [f for f in files if not any([at in f for at in avoid_terms])]
        if search != None:
            files = [f for f in files if search in f]
        return files

    def encrypt(self,data: Union[str, bytes], key: str = None, password: str = None, **kwargs ) -> bytes:
        return self.get_key(key).encrypt(data, password=password)
    def decrypt(self, data: Any,  password : str = None, key: str = None, **kwargs) -> bytes:
        return self.get_key(key).decrypt(data, password=password)

    def encrypt_test(self, data: Union[str, bytes], key: str = None, password: str = None, **kwargs) -> bool:
        encrypted = self.encrypt(data, key=key, password=password, **kwargs)
        decrypted = self.decrypt(encrypted, key=key, password=password, **kwargs)
        return data == decrypted
        
    def sign(self, data:dict  = None, key: str = None,  crypto_type='sr25519', mode='str', **kwargs) -> bool:
        return self.get_key(key, crypto_type=crypto_type).sign(data, mode=mode, **kwargs)

    def signtest(self, data:dict  = 'hey', key: str = None,  crypto_type='sr25519', mode='str', **kwargs) -> bool:
        signature = self.sign(data, key, crypto_type=crypto_type, mode=mode, **kwargs)
        return self.verify(data, signature, key=key, crypto_type=crypto_type, **kwargs)
    
    def size(self, module) -> int:
        return len(str(self.code_map(module)))

    def verify(self, data, signature=None, address=None,  crypto_type='sr25519',  key=None, **kwargs ) -> bool:  
        key = self.get_key(key, crypto_type=crypto_type)
        return key.verify(data=data, signature=signature, address=address, **kwargs)

    def get_utils(self, search=None):
        utils = self.path2fns(self.root_path + '/utils.py', tolist=True)
        if search != None:
            utils = [u for u in utils if search in u]
        return sorted(utils)
        
    def util2path(self, search=None):
        utils_paths = self.get_utils(search=search)
        util2path = {}
        for f in utils_paths:
            util2path[f.split('.')[-1]] = f
        return util2path

    def routes(self):
        routes = self.config['routes']
        for util in  self.get_utils():
            k = '.'.join(util.split('.')[:-1])
            v = util.split('.')[-1]
            routes[k] = routes.get(k , [])
            routes[k].append(v)
        return routes

    # def servers(self, *args, **kwargs) ->  Dict[str, str]:
    #     return self.fn('server/servers')(*args, **kwargs)

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

    def get_config(self, module=None, mode='dict', fn='__init__', file_types=['json', 'yaml', 'yml']) -> 'Munch':
        """
        check if there is a config 
        """
        path = None
        dirpath_options = [ self.lib_path , self.root_path,  self.pwd()]
        path_options = [os.path.join(dp, f'config.{file_type}') for dp in dirpath_options for file_type in file_types]
        for p in path_options:
            if os.path.exists(p):
                path = p
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
        
        # parse output
        if mode == 'dict':
            pass
        elif mode == 'munch':
            from munch import Munch
            config =  Munch(config)
        else:
            raise Exception(f'Invalid mode {mode}')
        return config

    config = get_config

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
        return self.put_json(k, data)
    
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
                if verbose:
                    print(f'TooOld(path={k} age={age} max_age={max_age})')
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

    def fn_schema(self, fn:str = '__init__', code=False, **kwargs)->dict:
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
        schema['source'] = self.source(fn_obj, code=code)
        return schema

    fnschema = fn_schema
    def source(self, obj, code=True):
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
                             'code': source if code else None,
                             'hash': self.hash(source),
                             'end': len(sourcelines[0]) + sourcelines[1]
                             }
    
    def schema(self, obj = None, verbose=False, **kwargs)->dict:
        '''
        Get function schema of function in self
        '''   
        if '/' in str(obj) or callable(obj) :
            schema = self.fn_schema(obj,  **kwargs)
        elif isinstance(obj, str) and hasattr(self, obj):
            obj = getattr(self, obj)
            schema = self.fn_schema(obj, **kwargs)
        else:
            module = self.resolve_module(obj)
            fns = self.fns(module)
            schema = {}
            for fn in fns:
                try:
                    schema[fn] = self.fn_schema(getattr(module, fn), **kwargs)
                except Exception as e:
                    self.print(f'Error {e} {fn}', color='red', verbose=verbose)
                
        return schema
 
    def get_obj(self, obj = None, search=None, *args, **kwargs) -> Union[str, Dict[str, str]]:
        if isinstance(obj, str):

            if '/' in obj:
                obj = self.fn(obj)
            elif self.module_exists(obj):
                obj = self.module(obj)
            elif hasattr(self, obj):
                obj = getattr(self, obj)
            else:
                util2path = self.util2path()
                if  obj in util2path:
                    obj = self.obj(util2path[obj])
        else:
            obj = self.resolve_module(obj)
        return obj

    def code(self, obj = None, search=None, full=False,  *args, **kwargs) -> Union[str, Dict[str, str]]:
        if full:
            return self.code_map(obj, search=search)
        return  inspect.getsource(self.obj(obj))
    
    def code_map(self, module = None , search=None, ignore_folders = ['modules'], *args, **kwargs) ->  Dict[str, str]:
        dirpath = self.dirpath(module)
        path = dirpath if self.is_module_folder(module) else self.filepath(module)
        code_map = self.file2text(path)
        code_map = {k[len(dirpath+'/'): ]:v for k,v in code_map.items()}
        # ignore if .modules. is in the path
        code_map = {k:v for k,v in code_map.items() if not any(['/'+f+'/' in k for f in ignore_folders])}
        
        return code_map

    codemap = code_map

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
        obj = self.obj(obj)
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


    def module2fns(self,max_age=30, update=False, core=True) -> List[str]:
        module2schema = self.module2schema(max_age=max_age, update=update, core=core)
        module2fns = {}
        for module in module2schema:
            module2fns[module] = list(module2schema[module].keys())
        return module2fns
    
    def fn2module(self, max_age=30, update=False, core=True) -> List[str]:
        module2fns = self.module2fns(max_age=max_age, update=update, core=core)
        fn2module = {}
        for module in module2fns:
            for fn in module2fns[module]:
                fn2module[fn] = module
        return fn2module

    def module2schema(self, module=None, max_age=30, update=False, core=False) -> List[str]:
        module2schema = self.get('module2schema', default=None, max_age=max_age, update=update)
        if module2schema == None:
            modules = self.core_modules if core else self.modules()
            module2schema = {}
            for module in modules:
                module2schema[module] = self.schema(module)
            self.put('module2schema', module2schema)
        return module2schema 

    def info(self, module:str='module',  # fam
            max_age : Optional[int]=1000, 
            keep_last_n : int = 10,
            relative=True,
            update: bool =False, 
            key=None,
            **kwargs):
            
        path = self.resolve_info_path(module)
        info = self.get(path, None, max_age=max_age, update=update)
        if info == None:
            info =  {
                    'schema': self.schema(module), 
                    'name': module, 
                    'key': self.get_key(key or module).key_address,  
                    'founder': self.founder().address, 
                    'cid': self.cid(module),
                    'time': time.time()
                    }
            info['signature'] = self.sign(info)
            self.put(path, info)
            assert self.verify_info(info), f'Invalid signature {info["signature"]}'
        return  info

    def verify_info(self, info=None, **kwargs) -> bool:
        info = self.copy(info)
        if isinstance(info, str):
            info = self.info(info, **kwargs)
        signature = info.pop('signature')
        verify = self.verify(data=info, signature=signature)  
        assert verify, f'Invalid signature {signature}'
        return True

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
        return self.module('executor')(max_workers=max_workers, mode=mode).submit(fn=self.get_fn(fn), params=params, args=args, kwargs=kwargs, timeout=timeout)

    def get_fn(self, fn:Union[callable, str], params:str=None, splitter='/', default_fn='forward', default_module = 'module') -> 'Callable':
        """
        Gets the function from a string or if its an attribute 
        """
        if isinstance(fn, str):
            if hasattr(self, fn):
                fn_obj = getattr(self, fn)
            elif fn.startswith('/'):
                fn_obj = getattr(self.module(default_module)(), fn[1:])
            elif fn.endswith('/'):
                fn_obj = getattr( self.module(fn[:-1])(), default_fn)
            elif '/' in fn:
                module, fn = fn.split('/')
                module = self.module(module)()
                if hasattr(module, fn):
                    fn_obj = getattr(module, fn)
                else:
                    raise Exception(f'Function {fn} not found in {module}')
            elif self.object_exists(fn):
                fn_obj =  self.obj(fn)
        elif callable(fn):
            fn_obj = fn
        else:
            raise Exception(f'Invalid function {fn}')
        if params:
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

    def how(self, module, query, *extra_query) : 
        code = self.code(module)
        query = ' '.join([query, *extra_query])
        return self.fn('model.openrouter/')(f'query={query} code={code}')


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

    def path2fns(self, path = './', tolist=False,**kwargs):
        path2fns = {}
        fns = []
        path = os.path.abspath(path)
        if os.path.isdir(path):
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


    def obj(self, key:str, **kwargs)-> Any:
        if not hasattr(self, 'included_pwd_in_path'):
            for p in [self.modules_path, os.getcwd()]:
                if not p in sys.path:
                    sys.path.append(p)
            self.included_pwd_in_path = True
        if not hasattr(self, 'obj_cache'): 
            self.obj_cache = {}
        if key in self.obj_cache:
            return self.obj_cache[key]
        else:
            from commune.utils import import_object
            obj =  import_object(key, **kwargs)
            self.obj_cache[key] = obj
        return obj

    get_obj = obj
    
    def obj_exists(self, path:str, verbose=False)-> Any:

        # better way to check if an object exists?

        try:
            self.obj(path, verbose=verbose)
            return True
        except Exception as e:
            return False

    def object_exists(self, path:str, verbose=False)-> Any:

        # better way to check if an object exists?

        return self.obj_exists(path, verbose=verbose)

    def m(self):
        """enter modules path in vscode"""
        return self.cmd(f'code {self.modules_path}')
    
    def module_exists(self, module:str, **kwargs) -> bool:
        '''
        Returns true if the module exists
        '''
        try:
            module = self.module(module, **kwargs)
            module_exists = True
        except Exception as e:
            module_exists =  False
        return module_exists
    
    def objectpath2name(self, 
                        p:str,
                        avoid_terms=['modules', 'agents', 'module', '_modules', '_agents', 'core', 'src'],
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
            avoid_right = f'{avoid}.'
            avoid_left = f'.{avoid}'
            if avoid_right in path:
                path = path.replace(avoid_right, '')
            
            elif avoid_left in path:
                path = path.replace(avoid_left, '')
        if len(path) == 0:
            return 'module'
        for avoid in avoid_suffixes:
            if path.endswith(avoid) and path != avoid:
                path = path[:-(1+len(avoid))]
        return path
    
    def local_tree(self , **kwargs):
        return self.get_tree(os.getcwd(), **kwargs)

    def locals(self, **kwargs):
        return list(self.get_tree(self.pwd(), **kwargs).keys())

    def core_tree(self, **kwargs):
        return {**self.get_tree(self.core_path,  **kwargs)}

    def modules_tree(self, **kwargs):
        return self.get_tree(self.modules_path, depth=10,  **kwargs)
    
    def tree(self, search=None,  max_age=60,update=False, **kwargs):
        local_tree = self.local_tree(update=update, max_age=max_age)
        core_tree = self.core_tree(update=update, max_age=max_age)
        modules_tree = self.modules_tree(update=update, max_age=max_age)
        tree = { **modules_tree, **local_tree,  **core_tree }
        if search != None:
            tree = {k:v for k,v in tree.items() if search in k}
        return tree

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
            filter_path = lambda p: p.replace('src.', '') if p.startswith('src.' + self.repo_name) else p
            class_paths = [filter_path(p) for p in class_paths]
            simple_paths = [self.objectpath2name(p) for p in class_paths]
            tree = dict(zip(simple_paths, class_paths))
            self.put(tree_cache_path, tree)
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
    
    def new( self, name= None, base_module : str = 'base', update=0):
        """
        make a new module
        """
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
        self.put_text(path, 'import commune as c \n'+code)
        return {'name': name, 'path': path, 'msg': 'Module Created'}
    
    add_module = new_module = new
    
    def util(self, util:str, prefix='commune.utils'):
        return self.obj(self.util2path().get(util))

    def up(self, image = 'commune'):
        return self.cmd('make up', cwd=self.lib_path)

    def enter(self, image = 'commune'):
        import os
        return os.system('docker exec -it commune bash')


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

    def help(self, fn='help',  query:str = 'what is this', module=None, **kwargs):
        return self.module("agent")().ask(f'given {self.code(fn)} what is the answer to the question {query}')
    
    def ask(self, *args, module=None, mod=None, path='./' , ai=0, **kwargs):
        module = module or mod
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

        module_obj = self.module(module)()
        if not hasattr(module, 'test') and self.module_exists(module + '.test'):
            module = module + '.test'
            module_obj = self.module(module)()
        fn2result = {}
        for i, fn in enumerate(self.test_fns(module_obj)):
            try:
                fn2result[fn] = getattr(module_obj, fn)()
            except Exception as e:
                print(f'TestError({e})')
                fn2result[fn] = self.detailed_error(e)
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
                    fn_obj = self.fn(route)
                except Exception as e:
                    fn_obj = self.obj(route)
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

    def expand_ref(self, module:str = 'datura', expected_features = ['api', 'app', 'code']):
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
    
    def links(self, module:str = 'datura', expected_features = ['api', 'app', 'code']):
        modules = self.modules()
        filtered_modules = []
        for module in modules:
            isref = self.isref(module, expected_features=expected_features)
            if isref:
                filtered_modules += [module]
        return filtered_modules

    def push(self, path = None, comment=None):

        path = path or (self.modules_path + '/' + module.replace('.', '/'))
        assert os.path.exists(path), f'Path {path} does not exist'
        if comment == None:
            comment = input('Enter the comment for the commit: ')
        cmd = f'cd {self.modules_path}; git add {path} ; git commit -m "{comment}" ; git push'

        self.cmd(cmd, cwd=self.modules_path)
        

    def git_info(self, path:str = None, name:str = None, n=10):
        return self.fn('git/git_info', {'path': path, 'name': name, 'n': n})
    
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

    def build(self, name:str = None, query=None): 
        return self.fn('build/forward')(name=name, query=query)

    def cp_mod(self, module:str = 'dev', name:str = 'dev2'):
        """
        Copy the module to the git repository
        """
        from_path = self.dirpath(module)
        to_path = self.modules_path + '/' + name
        if not os.path.exists(to_path):
            raise Exception(f'Module {from_path} does not exist')
        if os.path.exists(to_path):
            if input(f'Path {to_path} already exists. Do you want to remove it? (y/n)'):
                self.rm(to_path)
        self.cp(from_path, to_path)
        assert os.path.exists(to_path), f'Failed to copy {from_path} to {to_path}'
        return {'success': True, 'msg': 'copied module', 'to': to_path}

    def add_mod(self, module:str = 'dev', name:str = None):

        repo2path = self.repo2path()
        if os.path.exists(module):
            to_path =  self.modules_path + '/' + module.split('/')[-1]
            from_path = module
            self.rm(to_path)
            self.cp(from_path, to_path)
            assert os.path.exists(to_path), f'Failed to copy {from_path} to {to_path}'

        elif 'github.com' in module:
            giturl = module
            module = module.split('/')[-1].replace('.git', '')
            
            # clone ionto the modules path
            to_path = self.modules_path + '/' + module.replace('.', '/')
            cmd = f'git clone {giturl} {self.modules_path}/{module}'
            self.cmd(cmd, cwd=self.modules_path)

        elif module in repo2path:
            from_path = repo2path[module]
            to_path =  self.modules_path + '/' + module.replace('.', '/')
            if os.path.exists(to_path):
                self.rm(to_path)
            self.cp(from_path, to_path)
            assert os.path.exists(to_path), f'Failed to copy {from_path} to {to_path}'
        else:
            raise Exception(f'Module {module} does not exist')
        
        git_path = to_path + '/.git'
        if os.path.exists(git_path):
            self.rm(git_path)
        self.tree(update=1)
        return {'success': True, 'msg': 'added module',  'to': to_path}

    add = add_mod

    def rm_mod(self, module:str = 'dev'):
        """
        Remove the module from the git repository
        """
        path = self.dirpath(module)
        if not os.path.exists(path):
            raise Exception(f'Module {path} does not exist')
        self.rm(path)
        return {'success': True, 'msg': 'removed module'}

    rmmod = rm_mod
    addmod = add_mod

    def add_globals(self, globals_input:dict = None):
        """
        add the functions and classes of the module to the global namespace
        """
        if globals_input == None:
            return {}
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
        self.repo_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
        self.home_path = os.path.expanduser('~')
        self.storage_path = f'{self.home_path}/.{self.repo_name}'
        self.root_path =os.path.dirname(__file__)
        self.core_path = self.root_path + '/core' # the path to the core
        self.lib_path  = self.libpath = self.repo_path  = self.repopath = os.path.dirname(self.root_path) # the path to the repo
        self.home_path = self.homepath  = os.path.expanduser('~') # the home path
        self.modules_path = self.modspath = self.root_path + '/modules'
        self.tests_path = f'{self.lib_path}/tests'

        # config attributes
        self.config  = config = self.get_config()
        self.core_modules = config['core_modules'] # the core modules
        self.repo_name  = config['repo_name'] # the name of the library
        self.endpoints = config['endpoints']
        self.port_range = config['port_range'] # the port range between 50050 and 50150
        self.shortcuts = config["shortcuts"]
        self.sync_routes()
        self.modules_url = self.config['modules_url']
        if not os.path.exists(self.modules_path):
            os.makedirs(self.modules_path, exist_ok=True)
        if not os.path.exists(self.modules_path+'/.git'):
            cmd = f'git clone {self.modules_url} {self.modules_path}'
            self.cmd(cmd, cwd=self.module_path, verbose=True)
        globals_input = self.add_globals(globals_input)
        return {'success': True, 'msg': 'synced config'}

    def main(self, *args, **kwargs):
        """
        Main function to run the module
        """
        self.module('cli')().forward()

    def hash(self, obj, *args, **kwargs):
        from commune.utils import get_hash
        return get_hash(obj, *args, **kwargs)

    def __getattr__(self, k):
        if k in self.__dict__:
            return self.__dict__[k]
        else:
            raise AttributeError(f'{k} not found in {self.__class__.__name__}')


if __name__ == "__main__":
    Module().run()


