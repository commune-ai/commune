
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

class Mod: 

    def __init__(self, 
                  config = None,
                  **kwargs):
        """
        Initialize the module by sycing with the config
        """
        self.sync(config=config)

    def sync(self, mod=None, verbose=False, config = None):

        self.root_path =os.path.dirname(__file__)
        self.lib_path  = self.libpath = self.repo_path  = self.repopath = os.path.dirname(self.root_path) # the path to the repo
        self.core_path = self.root_path + '/core' # the path to the core
        self.tests_path = f'{self.lib_path}/tests'
        self.modules_path = self.mp =  self.modspath = self.root_path + '/modules'
        self.home_path = self.homepath = os.path.expanduser('~')


        self.set_config(config)
        self.name  = self.config['name']
        self.storage_path = f'{self.home_path}/.{self.name}'
        self.port_range = self.config['port_range']
        self.expose = self.config['expose']
        self.shortcuts = self.config['shortcuts']
        if mod is not None:
            print(f'Syncing module {mod}')
            return self.fn(f'{mod}/sync')()

        routes = self.routes()
        for module, fns in routes.items():
            module = self.import_module(module)
            for fn in fns: 
                if hasattr(self, fn):
                    if verbose:
                        print(f'Warning: {fn} already exists')
                else:
                    if verbose:
                        print(f'Adding {fn} from {module.__name__}')
                    fn_obj = getattr(module, fn, None)
                    setattr(self, fn, fn_obj)

        self.sync_mods()

        return {'success': True, 'msg': 'synced modules and utils'}

    def sync_mods(self):
        modules_url = self.code_link(self.config['links']['modules'])
        modules_exist = os.path.exists(self.modules_path)
        update_modules =  len(os.listdir(self.modules_path)) == 0
        if update_modules:
            # os.makedirs(self.modules_path, exist_ok=True)
            print(f'Updating modules from {modules_url} to {self.modules_path}')
            cmd = f'git clone {modules_url} {self.modules_path}'
            print(f'Syncing modules from {modules_url} to {self.modules_path}')
            self.cmd(cmd)

    def module(self, 
                module: str = 'module', 
                params: dict = None,  
                cache=True, 
                verbose=False, 
                update=False,
                max_age: int = 600,
                trials = 1,
                **kwargs) -> str:

        """
        imports the module core
        """
        # Load the module
        module = module or 'module'
        if not isinstance(module, str):
            return module
        # Try to load the module
        if module in ['module', 'commune', 'mod']:
            return Mod
        module = module.replace('/', '.')
        module = self.shortcuts.get(module, module)
        tree = self.tree(update=update)
        obj_path = tree.get(module, module)
        try:
            obj = self.obj(obj_path)
        except Exception as e:
            self.print(f'Error loading module {module} from {obj_path}: {e}', color='red', verbose=verbose)
            tree = self.tree(update=True)
            tree_options = [v for k,v in tree.items() if module in k]
            if any([v == module for v in tree_options]):
                module = [v for v in tree_options if v == module][0]
                obj = self.obj(module)
            elif len(tree_options) == 1:
                obj = self.obj(tree_options[0])
            else:
                raise e
        if params != None:
            if isinstance(params, dict):
                obj = obj(**params)
            elif isinstance(params, list):
                obj = obj(*params)
        return obj

    get_module  = mod = module
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

        if self.module_exists(module):
            path = self.dirpath(module)
        else:
            repo2path = self.repo2path()
            first_guess_path = self.abspath(f"{self.modules_path}/{module.replace('.', '/')}")
            if os.path.exists(first_guess_path):
                path = first_guess_path
            elif module in repo2path:
                path = repo2path[module]
            elif os.path.exists(module):
                path = module
            else:
                raise Exception(f'{module} not found')
        assert os.path.exists(path), f'{path} does not exist'
        return self.cmd(f'code {path}', **kwargs)

    def getfile(self, obj=None) -> str:
        return inspect.getfile(self.mod(obj))

    def path(self, obj=None) -> str:
        return inspect.getfile(self.mod(obj))

    def about(self, module, query='what is this?', *extra_query):
        """
        Ask a question about the module
        """
        query = query + ' '.join(extra_query)
        return self.ask(f' {self.schema(module)} {query}')

    def abspath(self,path:str=''):
        return os.path.abspath(os.path.expanduser(path))

    abs = abspath

    def filepath(self, obj=None) -> str:
        """
        get the file path of the module
        """
        return inspect.getfile(self.mod(obj)) 

    fp = filepath

    def dockerfiles(self, module=None):
        """
        get the dockerfiles of the module
        """
        dirpath = self.dirpath(module)
        dockerfiles = [f for f in os.listdir(dirpath) if f.startswith('Dockerfile')]
        return [os.path.join(dirpath, f) for f in dockerfiles]
        
    def dirpath(self, module=None) -> str:
        """
        get the directory path of the module
        """
        module = self.shortcuts.get(module, module)
        module = (module or 'module').replace('/', '.')
        if module in ['module', 'commune']:
            return self.lib_path
        else:
            possible_paths = [ self.core_path + '/' + module, self.modules_path + '/' + module ]
            if any(os.path.exists(pp)  for pp in possible_paths):
                for pp in possible_paths:
                    if os.path.exists(pp):
                        dirpath = pp 
                        break
            elif self.module_exists(module):
                filepath = self.filepath(module)
                dirpath =  os.path.dirname(filepath)
            else: 
                dirpath = self.modules_path + '/' + module.replace('.', '/')

             
            src_tag =  module + '/src' 
            if src_tag in dirpath:
                dirpath = dirpath.split(src_tag)[0] + module
            
            if dirpath.endswith('/src'):
                dirpath = dirpath[:-4]  # remove the trailing /src
            return dirpath

    dp = dirpath
    
    def modname(self, obj=None):
        obj = obj or Mod
        if  isinstance(obj, str):
            obj = self.module(obj)
        module_file =  inspect.getfile(obj)
        return self.path2name(module_file)

    def vs(self, path = None):
        path = path or __file__
        path = os.path.abspath(path)
        return self.cmd(f'code {path}')
    
    def module_class(self, obj=None) -> str:
        return (obj or self).__name__

    def class_name(self, obj= None) -> str:
        if obj == None: 
            objx = self 
        return obj.__name__


    def config_path(self, obj = None) -> str:
        if obj in [None, 'module']:
            filename =  '/'.join(__file__.split('/')[:-2]+['config']) 
            config_path = None
        else:
            filename = self.dirpath(obj) + '/config'
        for filetype in ['json', 'yaml']:
            config_path = filename + '.' + filetype 
        assert config_path != None
        return config_path

    def storage_dir(self, module=None):
        module = self.mod(module)
        return os.path.abspath(os.path.expanduser(f'~/.commune/{self.modname(module)}'))
    
    def is_admin(self, key:str) -> bool:
        return self.get_key().key_address == key

    def is_home(self, path:str = None) -> bool:
        """
        Check if the path is the home path
        """
        if path == None:
            path = self.pwd()
        return os.path.abspath(path) == os.path.abspath(self.home_path)

    def print(self,  *text:str,  **kwargs):
        return self.obj('commune.utils.print_console')(*text, **kwargs)

    def time(self, t=None) -> float:
        import time
        return time.time()
        
    def pwd(self):
        return os.getcwd()

    def token(self, data, key=None, module='auth.jwt',  **kwargs) -> str:
        token = self.module(module)().get_token(data=data, key=key, **kwargs)
        assert self.verify_token(token), f'Token {token} is not valid'
        return token
    def verify_token(self, token:str = None,  module='auth.jwt',  *args, **kwargs) -> str:
        return self.module(module)().verify_token(token=token, *args, **kwargs)

    def run(self, fn:str='info', params: Union[str, dict]="{}", **_kwargs) -> Any: 
        module = 'module'
        if '/' in fn:
            module, fn = fn.split('/')
        parser = argparse.ArgumentParser(description='Argparse for the module')
        parser.add_argument('--module', dest='module', help='The function', type=str, default=module)
        parser.add_argument('--fn', dest='fn', help='The function', type=str, default=fn)
        parser.add_argument('--params', dest='params', help='key word arguments to the function', type=str, default=params) 
        argv = parser.parse_args()
        params = argv.params
        module = argv.module.replace('/', '.')
        fn = argv.fn
        args, kwargs = [], {}
        if isinstance(params, str):
            params = json.loads(params.replace("'",'"')) 
        print(f'Running {module}.{fn} with params {params}')
        if isinstance(params, dict):
            if 'args' in params and 'kwargs' in params:
                args = params['args']
                kwargs = params['kwargs']
            else:
                kwargs = params
        elif isinstance(params, list):
            args = params
        else:
            raise Exception('Invalid params', params)
        module = self.module(module)()
        return getattr(module, fn)(*args, **kwargs)     
        
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
        module = self.module(module)()
        fn_obj =  getattr(module, fn)
        params = params or {}
        if isinstance(params, str):
            params = json.loads(params.replace("'",'"'))
        if isinstance(params, list):
            args = params
        elif isinstance(params, dict):
            kwargs = params
        args = args or []
        kwargs = kwargs or {}
        return fn_obj(*args, **kwargs)

    def is_module_file(self, module = None, exts=['py', 'rs', 'ts'], folder_filenames=['module', 'agent', 'block',  'server']) -> bool:
        dirpath = self.dirpath(module)
        try:
            filepath = self.filepath(module)
        except Exception as e:
            self.print(f'Error getting filepath for {module}: {e}', color='red', verbose=False)
            return False
        folder_filenames.append(module.split('.')[-1]) # add the last part of the module name to the folder filenames
        for ext in exts:
            for fn in folder_filenames:
                if filepath.endswith(f'/{fn}.{ext}'):
                    return False
                non_folder_name = module.split('.')[-1]
                if filepath.endswith(f'/{non_folder_name}.{ext}'):
                    return True
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
              include_modules:bool = False,
              include_hidden_files:bool = False, 
              relative = False, # relative to the current working directory
              shorten_home = True, # null if relative is True
              startswith:str = None,
              **kwargs) -> List[str]:
        """
        Lists all files in the path
        """
        
        if self.module_exists(path):
            path = self.dirpath(path)
            # if not include_modules:
            #     avoid_terms.append('/modules/')
        files =self.glob(path, **kwargs)
        if not include_hidden_files:
            files = [f for f in files if not '/.' in f]

        files = list(filter(lambda f: not any([at in f for at in avoid_terms]), files))
        # search terms
        if relative: 
            files = [f.replace(self.pwd() + '/', './') for f in files]
        elif shorten_home: 
            files = [f.replace(self.home_path + '/', '~/') for f in files]
        if search != None:
            files = [f for f in files if search in f]
        return files


    def files_size(self):
        return len(str(self.files()))


    def envs(self, key:str = None, **kwargs) -> None:
        return self.get_key(key, **kwargs).envs()

    def encrypt(self,data: Union[str, bytes], key: str = None, password: str = None, **kwargs ) -> bytes:
        return self.get_key(key).encrypt(data, password=password)
    def decrypt(self, data: Any,  password : str = None, key: str = None, **kwargs) -> bytes:
        return self.get_key(key).decrypt(data, password=password)
        
    def sign(self, data:dict  = None, key: str = None,  crypto_type='sr25519', mode='str', **kwargs) -> bool:
        return self.get_key(key, crypto_type=crypto_type).sign(data, mode=mode, **kwargs)

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
        
    def utils(self, search=None):
        return self.get_utils(search=search)

    def routes(self, obj=None):
        obj = obj or self
        routes = {}
        if hasattr(self.config, 'routes'):
            routes.update(self.config.routes)
        for util in self.get_utils():
            k = '.'.join(util.split('.')[:-1])
            v = util.split('.')[-1]
            routes[k] = routes.get(k , [])
            routes[k].append(v)
        return routes

    def fn2route(self): 
        routes = self.routes()
        fn2route = {}
        for k,v in routes.items():
            for f in v:
                print(f, k)
                if isinstance(f, dict) and 'from' in f and 'to' in f:
                    from_fn = f['from']
                    to_fn = f['to']
                elif isinstance(f, list):
                    from_fn = f[0]
                    to_fn = f[1]
                else: 
                    from_fn = f
                    to_fn = f  
                fn2route[to_fn] = k + '/' + from_fn
        return fn2route

    def secret(self, key:str = None, seed=None, update=False, tempo=None, **kwargs) -> str:
        secret = self.get('secret', {}, update=update, max_age=tempo)
        if len(secret) > 0 :
            return secret
        time = self.time()
        seed = seed or self.random_int(0, 1000000) * self.time() / (self.random_int(1, 1000) + 1)
        nonce = str(int(secret.get('nonce', 0)) + 1)
        secret = self.sign({'time': time, 'nonce': nonce}, key=key,**kwargs)
        self.put('secret', secret)
        return secret

    def tempo_secret(self, key:str = None,  tempo=1, seed=None, **kwargs) -> str:
        """
        Get a secret that is valid for a certain time
        """
        return self.secret(key=key, seed=seed, update=True, tempo=tempo, **kwargs)


    def set_config(self, config:Optional[Union[str, dict]]=None ) -> 'Munch':
        '''
        Set the config as well as its local params
        '''
        config = config or {}
        if isinstance(config, str) :
            if os.path.exists(config):
                if config.endswith('.yaml') or config.endswith('.yml'):
                    import yaml
                    config = yaml.load(open(config, 'r'), Loader=yaml.FullLoader)
                elif config.endswith('.json'):
                    import json
                    config = json.load(open(config, 'r'))
            elif config == 'default':
                config = {}
        elif config == None:
            config = {}
            
        self.config = {**self.get_config(), **config }
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

    def put_json(self, 
                 path:str, 
                 data:Dict, 
                 meta = None,
                 verbose: bool = False,
                 **kwargs) -> str:
        if not path.endswith('.json'):
            path = path + '.json'
        path = self.get_path(path)
        if isinstance(data, dict):
            data = json.dumps(data)
        self.put_text(path, data)
        return path

    def env(self):
        """
        Get the environment variables
        """
        import os
        env = {}
        for k,v in os.environ.items():
            env[k] = v
        return env

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
    path = get_path
    
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
        module = self.mod(module)
        functions = self.fns(module)
        fn_code_map = {}
        for fn in functions:
            try:
                fn_code_map[fn] = self.code(getattr(module, fn))
            except Exception as e:
                self.print(f'Error {e} {fn}', color='red', verbose=False)
        return fn_code_map
    
    def fn_code(self,fn:str, module=None,**kwargs) -> str:
        '''
        Returns the code of a function
        '''
        fn = self.fn(fn)      
        return inspect.getsource(fn)       


    fn2cost = {}

    def fn_schema(self, fn:str = '__init__', code=True, **kwargs)->dict:
        '''
        Get function schema of function in self
        '''     
        schema = {}
        fn_obj = self.fn(fn)
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
        schema['cost'] = 1 if not hasattr(fn_obj, '__cost__') else fn_obj.__cost__ # attribute the cost to the function
        schema['name'] = fn_obj.__name__
        schema.update(self.source(fn_obj, code=code))
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
    
    def schema(self, obj = None , verbose=False, **kwargs)->dict:
        '''
        Get function schema of function in self
        '''   
        schema = {}
        obj = obj or 'module'
        if callable(obj):
            return self.fn_schema(obj, **kwargs)
        elif isinstance(obj, str):
            if '/' in obj :
                return self.fn_schema(obj,  **kwargs)
            elif self.module_exists(obj):
                obj = self.mod(obj)
            elif hasattr(self, obj):
                obj = getattr(self, obj)
                schema = self.fn_schema(obj, **kwargs)
            else: 
                raise  Exception(f'{obj} not found')
        elif hasattr(obj, '__class__'):
            obj = obj.__class__
        for fn in self.fns(obj):
            try:
                schema[fn] = self.fn_schema(getattr(obj, fn), **kwargs)
            except Exception as e:
                self.print(f'Error {e} {fn}', color='red', verbose=verbose)
        return schema

    def code(self, obj = None, search=None, full=False,  *args, **kwargs) -> Union[str, Dict[str, str]]:
        if full:
            return self.code_map(obj, search=search)
        if '/' in str(obj):
            obj = self.fn(obj)
        elif hasattr(self, obj):
            obj = getattr(self, obj)
        else:
            obj = self.module(obj)
        return  inspect.getsource(obj)

    def call(self, *args, **kwargs): 
        return self.fn('client/')(*args, **kwargs)
    
    def code_map(self, module = None , search=None, ignore_folders = ['modules', 'mods'], *args, **kwargs) ->  Dict[str, str]:
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

    def code_file_map(self, module , search=None, *args, **kwargs) ->  Dict[str, str]:
        return list(self.code_map(module=module, search=search,**kwargs).keys())

    def cid(self, module , search=None, *args, **kwargs) -> Union[str, Dict[str, str]]:
        return self.hash(self.code_hash_map(module=module, search=search,**kwargs))

    def getsource(self, module = None, search=None, *args, **kwargs) -> Union[str, Dict[str, str]]:
        if module != None:
            if isinstance(module, str) and '/' in module:
                fn = module.split('/')[-1]
                module = '/'.join(module.split('/')[:-1])
                module = getattr(self.module(module), fn)
            else:
                module = self.mod(module)
        else: 
            module = self
        return inspect.getsource(module)

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
        obj = self.mod(obj)
        fns = dir(obj)
        fns = sorted(list(set(fns)))
        if search != None:
            fns = [f for f in fns if search in f]
        if not include_hidden: 
            fns = [f for f in fns if not f.startswith('__') and not f.startswith('_')]
        return fns


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
                if fn in fn2module and len(fn2module[fn]) < len(module):
                    continue
                else:
                    fn2module[fn] = module
        return fn2module

    def mods(self, search=None,  startswith=None, endswith=None, **kwargs)-> List[str]:  
        return list(self.tree(search=search, endswith=endswith, startswith=startswith , **kwargs).keys())
    modules = mods
    def get_modules(self, search=None, **kwargs):
        return self.modules(search=search, **kwargs)

    

    def core_modules(self) -> List[str]:
        return list(self.core_tree().keys())
    core_mods = core_modules

    def module2schema(self, module=None, max_age=30, update=False, core=True, verbose=False) -> List[str]:
        module2schema = self.get('module2schema', default=None, max_age=max_age, update=update)
        if module2schema == None:
            modules = self.core_modules() if core else self.modules()
            module2schema = {}
            for module in modules:
                try:
                    module2schema[module] = self.schema(module)
                except Exception as e:
                    self.print(f'Module2schemaError({e})', color='red', verbose=verbose)
            # self.put('module2schema', module2schema)
        return module2schema 

    def info(self, module:str='module',  # fam
            max_age : Optional[int]=1000, 
            keep_last_n : int = 10,
            relative=True,
            code=False,
            update: bool =False, 
            key=None,
            ai_describe: bool = False,
            **kwargs):
        """
        Get the info of a module, including its schema, key, cid, and code if specified.
        """
            
        path = self.get_path('info/' + str(module)) 
        
        info = self.get(path, None, max_age=max_age, update=update)
        if info == None:
            info =  {
                    'schema': {}, 
                    'name': module, 
                    'key': self.get_key(key or module).key_address,  
                    'cid': self.cid(module),
                    'time': time.time()
                    }
            try:
                info['schema'] = self.schema(module)
            except Exception as e:
                self.print(f'Error getting schema for {module}: {e}', color='red', verbose=False)
            if code:
                info['code'] = self.code_map(module)
            info['signature'] = self.sign(info)
            if 'desc' not in info and ai_describe:
                prompt = 'given decribe this module in a few sentences, the module is a python module with the following schema: ' + json.dumps(info['schema'])

                desc = ''
                for ch in self.ask(prompt):
                    print(ch, end='', flush=True)
                    desc += ch
                info['desc'] = desc
            self.put(path, info)
            assert self.verify_info(info), f'Invalid signature {info["signature"]}'
        return  info

    card = info 

    def verify_info(self, info:Union[str, dict]=None, **kwargs) -> bool:
        """
        verify the info of the module
        """
        if isinstance(info, str):
            info = self.info(info, **kwargs)
        signature = info.pop('signature')
        verify = self.verify(data=info, signature=signature)  
        assert verify, f'Invalid signature {signature}'
        info['signature'] = signature
        return info

    def epoch(self, *args, **kwargs):
        return self.run_epoch(*args, **kwargs)

    def pwd2key(self, pwd, **kwargs) -> str:
        return self.module('key')().str2key(pwd, **kwargs)

    def is_property(self, fn: 'Callable') -> bool:
        '''
        is the function a property
        '''
        fn = self.fn(fn)
        return isinstance(fn, property)


    def tools(self):
        return self.fn('dev/tools')()
    
    _executors = {}
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

        executor = self.executor(mode=mode, max_workers=max_workers)
        args = args or []
        kwargs = kwargs or {}
        if isinstance(params, dict):
            kwargs = params
            params = None
        elif isinstance(params, list):
            args = params
            params = None
        if mode == 'thread':
            future = executor.submit(self.fn(fn), params={'args': args, 'kwargs':kwargs}, timeout=timeout)
        else:
            future =  executor.submit(self.fn(fn), *args, **kwargs)
        return future 

    def fn(self, fn:Union[callable, str], params:str=None, splitter='/', default_fn='forward', default_module = 'module') -> 'Callable':
        """
        Gets the function from a string or if its an attribute 
        """
        if callable(fn):
            return fn
        if isinstance(fn, str):
            if hasattr(self, fn):
                fn_obj = getattr(self, fn)
            elif fn.startswith('/'):
                fn_obj = getattr(self.module(default_module)(), fn[1:])
            elif fn.endswith('/'):
                fn_obj = getattr( self.module(fn[:-1])(), default_fn)
            elif '/' in fn:
                module, fn = fn.split('/')
                if self.module_exists(module):
                    module = self.module(module)()
                elif self.is_python_module(module):
                    module = self.import_module(module)
                else:
                    raise Exception(f'Mod {module} not found')  
                fn_obj = getattr(module, fn)
            elif self.object_exists(fn):
                fn_obj =  self.obj(fn)
        else:
            return fn
        if params:
            return fn_obj(**params)
        return fn_obj
    
    get_fn = fn


    

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

    def hosts(self):
        return self.fn('remote/hosts')()

    def how(self, module, query, *extra_query) : 
        code = self.code(module)
        query = ' '.join([query, *extra_query])
        return self.fn('model.openrouter/')(f'query={query} code={code}')

    def client(self, *args, **kwargs) -> 'Client':
        """
        Get the client for the module
        """
        return self.fn('client/client')( *args, **kwargs)
    
    def classes(self, path='./',  **kwargs):
        """
        Get the classes for each path inside the path variable
        """
        return  self.path2classes(path=path, tolist=True, **kwargs)

    def password(self, max_age=None, update=False, **kwargs):
        """
        Get the password for the module
        """
        path = self.get_path('password')
        pwd = self.get(path, None, max_age=max_age, update=update,**kwargs)
        if pwd == None:
            pwd = self.hash(self.mnemonic() + str(time.time()))
            self.put(path, pwd)
            self.print('Generating new password', color='blue')
        return pwd

    def mnemonic(self, words=24):
        """
        Generates a mnemonic phrase of the given length.
        """

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
        return self.mod('key')().generate_mnemonic(words=words)

    def path2relative(self, path='./'):
        """
        Converts a path to a relative path (for instance ~/foo/bar.py to ./foo/bar.py)
        """
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
        """
        Converts a path to an object path (for instance ./foo/bar.py to foo.bar)
        """
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
        if name_chunks[0] == self.name:
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

    ensure_syspath_flag = False

    def ensure_syspath(self):
        """
        Ensures that the path is in the sys.path
        """
        if not self.ensure_syspath_flag:
            import sys
            paths = [self.pwd(), self.repo_path]
            for path in paths:
                if path not in sys.path:
                    sys.path.append(path)
        return {'paths': sys.path, 'success': True}
            
    obj_cache = {}
    def obj(self, key:str, **kwargs)-> Any:
        # add pwd to the sys.path if it is not already there
        self.ensure_syspath()
        if key in self.obj_cache:
            return self.obj_cache[key]
        else:
            from commune.utils import import_object
            obj = self.obj_cache[key] = import_object(key, **kwargs)
        return obj

    def obj_exists(self, path:str, verbose=False)-> Any:
        # better way to check if an object exists?
        try:
            self.obj(path, verbose=verbose)
            return True
        except Exception as e:
            return False

    def object_exists(self, path:str, verbose=False)-> Any:
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
                        avoid_terms=['modules', 
                                     'agents',
                                     'agent',
                                     'module', 
                                     '_modules', 
                                     '_agents', 
                                     'core',
                                    'src', 
                                    'server',
                                    'servers', 
                                    ]):
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
        if path.startswith(self.name + '.'):
            path = path[len(self.name)+1:]
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
        return path

    def logs(self, *args, **kwargs):
        return self.fn('pm/logs')(*args, **kwargs)
    
    def local_tree(self , depth=10, **kwargs):
        pwd = os.getcwd()
        if os.path.expanduser('~') == pwd:
            depth = 2
        return self.get_tree(pwd, depth=depth , **kwargs)

    def locals(self, **kwargs):
        return list(self.get_tree(self.pwd(), **kwargs).keys())

    def core_modules(self, **kwargs) -> List[str]:
        return list(self.get_tree(self.core_path, **kwargs).keys())

    def core_tree(self, **kwargs):
        return {**self.get_tree(self.core_path,  **kwargs)}
    def modules_tree(self, **kwargs):
        return self.get_tree(self.modules_path, depth=10,  **kwargs)
    
    def tree(self, max_age=None, update=False, **kwargs):

        params = {'max_age': max_age, 'update': update, **kwargs}
        tree = { 
                **self.modules_tree(**params), 
                **self.local_tree(**params),  
                ** self.core_tree(**params) 
            }


        return tree

    def get_tree(self, path='./', depth = 10, max_age=None, update=False,
                    search=None, startswith=None, endswith=None,  **kwargs):
        """
        Get the tree of the modules in the path
        a tree is a dictionary of the form {modname: module_path}
        the modname is based on the directory path 
        """
    
        path = self.abspath(path)
        path_hash = self.hash(path)
        tree_cache_path = 'tree/'+self.hash(os.path.abspath(path))
        tree = self.get(tree_cache_path, None, max_age=max_age, update=update)
        if tree == None:
            class_paths = self.classes(path, depth=depth)
            
            def filter_path(p):
                if p.startswith('src.' + self.name):
                    return p.replace('src.' + self.name + '.', '')
                repo_prefix = self.name + '.' + self.name
                if p.startswith(repo_prefix):
                    return p.replace(repo_prefix, self.name)
                return p
            class_paths = [filter_path(p) for p in class_paths]
            simple_paths = [self.objectpath2name(p) for p in class_paths]
            tree = dict(zip(simple_paths, class_paths))
            self.put(tree_cache_path, tree)
        if startswith != None:
            tree = {k:v for k,v in tree.items() if k.startswith(startswith)}
        if endswith != None:
            tree = {k:v for k,v in tree.items() if k.endswith(endswith)}
        if search != None:
            tree = {k:v for k,v in tree.items() if search in k}
        return tree
    
    ltree = local_tree
    mtree = modules_tree

    
    def check_info(self,info, features=['key', 'hash', 'time', 'founder', 'name', 'schema']):
        try:
            assert isinstance(info, dict), 'info is not a dictionary'
            for feature in features:
                assert feature in info, f'{feature} not in info'
        except Exception as e:
            return False
        return True

    def cwd(self, mod=None):
        if mod:
            return self.dirpath(mod)
        return os.getcwd()

    def addmod(self, name= None, base_module : str = 'base', update=True):
        """
        make a new module
        """
        if not name:
            name = input('Mod name/github/ipfs/url')

        if os.path.exists(name):
            original_dirpath = self.abspath(name)
            name = original_dirpath.split('/')[-1]
            dirpath = self.abspath(self.modules_path + '/' + name.replace('.', '/'))
            if os.path.exists(dirpath):
                self.rm(dirpath)
            cmd = f'cp -r {original_dirpath} {dirpath}'
            
            self.cmd(cmd)
            return {'name': name, 'path': dirpath, 'msg': 'Mod Copied'}
        elif bool(name.endswith('.git') or name.startswith('http')):
            git_path = name
            name =  name.split('/')[-1].replace('.git', '')
            dirpath = self.abspath(self.modules_path +'/'+ name.replace('.', '/'))
            print(f'Cloning {git_path} to {dirpath}')
            self.cmd(f'git clone {git_path} {dirpath}')
            self.cmd(f'rm -rf {dirpath}/.git')
        else:
            dirpath = self.abspath(self.modules_path +'/'+ name.replace('.', '/'))
            module_class_name = ''.join([m[0].capitalize() + m[1:] for m in name.split('.')])
            code_map = self.code_map(base_module)
            new_code_map = {}
            new_class_name = name[0].upper() + name[1:]
            for k,v in code_map.items():
                k_path =  dirpath + '/' +  k.replace(base_module, name)
                new_code_map[k_path] = v
                self.put_text(k_path, v)
            code_map = new_code_map
        self.go(dirpath)
        return {'name': name, 'path': dirpath, 'msg': 'Mod Created'}
    
    create = new = add = addmod



    def urls(self, *args, **kwargs):
        return self.fn('pm/urls')(*args, **kwargs)


    def servers(self, *args, **kwargs):
        return self.fn('pm/servers')(*args, **kwargs)

    executor_cache = {}
    def executor(self,  max_workers=8, mode='thread', cache=True):
        path = "executor/" + mode + '/' + str(max_workers)
        if cache and path in self.executor_cache:
            return self.executor_cache[path]

        if mode == 'process':
            from concurrent.futures import ProcessPoolExecutor
            executor =  ProcessPoolExecutor(max_workers=max_workers)
        elif mode == 'thread':
            executor =  self.mod('executor')(max_workers=max_workers)
        elif mode == 'async':
            from commune.core.api.src.async_executor import AsyncExecutor
            executor = AsyncExecutor(max_workers=max_workers)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'thread', 'process' or 'async'.")
        if cache:
            self.executor_cache[path] = executor
        return executor

    def server_exists(self, server:str = 'commune', *args, **kwargs):
        return  self.fn('pm/server_exists')(server, *args, **kwargs)

    def namespace(self, *args, **kwargs):
        return self.fn('pm/namespace')(*args, **kwargs)

    def epoch(self, *args, **kwargs):
        return self.fn('vali/epoch')(*args, **kwargs)

    def up(self, image = 'commune'):
        return self.cmd('make up', cwd=self.lib_path)

    def enter(self, image = 'commune'):
        return self.fn('pm/enter')(image)

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

    def build_image(self, module:str = 'module'):
        return os.system(f'docker build -t {self.name} {self.lib_path}')

    def repos(self, search=None):
        return list(self.repo2path(search=search).keys())

    def help(self, query:str = 'what is this', *extra_query , mod='module', **kwargs):
        query = ' '.join([query, *extra_query])
        module =  mod or module
        context = self.context(path=self.core_path)
        return self.mod('agent')().ask(f'given the code {self.code(module)} and CONTEXT OF COMMUNE {context} anster wht following question: {query}', preprocess=False)
    
    def ask(self, *args, module=None, mod=None, path='./' , ai=0,  **kwargs):
        module = module or mod
        # commune_readmes = self.readmes(path=path)
        if module != None:
            args = [self.code(module)] + list(args)
        return self.module("agent")().ask(*args, **kwargs) 

    def readmes(self,  path='./', search=None, avoid_terms=['/modules/']):
        files =  self.files(path)
        files = [f for f in files if f.endswith('.md')]
        files = [f for f in files if all([avoid not in f for avoid in avoid_terms])]
        if search != None:
            files = [f for f in files if search in f]
        return files

    def context(self, path=None):
        path = path or self.core_path
        readme2text = self.readme2text(path)
        print('ctx size', len(str(readme2text)))
        return readme2text

    def context_size(self, path:str = './', search=None, **kwargs) -> int:
        return len(str(self.readme2text(path=path, search=search, **kwargs)))

    def readme2text(self, path:str = './', search=None, **kwargs) -> str:
        """
        Returns the text of the readme file in the path
        """
        files = self.readmes(path=path, search=search)
        readme2text = {}
        for f in files:
            readme2text[f] = self.get_text(f)
        return readme2text

    def import_module(self, module:str = 'commune.utils', lib_name = 'commune'):
        from importlib import import_module
        double_lib_name = f'{lib_name}.{lib_name}'
        if double_lib_name in module:
            module = module.replace(double_lib_name, lib_name)
        return import_module(module)

    def is_python_module(self, module:str = 'commune.utils'):
        try:
            module = self.import_module(module)
            return True
        except Exception as e:
            return False

    def kill(self, server:str = 'commune'):
        return self.fn('pm/kill')(server)

    def kill_all(self):
        return self.fn('pm/kill_all')()

    killall = kill_all

    def configs( path='./', 
                modes=['yaml', 'json'], 
                search=None, 
                config_name_options = ['config', 'cfg', 'module', 'block',  'agent', 'mod', 'bloc', 'server'],
                names=['config', 'cfg', 'module', 'block',  'agent', 'mod', 'bloc']):
        """
        Returns a list of config files in the path
        """
        def is_config(f):
            return any(f.endswith(f'{name}.{m}') for name in config_name_options for m in modes)
        configs =  [f for f in  self.files(path) if is_config(f)]
        if search != None:
            configs = [f for f in configs if search in f]
        return configs

    def serve(self, module:str = 'module', port:int=None, **kwargs):
        return self.mod('pm')().serve(module=module, port=port, **kwargs)

    def app(self, module=None, **kwargs):
        if module:
            return self.fn(module + '/app' )()
        return self.fn('app/')(**kwargs)

    def api(self, *args, **kwargs):
       return self.fn('app/api')(*args, **kwargs)

    def code_link(self, url:str='commune-ai/commune'):
        gitprefix = 'https://github.com/'
        gitsuffix = '.git'
        if not url.startswith(gitprefix):
            url = gitprefix + url
        if not url.endswith(gitsuffix):
            url = url + gitsuffix
        return url
    
    def links(self, module:str = 'datura', expected_features = ['api', 'app', 'code']):
        return self.config['links']

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

    def cpmod(self, from_module:str = 'dev', to_module:str = 'dev2'):
        """
        Copy the module to the git repository
        """
        from_path = self.dirpath(from_module)
        if not os.path.exists(from_path):
            raise Exception(f'Mod {from_path} does not exist')
        to_path = self.modules_path + '/' + to_module
        if os.path.exists(to_path):
            if input(f'Path {to_path} already exists. Do you want to remove it? (y/n)'):
                self.rm(to_path)
        self.cp(from_path, to_path)
        assert os.path.exists(to_path), f'Failed to copy {from_path} to {to_path}'
        return { 
                'from': {'module': from_module, 'path': from_path}, 
                'to': {'path': to_path, 'module': to_module}
                } 

    def mvmod(self, from_module:str = 'dev', to_module:str = 'dev2'):
        """
        Move the module to the git repository
        """
        from_path = self.dirpath(from_module)
        to_path = self.dirpath(to_module)
        result =  { 
                'from': {'module': from_module, 'path': from_path}, 
                'to': {'path': to_path, 'module': to_module}
                }

        return self.mv(from_path, to_path)
    mv_mod = mvmod

    def address2key(self, *args, **kwargs):
        return self.fn('key/address2key')(*args, **kwargs)

    def clone(self, module:str = 'dev', name:str = None):
        repo2path = self.repo2path()
        if os.path.exists(module):
            to_path =  self.modules_path + '/' + module.split('/')[-1]
            from_path = module
            self.rm(to_path)
            self.cp(from_path, to_path)
            assert os.path.exists(to_path), f'Failed to copy {from_path} to {to_path}'
        elif 'github.com' in module:
            code_link = module
            module = (name or module.split('/')[-1].replace('.git', '')).replace('/', '.')
            # clone ionto the modules path
            to_path = self.modules_path + '/' + module
            cmd = f'git clone {code_link} {self.modules_path}/{module}'
            self.cmd(cmd, cwd=self.modules_path)
        else:
            raise Exception(f'Mod {module} does not exist')
        git_path = to_path + '/.git'
        if os.path.exists(git_path):
            self.rm(git_path)
        self.tree(update=1)
        return {'success': True, 'msg': 'added module',  'to': to_path}

    def rm_mod(self, module:str = 'dev'):
        """
        Remove the module from the git repository
        """
        path = self.dirpath(module)
        if not os.path.exists(path):
            raise Exception(f'Mod {path} does not exist')
        self.rm(path)
        assert not os.path.exists(path), f'Failed to remove {path}'
        self.tree(update=1)
        return {'success': True, 'msg': 'removed module'}

    rmmod = rm_mod

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
        for f in self.fns(Mod, mode='self'):
            def wrapper_fn(f, *args, **kwargs):
                fn = getattr(Mod(), f)
                return fn(*args, **kwargs)
            globals_input[f] = partial(wrapper_fn, f)
        return globals_input

    def main(self, *args, **kwargs):
        """
        Main function to run the module
        """
        self.module('cli')().forward()

    def hasattr(self, module, k):
        """
        Check if the module has the attribute
        """
        return hasattr(self.module(module)(), k)

    def hash(self, obj, *args, **kwargs):
        from commune.utils import get_hash
        return get_hash(obj, *args, **kwargs)
    def test(self, module = None,  **kwargs) ->  Dict[str, str]:
        return self.fn('test/')( module=module,  **kwargs )

    def txs(self, *args, **kwargs) -> 'Callable':
        return self.fn('server/txs')( *args, **kwargs)

    def sand(self):
        auth = self.mod('auth')()
        data = {
            'fn': 'sand',
            'params': {
                'module': self.name,
                'args': [],
                'kwargs': {}
            }
        }
        token = auth.generate(data)
        return auth.verify(token)

if __name__ == "__main__":
    Mod().run()


