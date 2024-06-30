import sys
import os
import inspect
import concurrent
import threading
from copy import deepcopy
from typing import *
import json
from functools import partial
from glob import glob
import sys
import argparse
import asyncio
import nest_asyncio

try:
    from .config import Config
    from .schema import Schema
    from .misc import Misc
    from .logger import Logger
    from .storage import Storage
except:
    # this is for serving mostly
    from config import Config
    from schema import Schema
    from misc import Misc
    from logger import Logger
    from storage import Storage



nest_asyncio.apply()

# AGI BEGINS 
class c(Config, Schema, Misc, Logger, Storage ):

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
    root_path  = root  = os.path.dirname(__file__) # the path to the root of the library
    libpath = lib_path = os.path.dirname(root_path) # the path to the library
    libname = lib_name = lib = root_path.split('/')[-1] # the name of the library
    datapath = os.path.join(root_path, 'data') # the path to the data folder
    modules_path = os.path.join(lib_path, 'modules') # the path to the modules folder
    repo_path  = os.path.dirname(root_path) # the path to the repo
    blacklist = [] # blacklist of functions to not to access for outside use
    server_mode = 'http' # http, grpc, ws (websocket)
    default_network = 'local' # local, subnet
    cache = {} # cache for module objects
    home = os.path.expanduser('~') # the home directory
    __ss58_format__ = 42 # the ss58 format for the substrate address
    cache_path = os.path.expanduser(f'~/.{libname}')
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


    @property
    def key(self):
        if not hasattr(self, '_key'):
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
            c.print(f'Error: {e}', color='red')
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
        c.cmd(f'python3 {c.libpath}/sandbox.py', verbose=True)
        return 
    sand = sandbox

    included_pwd_in_path = False
    @classmethod
    def import_module(cls, 
                      import_path:str, 
                      included_pwd_in_path=True, 
                      try_prefixes = ['commune', 'commune.modules', 'modules', 'commune.subspace', 'subspace']
                      ) -> 'Object':
        from importlib import import_module
        if included_pwd_in_path and not cls.included_pwd_in_path:
            import sys
            pwd = c.pwd()
            sys.path.append(pwd)
            sys.path = list(set(sys.path))
            cls.included_pwd_in_path = True
        pwd = c.pwd()
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
    def import_object(cls, key:str, verbose: bool = 0)-> Any:
        '''
        Import an object from a string with the format of {module_path}.{object}
        Examples: import_object("torch.nn"): imports nn from torch
        '''
        module = '.'.join(key.split('.')[:-1])
        object_name = key.split('.')[-1]
        if verbose:
            c.print(f'Importing {object_name} from {module}')

        obj =  getattr(cls.import_module(module), object_name)
      
        return obj
    

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


    @classmethod
    def resolve_address(cls, address:str = None):
        if address == None:
            address = c.free_address()
        assert isinstance(address, str),  'address must be a string'
        return address


    @classmethod
    def free_address(cls, **kwargs):
        return f'{c.ip()}:{c.free_port(**kwargs)}'
    

    def kwargs2attributes(self, kwargs:dict, ignore_error:bool = False):
        for k,v in kwargs.items():
            if k != 'self': # skip the self
                # we dont want to overwrite existing variables from 
                if not ignore_error: 
                    assert not hasattr(self, k)
                setattr(self, k)




    def check_used_ports(self, start_port = 8501, end_port = 8600, timeout=5):
        port_range = [start_port, end_port]
        used_ports = {}
        for port in range(*port_range):
            used_ports[port] = self.port_used(port)
        return used_ports

    @classmethod
    def pm2_restart_all(cls):
        '''
        Kill the server by the name
        '''
        for p in c.pm2_list():
            c.print(f'Restarting {p}', color='red')
            c.pm2_restart(p)

        c.update()



    
    @classmethod
    def url2text(cls, *args, **kwargs):
        return c.module('web').url2text(*args, **kwargs).text

    module_cache = {}
    @classmethod
    def get_module(cls, 
                   path:str = 'module',  
                   cache=True,
                   trials = 3,
                   verbose = False,
                   update_tree_if_fail = True,
                   init_kwargs = None,
                   ) -> str:
        """
        params: 
            path: the path to the module
            cache: whether to cache the module
            tree: the tree to search for the module
            update_if_fail: whether to update the tree if the module is not found
        """


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
        
        if cache:
            c.module_cache[cache_key] = module

        if verbose:
            c.print(f'Loaded {path} in {c.time() - t0} seconds', color='green')
        
        if init_kwargs != None:
            module = module(**init_kwargs)
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
    

    def search_dict(self, d:dict = 'k,d', search:str = {'k.d': 1}) -> dict:
        search = search.split(',')
        new_d = {}

        for k,v in d.items():
            if search in k.lower():
                new_d[k] = v
        
        return new_d

  
    @classmethod
    def has_module(cls, module):
        return module in c.modules()
    
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
    

    @classmethod
    def storage_dir(cls):
        return f'{c.cache_path}/{cls.module_name()}'
    


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
    def tilde_path(cls):
        return os.path.expanduser('~')

    data_path = repo_path + '/data'
    

    @classmethod
    def readme(cls):
        # Markdown input
        markdown_text = "## Hello, *Markdown*!"
        path = cls.filepath().replace('.py', '_docs.md')
        markdown_text =  cls.get_text(path=path)
        return markdown_text
    
    docs = readme


    def is_dir_empty(self, path:str):
        return len(self.ls(path)) == 0


    @classmethod
    def get_file_size(cls, path:str):
        path = cls.resolve_path(path)
        return os.path.getsize(path)

    @classmethod
    def __str__(cls):
        return cls.__name__

    @classmethod
    def get_server_info(cls,name:str) -> Dict:
        return cls.namespace_local().get(name, {})

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


    @classmethod
    def nest_asyncio(cls):
        import nest_asyncio
        nest_asyncio.apply()


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
    def is_module(cls, obj=None) -> bool:
        
        if obj is None:
            obj = cls
        if all([hasattr(obj, k) for k in ['info', 'schema', 'set_config', 'config']]):
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


    @property
    def server_name(self):
        if not hasattr(self, '_server_name'): 
            self._server_name = None
        return self._server_name
            
    @server_name.setter
    def server_name(self, name):
        self._server_name = name
    
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
            logs.append(f'Waiting for {name} for {time_waiting}s/{timeout}s ')
            if time_waiting > timeout:
                raise TimeoutError(f'Timeout waiting for {name} to start')
        return True

    
    # NAMESPACE::MODULE
    namespace_module = 'module.namespace'
    
    @classmethod
    def server2key(self, *args, **kwargs):
        servers = c.servers()
        key2address = c.key2address()
        server2key = {s:key2address[s] for s in servers}
        return server2key



    
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
    def resolve_object(cls, module:str = None, **kwargs):
        if module == None:
            module = cls.module_name()
        if isinstance(module, str):
            module = c.module(module)
        return module
    


    def info(self , 
             module = None,
             features = ['schema', 'namespace', 'commit_hash', 'hardware','attributes','functions'], 
             lite_features = ['name', 'address', 'schema', 'key', 'description'],
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
            info['path'] = self.module_name() # get the path of the module
        if 'address' in features:
            info['address'] = self.address.replace(c.default_ip, c.ip(update=False))
        if 'key' in features:    
            info['key'] = self.key.ss58_address
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


    def self_destruct(self):
        c.kill(self.server_name)    
        
    def self_restart(self):
        c.restart(self.server_name)
        
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
    def argparse(cls, verbose: bool = False, **kwargs):
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
    def get_server_name(cls, name:str=None, tag:str=None, seperator:str='.'):
        name = name if name else cls.__name__.lower()
        if tag != None:
            name = tag + seperator + name
        return name
  

    @classmethod
    def module_fn(cls, module:str, fn:str , args:list = None, kwargs:dict= None):
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
        return fn(*args, **kwargs)
    fn = module_fn
    
    @classmethod
    def module(cls,module: Any = 'module' , **kwargs):
        '''
        Wraps a python class as a module
        '''
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
        ip = cls.get('ip', None, max_age=max_age, update=update)
        if ip == None:
            ip =  cls.module('network').external_ip(**kwargs)
            c.put('ip', ip)
        return ip
    

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

    # CPU LAND

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


    @classmethod
    def test_fns(cls, *args, **kwargs):
        return [f for f in cls.functions(*args, **kwargs) if f.startswith('test_')]
    
    @classmethod
    def has_test_module(cls, module=None):
        module = module or cls.module_name()
        return c.module_exists(cls.module_name() + '.test')
    
    @classmethod
    def test(cls,
              module=None,
              timeout=100, 
              trials=3, 
              parallel=False,
              ):
        module = module or cls.module_name()

        if cls.has_test_module(module):
            c.print('FOUND TEST MODULE', color='yellow')
            module = module + '.test'
        self = c.module(module)()
        test_fns = self.test_fns()
        print(f'testing {module} {test_fns}')
        fn2result = {}
        if parallel:
            future2fn = {}
            for fn in self.test_fns():
                c.print(f'testing {fn}')
                f = c.submit(getattr(self, fn), timeout=timeout)
                future2fn[f] = fn
            for f in c.as_completed(future2fn, timeout=timeout):
                fn = future2fn.pop(f)
                fn2result[fn] = f.result()
        else:
            for fn in self.test_fns():
                fn2result[fn] = getattr(self, fn)()

                
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

    @staticmethod
    def jsonable( value):
        import json
        try:
            json.dumps(value)
            return True
        except:
            return False
    

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
    def start(cls, *args, **kwargs):
        return cls(*args, **kwargs)
    
    def remove_user(self, key: str) -> None:
        if not hasattr(self, 'users'):
            self.users = []
        self.users.pop(key, None)
    
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
            module = cls.module_name()

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


    def update_config(self, k, v):
        self.config[k] = v
        return self.config

    @classmethod
    def rm_lines(cls, path:str, start_line:int, end_line:int) -> None:
        # Get the absolute path of the file
        text = cls.get_text(path)
        text = text.split('\n')
        text = text[:start_line-1] + text[end_line:]
        text = '\n'.join(text)
        c.put_text(path, text)
        return {'success': True, 'msg': f'Removed lines {start_line} to {end_line} from {path}'}
    @classmethod
    def rm_line(cls, path:str, line:int, text=None) -> None:
        # Get the absolute path of the file
        text =  cls.get_text(path)
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
    def resolve_object(cls, module=None):
        """
        Resolves the moduls from the class 
        Case type(module):
            if None -> cls, the class method of the object
            if str -> c.module({module}) 
            if 


        
        """
        if module == None:
            module = cls
        if isinstance(module, str):
            module = c.module(module)
        return module


    thread_map = {}


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
 
        c.print(f'[bold cyan]Launching[/bold cyan] [bold yellow] class:{module.__name__}[/bold yellow] [bold white]name[/bold white]:{name} [bold white]fn[/bold white]:{fn} [bold white]mode[/bold white]:{mode}', color='green')

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
    
    @staticmethod
    def is_mnemonic(s: str) -> bool:
        import re
        # Match 12 or 24 words separated by spaces
        return bool(re.match(r'^(\w+ ){11}\w+$', s)) or bool(re.match(r'^(\w+ ){23}\w+$', s))

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
    def pool(cls , n=5, **kwargs):
        for i in range(n):
            cls.serve(tag=str(i), **kwargs)

    
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
    def resolve_key_address(cls, key):
        key2address = c.key2address()
        if key in key2address:
            address = key2address[key]
        else:
            address = key
        return address

    @classmethod
    def is_root_key(cls, address:str)-> str:
        return address == c.root_key().ss58_address

    @classmethod
    def getcwd(cls):
        return os.getcwd()

    def __repr__(self) -> str:
        return f'<{self.class_name()} tag={self.tag}>'
    def __str__(self) -> str:
        return f'<{self.class_name()} tag={self.tag}>'

    @classmethod
    def routes_path(cls):
        return cls.dirpath() + '/routes.yaml'

    @classmethod
    def has_routes(cls):
        
        return os.path.exists(cls.routes_path()) or (hasattr(cls, 'routes') and isinstance(cls.routes, dict)) 
    
    @classmethod
    def routes(cls):
        if not cls.has_routes():
            return {}
        return c.get_yaml(cls.routes_path())

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
        """
        This ties other modules into the current module.
        The way it works is that it takes the module name and the function name and creates a partial function that is bound to the module.
        This allows you to call the function as if it were a method of the current module.
        for example
        """
        my_path = cls.class_name()
        if not hasattr(cls, 'routes_enabled'): 
            cls.routes_enabled = False

        t0 = c.time()

        def fn_generator(*args, fn, module, **kwargs):
            module = c.module(module)
            fn_type = module.classify_fn(fn)
            if fn_type == 'self':
                module = module()
            else:
                module = module
            return getattr(module, fn)(*args, **kwargs)

        if routes == None:
            if not hasattr(cls, 'routes'):
                return {'success': False, 'msg': 'routes not found'}
            routes = cls.routes() if callable(cls.routes) else cls.routes
        for m, fns in routes.items():
            for fn in fns: 
                c.print(f'Enabling route {m}.{fn} -> {my_path}:{fn}', verbose=verbose)
                # resolve the from and to function names
                from_fn, to_fn = cls.resolve_to_from_fn_routes(fn)
                # create a partial function that is bound to the module
                fn_obj = partial(fn_generator, fn=from_fn, module=m )
                fn_obj.__name__ = to_fn
                # set the function to the current module
                setattr(cls, to_fn, fn_obj)
        t1 = c.time()
        c.print(f'enabled routes in {t1-t0} seconds', verbose=verbose)
        cls.routes_enabled = True
        return {'success': True, 'msg': 'enabled routes'}
    

    
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
            cls.thread_map[name].join()

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
    def root_fns(cls):
        if not hasattr(c, '_root_fns'):
            route_fns = c.route_fns()
            fns = c.get_module('module').fns()
            c._root_fns = [f for f in fns if f not in route_fns]
        return c._root_fns


    @classmethod
    def functions(cls, search: str=None , **kwargs):
        return cls.get_functions(search=search, **kwargs)  
    fns = functions


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
    def simple2object(cls, path:str,  path2objectpath = {'tree': 'commune.tree.Tree'}, **kwargs) -> str:
        print('tree')
        if path in path2objectpath:
            path = path2objectpath[path]
        else:
            path =  c.module('tree').simple2objectpath(path, **kwargs)
        return c.import_object(path)

    

    

c.enable_routes()
Module = c # Module is alias of c
Module.run(__name__)




