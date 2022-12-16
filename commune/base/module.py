from __future__ import annotations
from commune.utils import get_object, dict_any, dict_put, dict_get, dict_has, dict_pop, deep2flat, Timer, dict_override, get_functions, get_function_schema, kill_pid
import datetime

import streamlit as st
import os
import subprocess, shlex
import ray
import torch
import gradio as gr
import socket
import json
from importlib import import_module
from munch import Munch
import types
import inspect
from commune.ray.utils import kill_actor
from commune.config import Config
from copy import deepcopy
import argparse
import psutil
import gradio
import asyncio
from ray.experimental.state.api import list_actors, list_objects, list_tasks
import streamlit as st
import nest_asyncio
from typing import *
from glob import glob

from .utils import enable_cache
import inspect

class Module:
    client = None
    pwd = os.getenv('PWD')
    client_module_class_path = 'commune.client.manager.ClientModule'
    root_dir = __file__[len(pwd)+1:].split('/')[0]
    root_path = os.path.join(pwd, root_dir)
    root = root_path

    def __init__(self, config=None, override={}, client=None , **kwargs):
        
        self.config = Config()
        self.config = self.resolve_config(config=config, override=override)
        self.client = self.get_clients(client=client) 
        
        self.start_timestamp =self.current_timestamp
        self.get_submodules(get_submodules_bool = kwargs.get('get_submodules', True))
        self.cache = {}

    @property
    def registered_clients(self):
        return self.clients.registered_clients if self.clients else []
    @property
    def client_config(self):
        for k in ['client', 'clients']:
            client_config =  self.config.get(k, None)
            if client_config != None:
                return client_config
        return client_config


    def get_clients(self, client=None):
        if self.class_name == 'client.manager':
            # if this is the client manager, do not return clients
            return None
        if client == False:
            return None
        elif client == None:
            if self.client_config == None and client == None:
                return None
            client_module_class = self.import_object(self.client_module_class_path)
            client_config = client_module_class.default_config()
            # does the config have clients
            if isinstance(self.client_config, list):
                client_config['include'] = self.client_config
            elif isinstance(self.client_config, dict):
                client_config  = self.client_config
            elif self.client_config == None:
                return 
            return client_module_class(client_config)
        elif isinstance(client, client_module_class):
            return client
        elif isinstance(client, dict):
            client_module_class = self.import_object(self.client_module_class_path)
            client_config = client
            return client_module_class(client_config)
        elif isinstance(client, list):
            client_module_class = self.import_object(self.client_module_class_path)
            client_config['include'] = client
            return client_module_class(client_config)
        else:
            raise NotImplementedError

    @staticmethod
    def get_function_signature(fn) -> dict: 
        return dict(inspect.signature(fn)._parameters)

        
    @staticmethod
    def dict_override(*args, **kwargs):
        return dict_override(*args,**kwargs)

    def resolve_path(self, path, extension = '.json'):
        path = path.replace('.', '/')
        path = os.path.join(self.tmp_dir,path)
        path_dir = os.path.dirname(path)
        os.makedirs(path_dir,exist_ok=True)
        if path[-len(extension):] != extension:
            path = path + extension
        return path

    ############ JSON LAND ###############

    def get_json(self,path, default=None, **kwargs):
        path = self.resolve_path(path=path)
        try:
            data = self.client.local.get_json(path=path, **kwargs)
        except FileNotFoundError as e:
            if isinstance(default, dict):
                data = self.put_json(path, default)
            else:
                raise e

        return data

    def put_json(self, path, data, **kwargs):
        path = self.resolve_path(path=path)
        self.client.local.put_json(path=path, data=data, **kwargs)
        return data

    def ls_json(self, path=None):
        path = self.resolve_path(path=path)
        if not self.client.local.exists(path):
            return []
        return self.client.local.ls(path)
        
    def exists_json(self, path=None):
        path = self.resolve_path(path=path)
        return self.client.local.exists(path)

    def rm_json(self, path=None, recursive=True, **kwargs):
        path = self.resolve_path(path)
        if not self.client.local.exists(path):
            return 
    
        return self.client.local.rm(path,recursive=recursive, **kwargs)

    
    def glob_json(self, pattern ='**',  tmp_dir=None):
        if tmp_dir == None:
            tmp_dir = self.tmp_dir
        paths =  glob(tmp_dir+'/'+pattern)
        return list(filter(lambda f:os.path.isfile(f), paths))
    
    def refresh_json(self):
        self.rm_json()

    @staticmethod
    def resolve_config_path(path=None):
        if path ==  None:
            path = 'config'
        return path
    def put_config(self, path=None):
        path = self.resolve_config_path(path)
        return self.put_json(path, self.config)

    def rm_config(self, path=None):
        path = self.resolve_config_path(path)
        return self.rm_json(path)

    refresh_config = rm_config
    def get_config(self,  path=None, handle_error =True):
        path = self.resolve_config_path(path)
        config = self.get_json(path, handle_error=handle_error)
        if isinstance(config, dict):
            self.config = config


    @classmethod
    def simple2path(cls, simple:str, mode:str='config') -> str: 
        simple2path_map = getattr(cls, f'simple2{mode}_map')()
        module_path = simple2path_map[simple]
        return module_path

    @classmethod
    def path2simple(cls, path:str) -> str:
        return os.path.dirname(path)[len(cls.pwd)+1:].replace('/', '.')


    @classmethod
    def simple2import(cls, simple:str) -> str:
        config_path = Module.simple2path(simple, mode='config')
        module_basename = os.path.basename(config_path).split('.')[0]
        config = Config(config=config_path)
        obj_name = config.get('module', config.get('name'))
        module_path = '.'.join([simple, module_basename,obj_name])
        return module_path

    @classmethod
    def get_simple_paths(cls) -> List[str]:
        return [cls.path2simple(f) for f in cls.get_module_python_paths()]
    list_modules = get_simple_paths
    @property
    def module_tree(self): 
        return self.list_modules()

    module_list = module_tree
    @classmethod
    def simple2python_map(cls) -> Dict[str, str]:
        return {cls.path2simple(f):f for f in cls.get_module_python_paths()}

    @classmethod
    def simple2config_map(cls) -> Dict[str, str]:
        return {cls.path2simple(f):f for f in cls.get_module_config_paths()}


    @classmethod
    def get_module_python_paths(cls) -> List[str]:
        modules = []
        failed_modules = []

        for f in glob(Module.root_path + '/**/*.py', recursive=True):
            if os.path.isdir(f):
                continue
            file_path, file_ext =  os.path.splitext(f)
            if file_ext == '.py':
                if os.path.exists(file_path+'.yaml'):
                    modules.append(f)
        return modules


    @staticmethod
    def get_module_config_paths() -> List[str]:
        return [f.replace('.py', '.yaml')for f in  Module.get_module_python_paths()]

    def submit_fn(self, fn:str, queues:dict={}, block:bool=True,  *args, **kwargs):

        if queues.get('in'):
            input_item = self.queue.get(topic=queues.get('in'), block=block)
            if isinstance(input_item, dict):
                kwargs = input_item
            elif isinstance(input_item, list):
                args = input_item
            else:
                args = [input_item]
        
        out_item =  getattr(self, fn)(*args, **kwargs)

        if isinstance(queues.get('out'), str):
            self.queue.put(topic=queues.get('out'), item=out_item)
    
    def stop_loop(self, key='default'):
        return self.loop_running_loop.pop(key, None)

    def running_loops(self):
        return list(self.loop_running_map.keys())

    loop_running_map = {}
    def start_loop(self, in_queue=None, out_queue=None, key='default', refresh=False):
        
        in_queue = in_queue if isintance(in_queue,str) else 'in'
        out_queue = out_queue if isintance(out_queue,str) else 'out'
        
        if key in self.loop_running_map:
            if refresh:
                while key in self.loop_running_map:
                    self.stop_loop(key=key)
            else:
                return 
        else:
            self.loop_running_map[key] = True

        while key in self.loop_running_map:
            input_dict = self.queue.get(topic=in_queue, block=True)
            fn = input_dict['fn']
            fn_kwargs = input_dict.get('kwargs', {})
            fn_args = input_dict.get('args', [])
            output_dict  = self.submit_fn(fn=fn, *fn_args, **fn_kwargs)
            self.queue.put(topic=out_queue,item=output_dict)



    @classmethod
    def launch(cls, module:str=None, fn:str=None ,kwargs:dict={}, args:list=[], actor:Union[bool, dict]=False, ray_init:dict={}, **additional_kwargs):
        

        cls.ray_init(ray_init)


        if module == None:
            module = cls.get_module_path()

        if cls.is_class(module):
            module_class = module
        elif isinstance(module, str):
            module_class = cls.import_object(module)
        else:
            raise NotImplementedError(f'Type ({type(module)}) for module is not implemented')


 

        module_init_fn = fn
        module_kwargs = {**kwargs}
        module_args = [*args]
        
        if module_init_fn == None:
            if actor:
                # ensure actor is a dictionary
                if actor == True:
                    actor = {}
                assert isinstance(actor, dict), f'{type(actor)} should be dictionary fam'
                parents = cls.get_parents(module_class)
                if cls.is_module(module_class):
                    default_actor_name = module_class.get_default_actor_name()
                else:
                    default_actor_name = module_class.__name__

                actor['name'] = actor.get('name', default_actor_name )
                module_object = cls.create_actor(cls=module_class, cls_kwargs=module_kwargs, cls_args=module_args, **actor)

            else:
                module_object =  module_class(*module_args,**module_kwargs)

        else:
            module_init_fn = getattr(module_class,module_init_fn)
            module_object =  module_init_fn(*module_args, **module_kwargs)


        return module_object
    launch_module = launch

    @classmethod
    def load_module(cls, path):
        prefix = f'{cls.root_dir}.'
        if cls.root_dir+'.' in path[:len(cls.root_dir+'.')]:
            path = path.replace(f'{cls.root_dir}.', '')
        module_path = cls.simple2path(path)
        module_class =  cls.import_object(module_path)
        return module_class

    ############# TIME LAND ############
    @property
    def current_timestamp(self):
        return self.get_current_timestamp()

    def current_datetime(self):
        datetime.datetime.fromtimestamp(self.current_timestamp)
    
    def start_datetime(self):
        datetime.datetime.fromtimestamp(self.start_timestamp)
    
    def get_age(self) :
        return  self.get_current_timestamp() - self.start_timestamp

    age = property(get_age) 
    @staticmethod
    def get_current_timestamp():
        return  datetime.datetime.utcnow().timestamp()


    ############# CONFIG LAND ############


    @classmethod
    def get_config_path(cls, simple=False):

        config_path = cls.get_module_path(simple=simple)
        if os.getenv('PWD') != config_path[:len(os.getenv('PWD'))]:
            config_path = os.path.join(os.getenv('PWD'), cls.get_module_path(simple=simple))


        if simple == False:
            config_path = config_path.replace('.py', '.yaml')
        return config_path

    @staticmethod
    def check_config(config):
        assert isinstance(config, dict)
        assert 'module' in config

    @staticmethod
    def get_base_config_path()-> str:
        return Module.get_config_path()

    @staticmethod
    def get_base_config()-> str:
        return Config(Module.get_config_path())

    @property
    def config_path(self):
        config_path = self.get_config_path()
        if not os.path.exists(config_path):
            base_config = self.get_base_config()
            Config(base_config).save_config(path=config_path)
        
        assert os.path.exists(config_path)
        
        return config_path

    def resolve_config(self, config, override={}, **kwargs):
        config = self.load_config(config=config if config else self.config_path)

        if config == None:
            config = 'base'
            config = cls.resolve_config(config=config)
            # assert isinstance(config, dict),  f'bruh the config should be {type(config)}'
            self.save_config(config=config)

        return config


    def config_set(self,k,v, **kwargs):
        return dict_put(self.config, k,v)

    def config_get(self,k, ):
        return dict_get(self.config, k,v)

    def override_config(self,override:dict={}):
        self.dict_override(input_dict=self.config, override=override)

    @classmethod
    def load_config(cls, config_path:Optional[str]=None, config:Dict[str]=None, override={}):

        return Config(config= config if config else config_path)

    def save_config(self, path=None):
        """
        config: 
            Option 1: dictionary config (passes dictionary) 
            Option 2: absolute string path pointing to config
        """
        return Config.save_config(path=path)
    @classmethod
    def default_cfg(cls):
        config_path = cls.get_config_path()
        return Config(config_path=config_path)

    config_template = default_config = default_cfg

    @classmethod
    def import_module(cls, import_path:str) -> 'Object':
        # imports a python module or a module
        try:
            import_path = cls.simple2import(import_path)
            return cls.import_object(import_path)

        except KeyError as e:
            raise e
            return import_module(import_path)

    @classmethod
    def import_object(cls, key:str)-> 'Object':
        try:
            key = cls.simple2import(key)
        except KeyError as e:
            print(key, 'KEYERROR')
            pass
  

        module = '.'.join(key.split('.')[:-1])
        object_name = key.split('.')[-1]
        print(module, object_name,  'BROOO1',  import_module(module)) 
        obj =  getattr(import_module(module), object_name)
        print(obj,  'BROOO2') 
        return obj
    get_object = import_object
    
    @property
    def module(self):
        return self.config.get('module', self.config.get('name'))

    @property
    def name(self):
        return self.config.get('name', self.module)
    
    def class_name(self):
        return self.__class__.__name__

    ################# ATTRIBUTE LAND ####################
    #####################################################


    def get(self, key):
        return self.getattr(key)

    def getattr(self, key):
        return getattr(self, key)

    def hasattr(self, key):
        return hasattr(self, key)

    def setattr(self, key, value):
        return self.__setattr__(key,value)

    def deleteattr(self, key):
        del self.__dict__[key]
        return key
    
    rmattr = rm = delete = deleteattr


    def mapattr(self, from_to_attr_dict:dict={}) -> Dict :
        '''
        from_to_attr_dict: dict(from_key:str->to_key:str)
        '''
        for from_key, to_key in from_to_attr_dict.items():
            self.copyattr(from_key=from_key, to_key=to_key)

    def copyattr(self, from_key, to_key):
        '''
        copy from and to a desintatio
        '''
        attr_obj = getattr(self, from_key)  if hasattr(self, from_key) else None
        setattr(self, to, attr_obj)

    def dict_keys(self):
        return self.__dict__.keys()

    @staticmethod
    def is_class(cls):
        '''
        is the object a class
        '''
        return type(cls).__name__ == 'type'


    @staticmethod
    def is_hidden_function(fn):
        if isinstance(fn, str):
            return fn.startswith('__') and fn.endswith('__')
        else:
            raise NotImplemented(f'{fn}')

    @staticmethod
    def get_functions(object):
        functions = get_functions(object)
        return functions

    @staticmethod
    def get_function_schema( fn, *args, **kwargs):
        return get_function_schema(fn=fn, *args, **kwargs)

    @classmethod
    def get_function_schemas(cls, obj=None, *args,**kwargs):
        if obj == None:
            obj = cls
        
        fn_map = {}
        for fn_key in obj.get_functions(obj):
            fn = getattr(obj, fn_key)
            if not callable(fn) or isinstance(fn, type) or isinstance(fn, types.BuiltinFunctionType):
                continue
            fn_map[fn_key] = cls.get_function_schema(fn=fn, *args, **kwargs)
        return fn_map
    
    @staticmethod
    def get_annotations(fn:callable) -> dict:
        return fn.__annotations__
    
    def is_parent(child, parent):
        return bool(parent in Module.get_parents(child))

    @classmethod
    def get_parents(cls, obj=None):
        if obj == None:
            obj = cls

        return list(obj.__mro__[1:-1])

    @classmethod
    def is_module(cls, obj=None):
        if obj == None:
            obj = cls
        return Module in cls.get_parents(obj)

    @classmethod
    def functions(cls, obj=None, return_type='str', **kwargs):
        if obj == None:
            obj = cls
        functions =  get_functions(obj=obj, **kwargs)
        if return_type in ['str', 'string']:
            return functions
        
        elif return_type in ['func', 'fn','functions']:
            return [getattr(obj, f) for f in functions]
        else:
            raise NotImplementedError

    @classmethod
    def hasfunc(cls, key):
        fn_list = cls.functions()
        return bool(len(list(filter(lambda f: f==key, fn_list)))>0)

    @classmethod
    def filterfunc(cls, key):
        fn_list = cls.functions()
        ## TODO: regex
        return list(filter(lambda f: key in f, fn_list))


    ############ STREAMLIT LAND #############

    @classmethod
    def run_streamlit(cls, port=8501):
        path = cls.get_module_path(simple=False)
        cls.run_command(f'streamlit run {path} --server.port={port} -- -fn=streamlit')

    @classmethod
    def streamlit(cls):
        st.write(f'HELLO from {cls.__name__}')
        st.write(cls)


    @classmethod
    def describe(cls, obj=None, streamlit=False, sidebar=True,**kwargs):
        if obj == None:
            obj = cls

        assert is_class(obj)

        fn_list = cls.functions(return_type='fn', obj=obj, **kwargs)
        
        fn_dict =  {f.__name__:f for f in fn_list}
        if streamlit:
            import streamlit as st
            for k,v in fn_dict.items():
                with (st.sidebar if sidebar else st).expander(k):
                    st.write(k,v)
        else:
            return fn_dict
        
    ############ GRADIO LAND #############

    @classmethod
    def gradio(cls):
        functions, names = [], []

        fn_map = {
            'fn1': lambda x :  int(x),
            'fn2': lambda x :  int(x) 
        }

        for fn_name, fn in fn_map.items():
            inputs = [gr.Textbox(label='input', lines=3, placeholder=f"Enter here...")]
            outputs = [gr.Number(label='output', precision=None)]
            names.append(fn_name)
            functions.append(gr.Interface(fn=fn, inputs=inputs, outputs=outputs))
        
        return gr.TabbedInterface(functions, names)


    @classmethod
    def run_gradio(cls, port=8501, host='0.0.0.0'):
        path = cls.get_module_path(simple=False)
        interface = cls.gradio()
 
        interface.launch(server_port=port,
                        server_name=host,
                        inline= False,
                        share= None,
                        debug=False,
                        enable_queue= None,
                        max_threads=10,
                        auth= None,
                        auth_message= None,
                        prevent_thread_lock= False,
                        show_error= True,
                        show_tips= False,
                        height= 500,
                        width= 900,
                        encrypt= False,
                        favicon_path= None,
                        ssl_keyfile= None,
                        ssl_certfile= None,
                        ssl_keyfile_password= None,
                        quiet= False)
        


    @classmethod
    def run_python(cls):
        cls.run_command(f'python {path}')

    @classmethod
    def argparse(cls):
        parser = argparse.ArgumentParser(description='Gradio API and Functions')
        parser.add_argument('-fn', '--function', dest='function', help='run a function from the module', type=str, default="streamlit")
        parser.add_argument('-kwargs', '--kwargs', dest='kwargs', help='arguments to the function', type=str, default="{}")  
        parser.add_argument('-args', '--args', dest='args', help='arguments to the function', type=str, default="[]")  
        return parser.parse_args()


    @classmethod
    def parents(cls):
        return get_parents(cls)

    # timer
    timer = Timer

    @classmethod
    def describe_module_schema(cls, obj=None, **kwargs):
        if obj == None:
            obj = cls
        return get_module_function_schema(obj, **kwargs)

    @property
    def module_path(self):
        return self.get_module_path()

    @staticmethod
    def run_command(command:str):

        process = subprocess.run(shlex.split(command), 
                            stdout=subprocess.PIPE, 
                            universal_newlines=True)
        return process

    @property
    def tmp_dir(self):
        return f'/tmp/{self.root_dir}/{self.class_name}'

    @classmethod
    def get_module_path(cls, obj=None,  simple=True):
        if obj == None:
            obj = cls
        module_path =  inspect.getmodule(obj).__file__
        # convert into simple
        if simple:
            module_path = cls.path2simple(path=module_path)

        return module_path


    ###########################
    #   RESOURCE LAND
    ###########################
    @staticmethod
    def check_pid(pid):        
        return check_pid(pid)

    @staticmethod
    def kill_pid(pid):        
        return kill_pid(pid)
    @property
    def pid(self):
        return os.getpid()

    def memory_usage(self, mode='gb'):
        '''
        get memory usage of current process in bytes
        '''
        import os, psutil
        process = psutil.Process(self.pid)
        usage_bytes = process.memory_info().rss
        usage_percent = process.memory_percent()
        mode_factor = 1

        if mode  in ['gb']:
            mode_factor = 1e9
        elif mode in ['mb']:
            mode_factor = 1e6
        elif mode in ['b']:
            mode_factor = 1
        elif mode in ['percent','%']:
            return usage_percent
        elif mode in ['ratio', 'fraction', 'frac']: 
            return usage_percent / 100
        else:
            raise Exception(f'{mode} not supported, try gb,mb, or b where b is bytes')

        return usage_bytes / mode_factor

    @staticmethod
    def memory_available(mode ='percent'):

        memory_info = Module.memory_info()
        available_memory_bytes = memory_info['available']
        available_memory_ratio = (memory_info['available'] / memory_info['total'])
    
        mode_factor = 1
        if mode  in ['gb']:
            mode_factor = 1e9
        elif mode in ['mb']:
            mode_factor = 1e6
        elif mode in ['b']:
            mode_factor = 1
        elif mode in ['percent','%']:
            return  available_memory_ratio*100
        elif mode in ['fraction','ratio']:
            return available_memory_ratio
        else:
            raise Exception(f'{mode} not supported, try gb,mb, or b where b is bytes')

        return usage_bytes / mode_factor


    @staticmethod
    def memory_used(mode ='percent'):

        memory_info = Module.memory_info()
        available_memory_bytes = memory_info['used']
        available_memory_ratio = (memory_info['used'] / memory_info['total'])
    
        mode_factor = 1
        if mode  in ['gb']:
            mode_factor = 1e9
        elif mode in ['mb']:
            mode_factor = 1e6
        elif mode in ['b']:
            mode_factor = 1
        elif mode in ['percent','%']:
            return  available_memory_ratio*100
        elif mode in ['fraction','ratio']:
            return available_memory_ratio
        else:
            raise Exception(f'{mode} not supported, try gb,mb, or b where b is bytes')

        return usage_bytes / mode_factor

    @staticmethod
    def memory_info():
        virtual_memory = psutil.virtual_memory()
        return {k:getattr(virtual_memory,k) for k in ['available', 'percent', 'used', 'shared', 'free', 'total', 'cached']}

    @staticmethod
    def get_memory_info(pid:int = None):
        if pid == None:
            pid = os.getpid()
        # return the memory usage in percentage like top
        process = psutil.Process(pid)
        memory_info = process.memory_full_info()._asdict()
        memory_info['percent'] = process.memory_percent()
        memory_info['ratio'] = memory_info['percent'] / 100
        return memory_info


    def resource_usage(self):
        resource_dict =  self.config.get('actor', {}).get('resources', None)
        resource_dict = {k.replace('num_', ''):v for k,v in resource_dict.items()}
        resource_dict['memory'] = self.memory_usage(mode='ratio')
        return  resource_dict


    ##############
    #   RAY LAND
    ##############
    @classmethod
    def get_default_actor_name(cls):
        return cls.get_module_path(simple=True)


    @classmethod
    def ray_stop(cls):
        cls.run_command('ray stop')

    @classmethod
    def ray_start(cls):
        cls.run_command('ray start --head')


    @classmethod
    def ray_restart(cls):
        cls.ray_stop()
        cls.ray_start()

    @classmethod
    def ray_status(cls):
        cls.run_command('ray status')

    @staticmethod
    def ray_initialized():
        return ray.is_initialized()

    @property
    def actor_id(self):
        return self.get_id()

    def get_id(self):
        return dict_get(self.config, 'actor.id')
        
    def get_name(self):
        return dict_get(self.config, 'actor.name')

    def actor_info(self):
        actor_info_dict = dict_get(self.config, 'actor')
        actor_info_dict['resources'] = self.resource_usage()
        return actor_info_dict


    @classmethod 
    def deploy(cls, actor=False ,  **kwargs):
        """
        deploys process as an actor or as a class given the config (config)
        """

        config = kwargs.pop('config', None)
        path = kwargs.pop('path', None)
        if isinstance(config, str):
            path = config
        elif isinstance(config, dict):
            path = kwargs.pop('path', None)
         
        config = Config(config_path = path, config=config)

        ray_config = config.get('ray', {})
        if not cls.ray_initialized():
            ray_context =  cls.init_ray(init_kwargs=ray_config)
        
        if actor:
            actor_config =  config.get('actor', {})

            assert isinstance(actor_config, dict), f'actor_config should be dict but is {type(actor_config)}'
            if isinstance(actor, dict):
                actor_config.update(actor)
            elif isinstance(actor, bool):
                pass
            else:
                raise Exception('Only pass in dict (actor args), or bool (uses config["actor"] as kwargs)')  

            try:

                actor_config['name'] =  actor_config.get('name', cls.get_default_actor_name())                
                config['actor'] = actor_config
                kwargs['config'] = config
                actor = cls.create_actor(cls=cls,  cls_kwargs=kwargs, **actor_config)

            except ray.exceptions.RayActorError:
                actor_config['refresh'] = True
                config['actor'] = actor_config
                kwargs['config'] = config
                actor = cls.create_actor(cls=cls, cls_kwargs=kwargs, **actor_config)


            return actor 
        else:
            kwargs['config'] = config
            kwargs['config']['actor'] = None
            return cls(**kwargs)

    default_ray_env = {'address':'auto', 
                     'namespace': 'default',
                      'ignore_reinit_error': False,
                      'dashboard_host': '0.0.0.0'}
    @classmethod
    def ray_init(cls,init_kwargs={}):

        # init_kwargs['_system_config']={
        #     "object_spilling_config": json.dumps(
        #         {"type": "filesystem", "params": {"directory_path": "/tmp/spill"}},
        #     )
        # }
        init_kwargs =  {**cls.default_ray_env, **init_kwargs}
        if cls.ray_initialized():
            # shutdown if namespace is different
            if cls.ray_namespace() == cls.default_ray_env['namespace']:
                return cls.ray_runtime_context()
            else:
                ray.shutdown()
  
        ray_context = ray.init(**init_kwargs)
        return ray_context

    init_ray = ray_init
    @staticmethod
    def create_actor(cls,
                 name, 
                 cls_kwargs,
                 cls_args =[],
                 detached=True, 
                 resources={'num_cpus': 1.0, 'num_gpus': 0},
                 cpus = 0,
                 gpus = 0,
                 max_concurrency=50,
                 refresh=False,
                 return_actor_handle=False,
                 verbose = True,
                 redundant=False,
                 tag_seperator = '-',
                 tag = None,
                 wrap = False,
                 **kwargs):

        if cpus > 0:
            resources['num_cpus'] = cpus
        if gpus > 0:
            resources['num_gpus'] = gpus

        if not torch.cuda.is_available() and 'num_gpus' in resources:
            del resources['num_gpus']

        # configure the option_kwargs

        if tag != None:
            tag = str(tag)
            name = tag_seperator.join([name, tag])

        options_kwargs = {'name': name,
                          'max_concurrency': max_concurrency,
                           **resources}
        if detached:
            options_kwargs['lifetime'] = 'detached'
        # setup class init config
        # refresh the actor by killing it and starting it (assuming they have the same name)
        
        if refresh:
            if Module.actor_exists(name):
                kill_actor(actor=name,verbose=verbose)
                # assert not Module.actor_exists(name)

        if redundant:
            # if the actor already exists and you want to create another copy but with an automatic tag
            actor_index = 0
            while Module.actor_exists(name):
                name =  f'{name}-{actor_index}' 
                actor_index += 1

        if not Module.actor_exists(name):
            actor_class = ray.remote(cls)
            actor_handle = actor_class.options(**options_kwargs).remote(*cls_args, **cls_kwargs)

        actor = Module.get_actor(name)

        if wrap:
            actor = Module.wrap_actor(actor)

        return actor

    @staticmethod
    def get_actor_id( actor):
        assert isinstance(actor, ray.actor.ActorHandle)
        return actor.__dict__['_ray_actor_id'].hex()

    @classmethod
    def create_pool(cls, replicas=3, actor_kwargs_list=[], **kwargs):
        if actor_list == None:
            actor_kwargs_list = [kwargs]*replicas

        actors = []
        for actor_kwargs in actor_kwargs_list:
            actors.append(cls.deploy(**a_kwargs))

        return ActorPool(actors=actors)

    @classmethod
    def wrap_actor(cls, actor):
        wrapper_module_path = 'commune.ray.client.module.ClientModule'
        return Module.get_module(module=wrapper_module_path, server=actor)

    @classmethod
    def deploy_module(cls, module:str, **kwargs):
        module_class = cls.import_object(module)
        return module_class.deploy(**kwargs)
    get_module = deploy_module

    @staticmethod
    def kill_actor(actor, verbose=True):

        if isinstance(actor, str):
            if Module.actor_exists(actor):
                actor = ray.get_actor(actor)
            else:
                if verbose:
                    print(f'{actor} does not exist for it to be removed')
                return None
        
        return ray.kill(actor)
        
    @staticmethod
    def kill_actors(actors):
        return_list = []
        for actor in actors:
            return_list.append(Module.kill_actor(actor))
        
        return return_list
            
    @staticmethod
    def actor_exists(actor):
        if isinstance(actor, str):
            try:
                ray.get_actor(actor)
                actor_exists = True
            except ValueError as e:
                actor_exists = False
            
            return actor_exists
        else:
            raise NotImplementedError

    @staticmethod
    def get_actor(actor_name, wrap=False):
        actor =  ray.get_actor(actor_name)
        # actor = Module.add_actor_metadata(actor)
        if wrap:
            actor = Module.wrap_actor(actor=actor)
        return actor

    @property
    def ray_context(self):
        return self.init_ray()

    @staticmethod
    def ray_runtime_context():
        return ray.get_runtime_context()

    @classmethod
    def ray_namespace(cls):
        return ray.get_runtime_context().namespace

    @staticmethod
    def get_ray_context():
        return ray.runtime_context.get_runtime_context()
    @property
    def context(self):
        if Module.actor_exists(self.actor_name):
            return self.init_ray()

    @property
    def actor_name(self):
        actor_config =  self.config.get('actor', {})
        if actor_config == None:
            actor_config = {}
        return actor_config.get('name')
    
    @property
    def default_actor_name(self):
        return self.get_module_path(simple=True)
    @property
    def actor_running(self):
        return self.is_actor_running

    def is_actor_running(self):
        return isinstance(self.actor_name, str)

    @property
    def actor_config(self):
        return self.config.get('actor',None)

    @property
    def actor_handle(self):
        if not hasattr(self, '_actor_handle'):
            self._actor_handle = self.get_actor(self.actor_name)
        return self._actor_handle


    @staticmethod
    def list_objects( *args, **kwargs):
        return ray.experimental.state.api.list_objects(*args, **kwargs)

    @staticmethod
    def list_actors(state='ALIVE', detail=True, *args, **kwargs):
        kwargs['filters'] = kwargs.get('filters', [("state", "=", state)])
        kwargs['detail'] = detail

        actor_info_list =  list_actors(*args, **kwargs)
        final_info_list = []
        for i, actor_info in enumerate(actor_info_list):
            resource_map = {'memory':  Module.get_memory_info(pid=actor_info['pid'])}
            resource_list = actor_info_list[i].pop('resource_mapping', [])

            for resource in resource_list:
                resource_map[resource['name'].lower()] = resource['resource_ids']
            actor_info_list[i]['resources'] = resource_map

            try:
                ray.get_actor(actor_info['name'])
                final_info_list.append(actor_info_list[i])
            except ValueError as e:
                pass

        return final_info_list

    @staticmethod
    def actor_map(*args, **kwargs):
        actor_list = Module.list_actors(*args, **kwargs)
        actor_map  = {}
        for actor in actor_list:
            actor_name = actor.pop('name')
            actor_map[actor_name] = actor
        return actor_map

    @staticmethod   
    def list_actor_names():
        return list(Module.actor_map().keys())

    @staticmethod
    def list_tasks(running=False, name=None, *args, **kwargs):
        filters = []
        if running == True:
            filters.append([("scheduling_state", "=", "RUNNING")])
        if isinstance(name, str):
            filters.append([("name", "=", name)])
        
        if len(filters)>0:
            kwargs['filters'] = filters

        return ray.experimental.state.api.list_tasks(*args, **kwargs)


    @staticmethod
    def list_nodes( *args, **kwargs):
        return list_nodes(*args, **kwargs)

    @staticmethod
    def ray_get(self, *jobs):
        return ray.get(jobs)

    @staticmethod
    def ray_wait( *jobs):
        finished_jobs, running_jobs = ray.wait(jobs)
        return finished_jobs, running_jobs

    @staticmethod
    def ray_put(*items):
        return [ray.put(i) for i in items]


    ##############
    #   ASYNCIO
    ##############
    @staticmethod
    def new_event_loop(set_loop=True):
        loop = asyncio.new_event_loop()
        if set_loop:
            asyncio.set_event_loop(loop)
        return loop
    new_loop = new_event_loop 

    def set_event_loop(self, loop=None, new=False):
        if loop == None:
            loop = self.new_event_loop()
        return loop
    set_loop = set_event_loop
    def get_event_loop(self):
        return asyncio.get_event_loop()     
    def async_run(self, job, loop=None): 
        if loop == None:
            loop = self.loop
        return loop.run_until_complete(job)

    @staticmethod
    def port_connected( port : int,host:str='0.0.0.0'):
        """
            Check if the given param port is already running
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)       
        result = s.connect_ex((host, int(port)))
        return result == 0



    @classmethod
    def run(cls): 
        input_args = cls.argparse()
        assert hasattr(cls, input_args.function)
        kwargs = json.loads(input_args.kwargs)
        assert isinstance(kwargs, dict)

        args = json.loads(input_args.args)
        assert isinstance(args, list)
        getattr(cls, input_args.function)(*args, **kwargs)

    ############ TESTING LAND ##############

    ############################################

    @classmethod
    def test(cls):
        import streamlit as st
        for attr in dir(cls):
            if attr[:len('test_')] == 'test_':
                getattr(cls, attr)()
                st.write('PASSED',attr)


    ##### SOON TO BE REMOVED ##########


    def get_submodules(self, submodule_configs=None, get_submodules_bool=True):
        
        if get_submodules_bool == False:
            return None
        '''
        input: dictionary of modular configs
        '''
        if submodule_configs == None:
            submodule_configs = self.config.get('submodule',self.config.get('submodules',{}))
    
        assert isinstance(submodule_configs, dict)
        for submodule_name, submodule in submodule_configs.items():
            submodule_kwargs, submodule_args = {},[]
            if isinstance(submodule, str):
                submodule_kwargs = {'module':submodule }
            elif isinstance(submodule, list):
                submodule_args = submodule
            elif isinstance(submodule, dict):
                submodule_kwargs = submodule
                
            submodule = self.get_module(*submodule_args,**submodule_kwargs)
            dict_put(self.__dict__, submodule_name, submodule)


    @property
    def state_staleness(self):
        return self.current_timestamp - self.last_saved_timestamp

    ############ LOCAL CACHE LAND ##############

    ############################################

    cache = {}

    @enable_cache()
    def put_cache(self, k, v, **kwargs):
        dict_put(self.cache, k, v)
    @enable_cache()
    def get_cache(self, k, default=None, **kwargs):
        return dict_get(self.cache, k,default)

    @enable_cache(save= {'disable':True})
    def in_cache(self, k):
        return dict_has(self,cache, k)
    has_cache = in_cache
    @enable_cache()
    def pop_cache(self, k):
        return dict_pop(self.cache, k)


    def load_cache(self, **kwargs):
        enable_bool =  kwargs.get('enable', True)
        assert isinstance(enable_bool, bool), f'{disable_bool}'
        if not enable_bool:
            return None
        path = kwargs.get('path',  self.cache_path)

        self.client.local.makedirs(os.path.dirname(path), True)
        data = self.client.local.get_json(path=path, handle_error=True)
        
        if data == None:
            data  = {}
        self.cache = data

    def save_cache(self, **kwargs):
        enable_bool =  kwargs.get('enable', True)
        assert isinstance(enable_bool, bool), f'{disable_bool}'
        if not enable_bool:
            return None

        path = kwargs.get('path',  self.cache_path)

        staleness_period=kwargs.get('statelness_period', 100)
  
        self.client.local.makedirs(os.path.dirname(path), True)
        data =  self.cache
        self.client.local.put_json(path=path, data=data)

    save_state = save_cache
    load_state = load_cache
    
    @property
    def refresh_cache_bool(self):
        refresh_bool = self.config.get('refresh_cache', False)
        if refresh_bool == False:
            refresh_bool = self.config.get('cache', False)
        
        return refresh_bool

    def init_cache(self):
        if self.refresh_cache_bool:
            self.cache = {}
            self.save_cache()
        self.load_cache()

    def reset_cache(self):
        self.cache = {}
        self.save_cache()


    del_cache = delete_cache = pop_cache 
    has_cache = cache_has = cache_exists = exists_cache =in_cache
    last_saved_timestamp=0
    @staticmethod
    def enable_cache(**input_kwargs):
        return enable_cache(**input_kwargs)

    @classmethod
    def cache(cls,keys=None,**kwargs):
        return cache(keys=keys, **kwargs)
    enable_cache = cache_enable = cache_wrap = enable_cache

    @property
    def cache_path(self):
        return os.path.join(self.tmp_dir, 'cache.json')


    @staticmethod
    def gradio_build_interface( fn_map:dict) -> 'gradio.TabbedInterface':
        names, functions = [], []
        for fn_name, fn_obj in fn_map.items():
            inputs = fn_obj.get('inputs', [])
            outputs = fn_obj.get('outputs',[])
            fn = fn_obj['fn']
            names.append(fn_name)
            functions.append(gradio.Interface(fn=fn, inputs=inputs, outputs=outputs))
        
        return gradio.TabbedInterface(functions, names)


if __name__ == '__main__':
    Module.run()
