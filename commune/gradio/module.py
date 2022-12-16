

import os, sys
sys.path.append(os.environ['PWD'])
import gradio
from commune import Module
from inspect import getfile
import inspect
import socket
from commune.utils import *
from copy import deepcopy
# from commune.thread import PriorityThreadPoolExecutor
import argparse
import streamlit as st
import argparse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn



class GradioModule(Module):


    def __init__(host='0.0.0.0',
                 num_ports=10, 
                 port_range=[7865, 7870]):
        
        self.host  = host
        self.num_ports = num_ports
        self.port_range =  port_range
        
    def find_registered_functions(self, module:str):
        '''
        find the registered functions
        '''
        fn_keys = []
        self.get_module
        for fn_key in self.get_functions(module):
            try:
                if getattr(getattr(getattr(self,fn_key), '__decorator__', None), '__name__', None) == GradioModule.register.__name__:
                    fn_keys.append(fn_key)
            except:
                continue
        return fn_keys

    @staticmethod
    def has_registered_functions(self):
        '''
        find the registered functions
        '''
        for fn_key in GradioModule.get_funcs(self):
            if getattr(getattr(getattr(self,fn_key), '__decorator__', None), '__name__', None) == GradioModule.register.__name__:
                return True


        return False

    def compile_module(self, module:str, live=False, flagging='never', theme='default', **kwargs):
        print("Just putting on the finishing touches... 🔧🧰")
        module_class = self.get_object(module)
        module = module_class()

        gradio_functions_schema = self.get_gradio_function_schemas(module)


        interface_fn_map = {}

        for fn_key, fn_params in gradio_functions_schema.items():                
            interface_fn_map[fn_key] = gradio.Interface(fn=getattr(module, fn_key),
                                        inputs=fn_params['input'],
                                        outputs=fn_params['output'],
                                        theme=theme)
            print(f"{fn_key}....{bcolor.BOLD}{bcolor.OKGREEN} done {bcolor.ENDC}")


        print("\nHappy Visualizing... 🚀")
        demos = list(interface_fn_map.values())
        names = list(interface_fn_map.keys())
        return gradio.TabbedInterface(demos, names)

    @staticmethod
    def register(inputs, outputs):
        def register_gradio(func):
               
            def wrap(self, *args, **kwargs):     
                if not hasattr(self, 'registered_gradio_functons'):
                    print("✨Initializing Class Functions...✨\n")
                    self.registered_gradio_functons = dict()

                fn_name = func.__name__ 
                if fn_name in self.registered_gradio_functons: 
                    result = func(self, *args, **kwargs)
                    return result
                else:
                    self.registered_gradio_functons[fn_name] = dict(inputs=inputs, outputs=outputs)
                    return None

            wrap.__decorator__ = GradioModule.register
            return wrap
        return register_gradio
    
    def get_gradio_modules(self):
        return list(self.get_module_schemas().keys())

    
    @staticmethod
    def get_module_function_schema(module):
        if isinstance(module,str):
            module = get_object(module)
        module_schema = get_module_function_schema(module)
        return module_schema
        
    @staticmethod
    def python2gradio(self, value):
        v_type =type(value).__name__
        gradio_type = None
        if v_type == 'int':
            gradio_type = 'Number'
        elif v_type == 'str':
            gradio_type = 'Textbox'
        elif v_type == 'bool':
            gradio_type = 'Checkbox'
        elif v_type in ['dict']:
            gradio_type = 'JSON'
        else:
            raise NotImplementedError(v_type)

        return getattr(self, gradio_type)

    @staticmethod
    def schema2gradio(fn_schema, return_type='dict'):
        gradio_schema = {}
        fn_example = fn_schema['example']
        gradio_schema['example'] = fn_example
        for m in ['input', 'output']:
            gradio_schema[m] = []
            for k,v in fn_example[m].items():
                gradio_object = self.python2gradio(value=v)
                gradio_schema[m] += [gradio_object(value=v, label=k)]
        return gradio_schema

    def get_gradio_function_schemas(self, module, return_type='gradio'):
        if isinstance(module, str):
            module = get_object(module)
        function_defaults_dict = get_module_function_defaults(module)
        function_defaults_dict = get_full_functions(module_fn_schemas=function_defaults_dict)

        gradio_fn_schema_dict = {}

        for fn, fn_defaults in function_defaults_dict.items():
            module_fn_schema = get_function_schema(defaults_dict=fn_defaults)
            module_fn_schema['example'] = fn_defaults
            gradio_fn_schema_dict[fn] = self.schema2gradio(module_fn_schema)

            gradio_fn_list = []
            if return_type in ['json', 'dict']:
                for m in ['input', 'output']:
                    for gradio_fn in gradio_fn_schema_dict[fn][m]:
                        gradio_fn_list += [{'__dict__': gradio_fn.__dict__, 
                                            'module': f'gradio.{str(gradio_fn.__class__.__name__)}'}]
                    gradio_fn_schema_dict[fn][m] =  gradio_fn_list
            elif return_type in ['gradio']:
                pass
            else:
                raise NotImplementedError


        return gradio_fn_schema_dict

    def get_module_schemas(self,filter_complete=False):
        module_schema_map = {}
        module_paths = self.get_modules()

        for module_path in module_paths:

            module_fn_schemas = get_module_function_schema(module_path)

            if len(module_fn_schemas)>0:
                module_schema_map[module_path] = module_fn_schemas
        

        return module_schema_map


    def rm(self, port:int=None, module:str=None):
        module2port = self.port2module
        if port == None:
            port = None
            for p, m in self.port2module.items():
                if module == m:
                    port = p
                    break
        
        assert type(port) in [str, int], f'{type(port)}'
        port = str(port)

        if port not in self.port2module:
            print(f'rm: {port} is already deleted')
            return None
        return self.subprocess_manager.rm(key=port)
    def rm_all(self):
        for port in self.port2module:
            self.rm(port=port)

    def ls_ports(self):
        return self.subprocess_manager.ls()

    def add(self,module:str, port:int, mode:str):
        module = self.resolve_module_path(module)
        module = self.get_object(module).get_module_filepath()
        print(module, __file__ , 'DEBUG')
        command_map ={
            'gradio':  f'python {__file__} --module={module} --port={port}',
            'streamlit': f'streamlit run {module} --server.port={port}'
        }

        command  = command_map[mode]
        process = self.subprocess_manager.add(key=str(port), command=command, add_info= {'module':module })
        return {
            'module': module,
            'port': port,
        }
    submit = add

    def resolve_module_path(self, module):
        simple2path = deepcopy(self.simple2path)
        module_list = list(simple2path.values())

        if module in simple2path:
            module = simple2path[module]
    
        assert module in module_list, f'{module} not found in {module_list}'
        return module

    def launch(self, interface:gradio.Interface=None, 
                    module:str=None, 
                    **kwargs):
        """
            @params:
                - name : string
                - interface : gradio.Interface(...)
                - **kwargs
            
            Take any gradio interface object 
            that is created by the gradio 
            package and send it to the flaks api
        """

        default_kwargs = dict(
                    
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
                        quiet= False
        )


        kwargs = {**default_kwargs, **kwargs}

        kwargs["port"] = kwargs.pop('port', self.suggest_port()) 
        if kwargs["port"] == None:
            return {'error': 'Ports might be full'}
        kwargs["server_port"] = kwargs.pop('port')
        kwargs['server_name'] = self.host

        
        module = self.resolve_module_path(module)
        
        if interface == None:
    
            assert isinstance(module, str)
            module_list = self.get_modules()
            assert module in module_list, f'{args.module} is not in {module_list}'
            interface = self.compile(module=module)

        return interface.launch(**kwargs)


register = GradioModule.register


