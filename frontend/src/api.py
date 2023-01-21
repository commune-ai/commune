from fastapi import FastAPI
from pydantic import BaseModel
from typing import *

import json 

class Type:
    @property
    def state(self):
        state_dict =  {k:v for k, v in self.__dict__.items() if k != 'self'}
        return state_dict
    
class Task(Type):

    def __init__(self,
                 name: str = 'CausalLMNext',
                 metrics = {'CE': {'score': 1.27 , 'rank': 1}},
                 calls: int = 10,
    ):
        self.__dict__.update(locals())
import torch
class DataType(Type):
    def __init__(self, value = torch.randn([10, 10])):
        
        self.type = self.get_str_type(value)

        if self.type == 'torch.Tensor':
            self.shape = value.shape
            self.dtype = str(value.dtype)

    @staticmethod
    def get_str_type(data):
        return str(type(data)).split("'")[1]
                 
print(DataType(value=torch.randn([10, 10])).state)

class Model(Type):
    def __init__(self, 
                 name: str = 'gpt',
                 tasks=[Task().state],
                 input: dict = {'input_ids': DataType(torch.randn([10, 32])).state},
                 output: dict = {'logits': DataType(torch.randn([10, 32])).state},
                 ip= '0.0.0.0:8505',
            
                 ):
        self.__dict__.update(locals())

class Dataset(Type):
    def __init__(self, 
                 name: str = 'gpt',
                 tasks=[Task().state],
                 input: dict = {'batch_size': 10, 'tokenize': True},
                 output: dict = {'logits': DataType().state},
                 ip= '0.0.0.0:8505',
            
                 ):
        self.__dict__.update(locals())
        
                   
module_tree = {}



class Module:
    module_tree = module_tree
    def __init__(self, demo_mode:bool = True):
        if demo_mode:
            self.set_demo_mode_params()
        self.module_name = 'module'

    def predict(self, data: dict= {}):
        return f"Prediction for {self.name}: {data}"

    def list_modules(self):
        return list(self.module_tree.values())
    
    @property
    def module_list(self): 
        return self.list_modules()
    
    def get_module(self, name:str):
        module = self.module_tree.get(name)
        print(module)
        return module
        
    def resolve_module(self, module: Any = None):
        
        module = module if module else self
        
        return module


    @classmethod
    def api(cls, *cls_args, **cls_kwargs):
        app = FastAPI()
        self = cls(*cls_args, **cls_kwargs)


        @app.post("/list_modules")
        async def list_modules():
            return self.list_modules()

        @app.post("/get_module")
        async def get_module(name:str):
            return self.get_module(name)
        @app.post("/fn/{fn}")
        async def call_api(fn:str, module:str = None, kwargs: str = None, args:str = None):
            
            kwargs = {} if kwargs == None else json.loads(kwargs)
            args = [] if args == None else json.loads(args)
            
            module = self.resolve_module(module=module)
            print(module, 'BROOO')
            fn_obj =  getattr(module, fn)
            kwargs = kwargs if kwargs != None else {}
            args = args if args != None else []
            print(args, kwargs, 'DEBUG')
            if callable(fn_obj):
                return fn_obj(*args, **kwargs)
            else: 
                return fn_obj

        return app
    
    def set_demo_mode_params(self):
        for m in ['gpt2', 'gpt3', 'gpt4', 'gpt5']:
            self.module_tree['model.'+m] =  Model(name=m).state

        for m in ['gpt2', 'gpt3', 'gpt4', 'gpt5']:
            self.module_tree['dataset.'+m] =  Dataset(name=m).state


    

app = Module.api()

from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



    
