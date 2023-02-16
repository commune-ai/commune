
from fastapi import FastAPI
from pydantic import BaseModel
from typing import *     
import json 
import torch

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
class DataType(Type):
    def __init__(self, value = torch.randn([10, 10])):
        
        self.type = self.get_str_type(value)

        if self.type == 'torch.Tensor':
            self.shape = value.shape
            self.dtype = str(value.dtype)

    @staticmethod
    def get_str_type(data):
        return str(type(data)).split("'")[1]
                 
class Model(Type):
    def __init__(self, 
                 name: str = 'gpt',
                 tasks=[Task().state],
                 input: dict = {'token_batch': DataType(torch.randn([10, 32])).state},
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
        
    