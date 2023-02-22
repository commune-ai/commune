      
import torch
import commune
from typing import *

class BaseMetric(commune.Module):
    
    def __init__(self,value: Union[int, float] = None, 
                 **kwargs): 
    
        self.metric_type = self.module_name()
        
        
    def update(self, value):
        self.value = value
        
    
    def to_dict(self) -> Dict:
        state_dict = self.__dict__
        return state_dict   
    
     
    @classmethod
    def from_dict(cls, state_dict:Dict):
        
        return cls(**state_dict)

    @classmethod
    def test(cls):
        
        # testing constant value
        constant = 10
        self = cls(value=constant)
        
        self.value == constant
        state_dict = self.to_dict()
        cls.from_dict(state_dict)
        self.value == constant
        print(self.window_values)
            