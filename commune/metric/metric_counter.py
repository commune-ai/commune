      
import torch
import commune
from typing import *

class MetricCounter(commune.Module):
    def __init__(self,value: Union[int, float] = 0, **kwargs): 
    
        self.value = value
        
            
    def update(self, *args, **kwargs):
        '''
        Update the moving window average with a new value.
        '''
        self.value += 1 


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
        
        for i in range(10):
            self.update(10)
            assert constant == self.value
            
        variable_value = 100
        window_size = 10
        self = cls(value=variable_value, window_size=window_size+1)
        for i in range(variable_value+1):
            self.update(i)
        print(self.value)
        assert self.value == (variable_value - window_size/2)
        print(self.window_values)
            