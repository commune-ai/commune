      
import torch
import commune
from typing import *

class MovingAverage(commune.Module):
    
    def __init__(self,value: Union[int, float] = None, 
                 alpha = 0.9,
                 **kwargs): 
        
        
        self.value = value if value is not None else 0
        self.metric_type = self.module_name()

    def set_alpha(self, alpha = 0.9):
        assert alpha >= 0 and alpha <= 1, 'alpha must be between 0 and 1'
        self.alpha = alpha
        return alpha
    
    def update(self, *values, window_size:int=None):
        '''
        Update the moving window average with a new value.
        '''
        for value in values:
            self.value = self.value * self.alpha + value * (1 - self.alpha)
        
        self.value = sum(self.window_values)/len(self.window_values)
        
        return self.value


    
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
            