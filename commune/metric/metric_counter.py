      
import torch
import commune
from typing import *

class MetricCounter(commune.Module):
    metric_id = 'counter'
    def __init__(self,value: Union[int, float] = 0, **kwargs): 
    
        self.metric_type = self.module_name()
        self.value = value
        
         
    def update(self, *args, **kwargs):
        '''
        Update the moving window average with a new value.
        '''
        
        self.value += 1 
        
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
        iter_count = 10
        for i in range(iter_count):
            self.update()
            
        assert constant == self.value
        state_dict = self.to_dict()
        self = cls.from_dict(state_dict)
        self.value == iter_count
            