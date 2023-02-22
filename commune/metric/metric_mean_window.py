      
import torch
import commune
from typing import *

class MetricMeanWindow(commune.Module):
    metric_id = 'mean_window'
    
    def __init__(self,
                 window_size:int=100, 
                 window_values:List[float]=None,
                 **kwargs): 
    
        self.metric_type = self.module_name()
        self.set_window_size(window_size=window_size, window_values=window_values)
        self.value 
            

    def set_window_size(self, window_size:int=100, window_values:List[Union[int, float]] = None)-> int:
        self.window_values = window_values if window_values != None else []  
        self.window_size = window_size
        return window_size

    def update(self, *values, window_size:int=None):
        '''
        Update the moving window average with a new value.
        '''
        for value in values:
            self.window_values +=  [value]
            if len(self.window_values) > self.window_size:
                self.window_values = self.window_values[-self.window_size:]

        return self.value
    @property
    def value(self) -> Dict:
        return sum(self.window_values)/len(self.window_values)
        
    def __str__(self):
        return str(self.value)


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
            