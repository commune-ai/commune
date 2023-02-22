      
import torch
import commune
from typing import *

class MetricDefault(commune.Module):
    
    def update(self, value , *args, **kwargs):
        '''
        Update the moving window average with a new value.
        '''
        
        self.value = value
        
        return self.value


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
            