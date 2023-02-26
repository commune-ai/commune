      
import torch
import commune
from commune.metric import Metric
from typing import *

class MetricMA(Metric):

    def setup(self,
              value: Union[int, float] = None, 
              alpha: float = 0.9,
              **kwargs):
        self.value = value
        self.set_alpha(alpha)
                
    def set_alpha(self, alpha:float = 0.9) -> float:
        assert alpha >= 0 and alpha <= 1, 'alpha must be between 0 and 1'
        self.alpha = alpha
        return alpha
    
    def set_params(self, params: dict) -> None:
        '''
        set the parameters of the metric
        '''
        for key, value in params.items():
            getattr(self, f'set_{key}')(value)
         
    def update(self, value, **params):
        '''
        Update the moving window average with a new value.
        '''
        self.set_params(params)
        
        if self.value == 0 or self.value is None:
            self.value = value
        else:
            self.value = self.value * self.alpha + value * (1 - self.alpha)
            
        return self.value


    @classmethod
    def test(cls):
        
        # testing constant value
        constant_list  = [10, 10, 0, 0]
        self = cls()
        for constant in constant_list:
            self.update(constant)
        
        # self.value == sum(constant_list)/len(constant_list)
        state_dict = self.to_dict()
        cls.from_dict(state_dict)
        self.value == sum(constant_list)/len(constant_list)
        
            
        