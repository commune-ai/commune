      
import torch
import commune
from commune.metric import Metric
from typing import *

class MovingAverage(Metric):

    def setup(self,
              value: Union[int, float] = None, 
              alpha: float = 0.9,
              **kwargs):
        self.value = value if value is not None else 0
        self.set_alpha(alpha)
                
    def set_alpha(self, alpha:float = 0.9) -> float:
        assert alpha >= 0 and alpha <= 1, 'alpha must be between 0 and 1'
        self.alpha = alpha
        return alpha
    
    def update(self, *values, window_size:int=None):
        '''
        Update the moving window average with a new value.
        '''
        for value in values:
            self.value = self.value * self.alpha + value * (1 - self.alpha)
        
        return self.value


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
            