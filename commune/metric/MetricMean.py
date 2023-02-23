      
import torch
import commune
from commune.metric import Metric
from typing import *

class MetricMean(Metric):


    def setup(self, **kwargs):
        
        self.value = kwargs.get('value', 0)
        self.count = kwargs.get('count', 0)
        

        
    def update(self, value, *args, **kwargs):
        '''
        Update the moving window average with a new value.
        '''
        
        self.value = (self.value * self.count + value) / (self.count + 1)
        self.count += 1
        return self.value


    @classmethod
    def test(cls):
        
        # testing constant value
        self = cls()
        for i in range(10):
            self.update(10)
            
        assert 10 == self.value
        
        print(self.to_dict())
        print(self.metric_map)
        

if __name__ == '__main__':
    MetricMean.test()
            