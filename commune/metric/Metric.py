      
import torch
import commune
from typing import *

class Metric(commune.Module):
    
    def __init__(self, **kwargs): 
        self.value = 0
        self.setup(**kwargs)
        # self.value = value
        
    def set_value(self, value:Union[float, int], ) -> float:
        if value == None:
            value = 0
        self.value = value 

    def setup(self, **kwargs):
        pass
        
        

    def update(self, value, **kwargs):
        self.value = value


    @classmethod
    def test(cls, recursive=True):
        # testing constant value
        constant = 10
        self = cls()
        self.update(constant)
        print(self.to_dict())
        self = cls.from_dict(self.to_dict())
        print(self.to_dict())
        self.value == constant
        
    @classmethod
    def test_metrics(cls):
        metric_map = cls.get_metric_map()
        for metric_key in cls.metrics():
            commune.log(f'Testing {metric_key}')
            metric = metric_map[metric_key]
            metric.test()
            commune.log(f'Testing {metric_key} passed', 'success')
            
            
    @classmethod
    def get_metric_map(cls) -> Dict[str, 'Metric']:
        import glob
        metric_map = {}
        import os
        for f in glob.glob(cls.__module_dir__()+'/*'):
            if os.path.basename(f).startswith('Metric'):
                metric_name = os.path.basename(f).split('.')[0]

                metric_key = metric_name if metric_name == 'Metric' else metric_name.replace('Metric','')
                metric_key = metric_key.lower()           
                metric_map[metric_key] = cls.import_object(f'commune.metric.{metric_name}.{metric_name}')
                
        return metric_map
    
    def state_dict(self):
        return self.to_dict()
    @property
    def metric_map(self):
        return self.get_metric_map()
    
    @classmethod
    def metrics(self) -> List[str]:
        return list(self.get_metric_map().keys())
    

    def set_params(self, params: dict) -> None:
        '''
        set the parameters of the metric
        '''
        for key, value in params.items():
            getattr(self, f'set_{key}')(value)
if __name__ == '__main__':
    Metric.test_metrics()
    