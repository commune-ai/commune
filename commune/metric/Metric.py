      
import torch
import commune
from typing import *


class Metric(commune.Module):
    
    def __init__(self,value: Union[int, float] = None, 
                 **kwargs): 
    
        self.metric_key =  self.module_name().replace('Metric','').lower() 
        if self.metric_key == '':
            self.metric_key = 'metric'
        self.setup(value, **kwargs)
        # self.value = value
        
    def set_value(self, value:Union[float, int], ) -> float:
        self.value = value if value is not None else 0


    def setup(self, value, **kwargs):
        self.set_value(value)
        

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
    def get_default_metric_map(cls) -> Dict[str, 'Metric']:
        import glob
        metric_map = {}
        import os
        print(cls.__module_dir__())
        for f in glob.glob(cls.__module_dir__()+'/*'):
            if os.path.basename(f).startswith('Metric'):
                metric_name = os.path.basename(f).split('.')[0]

                metric_key = metric_name if metric_name == 'Metric' else metric_name.replace('Metric','')
                metric_key = metric_key.lower()           
                metric_map[metric_key] = cls.import_object(f'commune.metric.{metric_name}.{metric_name}')
                
        print(metric_map)
        return metric_map
    
    @property
    def default_metric_map(self):
        return self.get_default_metric_map()
if __name__ == '__main__':
    Metric.test()