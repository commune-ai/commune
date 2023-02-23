      
import torch
import commune
from typing import *

class Metric(commune.Module):
    
    def __init__(self,value: Union[int, float] = None, 
                 **kwargs): 
    
        self.metric_key =  self.module_name().replace('Metric','').lower() 
        if self.metric_key == '':
            self.metric_key = 'metric'
        kwargs['value'] = value
        self.setup(**kwargs)
        if not hasattr(self, 'value'):
            self.value = value
        self.set_value(self.value)
        # self.value = value
        
    def set_value(self, value:Union[float, int], ) -> float:
        if not hasattr(self, 'value'):
            self.value = None
        self.value = value if value is not None else 0


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
    def metric_map(self):
        return self.get_metric_map()
    
    @classmethod
    def metrics(self) -> List[str]:
        return list(self.get_metric_map().keys())
if __name__ == '__main__':
    # Metric.test_metrics()
    t = commune.timer()
    og_t = commune.time()
    commune.sleep(1)
    
    print( commune.time() - og_t)
    # print(t.seconds-og_t)
    