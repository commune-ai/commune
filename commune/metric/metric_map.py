
from typing import *
import commune


class MetricMap(commune.Module):
    
    default_metric = 'commune.metric.MetricMeanWindow'
    def __init__(self, metrics:Dict[str, commune.Module] = {}):
        self.metrics = metrics
        
    def set_metric(self, key:str, value: float, metric: str = None, **metric_kwargs):
        
        metric = metric if metric is not None else self.default_metric
        if not hasattr(self, 'metrics'):
            self.metrics = {}
            
        if key not in self.metrics:
            metric_kwargs['value'] = value
            if isinstance(metric,str): 
                metric_class = commune.get_module(metric)
            self.metrics[key] =  metric_class(**metric_kwargs)
        else:
            self.metrics[key].update(value)
        print(self.metrics)
        return self.metrics[key].value
    def get_metric(self, name:str, return_value:bool=True):
        return self.metrics[name].value
    
    def to_dict(self):
        state_dict = {}
        
        for metric_key, metric in self.metrics.items():
            state_dict[metric_key] = metric.to_dict()
        return state_dict
    
    def add_metrics(self, metrics:Dict[str, commune.Module]):
        for metric_key, metric in metrics.items():
            self.metrics[metric_key] = metric
        return self
    
    @classmethod
    def from_dict(cls, state_dict:Dict):
        metrics = {}
        for metric_key, metric_dict in state_dict.items():
            print(metric_dict)
            if 'metric_type' not in metric_dict:
                metric_path = cls.default_metric
            else:
                metric_path = f"commune.metric.{metric_dict['metric_type']}"
            metrics[metric_key] = commune.get_module(metric_path).from_dict(metric_dict)
            print(metrics[metric_key].value)
        print(metrics)
        
        return cls(metrics=metrics)
    

        
    @property
    def value_map(self):
        return {metric_key: metric.value for metric_key, metric in self.metrics.items()}
    

    @classmethod
    def test(cls):
        self = MetricMap()
        test_values = {
            'a': [10, 4, 4],
            'b': [10, 4, 4]
        }
        
        for metric_key, values in test_values.items():
            for value in values:
                self.set_metric(metric_key, value)
            assert self.get_metric(metric_key) == sum(values)/len(values)
        # some other tests
    
        
        state_dict = self.to_dict()
        print(state_dict)
        regen_self = MetricMap.from_dict(state_dict)

        for metric_key, values in test_values.items():
            assert self.get_metric(metric_key) == sum(values)/len(values)
        
if __name__ == "__main__":
    MetricMap.test()
    
