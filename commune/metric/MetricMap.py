
from typing import *
import commune
from commune.metric import Metric


class MetricMap(Metric):
    
    default_metric_path = 'metric'
    def __init__(self, metrics:Dict[str, commune.Module] =None):
        self.set_metrics(metrics)
    
        
    def set_metrics(self, metrics:Dict[str, commune.Module]) -> None:
        self.metrics = metrics if metrics is not None else {}
        if metrics == None:
            self.metrics = {}
        elif  isinstance(metrics, dict):
            for metric_key, metric in metrics.items():
                if isinstance(metric, dict):
                    metric_path = metric.pop('metric_key', self.default_metric_path)
                    metric_class = self.metric_map[metric_path]
                    metric = metric_class(**metric)
                else:
                    metric_class = self.metric_map[self.default_metric_path]
                self.metrics[metric_key] = metric
        else:
            raise ValueError('metrics must be a dictionary')

    def update(self, key:str, value: Any = None, 
                   metric: str = 'metric',
                   params: dict = None,
                   refresh: bool = False) -> Any:
        
        if refresh:
            self.metrics.pop(key, None)
        params = params if params is not None else {}
        if key not in self.metrics:
            metric_class = self.metric_map[metric]
            self.metrics[key] =  metric_class(**params)
        
        return self.metrics[key].update(value)
    def set_metric(self, key:str, 
                   value: Any = None, 
                   metric: str = 'metric',
                   params: dict = None,
                   refresh: bool = False) -> Any:
        
        if refresh:
            self.metrics.pop(key, None)
        params = params if params is not None else {}
        if key not in self.metrics:
            metric_class = self.metric_map[metric]
            self.metrics[key] =  metric_class(**params)
        
        return self.metrics[key].update(value, **params)
    
    
    def get_metric(self, name:str, return_value:bool=True):
        return self.metrics[name].value
    
    def to_dict(self):
        state_dict = {}
        
        for metric_key, metric in self.metrics.items():
            state_dict[metric_key] = metric.to_dict()
        return state_dict
    
    def rm_metric(self, name:str) -> str:
        self.metrics.pop(name, None)
        return name
    
    def reset(self) -> None:
        # reset metrics
        self.set_metrics({})
        
    def leaderboard(self, descending: bool=False) -> List[Dict]:
        metric_values = self.get_metrics()
        sorted_values = [{'name':k, 'value': v } for k, v in sorted(metric_values.items(), key=lambda item: item[1], reverse=descending)]  
        return sorted_values
    
    def best_metric(self, return_value:bool = False, descending:bool=False) -> Union[float, Dict[str, Any]]:
        best_metric_row = self.leaderboard()[0]
        if return_value:
            return best_metric_row['value']

        return best_metric_row
    
    def add_metric(self, key, metric: str = 'metric', params: dict = None, refresh: bool = False) -> None:
        if refresh:
            self.metrics.pop(key, None)
        metric_class = self.get_metric_map()[key]
        params = params if params is not None else {}
        self.metric[key] = metric_class(**params)
    
    @classmethod
    def from_dict(cls, state_dict:Dict):
        metrics = {}
        for metric_key, metric_dict in state_dict.items():
            metric_path = metric_dict.pop('metric_key', cls.default_metric_path)
            metric_class = cls.get_metric_map()[metric_path]
            metrics[metric_key] = metric_class.from_dict(metric_dict)        
        return cls(metrics=metrics)
    
 

    def get_metrics(self):
        return {metric_key: metric.value for metric_key, metric in self.metrics.items()}
    

    @classmethod
    def test_leaderboard(cls):
        self = cls()
        num_peers = 10
        for i in range(num_peers+1):
            self.set_metric(str(i), i)
        descending = True
        self.leaderboard(descending=descending)[num_peers]['value'] == num_peers
        self.best_metric(descending=descending) == num_peers
        descending = False
        self.leaderboard(descending=descending)[num_peers]['value'] == 0
        self.best_metric(descending=descending) == 0
        commune.log('Leaderboard Succeeded')
    @classmethod
    def test(cls):
        self = MetricMap()
        test_values = {
            'a': [10, 10, 10],
            'b': [4, 4, 4]
        }
        
        for metric_key, values in test_values.items():
            for value in values:
                self.set_metric(metric_key, value)
            assert self.get_metric(metric_key) == sum(values)/len(values)
        # some other tests
    
        
        state_dict = self.to_dict()
        regen_self = MetricMap.from_dict(state_dict)

        for metric_key, values in test_values.items():
            assert self.get_metric(metric_key) == sum(values)/len(values)
            
        self.test_leaderboard()
        
if __name__ == "__main__":
    MetricMap.run()
    
