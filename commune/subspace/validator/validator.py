import commune
import streamlit as st
import torch
from typing import Dict, List, Union, Any
import random
from copy import deepcopy

class Validator(commune.Module):
    
    def __init__(self, 
                 dataset: str = 'dataset',
                 models: List[str]= ['model::a', 'model::b'],
                 key: Union[Dict, str] = None,
                 metric: Union[Dict, str] = None,
                 stats: Union[Dict, None] = None,
                 ):
        
        self.set_dataset(dataset)
        self.set_models(models)
        self.set_key(key)
        self.set_metric(metric)
        self.set_stats(stats)
        
    def add_model(self, model: str, signature: Dict = None) -> None:
        if not hasattr(self, 'models'):
            self.models = {}
        self.verify_signature(signature)
        self.models[model] = commune.connect(model)
            
    def set_models(self, models: List[str]) -> None:
        for model in models:
            self.add_model(model)
        

    def set_key(self, key) -> None:
        key_class = commune.get_module('web3.account.substrate')
        if isinstance(key, dict):
            key = key_class(**key)
        elif isinstance(key, str):
            key = key_class(key)
        elif key is None:
            key = key_class()
        self.key = key
        
    def verify_signature(self, signature: Dict) -> bool:
        # verify everything as default
        return True

        
    def set_dataset(self, dataset: str) -> None:
        if isinstance(dataset, str):
            dataset = commune.connect(dataset)
        else:
            raise ValueError(f'Invalid dataset type: {type(dataset)}')
        
        self.dataset = dataset
        
    def validate(self, value):
        return value
    
    def set_metric(self, metric = None) -> None:
        if metric is None:
            metric = torch.nn.CrossEntropyLoss()
        self.metric = metric
    def calculate_metric(self, x):
        
        import torch
        input_ids = x.get('input_ids', None)
        pred = x.get('logits', None)
        if input_ids != None:
            gt = input_ids[:, -(pred.shape[1]-1):].flatten()
            pred = pred[:, :-1]
            
        assert isinstance(gt, torch.Tensor), f'gt is not a torch.Tensor. gt: {gt}'
        assert isinstance(pred, torch.Tensor), f'gt is not a torch.Tensor. gt: {gt}'
            
        if len(pred.shape) == 3:
            pred = pred.reshape(-1, pred.shape[-1])
        
        assert gt.shape == pred.shape[:1], f'gt.shape: {gt.shape} pred.shape: {pred.shape}'

        metric =  self.metric(pred, gt.to(pred.device))
        
        
        return metric.item()
    
    
    
    
    def get_sample(self, **kwargs):
        kwargs.update(dict(
            tokenize=True, sequence_length=10, batch_size=2
        ))
        return self.dataset.sample(**kwargs)
    @property
    def model_keys(self):
        return list(self.models.keys())
    
    def set_stats(self, stats: Dict[str, Any]) -> None:
        if stats is None:
            stats = {}
        self.stats = stats
        

    def get_sample_metatdata(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample_metadata = {}
        for k, v in sample.items():
            metadata_k = {'type': type(v)}
            
            if isinstance(v, torch.Tensor):
                metadata_k.update({
                    'shape': v.shape,
                    'dtype': v.dtype,
                })
            elif type(v) in [list, set, tuple]:
                metadata_k.update({
                    'length': len(v),
                })
            elif isinstance(v, dict):
                metadata_k.update({
                    'length': len(v),
                })
            sample_metadata[k] = metadata_k

        return sample_metadata
            
                
            
    
    def validate_model(self, model_key: str = None, **kwargs):
        model_key = model_key if model_key else self.random_model_key()
        model = self.models[model_key]
        sample = self.get_sample()
        
        
        t= commune.timer()
        output = model.forward(**sample,return_keys=['logits'])
        elapsed_time =  t.seconds
        output['input_ids'] = sample['input_ids']
        
        # calculate metric
        metric = self.calculate_metric(output)
        
        
        # add to stats
        if model in self.stats:
            old_stats = deepcopy(self.stats[model])
        
        self.stats[model_key] = {
                            'metric': metric,
                            'timestamp': commune.time(),
                            'elapsed_time': elapsed_time,
                            'sample_metadata': self.get_sample_metatdata(sample),
                             }
        
        
        return metric

    def random_model_key(self):
        random_model_key = random.choice(self.model_keys)
        return random_model_key

    def random_model(self):
        random_model_key = self.random_model_key()
        return self.models[random_model_key]
    
if __name__ == '__main__':
    models = [m for m in commune.servers() if m.startswith('model')]
    self = Validator(models=models)
    for _ in range(10):
        st.write(self.validate_model())
        
    st.write(self.stats)
    
    

    st.write()