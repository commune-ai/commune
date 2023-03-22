import commune
import streamlit as st
import torch
from typing import Dict, List, Union, Any
import random
from copy import deepcopy
from commune import subspace

class Miner(commune.Module):
    
    def __init__(self, 
                 model: Any= 'model',
                 subspace: 'Subspace' = None,
                 key: Union[Dict, str] = None,
                 stats: Union[Dict, str] = None,
                 ):
        
        self.set_model(model)
        self.set_subspace(subspace)
        self.set_key(key)
        self.set_stats(stats)


    def set_model(self, models: List[str]) -> None:
        self.model = commune.connect(model)
            
    def set_subspace(self, subspace = None ) -> None:
        module_class commune.get_module('subspace')
        
        if isinstance(subspace, str):
            subspace = module_class(subspace)
        elif subspace is None:
            subspace = module_class()
        elif isinstance(subspace, dict):
            subspace = module_class(**subspace)
        else: 
            raise ValueError(f'Invalid subspace type: {type(subspace)}')
        
        self.subspace = subspace
        
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
                    'shape': str(v.shape),
                    'dtype': str(v.dtype),
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
        
        

        model_stat={ 
                        'metric': metric,
                        'timestamp': commune.time(),
                        'elapsed_time': elapsed_time,
                        'sample_metadata': self.get_sample_metatdata(sample),
                             }
        
        
        self.set_stat(key=model_key, stat = model_stat)
        
        
        return metric


    def set_stat(self, key: str, stat: Dict[str, Any]) -> None:
        
        prev_stat = deepcopy(self.stats.pop(key, {}))
        if 'metric' in prev_stat:
            stat['metric'] =  self.alpha*prev_stat['metric'] + (1-self.alpha)*stat['metric']
        
        self.stats[key] = stat
        
    def calculate_weights(self):
        
        
        total_weights = 0 
        weight_map = {}
        for k in self.stats.keys():
            weight_map[k] =  1 / (self.stats[k]['metric'] + 1e-8)
            total_weights = total_weights + weight_map[k]


        for k in self.stats.keys():
            weight_map[k] = weight_map[k] / total_weights
            self.stats[k]['weight'] = weight_map[k]
            
    def random_model_key(self):
        random_model_key = random.choice(self.model_keys)
        return random_model_key

    def random_model(self):
        random_model_key = self.random_model_key()
        return self.models[random_model_key]
    
    
    @classmethod
    def test(cls):
        models = [m for m in commune.servers() if m.startswith('model')]
        self = Validator(models=models)
        for _ in range(10):
            st.write(self.validate_model())
        self.calculate_weights()
        st.write(self.stats)
      
    @classmethod
    def test_validation_keys(cls):
        vals = [Validator() for _ in range(10)]
        st.write([v.key.address for v in vals])
        hash = vals[0].key.hash({'hey': 'whadup'})
        sig = vals[0].key.sign(hash)
        
        assert not vals[0].key.verify(hash, signature = sig, public_key = vals[1].key.public_key )
        assert vals[0].key.verify(hash, signature = sig, public_key = vals[0].key.public_key )
        
        
    def sign(self, message: Dict[str, Any]) -> Dict[str, Any]:
        hash = self.key.hash(message)
        signature = self.key.sign(hash)
        return signature
    
    
    def verify(self, message: Dict[str, Any],
               signature: Dict[str, Any],
               public_key : str = None, 
               use_hash: bool = True) -> bool:
        if use_hash:
            message = self.key.hash(message)
            
        public_key = public_key if public_key else self.key.public_key
        
        return self.key.verify(message, signature)
        
if __name__ == '__main__':

    self = Validator()
    st.write(self.key)
    