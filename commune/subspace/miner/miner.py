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


    def set_model(self, model) -> None:
        if isinstance(model, str):
            model = commune.connect(model)
        elif isinstance(model, dict):
            model = commune.launch(**model)
        else:
            raise ValueError(f'Invalid model type: {type(model)}')
        
        self.model = model
            
            
    def set_subspace(self, subspace = None ) -> None:
        module_class = commune.get_module('subspace')
        if isinstance(subspace, str):
            subspace = module_class(subspace)
        elif subspace is None:
            subspace = module_class()
        elif isinstance(subspace, dict):
            subspace = module_class(**subspace)
        else: 
            raise ValueError(f'Invalid subspace type: {type(subspace)}')
        
        self.subspace = subspace

    def verify_signature(self, signature: Dict) -> bool:
        # verify everything as default
        return True

        

    def get_sample(self, **kwargs):
        kwargs.update(dict(
            tokenize=True, sequence_length=10, batch_size=2
        ))
        return self.dataset.sample(**kwargs)
    

    def set_stats(self, stats: Dict[str, Any]) -> None:
        if stats is None:
            stats = {}
        self.stats = stats
         
            
    
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


    @classmethod
    def launch_model(cls,name='', *args, **kwargs):
        if kwargs is None:
            kwargs = {}
        kwargs['module'] = 'model.transformer'
        model = commune.launch(module)
        model.deploy(**kwargs)
        

    def set_stat(self, key: str, stat: Dict[str, Any]) -> None:
        
        prev_stat = deepcopy(self.stats.pop(key, {}))
        if 'metric' in prev_stat:
            stat['metric'] =  self.alpha*prev_stat['metric'] + (1-self.alpha)*stat['metric']
        
        self.stats[key] = stat
      

if __name__ == '__main__':

    self = Miner(model='model::a')
    st.write(self.key)
    