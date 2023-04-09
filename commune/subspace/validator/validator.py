import commune
import streamlit as st
import torch
from typing import Dict, List, Union, Any
import random
from copy import deepcopy
import asyncio
        
class Validator(commune.Module):
    
    def __init__(self, 
                 batch_size: int = 32,
                 sequence_length: int = 256,
                 dataset: str = 'dataset',
                 models: List[str]= None,
                 key: Union[Dict, str] = None,
                 metric: Union[Dict, str] = None,
                 stats: Union[Dict, None] = None,
                 max_stats_history: int = 100,
                 alpha: float = 0.5,
                 ):
        commune.nest_asyncio()

        self.set_max_stats_history(max_stats_history)
        self.set_batch_size(batch_size)
        self.set_sequence_length(sequence_length)
                
        self.set_dataset(dataset)
        self.set_models(models)
        self.set_key(key)
        self.set_metric(metric)
        self.set_stats(stats=stats)
        self.set_alpha(alpha)
        
        
    def set_max_stats_history(self, max_stats_history: int) -> None:
        self.max_stats_history = max_stats_history
    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size
    def set_sequence_length(self, sequence_length: int) -> None:
        self.sequence_length = sequence_length

    def set_alpha(self, alpha: float) -> None:
        # set alpha for exponential moving average
        self.alpha = alpha
        
    def verify_signature(self, signature: Dict) -> bool:
        return True
    
    def add_model(self, model: str, signature: Dict = None) -> None:
        if not hasattr(self, 'models'):
            self.models = {}
        self.models[model] = commune.connect(model)

            
    def set_models(self, models: List[str] = None) -> None:
        if models == None:
            models = self.default_models()
            
        for model in models:
            self.add_model(model)
    
    def set_dataset(self, dataset: str) -> None:
        if isinstance(dataset, str):
            dataset = commune.connect(dataset)
        else:
            raise ValueError(f'Invalid dataset type: {type(dataset)}')
        
        self.dataset = dataset
        

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
    
    

    def sample(self, **kwargs):
        kwargs.update(dict(
            # tokenize=True, 
            sequence_length=self.sequence_length,
            batch_size=self.batch_size
        ))
        sample = self.dataset.sample(**kwargs)
        if 'input_ids' not in sample:
            sample = self.sample()
            
        return sample
    @property
    def model_keys(self):
        return list(self.models.keys())
        

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
            
    @property
    def default_loss(self) -> float:
        return 10.0

    def resolve_model(self, model: str = None) -> Any:
        if model is None:
            model = self.random_model()
        elif isinstance(model, str):
            model = self.models[model]
        return model
    async def async_forward(self, 
                            sample: dict,
                            model:str=None,
                            topk: int = 512, 
                            **kwargs ):
        model_key = model
        model = self.resolve_model(model)
        model = self.models[model_key]
        
        # we want the client to return the future
        kwargs['return_future'] = True
        timer = commune.timer()
        output = await model.forward(**sample,**kwargs)
        inference_time = timer.seconds
        
    
        if 'topk' in output:
            output['logits'] = self.decode_topk(output['topk'], topk=topk, vocab_size=50400)
                
            output['input_ids'] = sample['input_ids']
            # calculate metric
            metric = self.calculate_metric(output)
        else:
            metric = self.default_loss
            
            
        output['stats'] = {
            'metric': metric,
            'timestamp': commune.time(),
            'inference_time': inference_time
        }
        return output
            
        
    
    def forward(self, 
                sample: dict,
                models:str=None, 
                topk: int = 4096,
                aggregate:bool = False,
                set_stats: bool = True,
                **kwargs ):
        model_keys = list(self.models.keys())
        if models == None:
            jobs = [self.async_forward(sample, model=model_key, topk=topk, **kwargs) for model_key in model_keys]
            
        loop = self.get_event_loop()
        model_outputs = loop.run_until_complete(asyncio.gather(*jobs))
        model_output_dict = {}
        for model_output, model_key in zip(model_outputs, model_keys):
            model_output_dict[model_key] = model_output
            
        if aggregate:
            raise NotImplementedError('aggregate not implemented')
        

        for model_key, output_dict in model_output_dict.items():
            sample_stats = output_dict['stats']
            if model_key in self.stats:
                stats = self.stats[model_key]
                stats['count'] += 1
            else:
                stats = sample_stats
                stats['count'] = 1
            
            for k in ['inference_time', 'metric']:
                stats[k] = ((stats[k]*(stats['count']-1)) + sample_stats[k])/stats['count']
            stats['history'] = stats.get('history', []) + [sample_stats]
            stats['history'] = stats['history'][-self.max_stats_history:]
            self.set_stats(key=model_key, stats = stats)
            
            
        return model_output_dict
    
    def validate(self, sample=None, models: str = None, topk:int=512, **kwargs):
        sample = self.sample() if sample == None else sample
        output = self.forward(sample=sample, models=models, topk=topk, **kwargs)
        
        return output


    def set_stats(self, key: str = None, stats: Dict[str, Any] = None) -> None:
        if stats is None:
            stats = {}
        assert isinstance(stats, dict), f'stats must be a dict. stats: {stats}'

        if key is None:
            self.stats = stats
            return key
            
        assert isinstance(key, str), f'key must be a str. key: {key}'
        
        self.stats[key] = stats
        return key
        
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
        
        
    @classmethod 
    def default_models(cls):
        return [m for m in commune.servers() if m.startswith('model')]
    
    
    @classmethod
    def decode_topk(cls,  forward_response_tensor: torch.Tensor, topk:int=4096, vocab_size:int=50257) -> torch.Tensor:
        """ Returns full logits by decoding topk-encoding input. """
        batch_size, sequence_len, _ = forward_response_tensor.shape
        encoded_probs = forward_response_tensor  # encoded probabilities: [batch_size, sequence_len, topk + topk]
        topk_values = encoded_probs[..., :topk]  # topk probs: [batch_size, sequence_len, topk]
        topk_indices = encoded_probs[..., topk:].long()  # topk probs indices: [batch_size, sequence_len, topk]

        topk_pmass = topk_values.sum(dim=-1)  # topk probability mass: [batch_size, sequence_len]
        remainder_pmass = torch.clamp(1 - topk_pmass, 1e-40, 1)  # remainder probability mass: [batch_size, sequence_len]
        
        remainder_floor = remainder_pmass / (vocab_size - topk)  # divide remainder: [batch_size, sequence_len]

        logits = torch.ones((batch_size, sequence_len, vocab_size), dtype=topk_values.dtype).to(topk_values.device)
        logits *= torch.log(remainder_floor)[:, :, None]  # set probability floor: [batch_size, sequence_len, vocab_size]

        logits.scatter_(-1, topk_indices, torch.log(topk_values + 1e-40))  # insert topk probs: [batch_size, sequence_len, vocab_size]

        return logits  # [batch_size, sequence_len, vocab_size]

        
if __name__ == '__main__':

    # self = Validator(
        
    validator =  Validator(models=None, dataset='dataset.text.bittensor')

    timer = commune.timer()
    for i in range(100):
        sample = validator.sample()
        validator.validate(sample=sample, topk=4096, models=None)
        commune.print(validator.stats)