import os, sys
from pprint import pp

from functools import partial
import asyncio
from copy import deepcopy
from typing import Union, Optional
from concurrent import futures
import os, sys
from typing import *
from loguru import logger
import time
from munch import Munch
import argparse
import torch
from commune.utils.torch import tensor_dict_info
from commune.utils.tokenizer import decode_topk
import streamlit as st
# logger = logger.opt(colors=True)
import commune
import os
# commune.utils
from torch import nn
from torch import Tensor
from commune.model.attention import MultiheadAttention
from commune.model.layer import LayerBlock
from typing import *
from torch import nn




class Trainer(commune.Module):
    default_adapter = dict(
                    module='commune.model.adapter.block.AdapterBlock', 
                    params = {'in_dim': 10, 'hidden_dim': 64,  'num_layers': 8},
                    key2attr = {'in_dim': 'hidden_dim', 'out_dim': 'vocab_size'},
                    device = None
                    )
    def __init__(self, model:str='model::gptj', 
                 optimizer:dict={'lr': 0.0001},
                 hidden_dim = 700,
                 device:str='cuda', 
                 tokenizer: str = 'gptj',
                 tag:str = None,
                 adapter:dict = default_adapter,
                 load:dict = False,
                 **kwargs):
        


        commune.model.Model.__init__(self, **kwargs )
        
        
        self.tag = tag if tag != None else 'base'
        self.model = model
        self.hidden_dim = hidden_dim
        self.set_tokenizer(tokenizer=tokenizer)

        self.set_model(model=model, device=device, adapter=adapter)
        self.set_optimizer(optimizer=optimizer)

        
        if load:
            self.load()


    def forward(self,
                input_ids: torch.Tensor, 
                output_length=10,
                topk=512, 
                alpha = 1,
                server_kwargs = dict(
                    output_hidden_states=True,
                    output_logits=False, 
                    output_topk=True, 
                    hidden_dim_bounds = None,
                    token_remap = False , 
                    logit_remap = False,
                ), 
                train: bool = False,
                save: bool = False,
                load: bool = False,
                tag : str = None,
                return_keys: List[str] = ['logits', 'hidden_states', 'adapter_logits', 'stats'],
                **kwargs):
        
        sample = kwargs
        sample.update(dict(
            input_ids = input_ids.to(self.device),
            output_length=output_length,
            topk = topk,
            **server_kwargs
            ))
                

        model_output = self.model.forward(**sample)

        model_output['logits'] = decode_topk(model_output['topk'], vocab_size=int(self.vocab_size), topk= topk).to(self.device)
        model_output['hidden_states'] = model_output['hidden_states'][..., :self.hidden_dim]
        model_output['adapter_logits'] = self.adapter(model_output['hidden_states'].to(self.device))


        model_output['logits'] = self.combine_logits(model_output['logits'], model_output['adapter_logits'], weights = [1, 0.2])
        
        
        
        
        if train:
            sample.pop('topk')
            
            for key in sample:
                if key not in model_output:
                    model_output[key] = sample[key]
            
            loss = self.calculate_loss(**model_output)   
            self.set_metric('loss', loss.item(), metric='ma')
            self.set_metric('learn_steps', metric='counter')
            
            loss.backward()
            self.optimizer.step()
            model_output['stats'] = deepcopy(self.stats)
            model_output['stats']['metrics'] = self.get_metrics()
        
        
            
        if save:
            self.save(tag)
            


        if return_keys != None:
            output_dict = {}
            for key in return_keys:
                
                if key in model_output:
                    output_dict[key] = model_output[key]
            return output_dict
                    

        return model_output


    def combine_logits(self, *logits, weights = None):
        combined_probs = 0
        combined_probs =torch.zeros_like(logits[0]).to(self.device)
        if weights == None:
            weights = [1] * len(logits)
        for i, logit in enumerate(logits):
            combined_probs = torch.softmax(logit, dim=-1)*weights[i] + combined_probs
            
        combined_probs = combined_probs / combined_probs.sum(dim=-1, keepdim=True)
        combined_logits = torch.log(combined_probs + 1e-8)
        return combined_logits
    
    def set_adapter(self,
                    module:str='commune.model.adapter.block.AdapterBlock', 
                    params:dict = {'in_dim': 10, 'hidden_dim': 64,  'num_layers': 8},
                    key2attr:dict = {'in_dim': 'hidden_dim', 'out_dim': 'vocab_size'},
                    device = None) -> None:
        
        device = device if device != None else self.device
        params = params if params != None else {}
        key2attr = key2attr if key2attr != None else {}
        # map attributes to params
        for key, attr in key2attr.items():
            params[key] = getattr(self, attr)
            
        adapter_block_class = commune.get_module(module)
        
        self.adapter = adapter_block_class(**params).to(self.device)
        self.hidden_dim = self.hidden_dim
        self.config['adapter'] = self.adapter.config
        
        return self.adapter
    
    def set_model(self, model:List[str], device:str = None, adapter:dict = None ):
        if isinstance(model, str):
            model = commune.connect(model)
        self.model = model
                
        if isinstance(model, str):
            self.model_name = model + f'::adapter' + f'::{self.tag}'
        else:
            self.model_name = model.model_id + f'::adapter' + f'::{self.tag}'
            
            
        self.config = Munch(self.model.model_config)
        self.set_adapter(**adapter)
        self.set_device(device)
        self.config.pad_token_id = self.tokenizer.pad_token_id
        self.config.eos_token_id = self.tokenizer.eos_token_id
        return self.model

    shortcuts =  {
        'gptj': 'EleutherAI/gpt-j-6B',
        'gpt2.7b': 'EleutherAI/gpt-neo-2.7B',
        'gpt125m': 'EleutherAI/gpt-neo-125M',
        'gptjt': 'togethercomputer/GPT-JT-6B-v1',
        'gptneox': 'EleutherAI/gpt-neox-20b',
        'gpt20b': 'EleutherAI/gpt-neox-20b',
        'opt13b': 'facebook/opt-13b',

         }

    def set_tokenizer(self, tokenizer:Union[str, 'tokenizer', None]):
        from transformers import AutoTokenizer
        tokenizer = self.shortcuts.get(tokenizer, tokenizer)
        
        if isinstance(tokenizer, str):
            if tokenizer == 'bittensor':
                import bittensor
                tokenizer = bittensor.tokenizer()
            else:
                
                try:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
                except ValueError:
                    print('resorting ot use_fast = False')
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        
        self.tokenizer = tokenizer
        
        if  self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer
    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size
    @classmethod
    def test_neuron(cls, tokenizer='bittensor', num_batches=10, dataset='dataset::bittensor', batch_size=32, sequence_length=12, topk=4096, **model_kwargs):
        from commune.block.bittensor.neuron.miner import neuron
        from bittensor.utils.tokenizer_utils import phrase_cross_entropy, topk_token_phrases, prep_tokenizer
        self = cls( tokenizer=tokenizer)
        self.to('cuda')
        nucleus = neuron(model=self).model
        nucleus.model.train()
        nucleus.model.eval()
        nucleus.model = nucleus.model.half()
        nucleus.model.config.hidden_size
        nucleus.model.config.pad_token_id
        nucleus.model.config.eos_token_id
        nucleus.model.named_parameters()
        state_dict = nucleus.model.state_dict()
        nucleus.model.load_state_dict(state_dict)
        
        dataset = commune.connect(dataset)
        sample = dataset.sample()
        
        for i in range(num_batches):
            sample = dataset.sample(batch_size=32, sequence_length=256)
            target = sample['input_ids'][:, -1:] 
            inputs_x = sample['input_ids'][:, :-1] 
            t = commune.timer()
            message, _model_output, topk_tensor = nucleus.encode_forward_causallmnext(inputs_x, topk=topk)
            loss_tuple = phrase_cross_entropy(topk_tensor=topk_tensor, target_phrases=target)
            commune.print(f'Loss : {loss_tuple[0].item()} Time: {t.seconds}', 'cyan')
 
    @classmethod
    def run_neuron(cls, tokenizer='bittensor'):
        import bittensor
        from commune.block.bittensor.neuron.miner import neuron
        self = cls( tokenizer=tokenizer)
        n = neuron(model=self)  
        n.run()

    
    @classmethod
    def train(cls,
             model:str='gptj', 
             dataset : Union[str, 'Module'] = 'dataset::bittensor',
             output_length:int=10,
             sequence_length:int=64,
             num_batches: int = 100, 
             tag:str=None,
             save : bool = True,
             refresh: bool = False,
             **kwargs):
        if refresh:
            load = False
        
        model = cls(model=model, tag=tag, **kwargs)
        
        if isinstance(dataset, str):
            dataset = commune.connect(dataset)

        for i in range(num_batches):
            sample = dataset.sample(sequence_length=sequence_length)
            sample['output_length'] =  output_length
            sample['return_keys'] = ['stats']
            sample['train'] = True
            output = model.forward(**sample)
            
            commune.print(output['stats'], 'cyan')
            
        if save:
            model.save(tag)
        return output['stats']
    
    
    
    @classmethod
    def calculate_loss( cls,  **kwargs) -> torch.Tensor:
        '''
        Calculate the loss for the model.
        '''
        pred = kwargs['logits']
        gt = kwargs['input_ids'][:, -(pred.shape[1]-1):].flatten()
        return_value = kwargs.get('return_value', False)
        pred = pred[:, :pred.shape[1]-1]
            
        if len(pred.shape) == 3:
            pred = pred.reshape(-1, pred.shape[-1])
        
        assert gt.shape == pred.shape[:1], f'gt.shape: {gt.shape} pred.shape: {pred.shape}'

        loss_fn = torch.nn.CrossEntropyLoss()
        loss =  loss_fn(pred, gt.to(pred.device))
        if return_value:
            return loss.item()
        return loss



    @classmethod
    def get_hyperopt_tag(cls, config:dict, prefix:str = None):
        tag = ''
        if prefix:
            tag += f'{prefix}::'
        for k, v in config.items():
            tag += f'{k}_{v}__'
                
        return tag
                
    @classmethod
    def hyperopt(cls, 
                 model: str = 'model::gptj',
                 dataset: str = 'dataset::bittensor',
                 num_batches: int =50, 
                 metric: str = 'loss',
                 mode :str = 'min',
                 num_samples: int = 1,


                 **kwargs):
        from ray import tune

        # 1. Define an objective function.
        

                
        def objective(config = {'lr': 1e-4, 'hidden_dim': 32, 'num_layers': 1}, 
                      model=model,
                      dataset=dataset):
            # tag = cls.get_hyperopt_tag(config, prefix=model)
            # if isinstance(dataset, str):
            #     dataset = commune.connect(dataset)
            # if isinstance(model, str):
            #     model = commune.connect(model)
            # train_stats  = cls.train(
            #                         model=model,
            #                     adapter={'params': {'hidden_dim': config['hidden_dim'], 'num_layers': config['num_layers']}},
            #                     optimizer={'lr': config['lr']}, 
            #                     dataset = dataset,
            #                     tag =tag ,
            #                     save=True,
            #                     num_batches=num_batches)
            from commune.utils.dict import dict_put
            tag = cls.get_hyperopt_tag(config, prefix=model)
            
            train_kwargs = dict(dataset = dataset,
                                model = model,
                                tag =tag ,
                                save=True,
                                num_batches=num_batches)
            from commune.utils.dict import dict_put
            for k, v in config.items():
                dict_put(train_kwargs, k,v)
            

            train_stats={}
            train_stats['metrics'] = 1
            
            print(train_kwargs)
            
            return train_stats['metrics']

        # 2. Define a search space.
        search_space = {
            'optimizer.lr': tune.loguniform(1e-4, 1e-2),
            "adapter.params.hidden_dim": tune.choice([32, 64, 128, 256, 512]),
            'adapter.params.num_layers': tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        }
        objective_with_resources = tune.with_resources(objective, {"cpu": 4, "gpu": 2})
        
        
        # 3. Start a Tune run and print the best result.
        tuner = tune.Tuner(objective_with_resources, 
                           param_space=search_space, 
                           tune_config=tune.TuneConfig(num_samples=num_samples))
        results = tuner.fit()
        print(results.get_best_result(metric=metric, mode=mode).config)


if __name__ == "__main__":
    
    # dataset = commune.connect('dataset::bittensor')
    AdapterModel.hyperopt(num_batches=1)
    # print(dataset.module_id)
    # for i in range(10):
    #     print('Alo')
    #     AdapterModel.train(num_batches=1, dataset=dataset)
    #     adapter = dict(
    #                 module='commune.model.adapter.block.AdapterBlock', 
    #                 params = {'in_dim': 10, 'hidden_dim': 64,  'num_layers': 8},
    #                 key2attr = {'in_dim': 'hidden_dim', 'out_dim': 'vocab_size'},
    #                 device = None
    #                 )
    # AdapterModel.run()
    # EnsembleModel.run_neuron()
    # AdapterModel.serve_module(wait_for_termination=False)
    # AdapterModel.run()
    # print('FUCK')
    # TransformerModel('gptj', tag='demo', load=True).save_pretrained()
    
    # TransformerModel.run()
    # TransformerModel.experiment()


