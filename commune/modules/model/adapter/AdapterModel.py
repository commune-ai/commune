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
from commune.module.utils.torch import tensor_dict_info
from commune.module.utils.tokenizer import decode_topk
import streamlit as st
# logger = logger.opt(colors=True)
import commune
import os
# commune.module.utils
from torch import nn
from torch import Tensor
from commune.model import Model
from commune.model.attention import MultiheadAttention
from commune.model.layer import LayerBlock
from typing import *
from torch import nn




class AdapterModel(Model): 
    def __init__(self, 
                 model:str='model::gptj', 
                 optimizer:dict={'lr': 0.0001},
                 hidden_dim = 700,
                 device:str='cuda', 
                 tokenizer: str = 'gptj',
                 metrics: dict = {},
                 tag:str = 'base',
                 adapter:dict = dict(
                                module='commune.model.adapter.block.AdapterBlock', 
                                in_dim= 10, 
                                hidden_dim= 64,  
                                num_layers= 8,
                                key2attr = {'in_dim': 'hidden_dim', 'out_dim': 'vocab_size'},
                                ),
                 **kwargs):
        


        Model.__init__(self,config=locals())
        
        self.model = model
        self.hidden_dim = hidden_dim

        self.set_params(model=model, 
                        device=device,
                        adapter=adapter, 
                        tag=tag,
                        optimizer = optimizer,
                        tokenizer=tokenizer)


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
            output_topk=True,
            output_logits=False,
            output_hidden_state=True,
            return_keys =['topk', 'hidden_states'],
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
            model_output['metrics'] = self.get_metrics()
        
            
        if save:
            self.save(tag)
            


        if return_keys != None:
            output_dict = {}
            for key in return_keys:
                
                if key in model_output:
                    output_dict[key] = model_output[key]
            return output_dict
                    

        return model_output


    def set_adapter(self, adapter: dict) -> None:
        
        # get the module class
        adapter_kwargs = {}
        adapter_module_path = adapter.pop('module', 'commune.model.adapter.block.AdapterBlock')
        key2attr = adapter.pop('key2attr',  {'in_dim': 'hidden_dim', 'out_dim': 'vocab_size'})
        adapter_params = adapter.get('params', adapter.get('kwargs', adapter))
        adapter_module_class = commune.get_module(adapter_module_path)
        
        # resolve params
        for key, attr in key2attr.items():
            adapter_params[key] = getattr(self, attr)
        
        
        adapter_kwargs = {}
        # set the config
        self.adapter = adapter_module_class(**adapter_params)
        self.config['adapter'] = self.adapter.config
        
        return self.adapter
    
    
    def set_model(self,model:str) -> None:
        if isinstance(model, str):
            model = commune.connect(model) 
        self.model = model
        self.config = Munch(self.model.model_config)
        
    
    

    def set_params(self, 
                   model:str = None, 
                   device:str = None, 
                   adapter:dict = None, 
                   optimizer:dict=None,
                   tokenizer: str = None,
                   stats: dict = None,
                   tag: str = None):
        
        # only set parts of the network when you want to

        if model != None :
            self.set_model(model)
        if tokenizer != None:
            self.set_tokenizer(tokenizer)
            
        if adapter != None:
            self.set_adapter(adapter)
            self.set_optimizer(optimizer)
            
        if tag != None:
            self.set_tag(tag)
            
        if stats != None:
            self.set_stats(metrics)
            
        self.set_device(device)
        for k in ['optimizer', 'adapter', 'model', 'tag', 'tokenizer']:
            assert hasattr(self, k)
            

        
        
        return self.model

    shortcuts =  {
        'gptj': 'EleutherAI/gpt-j-6b',
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

        self.config.pad_token_id = self.tokenizer.pad_token_id
        self.config.eos_token_id = self.tokenizer.eos_token_id
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

    
    def train_model(self,
             dataset : Union[str, 'Module'] = 'dataset::bittensor',
             params: dict = None,
            output_length:int=10,
            sequence_length:int=256,
            num_batches: int = 1, 
            tag : str = None,
            save : bool = False,
            load : bool = False,
            refresh: bool = False,
            **kwargs):
        st.write(self.config)

        params = params if params != None else {}
        params['tag'] = tag

        if load and (refresh == False):
            self.load(tag=tag)
        
        self.set_params(**params)
        
        if not hasattr(self, 'dataset'):
            if isinstance(dataset, str):
                dataset = commune.connect(dataset)
            self.dataset = dataset
            
            
            
        for i in range(num_batches):
            sample = self.dataset.sample(sequence_length=sequence_length)
            if isinstance(sample, str):
                continue
            sample.update(dict(
                output_length=output_length,
                return_keys=['stats'],
                train = True
            ))
            
            output = self.forward(**sample)
            commune.print(output, 'cyan')

        if save :
            self.save(tag=tag)
            
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
    def streamlit(cls):
        import streamlit as st
        self = cls(model = 'model:gptj:0')
        
        cls.train_model(self, num_batches=1, dataset='dataset::bittensor', sequence_length=256, output_length=10)
        

                

if __name__ == "__main__":
    
    # dataset = commune.connect('dataset::bittensor')
    AdapterModel.run()
    # print(dataset.module_name)
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
    # AdapterModel.serve(wait_for_termination=False)
    # AdapterModel.run()
    # print('FUCK')
    # TransformerModel('gptj', tag='demo', load=True).save_pretrained()
    
    # TransformerModel.run()
    # TransformerModel.experiment()


