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
from adapter_block import AdapterBlock
from torch import nn

class AdapterModel(commune.model.Model):
    
    def __init__(self, model:str='model::gptj', 
                 optimizer={'lr': 0.02},
                 device='cuda', 
                 tokenizer: str = 'gptj',
                 tag = 'bro',
                 params = None,
                 load = False,
                 **kwargs):
        self.model_name = model + ("::"+tag if tag != None else '')
        kwargs['tag'] = self.model_name
        commune.model.Model.__init__(self, **kwargs )
        
        self.model = model
        self.params = params if params != None else {}
        self.set_tokenizer(tokenizer=tokenizer)

        self.set_model(model=model, device=device, **self.params)
        self.set_optimizer(optimizer=optimizer)

        
        if load:
            self.load()


    def forward(self, *args, 
                output_length=10,
                topk=512, 
                alpha = 1,
                **kwargs):
        
        kwargs.update(dict(
            output_hidden_states=True,
            hidden_dim_bounds = None,
            output_logits=False, 
            output_topk=True, 
            output_length=output_length,
            token_remap = False , 
            logit_remap = False,
            topk=topk
        ))
        
        kwargs['input_ids'] = kwargs['input_ids'].to(self.device)
        
        
        model_output = self.model.forward(*args, **kwargs)

        model_output['logits'] = decode_topk(model_output['topk'], vocab_size=int(self.vocab_size), topk= topk).to(self.device)
        model_output['hidden_states'] = model_output['hidden_states'][..., :self.hidden_dim]
        model_output['adapter_logits'] = self.adapter(model_output['hidden_states'].to(self.device))

        
        
        model_output['logits'] = self.combine_logits(model_output['logits'], model_output['adapter_logits'], weights = [1, 0.2])
        return Munch(model_output)


    def combine_logits(self, *logits, weights = None):
        combined_probs = 0
        combined_probs =torch.zeros_like(logits[0]).to(self.device)
        if weights == None:
            weights = [1] * len(logits)
        for i, logit in enumerate(logits):
            print( logit.device, combined_probs.device)
            combined_probs = torch.softmax(logit, dim=-1)*weights[i] + combined_probs
            
        combined_probs = combined_probs / combined_probs.sum(dim=-1, keepdim=True)
        combined_logits = torch.log(combined_probs + 1e-8)
        return combined_logits
    def set_model(self, model:List[str], device:str = None, **model_kwargs ):
        
        
        self.model = commune.connect(model)
        
        self.config = Munch(self.model.model_config)
        model_kwargs['out_dim'] = self.tokenizer.vocab_size
        self.adapter = AdapterBlock(**model_kwargs).to(self.device)
        
        st.write(self.adapter.device)
        self.hidden_size = self.hidden_dim = self.config.hidden_dim = self.adapter.hidden_dim
        self.config['adapter'] = self.adapter.config
        
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
        
        self.vocab_size = self.tokenizer.vocab_size       
        self.tokenizer = tokenizer
        
        if  self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer
  

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
    def test(cls, topk=1024, output_length=10):
        
        model = cls(load=True)
        dataset = commune.connect('dataset::bittensor')
        for i in range(100):
            sample = dataset.sample(sequence_length=256)
            output = model.learn_step(**sample, save=True)
            print(output['stats'])
            
        # output['logits'] = decode_topk(output['topk'])
        
        # print(cls.calculate_loss(output['logits'].reshape(-1, output['logits'].shape[-1]), targets[:, -output_length:].flatten()))
     
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

     

    def learn_step(self, **sample):

        save = sample.pop('save', False)
        load = sample.pop('load', False)
        tag = sample.pop('tag', None)
        
        if load:
            self.load(tag)
            
            
        original_kwargs = {}
        original_kwargs['output_logits'] = sample.get('output_logits', True)
        # we need the logits and we need to 
        if  original_kwargs['output_logits'] == False:
            sample['output_logits'] = True 
        
        self.optimizer.zero_grad()
        model_output = self.forward(**sample, no_grad=False)
        loss = self.calculate_loss(**model_output, **sample)   
        loss.backward()
        self.optimizer.step()
        self.set_metric('loss2', loss.item(), metric='metric')
        self.set_metric('learn_steps', metric='counter')
        
        if not original_kwargs['output_logits']:
            del model_output['logits']
            
        model_output['stats'] = deepcopy(self.stats)
        model_output['stats']['metrics'] = self.get_metrics()
        
        if save:
            self.save(tag)
            
        
        
        
        return Munch(model_output)
    

if __name__ == "__main__":
    
    
    # EnsembleModel.run_neuron()
    AdapterModel.test()
    # print('FUCK')
    # TransformerModel('gptj', tag='demo', load=True).save_pretrained()
    
    # TransformerModel.run()
    # TransformerModel.experiment()


