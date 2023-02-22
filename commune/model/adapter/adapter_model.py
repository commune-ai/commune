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
commune.new_event_loop()
if os.getenv('USE_STREAMLIT') == 'true':
    import streamlit as st
import os
    
# import torch
import commune
# commune.utils
from torch import nn
from torch import Tensor
from commune.model.attention import MultiheadAttention
from commune.model.layer import LayerBlock
from typing import *
from adapter_block import AdapterBlock
from torch import nn

class AdapterModel(commune.Module, nn.Module):
    
    def __init__(self, model:str='model::gptj', 
                 optimizer=None,
                 device='cuda', 
                 tokenizer: str = 'gptj',
                 **model_kwargs):
        nn.Module.__init__(self)
        self.model = model
        self.set_model(model=model,**model_kwargs)
        self.set_optimizer(**(optimizer if optimizer != None else {}))
        self.set_tokenizer(tokenizer=tokenizer)
        self.set_device(device)
    
    def set_optimizer(self, **params) -> 'optimizer':
        self.optimizer = self.get_optimizer(**params)
        return self.optimizer
    
    def get_optimizer(self, optimizer=None, **params) -> 'optimizer':
        params = params.pop('params', {'lr': 0.001})

        if optimizer == None:
            optimizer =  torch.optim.Adam
        elif isinstance(optimizer, str):
            optimizer = commune.import_object(optimizer_class)
        elif isinstance(optimizer, type):
            return optimizer_class
        
        # assumes the params are the first arg
        optimizer = optimizer(self.parameters(), **params)
        
        return optimizer
    

    def forward(self, *args, 
                output_length=10,
                topk=512, 
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
        
        
        model_output = self.model.forward(*args, **kwargs)

        model_output['logits'] = decode_topk(model_output['topk'], vocab_size=int(self.vocab_size), topk= topk)
        model_output['hidden_states'] = model_output['hidden_states'][..., :self.hidden_dim]
        model_output['adapter_emb'] = self.adapter(model_output['hidden_states'])

        return Munch(model_output)
    


    @property
    def device(self) -> str:
        return self._device


    def set_model(self, model:List[str], device:str = None, **model_kwargs ):
        
        
        self.model = commune.connect(model)
        
        self.config = Munch(self.model.model_config)
        
        self.adapter = AdapterBlock(**model_kwargs)
        self.config.hidden_size = self.config.hidden_dim = self.adapter.hidden_dim
        self.hidden_size = self.hidden_dim = self.adapter.hidden_dim
        self.config['adapter'] = self.adapter.config
        
        self.set_device(device)
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
        self.config.pad_token_id = self.tokenizer.pad_token_id
        self.config.eos_token_id = self.tokenizer.eos_token_id
        return self.tokenizer


    def set_stats(self, stats:dict=None): 
        if stats == None:
            stats =  dict(
            )
        self.stats = Munch(stats)
        

    @property
    def module_tag(self): 
        return self.resolve_module_tag()

    def save(self, tag:str = None, trainable_only:bool = True):
        module_tag = self.resolve_module_tag(tag=tag)
        path = self.resolve_path(tag)
        model_state_dict = self.state_dict()
        
        if trainable_only:
            model_state_dict = {k:v for k,v in model_state_dict.items() if v.requires_grad} 
    
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state_dict = {
            'model': model_state_dict,
            'optimizer': self.optimizer.state_dict(),
            'stats': dict(self.stats)
        }
    
        torch.save(state_dict, path)
        
        return path
    
    def load(self, tag=None):
        module_tag = self.resolve_module_tag(tag=tag)
        path = self.resolve_path(module_tag)
        if not os.path.exists(path):
            logger.warning(f'No saved model found at {path}')
            return
        loaded_state  = torch.load( path)
        state_dict = self.state_dict()
        for k,v in loaded_state['model'].items():
            assert k in state_dict
            state_dict[k] = v
        self.load_state_dict(state_dict)
        self.optimizer.load_state_dict(loaded_state['optimizer'])
        self.set_stats(loaded_state['stats'])

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
 

    def set_device(self, device:str = None):
        if device == None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._device = torch.device(device)
        return self.device
    
    
    @classmethod
    def test(cls, topk=1024, output_length=10):
        
        model = cls()
        dataset = commune.connect('dataset::bittensor')
        model = model.to('cuda')
        for i in range(100):
            sample = dataset.sample(sequence_length=256)
            loss = model.learn_step(**sample)
            
        # output['logits'] = decode_topk(output['topk'])
        
        # print(cls.calculate_loss(output['logits'].reshape(-1, output['logits'].shape[-1]), targets[:, -output_length:].flatten()))
     

    @classmethod
    def calculate_loss( cls, pred, gt = None, input=None , *args, **kwargs):
        if input != None:

            gt = input[:, -pred.shape[1]:].flatten()
        if len(pred.shape) == 3:
            pred = pred.reshape(-1, pred.shape[-1])
        loss_fn = torch.nn.CrossEntropyLoss( *args, **kwargs)
        loss =  loss_fn(pred, gt.to(pred.device))
        return loss
    
if __name__ == "__main__":
    
    
    # EnsembleModel.run_neuron()
    AdapterModel.test()
    # print('FUCK')
    # TransformerModel('gptj', tag='demo', load=True).save_pretrained()
    
    # TransformerModel.run()
    # TransformerModel.experiment()


