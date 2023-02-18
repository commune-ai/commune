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
# logger = logger.opt(colors=True)

if os.getenv('USE_STREAMLIT') == 'true':
    import streamlit as st
    
    
# import torch
import commune
# commune.utils
from torch import nn
    
"""
Examples 



"""

class EnsembleModel( nn.Module, commune.Module):

    def __init__(self,
                models: List[str] = ['model::gpt2.7b',
                                     'model::gpt125m', 
                                     'model::opt13b', 
                                     'model::gptj', 
                                     'model::gptjt'],
                tokenizer: 'tokenizer' = 'bittensor',
                optimizer:  'torch.optimizer' = None,
                metrics: Dict= None,
                load: bool = True,
                tag= None,
                device = None,
                ):
        nn.Module.__init__(self)
        self.layer = commune.import_object('commune.model.layer.LayerBlock')()
        self.tag = tag
        
        self.model_device = 'cpu'
        
        self.model_name = 'ensemble'
        
        # set model and tokenizer
        self.set_models(models=models)

        # set tokenizer to model name (HF only) if tokenizer == None
        self.set_tokenizer(tokenizer=tokenizer if tokenizer != None else self.model_name)
        
        self.set_optimizer(optimizer=optimizer)
        
        self.set_metrics(metrics=metrics)
        
        self.set_stats()
        
        if load:
            self.load()
        
        
        
    def set_optimizer(self, optimizer:'torch.optim.Optimizer'=None, *args, **kwargs):
        
        if optimizer == None:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        
        self.optimizer = optimizer
        return self.optimizer


    def calculate_loss(self, pediction, gt):
        loss =  self.metrics['cross_entropy'](pediction, gt)
        return loss

    def set_metrics(self, metrics=None):
        self.metrics = {}
        if metrics == None:
            self.metrics['cross_entropy'] =  torch.nn.CrossEntropyLoss()
        return metrics
    
    async def async_model_forward(self, model, *args, **kwargs):
        return self.models[model].forward(*args, **kwargs)
        
        
        
    def aggregate(self, 
                  x: List[torch.Tensor], 
                  *args, **kwargs) -> Dict[str, torch.Tensor]:
        
        
        if isinstance(x, list):
            x = torch.stack(x, dim=0)
        x = torch.mean(x, dim=0)
        x = torch.softmax(x, dim=-1)
        x = torch.log(x + 1e-10)
        
        return x
    def forward(self, topk=4096, output_length=10, *args, **kwargs):
        jobs = []
        kwargs['logit_remap'] = False
        kwargs['token_remap'] = True
        kwargs['output_logits'] = False
        kwargs['topk'] = topk
        kwargs['output_length'] = output_length

        for model in self.models:
            kwargs['token_remap'] = True
            job = self.async_model_forward(model=model, *args, **kwargs)
            jobs.append(job)
            
        # return 
        
        tensor_outputs =  asyncio.run(asyncio.gather(*jobs))
        
        for model_i, tensor_output in enumerate(tensor_outputs):
            import streamlit as st
            if 'topk'  in tensor_output:
                max_token_index = int(torch.max(tensor_output['topk']).item()) 
                tensor_output['logits'] = decode_topk(tensor_output['topk'], vocab_size=int(50600))
                tensor_outputs[model_i]
            else:
                tensor_outputs[model_i] = tensor_output
              
        output_dict = dict(
            peer_outputs = torch.stack([x['logits'] for x in tensor_outputs], dim=0)
        )
        output_dict['ensemble_output'] = self.aggregate(output_dict['peer_outputs'])
        
        return output_dict
    

    @property
    def device(self):
        # deepspeed has .module.device to access device
        return self.model_device

    def set_models(self, models:List[str]):
        self.models = {}
        for model in models:
            self.models[model] = commune.connect(model)
        return self.models


    def list_models(self):
        return list(self.models.keys())


    def set_tokenizer(self, tokenizer:Union[str, 'tokenizer', None]):
        from transformers import AutoTokenizer
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

    @staticmethod
    def encode_topk( forward_response_tensor: torch.Tensor , topk:int=4096) -> torch.Tensor:
        """ Returns topk tokens/probabilities given unnormalized logits as input. """

        #import ipdb; ipdb.set_trace()

        logits = forward_response_tensor  # unnormalized logit scores: [batch_size, sequence_len, vocab_size]
        probs = torch.softmax(logits, dim=-1).to(torch.float32)  # normalized probabilities: [batch_size, sequence_len, vocab_size]

        topk_indices = torch.argsort(probs, dim=-1, descending=True)[...,:topk]
        # topk_values, topk_indices = torch.topk(probs, topk) # topk probs and indices: [batch_size, sequence_len, topk]

        topk_values = probs.gather( index=topk_indices, dim=-1)
        encoded_probs = torch.cat([topk_values, topk_indices], dim=-1)  # [batch_size, sequence_len, topk + topk]
        return encoded_probs  # [batch_size, sequence_len, topk + topk]

    def tokenize(self, text: str = 'Whadup', input_ids_only:bool = True, device: str=None) -> torch.Tensor:
        """ Returns tokenized text as torch tensor. """
        device = device if device != None else self.device
        tokenizer_output = self.tokenizer(text, return_tensors='pt')
        if input_ids_only:
            return tokenizer_output.input_ids.to(self.device)
        return self.tokenizer(text, return_tensors='pt').input_ids.to(self.device)

    
    def learn_step(self, **sample ):
        targets = sample['input_ids'][:,1:].to(self.device)
        sample['input_ids'] = sample['input_ids'][:,:-1].to(self.device)
        self.optimizer.zero_grad()
        
        
        with torch.autocast(device_type='cuda'):
            pred = self.forward(**sample, no_grad=False)
            logits =  pred['logits']
            targets = targets[:,-logits.shape[1]:]
            pred = logits.reshape(-1, logits.size(-1))
            loss = self.calculate_loss(pediction=logits.reshape(-1, logits.size(-1)), 
                                        gt=targets.flatten().to(self.device))              
        

        loss.backward()
        self.optimizer.step()
    
        
        return loss.item()

    def set_stats(self, stats:dict=None): 
        if stats == None:
            stats =  dict(
                steps = 0,
                loss = 0,
            )
        self.stats = Munch(stats)
        

    @property
    def module_tag(self): 
        return self.resolve_module_tag()
    
    def resolve_module_tag(self, tag=None):
        tag = tag if tag else self.tag
        module_tag = self.model_name.replace("/", "_")
        if tag:
            module_tag +=  f'_{tag}'
        return module_tag
    
    def save(self, tag:str = None, trainable_only:bool = True):
        module_tag = self.resolve_module_tag(tag=tag)
        path = self.resolve_path(module_tag)
        model_state_dict = self.models.state_dict()
        
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
        state_dict = self.models.state_dict()
        for k,v in loaded_state['model'].items():
            assert k in state_dict
            state_dict[k] = v
        self.models.load_state_dict(state_dict)
        self.optimizer.load_state_dict(loaded_state['optimizer'])
        self.set_stats(loaded_state['stats'])
        

    @classmethod
    def train(cls, 
                    tag:str = 'demo', 
                    num_batches:int = 200,
                    num_epochs:int = 200, 
                    dataset:str= 'BittensorDataset', **kwargs):
        model = cls(tag=tag, load=True,  **kwargs)
        dataset = cls.connect(dataset)
        
        best_loss = 10e10
        for epoch in range(num_epochs):
            total_epoch_loss = 0
            epoch_loss = 0
            if epoch > 0:
                model.load(tag=tag)
            for i in range(num_batches):
                sample = dataset.sample()
                loss = model.learn_step(**sample)
                try:
                    total_epoch_loss += loss
                except:
                    continue
                epoch_loss = total_epoch_loss/(i+1)
                info_str = f'Batch {i}/{num_batches} Epoch {epoch}/{num_epochs} CE: {loss} Epoch Loss: {epoch_loss} Best Loss: {best_loss}'
                logger.success(info_str)
                print('BROOO')
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                try:
                    model.save(tag=tag)
                except TypeError:
                    continue


    @classmethod
    def test(cls):
        import streamlit as st
        data = commune.connect('dataset::bittensor')
        sample = data.sample()
        model = EnsembleModel()
        t = commune.timer()
        pred = model.forward(**sample, output_hidden_states=False, output_logits=False, output_topk=True, output_length=10, topk=4096 )
        
        
        metrics = {
            'info': tensor_dict_info(pred),
            'seconds': t.seconds
        }
        st.write(metrics)
        
if __name__ == "__main__":
    
    EnsembleModel.test()
    # print('FUCK')
    # TransformerModel('gptj', tag='demo', load=True).save_pretrained()
    
    # TransformerModel.run()
    # TransformerModel.experiment()


