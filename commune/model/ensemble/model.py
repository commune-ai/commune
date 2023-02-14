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
                models: List[str] = ['model::gptj', 'model::gptjt', 'model::gpt2.7b'],
                **model_kwargs
                ):
    
        
        nn.Module.__init__(self)
        
        # set model and tokenizer
        self.set_model(model_name=model_name,device=device, autocast=autocast, **model_kwargs)

        # set tokenizer to model name (HF only) if tokenizer == None
        self.set_tokenizer(tokenizer=tokenizer if tokenizer != None else self.model_name)
        
        self.set_optimizer(optimizer=optimizer)
        
        self.set_metrics(metrics=metrics)
        
        self.set_stats()
        
        if load:
            self.load()
        
        
        
    def set_optimizer(self, optimizer:'torch.optim.Optimizer'=None, *args, **kwargs):
        
        if isinstance(optimizer, dict):
            module_path = optimizer.pop('module', None)
            assert module_name != None, f'Please specify a valid optimizer ex: torch.optim.Adam'
            optimizer_class = self.import_object(module_path) 
            optimizer_kwargs = optimizer.get('kwargs', optimizer)
            optimizer_args = optimizer.get('args', [])
            self.optimizeroptimizer_class(*optimizer_args,**optimizer_kwargs)
                
        elif optimizer == None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        
        else:
            raise NotImplementedError(optimizer)
        
        
        return self.optimizer


    def calculate_loss(self, pediction, gt):
        loss =  self.metrics['cross_entropy'](pediction, gt)
        return loss

    def set_metrics(self, metrics=None):
        self.metrics = {}
        if metrics == None:
            self.metrics['cross_entropy'] =  torch.nn.CrossEntropyLoss()
        return metrics
    
    @classmethod
    def forward(self, *args, **kwargs):
        model_output_list = []
        for model in self.models:
            model_output = model.forward(*args, **kwargs)
            model_output_list.append(model_output)


    def local_forward(self,  
                input_ids: torch.Tensor = None, 
                text: str = None,
                attention_mask: torch.Tensor= None, 
                topk:int=None, 
                output_hidden_states:bool=False, 
                output_logits:bool = True,
                verbose:bool = False,
                output_length:int = 10,
                **kwargs):

        # tokenizer the text if text is provided 

            
        # if input_ids is not provided, tokenize the text
        if input_ids == None:
            # if text is provided, tokenize the text
            if isinstance(text, str) or (isinstance(text, list) and isinstance(text[0], str)):
                input_ids = self.tokenize(text)
            else:
                raise ValueError('Please provide either input_ids or text')
        
        elif isinstance(input_ids, str) or (isinstance(input_ids, list) and isinstance(input_ids[0], str)):
            input_ids = self.tokenize(input_ids)

        input_dict = dict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states= output_hidden_states
                    )

        # ensure the input_ids and attention mask is a tensor
        for k in ['input_ids', 'attention_mask']:
            v = input_dict[k]
            if isinstance(v,  list):
                input_dict[k] = torch.tensor(v)
            elif isinstance(v, type(None)):
                del input_dict[k]
                continue
            if isinstance(v,  torch.Tensor):
                input_dict[k] = input_dict[k].to(self.device)

        # if verbose:
        #     print('INPUT_STATISTICS: ',tensor_info_dict(input_dict))

        model_output = self.model(**input_dict)
        output_length = output_length if output_length else model_output.logits.size(1)
            
        output_dict = {}
        if topk:
            topk_tensor = self.encode_topk(model_output.logits[:,-output_length:,:], topk=topk)
            output_dict['topk']=topk_tensor
            
        if output_logits:
            output_dict['logits']=model_output.logits[:,-output_length:,:]

        if output_hidden_states:
            output_dict['hidden_states'] = model_output.hidden_states[-1][:,-output_length:, :]

        # if verbose:
        #     print('OUTPUT_STATISTICS: ',tensor_info_dict(output_dict))

        return output_dict


    @property
    def device(self):
        # deepspeed has .module.device to access device
        return self.model.device

    def set_model(self, model_name:str, device:str = None, autocast:bool = False, **extra_model_kwargs):
        from transformers import  AutoModelForCausalLM, AutoModel, AutoConfig
        self.autocast = autocast
        self.model_name = self.shortcuts.get(model_name, model_name)
        # model_config = AutoConfig.from_pretrained(self.model_name)
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, 
                                            **extra_model_kwargs)        
        import json
        self.model_config = json.loads(self.model.config.to_json_string())
        self.model = self.model.to(device)
        if self.autocast:
            self.model = self.model.half()
            
        return self.model





    def set_tokenizer(self, tokenizer:Union[str, 'tokenizer', None]):
        from transformers import AutoTokenizer
        if isinstance(tokenizer, str):
            tokenizer = self.shortcuts.get(tokenizer, tokenizer)
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

    @classmethod
    def test_model(cls, batch_size=8, sequence_length=256, model_name='EleutherAI/gpt-neox-20b'):
        self = cls(serve=False, model_name=model_name)
        example = ["My name is Philipp and I"]*batch_size
        input_ids = self.tokenizer(example,return_tensors="pt", max_length=sequence_length, padding='max_length').input_ids.to(self.device)
        
        print('TESTING LOGITS OUTPUT')
        logits = self.forward(input_ids, output_hidden_states=True, topk=None,verbose=True)
        
        print('TESTING TOPK OUTPUT')
        logits = self.forward(input_ids, output_hidden_states=True, topk=None,verbose=True)
    
    
    def learn_step(self, **sample ):
        targets = sample['input_ids'][:,1:]
        sample['input_ids'] = sample['input_ids'][:,:-1]
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
        model_state_dict = self.model.state_dict()
        
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
        state_dict = self.model.state_dict()
        for k,v in loaded_state['model'].items():
            assert k in state_dict
            state_dict[k] = v
        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(loaded_state['optimizer'])
        self.set_stats(loaded_state['stats'])
        

    @classmethod
    def train(cls, 
                    model:str='gptj',
                    tag:str = 'demo', 
                    num_batches:int = 200,
                    num_epochs:int = 200, 
                    dataset:str= 'BittensorDataset', **kwargs):
        model = cls(model_name=model,tag=tag, load=True,  **kwargs)
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
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                try:
                    model.save(tag=tag)
                except TypeError:
                    continue


if __name__ == "__main__":
    # print('FUCK')
    # TransformerModel('gptj', tag='demo', load=True).save_pretrained()
    TransformerModel.run()
    # TransformerModel.run()
    # TransformerModel.experiment()


