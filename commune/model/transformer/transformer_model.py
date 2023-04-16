import os, sys
from pprint import pp

from functools import partial
import asyncio
from copy import deepcopy
from typing import Union, Optional, List
from concurrent import futures
import os, sys
from typing import *
from loguru import logger
import time
from munch import Munch
import argparse
import torch
import json

import streamlit as st


# logger = logger.opt(colors=True)
    
# import torch
import commune
from commune.model import Model
# commune.utils
from torch import nn
# commune.new_event_loop()
from commune.metric import MetricMap

from commune.utils.tokenizer import  decode_topk, get_translation_map, encode_topk, prep_tokenizer
 
"""
Examples 



"""
class TransformerModel( Model):
    shortcuts =  {
        # 0-1B models
        'gpt125m': 'EleutherAI/gpt-neo-125m',

        # 1-3B models
        'gpt2.7b': 'EleutherAI/gpt-neo-2.7B',
        'gpt3b': 'EleutherAI/gpt-neo-2.7B',
        'opt1.3b': 'facebook/opt-1.3b',
        'opt2.7b': 'facebook/opt-2.7b',

        # 0-7B models
        'gptjt': 'togethercomputer/GPT-JT-6B-v1',
        'gptjt_mod': 'togethercomputer/GPT-JT-Moderation-6B',
        'gptj': 'EleutherAI/gpt-j-6b',
        'gptj.pyg6b': 'PygmalionAI/pygmalion-6b',
        'gpt6b': 'cerebras/Cerebras-GPT-6.7B',
        'gptj.instruct': 'nlpcloud/instruct-gpt-j-fp16',
        'gptj.codegen': 'moyix/codegen-2B-mono-gptj',
        'gptj.hivemind': 'hivemind/gpt-j-6B-8bit',
        'gptj.adventure': 'KoboldAI/GPT-J-6B-Adventure',
        'gptj.pygppo': 'TehVenom/GPT-J-Pyg_PPO-6B', 
        'gptj.alpaca.gpt4': 'vicgalle/gpt-j-6B-alpaca-gpt4',
        'gptj.alpaca': 'bertin-project/bertin-gpt-j-6B-alpaca',
        'oa.galactia.6.7b': 'OpenAssistant/galactica-6.7b-finetuned',
        'opt6.7b': 'facebook/opt-6.7b',
        'llama': 'decapoda-research/llama-7b-hf',
        'vicuna.13b': 'lmsys/vicuna-13b-delta-v0',
        'vicuna.7b': 'lmsys/vicuna-7b-delta-v0',
        'llama-trl': 'trl-lib/llama-7b-se-rl-peft',
        'opt.nerybus': 'KoboldAI/OPT-6.7B-Nerybus-Mix',

        # # > 7B models
        'oa.pythia.12b': 'OpenAssistant/oasst-sft-1-pythia-12b',
        'gptneox': 'EleutherAI/gpt-neox-20b',
        'gpt20b': 'EleutherAI/gpt-neox-20b',
        'opt13b': 'facebook/opt-13b',
        'gpt13b': 'cerebras/Cerebras-GPT-13B'
        
         }
    

    def __init__(self,
                # model_name: str="EleutherAI/gpt-j-6b",
                model: str="gpt125m",
                tag :str = None,
                tokenizer:str = None,
                device: str = 'cuda',
                optimizer: dict = {'lr': 0.00001},
                finetune : dict = {'num_layers': 4},
                device_map: Union[dict, str]="auto", 
                load: bool = False,
                test: bool = True,
                **kwargs
                ):
        if tokenizer == None:
            tokenizer = model
        Model.__init__(self, config =locals())
        self.set_params(**self.config)
        if test:
            self.test(self)

    def set_tag(self,tag:str):
        if tag == None:
            if hasattr( self, 'tag'):
                return self.tag
            else:
                tag = 'base'
        self.tag = self.model_name + '::' +tag
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

    def _forward(self,  
                input_ids: torch.Tensor = None, 
                topk:int=32,
                output_length:int = 10,
                output_hidden_states : bool = True,
                hidden_state_index: int = -1,
                hidden_dim_bounds: List =  [0, -1],
                return_keys:List[str] = ['topk', 'stats'],
                train: bool = False,   
                map_tokens: bool = False,
                map_logits: bool = False,
                             
                **kwargs):

        sample = {
        'input_ids': input_ids,
        }
        if map_tokens:
            sample['input_ids'] = self.token_translator.translate_tokens(sample['input_ids'])
        
        for k,v in sample.items():
            if isinstance(v, torch.Tensor):
                sample[k] = sample[k].to(self.device)
        

            
        # clip the input ids to the vocab size
        sample['input_ids'] = torch.clip(sample['input_ids'], 0, self.tokenizer.vocab_size-1)
        if train:
            self.optimizer.zero_grad()
            
        device = self.get_model_device(self.model)
            
        self.stats['inference_start_time'] =  self.time() 
        sample['input_ids'] = sample['input_ids'].to(device)
        model_output = self.model(input_ids=sample['input_ids'].to(device),
                                  output_hidden_states=output_hidden_states)
        self.stats['inference_end_time'] = self.time() 
        self.stats['inference_time'] = self.stats['inference_end_time'] - self.stats['inference_start_time']
        self.stats['inference_steps'] = self.stats.get('inference_steps', 0) + 1
        # sometime we dont care about the begginning of the sequence
        
        output_length = output_length if output_length else model_output.logits.size(1)
        
        output_dict = {}
        # logits
        output_dict['logits']= model_output.logits[:,-output_length:,:]
        
        if map_logits:
            output_dict['logits'] = self.token_translator.translate_logits(output_dict['logits'])
        # topk
        output_dict['topk']=self.encode_topk(output_dict['logits'], topk=topk)
        
        # hidden state
        output_dict['hidden_states'] = model_output.hidden_states[hidden_state_index]
        output_dict['hidden_states'] = output_dict['hidden_states'][:,-output_length:,:]
        output_dict['hidden_states'] = output_dict['hidden_states'][:, :, hidden_dim_bounds[0]:hidden_dim_bounds[1]]
        
        output_dict.update(sample)
        loss = self.calculate_loss(**output_dict) 
        
        if train:
            loss.backward()
            self.optimizer.step()
            self.stats['learn_steps'] = self.stats.get('learn_steps', 0) + 1
        
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        output_dict['loss'] = loss
        # if 'loss_history' not in  self.stats:
        #     self.stats['loss_history'] = []
        # self.stats['loss_history'] += [append(loss.item()]
        # self.stats['mean_loss'] =loss.item()
        

        self.stats['loss'] = (self.stats.get('loss',0)*(self.stats['inference_steps']-1) + loss ) / self.stats['inference_steps']
        output_dict['stats'] = deepcopy(self.stats)         

        return {key:output_dict[key] for key in return_keys}

        
    def set_params(self, 
                   model:str = None,
                   optimizer:dict = None,
                   tokenizer: Union[str, 'tokenizer'] = None,
                   tag:str= None, 
                   finetune: dict = None,
                   stats: dict = None, 
                   device:str=None, 
                   load: bool = False,
                   **kwargs) -> None:   
        
        self.set_model(model)
        self.set_tokenizer(tokenizer)     
        self.set_optimizer(optimizer)
        self.set_finetune(finetune)
        # self.set_device(device)
        self.set_stats(stats)    
        self.set_tag(tag)
        
        if load:
            self.load()
        
        
    def set_model(self, model: Union[str, Dict],state_dict:Dict = None) -> None:
        
        
        from transformers import  AutoModelForCausalLM, AutoModel, AutoConfig


        if isinstance(model, str):
            model_name = model
        elif isinstance(model, dict):
            model_name = model['model_name']
            state_dict = model.get('state_dict', None)
        else:
            raise ValueError(f'invalid model type: {type(model)}')
        
        
        if hasattr(self, 'model_name') and self.model_name == model_name:
            pass

        else:
            self.model_name =  model_name
            self.model_path = self.shortcuts.get(model_name, model_name)
            # config = AutoConfig.from_pretrained(self.model_name)
            
            print(f'loading model from {self.model_path}...')

            model_kwargs = self.config.get('model_kwargs', {})
            model_kwargs['device_map'] = self.config.get('device_map')
            model_kwargs['load_in_8bit']=self.config.get('load_in_8bit', False)
            max_allocation_ratio = self.config.get('max_allocation_ratio', 0.6)
            model_kwargs['max_memory'] = self.free_gpu_memory(fmt='GB', max_allocation_ratio=max_allocation_ratio)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs) 

       
            
            # convert config to config
            model_config = json.loads(self.model.config.to_json_string())         
            self.config['model'] = model_config
            self.config['model']['model_name'] = self.model_name
            self.config['model']['model_path'] = self.model_path
            # yo 
        if state_dict:
            self.model.load_state_dict(state_dict)


    def set_tokenizer(self, tokenizer:Union[str, 'tokenizer', None]):
        from transformers import AutoTokenizer, AutoModel
        from commune.utils.tokenizer import prep_tokenizer

        
        if isinstance(tokenizer, str):

            tokenizer = self.shortcuts.get(tokenizer, tokenizer)
            self.config['tokenizer'] = tokenizer
            
            try:
                # HACK TO INCLUDE LLAMA TOKENIZER
                tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast= True)
            except ValueError:
                
                print('resorting ot use_fast = False')
                tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)

   
        
        self.tokenizer = tokenizer
        
    
        self.std_tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast= True)
        self.tokenizer = prep_tokenizer(self.tokenizer, self.std_tokenizer)
        self.token_translator = self.get_module('model.token_translator')(from_tokenizer=self.config['tokenizer'],to_tokenizer='gpt2')

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


    @classmethod
    def tokenizer_name(self):
        return self.config['tokenizer']

    def tokenize(self, text: str = 'Whadup',
                 padding=True, 
                 truncation=True, 
                 max_length=64,
                 return_tensors='pt',
                 add_special_tokens=False,
                 device:str = None, 
                 **kwargs) -> torch.Tensor:
        """ Returns tokenized text as torch tensor. """
        
        sample = self.tokenizer(text, 
                                             padding=padding, 
                                             truncation=truncation, 
                                             max_length=max_length, 
                                             return_tensors=return_tensors,
                                             add_special_tokens=add_special_tokens, 
                                             **kwargs)  # assume tokenizer.padding_side = 'left'

        device = device if device != None else self.device
        
        sample = dict(
            input_ids= sample['input_ids'].to(device),
            attention_mask= sample['attention_mask'].to(device)
        )
        
        return sample



    def detokenize(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """ Returns tokenized text as torch tensor. """
        
        text = self.tokenizer.batch_decode(input_ids,**kwargs)  # assume tokenizer.padding_side = 'left'

        return text


    @classmethod
    def test(cls, model = 'opt1.3b', topk:int=4096 ,
             dataset:str = 'dataset.text.bittensor',
             num_batches = 10,
             minimum_loss = 4, 
             ):
        
        if isinstance(model, str):
            self = cls(model= model)
        else:
            self = model
            

        dataset = commune.connect(dataset)

        for i in range(10):
            sample = dataset.sample(batch_size=32, no_tokenizer=True)
            sample = self.tokenize(sample['text'])  # assume tokenizer.padding_side = 'left'
            sample['topk'] = topk
            output = self.forward(**sample, train=False)
            cls.print(output['stats'])
        
        # print(cls.calculate_loss(output['logits'].reshape(-1, output['logits'].shape[-1]), targets[:, -output_length:].flatten()))
        

    @classmethod
    def run_train(cls,
              model:str='gptj', 
              dataset : Union[str, 'Module'] = 'dataset::bittensor',
             output_length:int=10,
             sequence_length:int=256,
             adapter: dict = None,
             num_batches: int = 10000, 
             tag:str=None,
             load: bool = False,
             save: bool= True,
             refresh: bool = False):
        if refresh:
            load = False
            

        model = cls(model=model, tag=tag, load=load)
        
        if isinstance(dataset, str):
            dataset = commune.connect(dataset)

        for i in range(num_batches):
            sample = dataset.sample(sequence_length=sequence_length)
            sample['output_length'] =  output_length
            sample['return_keys'] = ['stats']
            sample['train'] = True
            output = model.forward(**sample)
            print(output)
        if save:
            model.save(tag=tag)
            
        return output['stats']
    
    
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
    def models(cls):
        return list(cls.shortcuts.keys())
    
    
    @classmethod
    def remote_train(cls,
             model:str='model::gptj::5',  
             dataset : Union[str, 'Module'] = 'dataset::bittensor',
             params: dict = None,
            output_length:int=10,
            sequence_length:int=256,
            num_batches: int = 100, 
            num_epochs: int = 100,
            tag : str = None,
            save : bool = True,
            load : bool = False,
            refresh: bool = False,
            **kwargs):
        self = commune.connect(model)
        params = params if params != None else {}
        params['tag'] = tag

        if load and (refresh == False):
            self.load(tag=tag)
        
        self.set_params(**params)
        
  
        dataset = commune.connect(dataset)
            
        best_epoch_loss = self.stats.get('best_epoch_loss', 10)
        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch_idx in range(num_batches):
                
                sample = dataset.sample(sequence_length=sequence_length)
                
                print(sample)
                if isinstance(sample, str):
                    continue
                
                sample.update(dict(
                    output_length=output_length,
                    return_keys=['stats'],
                    train = True
                ))
                
                output = self.forward(**sample)
                epoch_loss = output['stats']['loss'] / (batch_idx + 1)
                commune.print(output, 'cyan')
                
                
            if epoch_loss < best_epoch_loss and save:
                output['stats']['epoch_loss'] = epoch_loss
                output['stats']['num_batches'] = num_batches
                output['stats']['best_epoch_loss'] = best_epoch_loss
                self.set_stats(stats=dict(epoch=epoch, loss=epoch_loss))
                self.save(tag=tag)

        return output['stats']
    

    default_models = list(shortcuts.keys())
          
          
    fleet_group = {
        
        '0': [ 'gpt125m', 'gpt2.7b','opt1.3b', 'opt2.7b','gptj', 'vicuna.7b', 'opt6.7b'],
        '1': [ 'gptj.alpaca', 'gptj.pygppo', 'opt6.7b', 'oa.galactia.6.7b', 'vicuna.7b', 'gptj'],
        '2': [ 'gptj.instruct', 'gpt6b', 'opt6.7b', 'oa.galactia.6.7b', 'vicuna.7b', 'gptj'],


        # '0': ['vicuna.7b', 'opt6.7b', 'oa.galactia.6.7b'],

        'all': default_models,
        'default': default_models,
    }
    @classmethod
    def deploy_fleet(cls, 
                     models: List[str] = '1',
                     replace: bool = False,
                     max_models: int = 8,
                     wait_for_server = False
                     ) -> List[str]:


        
        models = cls.fleet_group.get(models, models)
    
    
        deployed_model_tags = {}
        models = models[:max_models]
        deployed_models = []
        for model in models:
            commune.print(f'Deploying Model {model}', 'green')
            cls.deploy(model, wait_for_server=wait_for_server, replace=replace)
            deployed_models.append(model)
            commune.print(f'Deployed Model {model} ({len(deployed_models)}/{len(models)})', 'green')
            
            
        return deployed_models
        
        
    @classmethod
    def deploy(cls,
               *models: str,
               tokenizer: str=None, 
               name: str =None, 
               wait_for_server: bool = False, 
               mode:str = 'pm2',
               replace:bool = False,
               **kwargs):


        assert len(models) > 0
        model_names = []
        for model in models:
            model_kwargs =  {'model': model, 'tokenizer': tokenizer, **kwargs}
            name = f'model.{model}'
            module_exists = cls.module_exists(name)     
            if replace == False:
                cls.print(f'Model {name} already exists', color='yellow')
                continue
            cls.launch(name=name,kwargs=model_kwargs, mode=mode)
            if wait_for_server:
                cls.wait_for_server(name=name, sleep_interval=20, timeout=1000)
            model_names.append(name) 
        return model_names
            
    @classmethod
    def sandbox(cls):
        self = cls(model='opt2.7b')
        
        
if __name__ == "__main__":
    
    TransformerModel.run()


