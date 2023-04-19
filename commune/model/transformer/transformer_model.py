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
    

    def __init__(self, model = 'gp125m',
                **kwargs
                ):
        
        Model.__init__(self, locals())         
        config = self.config
        self.set_model(config)
        
        if config.test:
            self.test(self)

    default_tag = 'base'
    def set_tag(self,tag:str):
        if tag is None:
            tag = self.default_tag
        self.tag = self.model_path.replace('/','--')+'--'+tag
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
                input_ids: torch.Tensor, 
                topk:int=32,
                output_length:int = 10,
                output_hidden_states : bool = True,
                hidden_state_index: int = -1,
                hidden_dim_bounds: List =  [0, -1],
                return_keys:List[str] = ['topk', 'stats'],
                train: bool = False,   
                map_tokens: bool = False,
                map_logits: bool = False,  
                tag : str = None,                           
                **kwargs):

        sample = {
        'input_ids': input_ids,
        }
    
        if map_tokens:
            offset_mapping, offset_mapping_std, original_input_ids = None, None, None

            original_input_ids = self.copy(sample['input_ids'])
            tokens = self.token_translator.translate_tokens(input_ids=sample['input_ids'], return_offsets_mapping=True)
            offset_mapping = tokens.offset_mapping
            offset_mapping_std = tokens.offset_mapping_std
            sample['input_ids'] = tokens.input_ids
        
        for k,v in sample.items():
            if isinstance(v, torch.Tensor):
                sample[k] = sample[k].to(self.device)
        

            
        # clip the input ids to the vocab size
        sample['input_ids'] = torch.clip(sample['input_ids'], 0, self.tokenizer.vocab_size-1)
        if train:
            self.optimizer.zero_grad()
            
        device = self.get_model_device(self.model)
        
        self.stats['time'] =  self.time()
        sample['input_ids'] = sample['input_ids'].to(device)
        
        good_logits = False
        model_output = self.model(input_ids=sample['input_ids'].to(device),
                                output_hidden_states=output_hidden_states)
        
    
        # check if there are any nans in the logits
        logits_has_nans =  torch.isnan(model_output.logits).any()
        if logits_has_nans:
            self.print('logits has nans with sample input_ids: ', sample['input_ids'])
                
        self.stats['latency'] = self.round(self.time() - self.stats['time'], sig=2)
        
        self.stats['inference_steps'] = self.stats.get('inference_steps', 0) + 1
        # sometime we dont care about the begginning of the sequence
        
        output_length = output_length if output_length else model_output.logits.size(1)
        
        output_dict = {}
        # logits
        output_dict['logits']= model_output.logits[:,-output_length:,:]
        
        if map_logits:
            output_dict['logits'] = self.token_translator.translate_logits(logits = output_dict['logits'],
                                                                           offset_mapping=offset_mapping_std,
                                                                           offset_mapping_std=offset_mapping_std,
                                                                           tokens=sample['input_ids'],
                                                                           tokens_std=original_input_ids)
        # topk
        output_dict['topk']=self.encode_topk(output_dict['logits'], topk=topk)
        
        # hidden state
        output_dict['hidden_states'] = model_output.hidden_states[hidden_state_index]
        output_dict['hidden_states'] = output_dict['hidden_states'][:,-output_length:,:]
        output_dict['hidden_states'] = output_dict['hidden_states'][:, :, hidden_dim_bounds[0]:hidden_dim_bounds[1]]
        
        output_dict['input_ids'] = sample['input_ids']
        loss = self.calculate_loss(**output_dict) 
        
        # check if loss is nan
        if torch.isnan(loss):
            self.print(output_dict)
            self.print('Loss is nan, skipping backward pass')
            train = False
            loss = torch.tensor(10)
            raise Exception('Loss is nan, skipping backward pass')
        
        if train:
            loss.backward()
            self.optimizer.step()
            loss = loss.item()
                
            self.stats['learn_steps'] = self.stats.get('learn_steps', 0) + 1
            self.stats['lr'] = self.optimizer.param_groups[0]['lr']
            self.stats['epoch_loss'] = (self.stats.get('epoch_loss', 0)*(self.stats['learn_steps']-1) + loss)/self.stats['learn_steps']
        else:
            loss = loss.item()
        
        inference_steps = self.stats['inference_steps']
        alpha = self.config.get('loss_alpha', 0.9)
        assert 0 < alpha < 1, 'loss_alpha must be between 0 and 1'
        past_loss = self.stats.get('loss', 0)
        self.stats['ma_loss'] = (past_loss*(1-alpha) + alpha*loss) if past_loss != 0 else loss
        self.stats['alpha'] = alpha
        self.stats['sample_loss'] = loss

        if train and self.stats['learn_steps'] % self.config['epoch_length'] == 0:
            self.stats['epoch'] = self.stats.get('epoch', 0) + 1
            self.stats['epoch_loss_history'] = self.stats.get('epoch_loss_history',[]) + [{'loss': self.stats['epoch_loss'], 'time': self.time()}]
            self.stats['learn_steps'] = 0

            self.print('saving model...')
            self.save(tag)

        output_dict['stats'] = deepcopy(self.stats)
        
        return {key:output_dict[key] for key in return_keys} 
        
        
        
    @classmethod
    def infer_model_size(cls, model, buffer_ratio=1):
        from transformers import  AutoModelForCausalLM, AutoModel, AutoConfig
        from accelerate import init_empty_weights
        
        model = cls.shortcuts.get(model, model)

        print(f'loading config model from {model}...')
        if isinstance(model, str):
            model_config = AutoConfig.from_pretrained(model)
            model_config_dict = model_config.to_dict()
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(model_config)
    
        model_size = cls.get_model_size(model)*buffer_ratio
        return model_size


    @classmethod
    def infer_max_memory(cls, model, buffer_ratio:int=1.2):
        
        
        model_size = cls.infer_model_size(model)*buffer_ratio
        free_gpu_memory = cls.free_gpu_memory(fmt='b')
        gpus = list(free_gpu_memory.keys()) 
        total_gpu_memory = sum(free_gpu_memory.values())
        
        
        assert model_size < total_gpu_memory, f'model size {model_size} is larger than total gpu memory {total_gpu_memory}, over gpus {gpus}'



        cls.print(f'{model_size}')

        unallocated_model_memory = model_size
        # max_memory = {}
        max_memory = {}
        
        
        while unallocated_model_memory > 0:
            most_free_gpu, most_free_gpu_memory = cls.most_free_gpu(free_gpu_memory=free_gpu_memory, return_tuple=True)

            
            allocated_memory = min(unallocated_model_memory, most_free_gpu_memory)
            unallocated_model_memory -= allocated_memory
            max_memory[most_free_gpu] = allocated_memory
            free_gpu_memory[most_free_gpu] -= allocated_memory
            
            
        for k,v in max_memory.items():
            max_memory[k] = int(v//1e9)
            
        return max_memory
            

        
        
    def set_model(self, config) -> None:
        
        
        from transformers import  AutoModelForCausalLM, AutoModel, AutoConfig
        from accelerate import init_empty_weights
        

        model_name = config['model_name'] = config['model']
        self.model_path = config['model_path'] =self.shortcuts.get(model_name, model_name)
        # config = AutoConfig.from_pretrained(self.model_name)
                

        
        model_config = AutoConfig.from_pretrained(self.model_path)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(model_config)
            
            
        model_config_dict = model_config.to_dict()
        for k,v in model_config_dict.items():
            assert k not in config, f'config key {k} not found in config'
            config[k] = model_config_dict[k]        
        config = self.munch(config)
        free_gpu_memory = self.free_gpu_memory(fmt='b')
        
        model_kwargs = {}
        infer_model_size = self.infer_model_size(self.model_path)
        infered_max_memory = self.infer_max_memory(self.model_path)
        model_kwargs['load_in_8bit'] = bool(config.load_in_8bit)
        model_kwargs['max_memory'] = free_gpu_memory
        
        if config.device != None:
            if self.is_number(config.device):
                config.device = f'cuda:{config.device}'
            if 'cuda' in config.device:
                if ':' in config.device:
                    device_id = int(config.device.split(':')[-1])
                else:
                    device_id = self.most_free_gpu(free_gpu_memory=free_gpu_memory)
                    
                gpy_memory = free_gpu_memory[device_id]
                assert gpy_memory > infer_model_size, f'gpu memory {gpy_memory} is less than model size {infer_model_size}'
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs) 
            self.model.to(config.device)
            
        else:
            
            model_kwargs['device_map'] = config.device_map
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs) 

        
        
        self.set_tokenizer(config)
        self.set_optimizer(config.optimizer)
        self.set_finetune(config.finetune) 
          
        self.set_tag(config.tag)
        self.set_stats(config.stats)    
        self.set_epoch_length(config.epoch_length)        
        if config.load:
            self.load() 
            
        self.config = config


    def set_epoch_length(self, epoch_length:int) -> int:
        assert isinstance(epoch_length, int)
        self.epoch_length = self.config['epoch_length']=  epoch_length
        return self.epoch_length

    def set_tokenizer(self, config):
        from transformers import AutoTokenizer, AutoModel
        from commune.utils.tokenizer import prep_tokenizer

        self.print('setting tokenizer...')
        
        if config.tokenizer is None:
            tokenizer = config.model_path
        assert isinstance(tokenizer, str, )
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
        self.std_tokenizer = prep_tokenizer(self.std_tokenizer)
        self.tokenizer = prep_tokenizer(self.tokenizer, self.std_tokenizer)
        self.token_translator = self.get_module('model.token_translator')(tokenizer=tokenizer, std_tokenizer=self.std_tokenizer)

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
    def test(cls, model = 'gpt125m', 
             topk:int=256 ,
             dataset:str = 'dataset.text.bittensor',
             num_batches = 3,
             sequence_length = 256,
             batch_size = 32,
             device = None, 
             remote = False, 
             train = False,
             load = False,
             ):
        
        if not commune.server_exists(dataset):
            commune.deploy(dataset)
        
        dataset = commune.connect(dataset, wait_for_server=True)
        
        
    
        namespace = commune.namespace()
        
        if model in namespace:
            model_name = model
            model = commune.connect(model_name)
        elif isinstance(model, str):
            model = cls(model= model, test=False, device=device)
        else:
            model = model
        if load:
            model.load()

        for i in range(num_batches):
            sample = dataset.sample(batch_size=batch_size,sequence_length=sequence_length, no_tokenizer=False)
            sample['topk'] = topk
            sample['map_tokens'] = False
            sample['map_logits'] = False
            sample['train'] = train
            sample['autocast'] = True
            sample['timeout'] = 6
            sample['return_keys'] = [ 'topk', 'stats']
            
            output = model.forward(**cls.copy(sample))
            output['logits'] = decode_topk(output['topk'] )
            
            output['input_ids'] = sample['input_ids']
            cls.print(f"step: {i}/{num_batches} stats: {output['stats']}")
            # cls.print(outpu
            # t)
            # cls.print(output['stats'])
        
        # print(cls.calculate_loss(output['logits'].reshape(-1, output['logits'].shape[-1]), targets[:, -output_length:].flatten()))
        
        model.save()
    
    
    @classmethod
    def train(cls, model = 'gpt125m', 
             topk:int=3096 ,
             dataset:str = 'dataset.text.bittensor',
             num_batches = 100,
             sequence_length = 256,
             batch_size = 32,
             minimum_loss = 4, 
             lr = 1e-4,
             remote = False, 
             tag = None,
             load = False,
             device = None, 
             ):
        
        
        
        
        if remote:
            namespace = cls.namespace()
            model_name = f'model.{model}'
            model = cls.connect(model_name)
        
        if isinstance(model, str):
            model = cls(model= model, tag=tag, device=device)
        else:
            model = model
        if load:
            model.load()
        dataset = commune.connect(dataset)

        for i in range(num_batches):
            sample = dataset.sample(batch_size=batch_size,sequence_length=sequence_length, no_tokenizer=False)
            sample['topk'] = topk
            sample['map_tokens'] = False
            sample['map_logits'] = False
            sample['train'] = True
            sample['autocast'] = True
            sample['timeout'] = 6
            sample['return_keys'] = [ 'logits', 'stats']
            
            output = model.forward(**cls.copy(sample))
            output['input_ids'] = sample['input_ids']
            cls.print(f"step: {i}/{num_batches} stats: {output['stats']}")
        
        # print(cls.calculate_loss(output['logits'].reshape(-1, output['logits'].shape[-1]), targets[:, -output_length:].flatten()))
        
        model.save()
    
    

    
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
        
        '0': [ 'gpt125m', 'gpt2.7b', 'opt2.7b','gptj'],
        '1': [ 'gptj.alpaca', 'gptj.pygppo', 'opt6.7b', 'oa.galactia.6.7b', 'vicuna.7b', 'gptj'],
        '2': [ 'gptj.instruct', 'gpt6b', 'opt6.7b', 'oa.galactia.6.7b', 'vicuna.7b', 'gptj'],


        # '0': ['vicuna.7b', 'opt6.7b', 'oa.galactia.6.7b'],

        'all': default_models,
        'default': default_models,
    }
    @classmethod
    def deploy_fleet(cls, 
                     model = 'gptj',
                     tags= ['alan', 'bob', 'chris', 'dan', 'elon', 'frank', 'greg', 'huck' ], 
                     replace: bool = True,
                     max_models: int = -1,
                     device: str = None,
                     wait_for_server = False, 
                     ) -> List[str]:
        free_gpu_memory = cls.free_gpu_memory()
        deployed_models = []
        for i, tag in enumerate(tags):
            
            cls.deploy(model, tag=tag, device=i, replace=replace, wait_for_server=wait_for_server)
            deployed_models+= [f'{model}.{tag}']

            
        return deployed_models
        
    @classmethod
    def undeployed_models(cls, models: List[str] = 'all'):
        models = cls.fleet_group.get(models, models)
        undeployed_models = []
        for model in models:
            if cls.module_exists(f'model.{model}') == False:
                undeployed_models.append(model)
        return undeployed_models
        
    @classmethod
    def deploy(cls,
               *models: str,
               tokenizer: str=None, 
               name: str =None, 
               wait_for_server: bool = False, 
               mode:str = 'pm2',
               tag = None,
               device = None, 
               replace:bool = True,
               **kwargs):


        assert len(models) > 0
        model_names = []
        
        free_gpu_memory = cls.free_gpu_memory()
        for model in models:
            model_size = cls.infer_model_size(model)
            cls.print(f'Infered model size for {model} is {model_size}', color='yellow')
            model_kwargs =  {'model': model, 'tokenizer': tokenizer, **kwargs}
            name = f'model.{model}'
            if tag != None:
                name = f'{name}.{tag}'
            model_kwargs['tag'] = tag
            model_kwargs['device'] = device
            module_exists = cls.module_exists(name)     
            if replace == False and module_exists:
                cls.print(f'Model {name} already exists', color='yellow')
                continue
            cls.launch(name=name,kwargs=model_kwargs, mode=mode)
            if wait_for_server:
                cls.wait_for_server(name=name, sleep_interval=5, timeout=1000)
            model_names.append(name) 
        return model_names
            
    @classmethod
    def sandbox(cls):
        self = cls(model='opt2.7b')
        
        

        
        
if __name__ == "__main__":
    
    TransformerModel.run()


