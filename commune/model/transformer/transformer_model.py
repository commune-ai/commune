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
import pandas as pd

from munch import Munch
import argparse
import torch
import json
import random

import streamlit as st


# logger = logger.opt(colors=True)
    
# import torch
import commune as c
from torch import nn

from commune.utils.tokenizer import  decode_topk, get_translation_map, encode_topk, prep_tokenizer

"""
Examples 



"""

shortcuts =  {
    # 0-1B models
    'gpt125m': 'EleutherAI/gpt-neo-125m',

    # 1-3B models
    'gpt2.7b': 'EleutherAI/gpt-neo-2.7B',
    'gpt3b': 'EleutherAI/gpt-neo-2.7B',
    'opt1.3b': 'facebook/opt-1.3b',
    'opt2.7b': 'facebook/opt-2.7b',
    # 'gpt3btuning' : ''

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
    'pygmalion-6b': 'PygmalionAI/pygmalion-6b',
    # # > 7B models
    'oa.pythia.12b': 'OpenAssistant/oasst-sft-1-pythia-12b',
    'gptneox': 'EleutherAI/gpt-neox-20b',
    'gpt20b': 'EleutherAI/gpt-neox-20b',
    'opt13b': 'facebook/opt-13b',
    'gpt13b': 'cerebras/Cerebras-GPT-13B',
    
        }


from torch import nn
class TransformerModel(c.Model):
    shortcuts = shortcuts
    model_options = list(shortcuts.keys()) + list(shortcuts.values())



    default_tag = 'base'
    
    def __init__(self,
                 config = None,
                 **kwargs
                ):
        
        nn.Module.__init__(self) 
        # sets to self.config (with kwargs injected)
        config = self.set_config(config, kwargs=kwargs)
        self.set_stats(config.stats)
        self.set_model(config)
        
    
    def set_tag(self,tag:str):
        if tag is None:
            tag = self.default_tag
        self.tag = self.model_path.replace('/','--')+'--'+tag
    @classmethod
    def calculate_loss( cls, logits: torch.Tensor,
                       input_ids:torch.Tensor,
                       return_value = False,
                       **kwargs) -> torch.Tensor:
        '''
        Calculate the loss for the model.
        '''
        gt = input_ids[:, -(logits.shape[1]-1):].flatten()
        pred = logits[:, :logits.shape[1]-1]
            
        if len(pred.shape) == 3:
            pred = pred.reshape(-1, pred.shape[-1])
        
        assert gt.shape[0] == pred.shape[0], f'gt.shape: {gt.shape} pred.shape: {pred.shape}'

        loss_fn = torch.nn.CrossEntropyLoss()
        loss =  loss_fn(pred, gt.to(pred.device))
        
        # check if loss is nan
        if torch.isnan(loss):
            self.print('Loss is nan, skipping backward pass')
            train = False
            loss = torch.tensor(10)
            raise Exception('Loss is nan, skipping backward pass')
        
        if return_value:
            loss = loss.item()
        
        return loss

    hf = c.module('huggingface')()


    def generate(self, 
                 text:str,
                 **kwargs) -> List[str]:
        
        
        input_ids = self.tokenize(text)['input_ids']
        output_ids =  self.model.generate(input_ids, **kwargs)
        output_text = self.detokenize(output_ids, skip_special_tokens=True)
        return output_text
    
    @classmethod
    def test_generate(cls, text='Hello world', **kwargs):
        model = cls()
        output_text = model.generate(text, **kwargs)
        return output_text
    def _forward(self,  
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor = None,
                topk:int=32,
                output_length:int = None,
                output_hidden_states : bool = True,
                hidden_state_index: int = -1,
                hidden_dim_bounds: List =  [0, -1],
                return_keys:List[str] = ['topk', 'stats'],
                train: bool = False,   
                max_sequence_length : int = None,
                map_tokens: bool = False,
                map_logits: bool = False,  
                tag : str = None,                           
                **kwargs):
        

        
        # resolve the output length
        output_length = output_length or self.config.output_length or input_ids.shape[1]
        if output_length > input_ids.shape[1]:
            output_length = input_ids.shape[1]
        # resolve the max sequence length (sometimes we want to clip the input to make it faster)
        max_sequence_length = max_sequence_length or self.config.max_sequence_length or input_ids.shape[1]
        attention_mask = attention_mask if isinstance(attention_mask, torch.Tensor) else torch.ones_like(input_ids)


        sample = {
        'input_ids': input_ids[:, -max_sequence_length:],
        'attention_mask': attention_mask,
        }
        
        if map_tokens:
            offset_mapping, offset_mapping_std, original_input_ids = None, None, None
            original_input_ids = self.copy(sample['input_ids'])
            tokens = self.token_translator.translate_tokens(input_ids=sample['input_ids'], return_offsets_mapping=True)
            offset_mapping = tokens.offset_mapping
            offset_mapping_std = tokens.offset_mapping_std
            sample['input_ids'] = tokens.input_ids
            sample['attention_mask'] = tokens.attention_mask
        
        for k,v in sample.items():
            if isinstance(v, torch.Tensor):
                sample[k] = sample[k].to(self.device)
        
        
        # clip the input ids to the vocab size
        sample['input_ids'] = torch.clip(sample['input_ids'], 0, self.tokenizer.vocab_size-1)
        if train:
            self.optimizer.zero_grad()
            
        device = self.get_model_device(self.model)
        
        stats = self.copy(self.stats)
        stats['time'] =  self.time()
        sample['input_ids'] = sample['input_ids'].to(device)
        
        good_logits = False
        output = self.model(input_ids=sample['input_ids'].to(device),
                                output_hidden_states=output_hidden_states)
        
        # output = self.process_output(output)
        # check if there are any nans in the logits
        logits_has_nans =  torch.isnan(output.logits).any()
        if logits_has_nans:
            raise Exception('logits has nans with sample input_ids: ', sample['input_ids'])
                
        # sometime we dont care about the begginning of the sequence
        output_length = output_length if output_length else output.logits.size(1)
        
        output['hidden_states'] = output.hidden_states[hidden_state_index]
        output['input_ids'] = sample['input_ids']

        
        # map th elogits
        if map_logits:
            output['logits'] = self.token_translator.translate_logits(logits = output['logits'],
                                                                           offset_mapping=offset_mapping_std,
                                                                           offset_mapping_std=offset_mapping_std,
                                                                           tokens=sample['input_ids'],
                                                                           tokens_std=original_input_ids)
            
        
        output['logits']= output.logits[:,-output_length:,:]
        output['loss'] = loss = self.calculate_loss(**output)
        output['topk']=self.encode_topk(output['logits'], topk=topk)


        output = self.process_outputs(stats=stats, sample=sample, output=output)


        
        return {key:output[key] for key in return_keys}
        
       
    def encode(self, text:str, **kwargs):
        kwargs['return_keys'] = ['hidden_states']
        sample = self.tokenize(text)
        kwargs.update(sample)
        return self.forward(**kwargs)['hidden_states']
    def process_outputs(self, stats:dict, sample:dict, output:dict):

        loss = output['loss']
        
        stats['latency'] = self.round(self.time() - stats['time'], sig=2)
        stats['steps'] = stats.get('steps', 0) + 1
        stats['input_shape'] = list(sample['input_ids'].shape)
        num_samples = stats['input_shape'][0]
        num_tokens = stats['input_shape'][0]*stats['input_shape'][1]
        stats['tokens'] = stats.get('tokens', 0) +  num_samples
        stats['samples'] = stats.get('samples', 0) + num_tokens
        
        if self.training:
            train_stats = stats['train'] = stats.get('train', {})
            loss.backward()
            self.optimizer.step()
            loss = loss.item()
            train_stats['epoch'] = train_stats.get('epoch', 0)
            train_stats['samples'] = train_stats.get('samples', 0) + num_samples
            train_stats['tokens'] = train_stats.get('tokens',0) + num_tokens
            train_stats['steps'] = train_stats.get('steps', 0) + 1
            train_stats['epoch_length'] = self.config.epoch_length
            train_stats['batch_count'] = train_stats.get('batch_count', 0) + 1
            alpha = 1/self.config.epoch_length
            train_stats['epoch_loss'] = (train_stats.get('epoch_loss', loss)*(1-alpha)+ loss*alpha)
            stats['loss'] = loss
            train_stats['best_loss'] = train_stats.get('best_loss', self.config.default_metric)
            train_stats['time'] = stats['time']
            
            for k_r in ['best_loss', 'epoch_loss']:
                train_stats[k_r] = self.round(train_stats[k_r], self.config.loss_sigdigs)
            train_stats['loss_history'] =train_stats.get('loss_history',[])
            
            
            if train_stats['batch_count'] % self.config.epoch_length == 0:
                train_stats['loss_history'] += [self.round(train_stats['epoch_loss'], 3)]
                train_stats['epoch'] = train_stats['epoch'] + 1
                train_stats['batch_count'] = 0
                
                # check if the loss is better than the best loss
            is_better = bool(train_stats['epoch_loss'] <= train_stats['best_loss'])
            train_stats['is_better'] = is_better
            train_stats['saved_step'] = train_stats.get('saved_step', 0)
            
            train_stats['steps_since_saved'] = train_stats['steps'] - train_stats['saved_step']

            if is_better:
                if train_stats['steps_since_saved'] >= self.config.min_steps_since_saved :
                    self.set_stats(stats)
                    self.save() # save all
                    train_stats['saved_step'] = train_stats['steps']
                    
                train_stats['best_loss'] = train_stats['epoch_loss']
                
            else:
                if train_stats['steps_since_saved'] > self.config.min_steps_since_saved:
                    self.load()

            
            self.set_stats(stats)

        else:
            loss = loss.item()
            stats['loss'] = loss
            self.set_stats(stats)
        
        output['stats'] = self.munch2dict(stats)

        return output
    
    def check_config(self, config, ensure_keys=['model_path']):
        for k in ensure_keys:
            assert config[k] == self.config[k], f'{k} in config {config[k]} does not match {k} in model {self.config[k]}'

    
    def set_model(self, config) -> None:
        from transformers import  AutoModelForCausalLM, AutoModel, AutoConfig
        from accelerate import init_empty_weights
        
        self.model_path = config['model_path'] = config['model'] = self.shortcuts.get(config['model'], config['model'])

        self.set_tokenizer(config.tokenizer)

        if config == None:
            config = self.config
        


        model = self.get_empty_model(self.model_path, trust_remote_code=config.trust_remote_code)
        
        
        self.print(model.__dict__['_modules'])
        # assert False
        config.model_size = self.get_model_size(model)
        config.excpeted_model_size = config.model_size*self.config.model_inflation_ratio
      
        free_gpu_memory = self.free_gpu_memory()

              
        if config.max_memory == None:
            config.max_memory = self.max_gpu_memory(memory=config.excpeted_model_size,
                                                max_gpu_ratio=config.max_gpu_ratio,
                                                reserve=config.reserve_gpus)
        
        
            config.max_memory = {k:free_gpu_memory[k] for k,v in config.max_memory.items()}

        if config.device_map == None:
            config.device_map= self.infer_device_map(model, max_memory=config.max_memory)
        
        verbose = config.verbose
        

        if isinstance(config.device_map, dict):
            config.device_map = {k:v for k,v in config.device_map.items() }

        model_kwargs=dict(
            max_memory=config.max_memory,
            device_map= config.device_map,
            trust_remote_code=config.trust_remote_code,
        )
        
        if verbose:
            self.print(f'model_kwargs: {model_kwargs}')
       
        self.model = AutoModelForCausalLM.from_pretrained(config.model_path, **model_kwargs) 
        
        config.devices = list(set(list(self.model.hf_device_map.values())))
        config.device = config.devices[0]

        self.device_map = config.device_map 
        self.devices = config.devices
        self.device = config.device
        
        if config.reserve_gpus:
            self.unreserve_gpus(config.max_memory)
        
        self.print(f'device_map: {self.devices}')
        
        self.set_optimizer(config.optimizer)
        self.set_finetune(config.finetune) 
          
        self.set_tag(config.tag)
        self.set_epoch_length(config.epoch_length)      
          
        if config.load:
            self.load()
        self.set_stats(config.stats)    
        self.config = config


    def set_params(params:dict = None):
        params = params if params is not None else self.params
        
        return params


    def set_epoch_length(self, epoch_length:int) -> int:
        assert isinstance(epoch_length, int)
        self.epoch_length = self.epoch_size = self.config['epoch_length']=  epoch_length
        return self.epoch_length
    set_epoch_size = set_epoch_length


    def resolve_tokenizer(self, tokenizer:str):
        if tokenizer is None:
            tokenizer = self.config.model_path
        tokenizer = self.shortcuts.get(tokenizer, tokenizer)
        assert isinstance(tokenizer, str)
        return tokenizer
    def set_tokenizer(self, tokenizer):
        from transformers import AutoTokenizer, AutoModel
        from commune.utils.tokenizer import prep_tokenizer
        tokenizer = self.resolve_tokenizer(tokenizer)

        self.print(f'setting {tokenizer} tokenizer...')
        assert isinstance(tokenizer, str, )
        self.config['tokenizer'] = tokenizer
        
        # HACK TO INCLUDE LLAMA TOKENIZER
        if 'llama' in tokenizer:
            from transformers import LlamaTokenizer
            tokenizer_class = LlamaTokenizer
        else:
            tokenizer_class = AutoTokenizer
                
        try:
            tokenizer = tokenizer_class.from_pretrained(tokenizer, use_fast=True)
        except ValueError:
            
            print('resorting ot use_fast = False')
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)


        self.tokenizer = tokenizer
        
    
        self.std_tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast= True)
        self.tokenizer = prep_tokenizer(self.std_tokenizer)
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


    @staticmethod
    def decode_topk(  forward_response_tensor: torch.Tensor, topk=4096, vocab_size:int=50257) -> torch.Tensor:
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


    def tokenizer_name(self):
        return self.config['tokenizer']

    def tokenize(self, 
                 text: str = 'Whadup',
                 padding=True, 
                 truncation=True, 
                 max_length=64,
                 return_tensors='pt',
                 add_special_tokens=False,
                 device:str = None, 
                 **kwargs) -> torch.Tensor:
        """ Returns tokenized text as torch tensor. """
        
        sample = self.tokenizer(text, padding=padding, 
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
    def resolve_model(cls, model):
        return cls.shortcuts.get(model, model)
    
    @classmethod
    def learn(cls , *args, **kwargs):
        kwargs['train'] = True
        return cls.evaluate(*args, **kwargs)
    @classmethod
    def evaluate(cls, model = 'gpt125m', 
             topk:int=512 ,
             dataset:str = 'dataset.bittensor',
             num_batches = 1000,
             batch_delay = 0.3,
             sequence_length : int = 256,
             batch_size: int = 8,
             autocast : bool = True,
             train: bool= False,
             map_logits : bool = False,
             map_tokens : bool = False,
             timeout : int= 6,
             remote:bool = False,
             **kwargs
             ):
        
        if remote:
            kwargs = cls.locals2kwargs(locals())
            kwargs['remote'] = False
            return cls.remote_fn(fn='learn',kwargs=kwargs, name=f"train::{model}")
        
        
        if isinstance(model, str):
            if cls.module_exists(model):
                model  = cls.connect(model) 
            else:
                model = cls(model=model, **kwargs)
        
        
        
        def sample_check(sample):
            return bool(isinstance(sample, dict) and 'input_ids' in sample)
        
        datasets = c.connect_pool(dataset)
        
        data_idx = cls.choice(list(range(len(datasets))))
        
        @classmethod
        def resolve_model(cls, model):
            return cls.shortcuts.get(model, model)
        datasets = c.connect_pool(dataset)
        data_idx = 0
        for i in range(num_batches):
            
            try:
                dataset = datasets[data_idx]
                sample = dataset.sample(batch_size=batch_size,
                                        sequence_length=sequence_length)
                if not sample_check(sample):
                    raise Exception('Sample check failed')
            except Exception as e:

                del datasets[data_idx]
                cls.print(f'failed to sample, removing dataset {data_idx}, {len(datasets)} remaining')
                data_idx = cls.choice(list(range(len(datasets))))
                
                continue
        
            c.sleep(batch_delay)
            sample['input_ids'] = sample['input_ids'][:batch_size, :sequence_length]
            sample.update(
                topk=topk,
                map_tokens=map_tokens,
                map_logits=map_logits,
                train=train,
                autocast=autocast,
                timeout=timeout,
                return_keys=[ 'topk', 'stats']
            )
            try:
                sample['timeout'] = timeout
                output = model.forward(**sample)
                cls.print('STATS: ',output.get('stats', output))
            except Exception as e:
                cls.print(model.forward)
                raise e
          
          
    test = evaluate
    
    @classmethod
    def train_fleet(cls, model = 'model.gptj', network='local', **kwargs):
        models = c.modules(model, network=network)
        for m in models:
            cls.print(f"Training {m}")
            cls.learn(model=m, **kwargs)
        
          
    # @classmethod
    # def train_fleet(cls, model = 'model.gptj',
    #                 dataset='dataset.bittensor',
    #                 selection_ratio= 1.0,
    #                 batch_size=8,
    #                 num_batches = 1000,
    #                 sequence_length=256,
    #                 remote:bool = False,
    #                 network='global',
    #                 tag = None,
    #                 **kwargs):
        
    #     kwargs = cls.locals2kwargs(locals())

        
    #     if remote:
    #         kwargs.update(remote=False) # otherwise we get a remote recursion error
    #         return cls.remote_fn(fn='train_fleet',kwargs=kwargs, name=f"train_fleet::{model}", tag=tag)
        
    #     models = c.modules(model, network=network)
    #     datasets = c.connect_pool(dataset)

    #     for i in range(num_batches):
    #         selected_models = cls.random_ratio_selection(models, selection_ratio )
    #         dataset = cls.choice(datasets)
    #         try:
    #             sample = dataset.sample(batch_size=batch_size, sequence_length=sequence_length)
    #             assert isinstance(sample, dict) and 'input_ids' in sample
    #         except Exception as e:
    #             continue
    #         sample['train'] = True
    #         sample['input_ids'] = sample['input_ids'][:batch_size, :sequence_length]
    #         sample['return_keys'] = ['stats']
    #         results = cls.call(selected_models, fn='forward', **sample)
    #         stats = {k:v.get('stats', {}) for k,v in results.items() if isinstance(v, dict)}
    #         print_keys = ['epoch_loss', 'best_loss',  'steps']
    #         print_stats = [{**{_k: v.get('train',{}).get(_k) for _k in print_keys }, 'name': k} for k,v in stats.items()]
    #         print_stats = pd.DataFrame(print_stats)
    #         print_stats = print_stats.sort_values(by=['best_loss'])
    #         cls.print(f'\nRESULTS {i}/{num_batches} \n',print_stats)
    
    # train_fleet = learn_fleet
    

    @classmethod
    def test_encode(cls, text=['encode, hey whadup fam how is it going']*4, num_samples:int=10):
        self = cls()
        t = cls.timer()
        for i in range(num_samples):
            cls.print(self.encode(text).shape)
            cls.print(num_samples/t.seconds, 'samples per second')

    

    @classmethod
    def models(cls):
        return list(cls.shortcuts.keys())
    
    
    
    @classmethod
    def default_models(cls):
        return list(cls.shortcuts.keys())
          
          
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
                     *tags, 
                     model = 'gptj',
                     max_models = 4,
                     **kwargs
                     ) -> List[str]:
        c.update()
        if len(tags) == 0:
        
            tags = ['alice', 'bob', 'chris', 'dan', 'elon', 'frank', 'greg', 'huck' ]
        tag_seperator = kwargs.get('tag_seperator', '::')
        free_gpu_memory = cls.free_gpu_memory()
        models = [ model+tag_seperator+t for t in tags]
        deployed_models = cls.deploy(*models, **kwargs)
        return {'deployed': deployed_models}
        
    @classmethod
    def undeployed_models(cls, models: List[str] = 'all'):
        models = cls.fleet_group.get(models, models)
        undeployed_models = []
        for model in models:
            if cls.module_exists(f'model.{model}') == False:
                undeployed_models.append(model)
        return undeployed_models
       
    
    @classmethod   
    def infer_device_map(cls, model, 
                         max_memory: dict = None,
                         **kwargs,
                         ):
        if max_memory == None:
            max_memory = cls.max_gpu_memory()    
            
        from accelerate import infer_auto_device_map
        if isinstance(model, str):
            model = cls.get_empty_model(model)
        device_map = infer_auto_device_map(model, max_memory=max_memory, **kwargs) 
        
        return device_map
    
    

      
    @classmethod
    def deploy(cls,
               *models: str,
               name: str =None, 
               wait_for_server: bool = False, 
               device = None, 
               namespace = None,
               update:bool = False,
               mode:str = 'pm2',
               refresh = False,
               tag_seperator:str = '::',     
               **kwargs):
        
        
        if update:
            cls.update()
        tag = kwargs.get('tag', None)
        assert len(models) > 0

        
        free_gpu_memory = cls.free_gpu_memory()
        
        config = cls.get_config(kwargs=kwargs)
        
        
        
        deployed_models = {}
        for model in models:
            if tag_seperator in model:
                model, tag = model.split(tag_seperator)
            name = f'model.{model}'
            if tag == None:
                tag =  'base'
            if tag:
                name = name+tag_seperator+str(tag)
                  
            if cls.module_exists(name) and refresh == False:
                cls.print(f'{name} already exists, skipping...', color='red')
                continue
            else:
                cls.print(f'{name} does not exist, deploying...', color='green')
    
            model_size_bytes = cls.get_model_size(model)*config.model_inflation_ratio
            max_gpu_memory = cls.max_gpu_memory(model_size_bytes,
                                                max_gpu_ratio=config.max_gpu_ratio ,
                                                free_gpu_memory=free_gpu_memory,
                                                saturate=True)
    
            cls.print(max_gpu_memory)
            for k,v in max_gpu_memory.items():
                free_gpu_memory[k]-= v
                free_gpu_memory[k] = max(0, free_gpu_memory[k])
            devices = list(max_gpu_memory.keys())
            # cls.print(c.reserved_gpus(), 'fam') 
            config.model = model
            config.tag = tag
            kwargs = {'config': config, 'tag': tag}
            
            cls.launch(name=name,
                       kwargs=kwargs,
                       mode=mode, 
                       device=device, 
                       wait_for_server=wait_for_server,
                       verbose=False)
            
            
            
            deployed_models[name] = {'model': model, 'tag': tag, 'devices': devices, 'max_gpu_memory': max_gpu_memory}
            
        return deployed_models
            
    @classmethod
    def sandbox(cls):
        self = cls(model='opt2.7b')
        
        
    

        
if __name__ == "__main__":
    
    TransformerModel.run()


