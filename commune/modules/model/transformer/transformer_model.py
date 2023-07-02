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


from torch import nn
Model = c.module('model')
class TransformerModel(Model):
    default_config = c.config('model.transformer')
    shortcuts = default_config.shortcuts
    model_options = list(shortcuts.keys()) + list(shortcuts.values())
    default_tag = default_config.tag
    
    def __init__(self,
                 config = None,
                 **kwargs
                ):
        config = self.set_config(config=config, kwargs=kwargs)
        self.set_model(config)
        

        


    
    @classmethod
    def test_generate(cls, *args, **kwargs):
        model = cls( *args, **kwargs)
        output_text = model.generate(text='Hello world',)
        return output_text

    @staticmethod
    def check_output( output):
        assert hasattr(output, 'logits'), 'output does not have logits'
        
        # check if logits has nans
        logits_has_nans =  torch.isnan(output.logits).any()
        assert not logits_has_nans, 'logits has nans'
            
        return logits_has_nans
    
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
        
        if isinstance(input_ids, str) or isinstance(input_ids, list):
            input_ids = self.tokenize(input_ids)['input_ids']
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
        

        # move to device for all tensors
        for k,v in sample.items():
            if isinstance(v, torch.Tensor):
                sample[k] = sample[k].to(self.device)
        
        
        # clip the input ids to the vocab size to avoid index errors
        sample['input_ids'] = torch.clip(sample['input_ids'], 0, self.tokenizer.vocab_size-1)
        
        
            
        stats = self.stats
        stats['sample'] =  {**stats.get('sample', {}), **{'time': self.time()}}
        
        # forward pass
        output = self.model(input_ids=sample['input_ids'].to(self.device),
                                output_hidden_states=output_hidden_states)
        
        
        # check if there are any nans in the logits
        self.check_output(output)
        # sometime we dont care about the begginning of the sequence
        output_length = output_length if output_length else output.logits.size(1)
        output['hidden_states'] = output.hidden_states[hidden_state_index]
        output['input_ids'] = sample['input_ids']

        
        output['loss'] = loss = self.calculate_loss(**output)
        output['logits']= output.logits[:,-output_length:,:]
        output['topk']=self.encode_topk(output['logits'], topk=topk)


        output = self.process_outputs(stats=stats, sample=sample, output=output)


        
        return {key:output[key] for key in return_keys}
        
    
    def encode(self, text:str, token_idx:int = None, **kwargs) -> torch.Tensor:
        kwargs['return_keys'] = ['hidden_states']
        sample = self.tokenize(text)
        kwargs.update(sample)
        hidden_states = self.forward(**kwargs)['hidden_states']
        if isinstance(token_idx, int):
            return hidden_states[:,token_idx, :]
        else:
            return hidden_states
    
    embed = encode
    def process_outputs(self, stats:dict, sample:dict, output:dict):
        
        loss = output['loss']
        # training stats
        
        
        
        sample_stats = {
            'input_shape':list(sample['input_ids'].shape),
            'latency': self.round(self.time() - stats['sample']['time'], sig=2),
            'timestamp': int(self.time()),
            'loss': self.round(loss.clone().item(), 3),
            # 'history': stats['sample'].get('history',[])
            
        }
        stats['sample'] = sample_stats

                
        for mode in ['train', 'eval']:
            mode_stats = stats[mode] = stats.get(mode, {})
            
            
            # skip if we are not training and we are in training mode
            if mode == 'train' and not self.training:
                continue
            

            # update the stats for the mode
            mode_stats['samples'] = mode_stats.get('samples', 0) + stats['sample']['input_shape'][0]
            mode_stats['tokens'] = mode_stats.get('tokens',0) + stats['sample']['input_shape'][0]*stats['sample']['input_shape'][1]
            mode_stats['steps'] = mode_stats.get('steps', 0) + 1
            mode_stats['epoch_length'] = self.config.epoch_length
            mode_stats['epoch'] = mode_stats.get('epoch', 0)
            
            # calculate the running average of the loss
            sample_loss = loss.item()
            mode_stats['batch_count'] = mode_stats['steps'] % self.config.epoch_length
            mode_stats['epoch_loss'] = mode_stats.get('epoch_loss', sample_loss)
            mode_stats['epoch_loss'] = (mode_stats['epoch_loss']*mode_stats['batch_count']+ sample_loss)/(mode_stats['batch_count']+1)
            mode_stats['epoch_loss'] = self.round(mode_stats['epoch_loss'], self.config.loss_sigdigs)
            mode_stats['epoch_loss_history'] =mode_stats.get('epoch_loss_history',[])
            

            # update the loss history
            if mode_stats['steps'] % self.config.epoch_length == 0:
                mode_stats['epoch_loss'] = loss.item()
                mode_stats['epoch_loss_history'] += [self.round(mode_stats['epoch_loss'], 3)]
                mode_stats['epoch'] = mode_stats['epoch'] + 1
                mode_stats['batch_count'] = 0
                    
            
            # update the stats for the mode
            if mode == 'train':
                mode_stats['saved_step'] = mode_stats.get('saved_step', 0)
                mode_stats['best_epoch_loss'] = mode_stats.get('best_epoch_loss', self.config.default_metric)
                mode_stats['checkpoint_step'] = mode_stats.get('checkpoint_step', 0)
                loss.backward()
                if mode_stats['steps'] % self.config.accumulate_grad_batches == 0:
                    # we want to accumulate the gradients over multiple batches, and then take an optimizer step while clipping the gradients
                    if self.config.get('clip_grad_norm', 0)> 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                mode_stats['steps_since_checkpoint'] = mode_stats['steps'] - mode_stats['checkpoint_step']
                mode_stats['patience_steps'] = self.config['patience_steps'] = self.config.get('patience_steps', self.config.epoch_length*self.config.patience_epochs)
                if mode_stats['steps_since_checkpoint'] >= self.config.patience_steps :
                    is_better = mode_stats['is_better'] = bool(mode_stats['epoch_loss'] <= (mode_stats['best_epoch_loss'] - self.config.best_epoch_loss_delta))
                    if  is_better:
                        mode_stats['checkpoint_step'] = mode_stats['saved_step']=  mode_stats['steps']
                        mode_stats['best_epoch_loss'] = mode_stats['epoch_loss']
                        self.stats = stats
                        self.save() # save all
                    else:
                        mode_stats['checkpoint_step'] = mode_stats['steps']
                        mode_stats['epoch_loss'] = loss.item()
                        self.load(keys=['model', 'optimizer'])
                        
            stats[mode] = mode_stats



            
        self.stats = stats
    
        output['stats'] = self.munch2dict(stats)

        return output
    
    def check_config(self, config, ensure_keys=['model_path']):
        for k in ensure_keys:
            assert config[k] == self.config[k], f'{k} in config {config[k]} does not match {k} in model {self.config[k]}'


    def set_model(self, config) -> None: 
        from transformers import  AutoModelForCausalLM, AutoModel
        from accelerate import init_empty_weights
        
        # init pytorch module state
        self.init_nn()
        
        # if we are using a shortcut, we need to set the model path
        config['model'] = self.shortcuts.get(config.model, config.model)
        config.tokenizer = config.tokenizer if config.tokenizer else config.model
        self.set_tokenizer(config.tokenizer)

        config.block_prefix = config.model2block_prefix.get(config.model, config.block_prefix)
        if config.device_map == None:
            config.device_map= c.infer_device_map(config.model,
                                                buffer_memory=config.buffer_memory,
                                                block_prefix=config.block_prefix)
            
        c.print('device map', config.device_map)
            
        assert config.device_map is not None, f'could not infer device map for {config.model}'
        

        model_kwargs=dict(

        )
        
        c.print('loading model', config.model)
        self.model = AutoModelForCausalLM.from_pretrained(config.model,
                                                            device_map= config.device_map,
                                                            trust_remote_code=config.trust_remote_code,) 
                                                        

        self.devices = config.devices = list(set(list(self.model.hf_device_map.values())))
        self.device = config.device = config.devices[0]
        
        
        self.set_optimizer(config.optimizer)
        self.set_finetune(config.finetune) 
        
        if config.load:
            self.load(keys=['model', 'optimizer']) 
            
        self.config = config
        
        


    def set_params(params:dict = None):
        params = params if params is not None else {}
        if params.get('lr', None) is not None:
            self.set_lr(lr)
            
        if params.get('optimizer', None) is not None:
            self.set_optimizer(params['optimizer'])
            
        if params.get('clip_grad_norm', None) is not None:
            self.set_optimizer(params['optimizer'])
            
        if params.get('finetune', None) is not None:
            self.set_finetune(params['finetune'])
        
        
        
        
        return params




    def resolve_tokenizer(self, tokenizer:str):
        if tokenizer is None:
            tokenizer = self.config.model
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
                
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
        except ValueError:
            
            print('resorting ot use_fast = False')
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)

        if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
            assert hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
            
        c.print('tokenizer', tokenizer.pad_token, tokenizer.eos_token)
        self.tokenizer = tokenizer
                
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
    def sample_check(cls, sample):
        return bool(isinstance(sample, dict) and 'input_ids' in sample)
    
    @classmethod
    async def async_get_sample(cls, dataset, max_trials=10, batch_size=1, sequence_length=64, num_batches=10):
        sample = None
        if not hasattr(cls, 'dataset_pool'):
            cls.dataset_pool = c.connect_pool(dataset)

        fail_count = 0
    
        while not cls.sample_check(sample) and fail_count < max_trials:
            if len(cls.dataset_pool) == 0:
                cls.dataset_pool = c.connect_pool(dataset)
            try:
                data_idx =cls.choice(list(range(len(cls.dataset_pool))))
                sample = cls.dataset_pool[data_idx].sample(batch_size=batch_size,
                                        sequence_length=sequence_length)
                
                if not cls.sample_check(sample):
                    raise Exception('Sample check failed')
                sample['input_ids'] = sample['input_ids'][:batch_size, -sequence_length:]
                
                
            except Exception as e:
                fail_count += 1
                del cls.dataset_pool[data_idx]
                cls.print(f'ERROR {e} failed to sample, removing dataset {data_idx}, {len(cls.dataset_pool)} remaining', color='red')
        assert cls.sample_check(sample), f'Failed to sample from {dataset} after {max_trials} trials.'
        return sample
    
    @classmethod
    def get_sample(cls, timeout=2, retries = 3, *args, **kwargs):
        try:
            if timeout:
                # Add timeout to the async_get_sample call
                coro = asyncio.wait_for(cls.async_get_sample(*args, **kwargs), timeout=timeout)
            else:
                coro = cls.async_get_sample(*args, **kwargs)
            
            return asyncio.run(coro)
        except asyncio.TimeoutError:
            # Handle the timeout error here
            print("Async function call timed out.")
            if retries > 0:
                return cls.get_sample(timeout=timeout, retries=retries-1, *args, **kwargs)
    
    
    
    @classmethod
    def resolve_model(cls, model, **kwargs):      
        if isinstance(model, str):
            if cls.exists(model):
                model  = cls.connect(model) 
            else:
                model = cls(model=model, **kwargs)
        elif isinstance(model, nn.Module):
            model = model
        elif isinstance(model, dict):
            model = cls(**model)
        elif model == None:
            model = cls()
        else:
            raise ValueError(f"Model type {type(model)} not supported.")
        
        
        return model
                

    @classmethod
    def learn(cls, model = 'gpt125m', 
            topk:int=512 ,
            dataset:str = 'dataset.bittensor',
            num_batches = 1000,
            batch_delay = 3,
            sequence_length : int = 256,
            batch_size: int = 32,
            autocast : bool = True,
            train: bool= True,
            map_logits : bool = False,
            map_tokens : bool = False,
            timeout : int= 8,
            remote:bool = False,
            **kwargs
            ):
        
        if remote:
            kwargs = cls.locals2kwargs(locals())
            kwargs['remote'] = False
            return cls.remote_fn(fn='learn',kwargs=kwargs, name=f"train::{model}")
        
        model = cls.resolve_model(model, **kwargs)

        for i in range(num_batches):
            cls.print('GETTING SAMPLE')
            sample = cls.get_sample(dataset=dataset,
                                                    batch_size=batch_size, 
                                                    sequence_length=sequence_length,
                                                    num_batches=num_batches)
        
        
            sample.update(
                topk=topk,
                map_tokens=map_tokens,
                map_logits=map_logits,
                train=train,
                autocast=autocast,
                timeout=timeout,
                return_keys=[ 'topk', 'stats']
                
            )
            
            cls.sleep(batch_delay)
            
            output = model.forward(**sample)
            cls.print('STATS: ',output.get('stats', output))

        
        
    test = evaluate = learn
    
    @property
    def tag(self):
        if self.config.get('tag', None) == None:
            self.config['tag'] = 'base'
            
        return  self.config['tag']
    
    @tag.setter
    def tag(self, tag):
        self.config['tag'] = tag
        
        
    def resolve_state_path(self, tag=None):
        tag = tag if tag != None else self.tag
        path = self.config.model+'_'+tag
        path = self.resolve_path(path)
        return path
    
        
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
    
    
    @staticmethod
    def get_configurations(params:dict)-> List[dict]:
        import itertools
        configurations = []
        keys = params.keys()
        values = [params[key] for key in keys]
        for combination in itertools.product(*values):
            config = {}
            for i, key in enumerate(keys):
                config[key] = combination[i]
            configurations.append(config)
        return configurations

    @classmethod
    def hyperfleet(cls, 
                    *tags, 
                    model = 'gptj',
                    params = {
                        'optimizer.lr': [1e-4, 1e-5],
                        'finetune': [1,2,3,4],
                    }, 
                    **kwargs
                    ) -> List[str]:
        params_configurations = cls.get_configurations(params)
        deployed_models = []
        free_gpu_memory  = cls.free_gpu_memory()
        for params in params_configurations:
            
            kwargs.update(params)
            tag = '_'.join([f'{k}:{v}' for k,v in params.items()])
            deployed_models += cls.deploy(model, tag=tag, free_gpu_memory=free_gpu_memory, **kwargs)
        
        return {'deployed': deployed_models}
    @classmethod
    def deploy_fleet(cls, 
                    *tags, 
                    model = 'gptj',
                    max_models = None,
                    **kwargs
                    ) -> List[str]:

        c.update()
        if len(tags) == 0:
        
            tags = ['base','alice', 'bob', 'chris', 'dan', 'eve', 'frank', 'gina', 'harry', 'ian', 'jane', 'kate', 'larry', 'mike', 'nancy', 'olivia', 'peter', 'quinn', 'rob', 'sarah', 'tom', 'ursula', 'victor', 'wanda', 'xavier', 'yolanda', 'zach']
            
        
        if max_models is None:
            max_models = len(cls.gpus())
        tags = tags[:max_models]
        
        print(f'deploying {tags} on {cls.gpus()}')
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
            if cls.exists(f'model.{model}') == False:
                undeployed_models.append(model)
        return undeployed_models
    

    
    @classmethod
    def serve(cls,
            model: str,
            tag = None,
            refresh = True,    
            **kwargs
            ):
        
        config = cls.get_config(kwargs=kwargs)
        config.tag = tag
        config.model = model
        c.print(config)
        c.serve(module=cls.module_path(),
                name= f'model.{model}',
                tag = tag,
                kwargs={'config': config},
                refresh = refresh,
                verbose=True, **kwargs)
        
            
    @classmethod
    def sandbox(cls):
        self = cls(model='gpt125m')
        
    @classmethod
    def calculate_loss( cls, logits: torch.Tensor,
                    input_ids:torch.Tensor,
                    return_value = False,
                    **kwargs) -> torch.Tensor:
        '''
        Calculate the loss for the model.
        '''
        gt = input_ids[:, -(logits.shape[1]-1):].flatten()
        pred = logits[:, :-1]
            
        if len(pred.shape) == 3:
            pred = pred.reshape(-1, pred.shape[-1])
        
        assert gt.shape[0] == pred.shape[0], f'gt.shape: {gt.shape} pred.shape: {pred.shape}'

        loss_fn = torch.nn.CrossEntropyLoss()
        loss =  loss_fn(pred, gt.to(pred.device))
        
        # check if loss is nan
        if torch.isnan(loss):
            c.print('Loss is nan, skipping backward pass')
            train = False
            loss = torch.tensor(10)
            raise Exception('Loss is nan, skipping backward pass')
        
        if return_value:
            loss = loss.item()
        
        return loss

    hf = c.module('huggingface')()


    @classmethod
    def sand(cls):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import transformers
        import torch

        model = "tiiuae/falcon-40b-instruct"

        tokenizer = AutoTokenizer.from_pretrained(model)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        sequences = pipeline(
        "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
            max_length=200,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
        for seq in sequences:
            print(f"Result: {seq['generated_text']}")

    def generate(self, text: str, max_length: int = 20, max_new_tokens: int = None,
                min_length: int = 0, min_new_tokens: int = None,
                early_stopping: bool or str = True, max_time: float = None, **kwargs) -> List[str]:
        input_ids = self.tokenize(text)['input_ids']
        output_ids = self.model.generate(input_ids, 
                                        max_length=max_length, 
                                        max_new_tokens=max_new_tokens,
                                        min_length=min_length, 
                                        min_new_tokens=min_new_tokens,
                                        early_stopping=early_stopping,
                                        max_time=max_time, **kwargs)
        output_text = self.detokenize(output_ids, skip_special_tokens=True)
        return output_text
    
    @classmethod
    def infer_device_map(cls, model, 
                         device_map: str = 'auto',
                         max_memory = None,
                         
                         ) -> str:
        from accelerate import infer_auto_device_map
        model_size = c.get_model_size(model)
        max_memory = c.get_max_memory(max_memory, buffer_memory='10gb')

        device_map = infer_auto_device_map(my_model, max_memory={0: "10GiB", 1: "10GiB", "cpu": "30GiB"})

TransformerModel.run(__name__)