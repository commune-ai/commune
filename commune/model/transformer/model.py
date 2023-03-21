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

from commune.utils.tokenizer import get_translation_map, translate_logits_to_probs_std, \
    translate_special_token_text, pad_offsets, topk_token_phrases, compact_topk_token_phrases, \
        encode_topk, decode_topk
 
"""
Examples 



"""
class TransformerModel( Model):
    shortcuts =  {
        'gptj': 'EleutherAI/gpt-j-6B',
        'gpt2.7b': 'EleutherAI/gpt-neo-2.7B',
         'gpt3b': 'EleutherAI/gpt-neo-2.7B',
        'gpt125m': 'EleutherAI/gpt-neo-125M',
        'gptjt': 'togethercomputer/GPT-JT-6B-v1',
        'gptneox': 'EleutherAI/gpt-neox-20b',
        'gpt20b': 'EleutherAI/gpt-neox-20b',
        'opt13b': 'facebook/opt-13b'

         }

    def __init__(self,
                # model_name: str="EleutherAI/gpt-j-6B",
                model: str="gpt125m",
                tag :str = None,
                tokenizer:Union[str, 'tokenizer'] = None,
                device: str = 'cpu',
                optimizer: dict = {'lr': 0.00001},
                finetune : dict = {'num_layers': 4},
                load: bool = False,
                **kwargs
                ):
        Model.__init__(self, config =locals())
        self.set_params(**self.config)

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

    def local_forward(self,  
                input_ids: torch.Tensor = None, 
                topk:int=4096,
                token_remap:bool = False,
                logit_remap:bool = False,
                output_length:int = 10,
                output_logits:bool = False,
                output_topk:bool = True,
                output_hidden_states:bool=True,
                hidden_state_index: int = -1,
                hidden_dim_bounds: List =  [0, -1],
                verbose:bool = False,
                return_keys:List[str] = None,
                train: bool = False,                
                save: bool=  False,
                load: bool = False,
                return_offsets_mapping: bool = True,
                **kwargs):

        if token_remap:
            sample = self.token_remap(input_ids=input_ids, 
                                      std_tokenizer=self.tokenizer, 
                                      return_offsets_mapping=return_offsets_mapping)  # remap to server tokenizer
    
        else:
            sample = {
            'input_ids': input_ids,
            }
        for k,v in sample.items():
            if isinstance(v, torch.Tensor):
                sample[k] = sample[k].to(self.device)
        
        if train:
            self.optimizer.zero_grad()
            
        model_output = self.model(input_ids=sample['input_ids'],
                                  output_hidden_states=True)
        
    
        print(model_output.logits.device)
    
        # sometime we dont care about the begginning of the sequence
        
        output_length = output_length if output_length else model_output.logits.size(1)
        model_output.logits = model_output.logits[:,-output_length:,:]
        
        output_dict = {}
        
        output_dict['topk']=self.encode_topk(model_output.logits, topk=topk)
        output_dict['logits']=model_output.logits
        output_dict['hidden_states'] = model_output.hidden_states[hidden_state_index]
        output_dict['hidden_states'] = output_dict['hidden_states'][:,-output_length:,:]
        output_dict['hidden_states'] = output_dict['hidden_states'][:, :, hidden_dim_bounds[0]:hidden_dim_bounds[1]]
        hidden_dim = output_dict['hidden_states'].size(-1)
        
        
             
        if train:
            for key in sample:
                if key not in output_dict:
                    output_dict[key] = sample[key]
            
            loss = self.calculate_loss(**output_dict)  
            loss.backward()
            self.optimizer.step()
            
            self.stats['loss'] =loss.item()
            self.stats['learn_steps'] = self.stats.get('learn_steps', 0) + 1
            output_dict['stats'] = deepcopy(self.stats)
        
            
        # remap back to original tokens if token_remap is True
        if logit_remap:
            output_dict['logits'] = self.logit_remap(logits = output_dict['logits'], input_ids=input_ids)

    
        hidden_dim_bounds = hidden_dim_bounds if hidden_dim_bounds else [0, hidden_dim+1]
        
        return_keys = return_keys if return_keys else ['topk']
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
        if model!= None:
            self.set_model(model)
        if tokenizer != None:
            self.set_tokenizer(tokenizer)
        if optimizer!= None:
            self.set_optimizer(optimizer)
        if finetune!= None:
            self.set_finetune(finetune)
        if device!= None:
            self.set_device(device)
        
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
            
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)        
            
            # convert config to config
            model_config = json.loads(self.model.config.to_json_string())         
            self.config['model'] = model_config
            self.config['model']['model_name'] = self.model_name
            self.config['model']['model_path'] = self.model_path
            # yo 
        if state_dict:
            self.model.load_state_dict(state_dict)

        self.set_tokenizer(tokenizer=self.model_path)


    def set_tokenizer(self, tokenizer:Union[str, 'tokenizer', None]):
        tokenizer = tokenizer if tokenizer else self.model_path
        from transformers import AutoTokenizer
        
        if isinstance(tokenizer, str):
            tokenizer = self.shortcuts.get(tokenizer, tokenizer)
            self.config['tokenizer'] = tokenizer

            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast= True)
            except ValueError:
                print('resorting ot use_fast = False')
                tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        
        self.tokenizer = tokenizer
        
    
        self.std_tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast= True)
        from commune.utils.tokenizer import prep_tokenizer
        self.tokenizer = prep_tokenizer(self.tokenizer, self.std_tokenizer)
        
        self.to_translation_map = get_translation_map(self.tokenizer, self.std_tokenizer)
        self.from_translation_map = get_translation_map(self.std_tokenizer, self.tokenizer)
        self.split_map_cache = {}

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


    def save_pretrained(self, path:str = None, tag:str = None,  *args, **kwargs):
        # Save the model and tokenizer
        module_tag = self.resolve_module_tag(tag)
        path = self.resolve_path('pretrained/'+module_tag)
        self.model.save_pretrained(path, *args, **kwargs)
        self.tokenizer.save_pretrained(path, *args, **kwargs)

    def logit_remap(self, logits:torch.Tensor, input_ids:torch.Tensor):
        raise NotImplementedError('Can you give me a sec fam')
        # pre_logits = model_output.logits.to(self.device)
                    
        # probs_std = translate_logits_to_probs_std(pre_logits,
        #                                             tokens['offset_mapping'], tokens['offset_mapping_std'],
        #                                             self.tokenizer, self.std_tokenizer,
        #                                             self.split_map_cache,
        #                                             self.to_translation_map, 
        #                                             self.from_translation_map,
        #                                             tokens['input_ids'], input_ids)
        # logits_std = torch.log(probs_std + 1e-40)            
        
        return logits_std
    def token_remap(self, token_batch, std_tokenizer=None, return_offsets_mapping=False):
        r""" Tokenizer remapping; decodes the message and then remaps the message using a new tokenizer
            Args:
                token_batch ( :obj:`torch.LongTensor`, `required`):
                    token_batch to be retokenized, [batch_size, sequence_len]
                std_tokenizer ( :obj:`transformers.Tokenizer`, `optional`):
                    The standard tokenizer which was used to tokenize the input.
                return_offsets_mapping ( :obj:`bool`, `required`):
                    Return offsets_mapping in tokenization to delineate token segment positions.
        """
        if std_tokenizer is None:
            std_tokenizer = self.std_tokenizer

        text_batch = std_tokenizer.batch_decode(token_batch)  # decode tokens to original text
        result = translate_special_token_text(text_batch, std_tokenizer, self.tokenizer)  # translate special tokens
        to_text_batch, from_offsets_batch, to_offsets_batch, pad_offsets_batch = result

        tokens = self.tokenizer(to_text_batch, padding=True, truncation=True, max_length=token_batch.size(1), return_tensors='pt',
                                add_special_tokens=False).to(self.device)  # assume tokenizer.padding_side = 'left'

        if return_offsets_mapping:  # get offsets_mapping in tokenization to delineate token segment positions
            server_tokens = self.tokenizer(to_text_batch, return_offsets_mapping=True, add_special_tokens=False)
            std_tokens = std_tokenizer(text_batch, return_offsets_mapping=True)  # encode again to get offsets mapping

            # pad offsets so that special token offset widths match for continued correct alignment
            tokens['offset_mapping'] = pad_offsets(server_tokens['offset_mapping'], to_offsets_batch, pad_offsets_batch)
            tokens['offset_mapping_std'] = pad_offsets(std_tokens['offset_mapping'], from_offsets_batch,
                                                       pad_offsets_batch)
        return tokens
    @classmethod
    def test(cls, topk=4096, output_length=20):
        self = cls(model_name='gpt125m', load=True)
        dataset = commune.connect('dataset')
        sample = dataset.sample(batch_size=2)
        sample = self.tokenize(sample['text'])  # assume tokenizer.padding_side = 'left'

        output = self.forward(**sample, train=False)

        print(output)
        # output['logits'] = decode_topk(output['topk'])
        
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
    
    
    @classmethod
    def sandbox(cls):
        datasets = [ 'Gutenberg_PG', 'BookCorpus2', 'HackerNews', 'Books3', 'NIHExPorter', 'OpenSubtitles']

        models = [ 'gptj', 'gpt3b']

        for model in models:
            for i in range(len(datasets)):
                dataset = datasets[i].lower()
                dataset_id = f'dataset:{dataset}'
                
                model_idx = i % 4
                model_id = f'model::{model}::{model_idx}'
                kwargs = dict(model=model_id, dataset=dataset_id, num_batches=300, num_epochs=100, save=True, load=False, refresh=False)
                
                train_id = f'train::{model_id}::{dataset}'.lower()
                cls.pm2_kill(train_id)
                cls.pm2_launch(name = train_id, fn='train_remote', kwargs=kwargs)
if __name__ == "__main__":
    
    TransformerModel.run()

    # TransformerModel.test()


