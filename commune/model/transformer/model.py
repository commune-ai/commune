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
        'gpt125m': 'EleutherAI/gpt-neo-125M',
        'gptjt': 'togethercomputer/GPT-JT-6B-v1',
        'gptneox': 'EleutherAI/gpt-neox-20b',
        'gpt20b': 'EleutherAI/gpt-neox-20b',
        'opt13b': 'facebook/opt-13b'

         }

    def __init__(self,
                # model_name: str="EleutherAI/gpt-j-6B",
                model_name: str="gpt125m",
                tag :str = None,
                topk: int = 4096,
                tokenizer:Union[str, 'tokenizer'] = None,
                device: str = 'cuda',
                optimizer: dict = {'lr': 0.0001},
                load: bool = False,
                finetune : dict = None,
                **kwargs
                ):
        
        
        
        Model.__init__(self, **kwargs)
        
        self.tag = tag if tag else model_name
        self.topk = topk
        
        # set model and tokenizer

        self.set_model(model_name=model_name,device=device,  **kwargs)

        # set tokenizer to model name (HF only) if tokenizer == None
        self.set_tokenizer(tokenizer=tokenizer if tokenizer != None else self.model_name)
        self.set_optimizer(optimizer)        
        
        if load:
            self.load(self.tag)
        

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
                output_logits:bool = True,
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
        
        if train:
            self.optimizer.zero_grad()
            
            
        model_output = self.model(input_ids=sample['input_ids'].to(self.device),
                                  output_hidden_states=True)
        
    
    
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
                if key not in model_output:
                    model_output[key] = sample[key]
            
            loss = self.calculate_loss(**model_output)   
            self.set_metric('loss', loss.item(), metric='metric')
            self.set_metric('learn_steps', metric='counter')
            
            loss.backward()
            self.optimizer.step()
            model_output['stats'] = deepcopy(self.stats)
            model_output['stats']['metrics'] = self.get_metrics()
        
            
        # remap back to original tokens if token_remap is True
        if logit_remap:
            output_dict['logits'] = self.logit_remap(logits = output_dict['logits'], input_ids=input_ids)

    
        if isinstance(hidden_dim_bounds, int):
            hidden_dim_bounds = [0, hidden_dim_bounds]
        hidden_dim_bounds = hidden_dim_bounds if hidden_dim_bounds else [0, hidden_dim+1]
        
        return_keys = return_keys if return_keys else []
        if output_logits:
            return_keys.append('logits')
        if output_topk:
            return_keys.append('topk')
        if output_hidden_states:
            return_keys.append('hidden_states')
        return {key:output_dict for key in return_keys}


    def set_model(self, model_name:str, device:str = None, finetune: dict = None,  **extra_model_kwargs):
        from transformers import  AutoModelForCausalLM, AutoModel, AutoConfig

        self.model_name = self.shortcuts.get(model_name, model_name)
        # config = AutoConfig.from_pretrained(self.model_name)
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, 
                                            **extra_model_kwargs)        
        
        # convert config to config
        self.config = json.loads(self.model.config.to_json_string())
        
        self.set_device(device=device)
        
        if finetune:
            self.set_fine_tuning_params(**finetune)
            
        return self.model



    def set_tokenizer(self, tokenizer:Union[str, 'tokenizer', None]):
        from transformers import AutoTokenizer
        if isinstance(tokenizer, str):
            tokenizer = self.shortcuts.get(tokenizer, tokenizer)
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast= True)
            except ValueError:
                print('resorting ot use_fast = False')
                tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        self.tokenizer = tokenizer
        
        
        try:
            import bittensor
        except RuntimeError:
            commune.new_event_loop()
            import bittensor
        self.std_tokenizer = bittensor.tokenizer()
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



    def tokenize(self, text: str = 'Whadup',device:str = None, **kwargs) -> torch.Tensor:
        """ Returns tokenized text as torch tensor. """
        
        device = device if device != None else self.device
        tokenizer_output = self.tokenizer(text, **kwargs)
        
        return tokenizer_output.input_ids.to(device)


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
        model = cls(model_name='gpt125m', load=True)
        sample = commune.connect('dataset::bittensor').sample()
        output = model.forward(**sample, train=True)

        # output['logits'] = decode_topk(output['topk'])
        
        # print(cls.calculate_loss(output['logits'].reshape(-1, output['logits'].shape[-1]), targets[:, -output_length:].flatten()))
        

    @classmethod
    def train(cls,
              model:str='gpt125m', 
              dataset : Union[str, 'Module'] = 'dataset::bittensor',
             output_length:int=10,
             sequence_length:int=256,
             adapter: dict = None,
             num_batches: int = 100, 
             tag:str=None,
             refresh: bool = False):
        if refresh:
            load = False
        
        model = cls()
        if isinstance(dataset, str):
            dataset = commune.connect(dataset)

        for i in range(num_batches):
            sample = dataset.sample(sequence_length=sequence_length)
            sample['output_length'] =  output_length
            sample['return_keys'] = ['stats']
            sample['train'] = True
            output = model.forward(**sample)
            
        return output['stats']
    
    

if __name__ == "__main__":
    
    TransformerModel.run()
    # TransformerModel.test()


