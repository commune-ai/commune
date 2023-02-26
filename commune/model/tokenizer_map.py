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

if os.getenv('USE_STREAMLIT') == 'true':
    import streamlit as st
    
    
# import torch
import commune
# commune.utils
from torch import nn
commune.new_event_loop()
import bittensor
from commune.utils.tokenizer import prep_tokenizer, get_translation_map, translate_logits_to_probs_std, \
    translate_special_token_text, pad_offsets, topk_token_phrases, compact_topk_token_phrases
 
"""
Examples 



"""
class TokenizerMap( nn.Module, commune.Module):
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
                tokenizer = 'gptneox',
                std_tokenizer = None,
                dataset = 'BittensorDataset',
                ):
        
        self.dataset = commune.connect(dataset)
        self.tokenizer_1 = self.set_tokenizer(tokenizer_1)
        self.tokenizer_2 = self.set_tokenizer(tokenizer_2)

        nn.Module.__init__(self)
        
        # set model and tokenizer

        self.set_model()

        # set tokenizer to model name (HF only) if tokenizer == None
        self.set_tokenizer(tokenizer=tokenizer if tokenizer != None else self.model_name)
        
        self.set_optimizer(optimizer=optimizer)
        
        self.set_metrics(metrics=metrics)
        
        self.set_stats()
        
        
        if load:
            self.load()
        
        self.set_finetune(**finetune)
        
        
    def set_optimizer(self, optimizer:'torch.optim.Optimizer'=None, *args, **kwargs):
        
        if isinstance(optimizer, dict):
            module_path = optimizer.pop('module', torch.optim.Adam)
            assert module_name != None, f'Please specify a valid optimizer ex: torch.optim.Adam'
            optimizer_class = self.import_object(module_path) 
            optimizer_kwargs = optimizer.get('kwargs', optimizer)
            optimizer_args = optimizer.get('args', [])
            self.optimizeroptimizer_class(*optimizer_args,**optimizer_kwargs)
                
        elif optimizer == None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.00002)
        
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
    

    def forward(self, *args,no_grad=True, autocast:bool=True, **kwargs):
        # import ipdb; ipdb.set_trace()
        if no_grad:
            with torch.no_grad():
                if autocast: 
                    with torch.cuda.amp.autocast():
                        result = self.local_forward(*args,**kwargs)
                else:
                    result = self.local_forward(*args,**kwargs)
        else:
            if autocast:
                with torch.cuda.amp.autocast():
                    result = self.local_forward(*args,**kwargs)
            else:
                result = self.local_forward(*args,**kwargs)
        # import ipdb; ipdb.set_trace()
        return result


    def local_forward(self,  
                input_ids: torch.Tensor = None, 
                topk:int=None, 
                hidden_state_index: int = -1,
                hidden_dim_bounds: List =  None,
                output_hidden_states:bool=False,
                output_logits:bool = True,
                output_length:int = 10,
                token_remap:bool = True,
                logit_remap:bool = True,
                verbose:bool = False,
                **kwargs):

        # if isinstance(input_ids, str) or ((isinstance(input_ids, list) and isinstance(input_ids[0], str))):
        #     input_ids = self.tokenize(input_ids)
        #     token_remap = False
        # transformers.set_seed(0)
        # transformers.enable_full_determinism(0)
        # remap the tokens if token_remap is True
        tokens = {
            'input_ids': input_ids,
        }
        if token_remap:
            tokens = self.token_remap(input_ids, std_tokenizer=self.tokenizer)  # remap to server tokenizer
    
        # if verbose:
        #     print('INPUT_STATISTICS: ',tensor_info_dict(input_dict))
        
        tokens['input_ids'] = tokens['input_ids'].to(self.device)

        model_output = self.model(input_ids=tokens['input_ids'],
                                  output_hidden_states=True)
        
        # sometime we dont care about the begginning of the sequence
        
        output_length = output_length if output_length else model_output.logits.size(1)
        model_output.logits = model_output.logits[:,-output_length:,:]
        
        # remap back to original tokens if token_remap is True
        if logit_remap:
            pre_logits = model_output.logits.to(self.device)
            probs_std = translate_logits_to_probs_std(pre_logits,
                                                        tokens['offset_mapping'], tokens['offset_mapping_std'],
                                                        self.tokenizer, self.std_tokenizer,
                                                        self.split_map_cache,
                                                        self.to_translation_map, 
                                                        self.from_translation_map,
                                                        tokens['input_ids'], input_ids)
            probs_std = probs_std.to(self.device)
            logits_std = torch.log(probs_std + 1e-40)            
            model_output.logits = logits_std
        
        output_dict = {}
        if topk:
            topk_tensor = self.encode_topk(model_output.logits, topk=topk)
            output_dict['topk']=topk_tensor
            
        if output_logits:
            output_dict['logits']=model_output.logits

        if output_hidden_states:
            output_dict['hidden_states'] = model_output.hidden_states[-1]
            hidden_dim = output_dict['hidden_states'].size(-1)
            hidden_dim_bounds = hidden_dim_bounds if hidden_dim_bounds else [0, hidden_dim+1]
            output_dict['hidden_states'] = output_dict['hidden_states'][:, :, hidden_dim_bounds[0]:hidden_dim_bounds[1]]
            


        return output_dict

    def get_loss_fct(self, logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        """
        Calculate loss_fct, CausalLM loss, next-token prediction loss.
            Args:
                logits (:obj:`torch.FloatTensor`, `required`):
                    [batch_size, sequence_len, bittensor.__network_dim__]
                labels (:obj:`torch.LongTensor`, `required`):
                    [batch_size, sequence_len]

            Returns:
                loss (:obj:`torch.FloatTensor`):
                    scalar
        """
        if not hasattr(self, 'loss_fct'):
            self.loss_fct = torch.nn.CrossEntropyLoss()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss


    @property
    def device(self):
        # deepspeed has .module.device to access device
        return self.model.device

    def set_model(self, model_name:str, device:str = None,hidden_dim = 128, **extra_model_kwargs):
        
        from commune.model.layer import Layer
        
        device = device if device else self.device
        
        self.encoder = Layer(in_dim=self.tokenizer.vocab_size, out_dim=hidden_dim, device=device)
        self.decoder = Layer(in_dim=hidden_dim, out_dim=self.std_tokenizer.vocab_size, device=device)
    
    
    
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
        self.std_tokenizer = bittensor.tokenizer()
        
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
        
        

        pred = self.forward(**sample, no_grad=False)
        logits =  pred['logits']
        targets = targets[:,-logits.shape[1]:]
        pred = logits.reshape(-1, logits.size(-1))
        loss = self.calculate_loss(pediction=logits.reshape(-1, logits.size(-1)), 
                                    gt=targets.flatten().to(self.device))              
        
        self.stats['learn_steps'] = self.stats.get('learn_steps', 0)+1
        
        
        loss.backward()
        self.optimizer.step()
    
        
        return loss.item()
    

    def set_stats(self, **stats) -> None: 
        self.stats = {**self.stats, **stats}
        
    def get_stats(self ) -> dict:
        return self.stats

    @property
    def module_tag(self): 
        return self.resolve_module_tag()
    
    def resolve_module_tag(self, tag=None):
        tag = tag if tag else self.tag
        module_tag = self.model_name.replace("/", "_")
        if tag:
            module_tag +=  f'_{tag}'
        return module_tag
    

    def save_pretrained(self, path:str = None, tag:str = None,  *args, **kwargs):
        # Save the model and tokenizer
        path = self.resolve_path(tag=tag)
        self.model.save_pretrained(path, *args, **kwargs)
        self.tokenizer.save_pretrained(path, *args, **kwargs)
        
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
            'stats': self.stats
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
        self.set_stats(**loaded_state['stats'])
        

    def set_finetune(self, num_layers:int=1, layer_name:str = None, all:bool = False) -> Tuple[bool, str]:
        r''' Set to tune only the parameter of the last layer
            Returns: 
                reached_last_layer (:type:`bool`):
                    If we have set partial of the model to requires grad.
                
                last_layer_name (:type:`string`):
                    The name of the last layer that user specified or we found.
                    None if the user did not specify and we couldnt find it. 
        '''
        def find_last_layer(model: torch.nn.Module) -> Optional[str]:    
            r''' Recursively find the last layer in a nn.ModuleList
                Args:
                    model (:obj:`torch.module`):
                        The model (or sub-model) to fine the last layer from. 
                Returns:
                    name (:type:`str`):
                        The name (or sub-name) of the last layer.
                        None if not found
            '''
            reverted_child_list = [(name, child) for name, child in model.named_children()]
            reverted_child_list.reverse()

            for name, child in reverted_child_list:    
                if isinstance(child, nn.ModuleList):
                    if num_layers > len(child):
                        logger.warning(f'Number of finetune layers was set higher then the layers avaliable {len(child)}')
                        return None
                    return (name + '.' +str(len(child) - num_layers))
                
            for name, child in reverted_child_list:    
                name_ = find_last_layer(child)
                if name_ != None:
                    return (name+'.'+ name_)

            return None     

        if layer_name == None:
            last_layer_name = find_last_layer(self.model)
        else:
            last_layer_name = layer_name

        reached_last_layer = False

        # set the non-last layer parameters not to require grads
        if (all) or (last_layer_name == None):
            return False, last_layer_name

        logger.success(f'Set to finetune layer {last_layer_name} and onwards')
        
        for name, param in self.model.named_parameters():
            if last_layer_name in name or reached_last_layer == True:
                param.requires_grad = True
                reached_last_layer = True
            else:
                param.requires_grad = False

        if reached_last_layer == False:
            if all:
                logger.warning('Set to finetune the whole model, this will significantly increase the memory usage.')
            else:
                logger.warning(f'Cannot identify the last layer of the model with name {last_layer_name}, setting to finetune on all of the parameters.')

        return reached_last_layer, last_layer_name


    @classmethod
    def local_train(cls, 
                    model:str='gptj',
                    tag:str = 'demo', 
                    num_batches:int = 10000,
                    window_size:int = 50,
                    backoff_window_size:int = 25,
                    max_iters_since_best:int = 100,
                    dataset:str= 'BittensorDataset',
                    best_loss: float = 10e10,
                    **kwargs
                    ):
        model = cls(model_name=model,tag=tag, load=True,  **kwargs)
        dataset = commune.connect(dataset)
        
        stats = model.get_stats()
        best_loss = stats.get('loss', best_loss)
        if best_loss < 0.1:
            best_loss = 10e10
        
        commune.print(f'Loaded {stats} from {tag}', 'yellow')

        metric_window = commune.get_module('commune.utils.math.MovingWindowAverage')(value=2, window_size=window_size)
        # if epoch > 0:
        #     model.load(tag=tag)
        fail_count = 0
        iters_since_best = 0
        for i in range(num_batches):
            
            if iters_since_best > max_iters_since_best:
                model.load(tag=tag)
            sample = dataset.sample()
            
            if not (isinstance(sample, dict) and 'input_ids' in sample):
                fail_count += 1
                commune.print(f'Failed to get sample {fail_count} times', 'red')
                continue
            
            
            loss = model.learn_step(**sample)
            
            # update the metric_window
            metric_window.update(loss)
            
            window_loss = metric_window.value
            info_str = f'Batch {i}/{num_batches} CE: {loss} Window Loss ({window_size}): {window_loss} Best Loss: {best_loss}'
            commune.print(info_str, 'purple')
            
            if window_loss < best_loss and i > window_size and iters_since_best > backoff_window_size:
                best_loss = window_loss
                model.set_stats(loss=best_loss)
                commune.print(f'Best Stats: {model.get_stats()} ', 'green')
                iters_since_best = 0
                model.save(tag=tag)

                
            else:
                iters_since_best += 1
       
    @classmethod
    def resolve_device(cls, device:str = None) -> str:
        return commune.resolve_device(device=device)

    def generate(self, 
                 text:str = "Today is a beautiful day, and", 
                 max_length:int=20):
    
        '''
        Generate text from a given text.
        '''
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            LogitsProcessorList,
            MinLengthLogitsProcessor,
            TopKLogitsWarper,
            TemperatureLogitsWarper,
            StoppingCriteriaList,
            MaxLengthCriteria,
        )
        import torch

        # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
        self.model.config.pad_token_id = self.model.config.eos_token_id
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids

        # instantiate logits processors
        logits_processor = LogitsProcessorList(
            [
                MinLengthLogitsProcessor(15, eos_token_id=self.model.config.eos_token_id),
            ]
        )
        # instantiate logits processors
        logits_warper = LogitsProcessorList(
            [
                TopKLogitsWarper(50),
                TemperatureLogitsWarper(0.7),
            ]
        )

        stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])

        torch.manual_seed(0)
        with torch.no_grad():
            outputs = self.model.sample(
                input_ids,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
            )
            
        commune.print(f'outputs: {outputs.shape}', 'purple')

        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return output_text
    

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
    def test(cls):
        model = cls(tokenizer='gptneox')
        sample = commune.connect('BittensorDataset').sample()
        print(model.forward(**sample, autocast=False))


if __name__ == "__main__":
    TransformerModel.run()


