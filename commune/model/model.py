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
# commune.utils
from torch import nn

"""
Examples 



"""
class  Model( nn.Module, commune.Module):

    def __init__(self,
                # model_name: str="EleutherAI/gpt-j-6B",
                device: str='cuda',
                tag :str = None,
                model: Union[dict,Munch] = None,
                optimizer: torch.optim  = None,
                fine_tune_params : dict = {'num_layers': 4},
                **kwargs
                ):
        
        
        self.tag = tag

        
        print('BROOO')
        self.device = self.set_device(device)
        
        
        self.stats = {'tag': self.tag}
        
        nn.Module.__init__(self)
        
        # set model and tokenizer

        self.set_model(model_name=model_name,device=device, autocast=autocast, **kwargs)

        
        self.set_optimizer(optimizer=optimizer)
        
        self.set_metrics(metrics=metrics)
        
        self.set_stats()
        
        
        
        
        if load:
            if isinstance(load, str):
                self.load(load)
            elif isinstance(load, dict):
                self.load(**load)
            
        self.load(tag=tag)
        self.set_device(device)
        
        self.set_fine_tuning_params(**finetune)
        
        
    def set_optimizer(self, optimizer:'torch.optim.Optimizer'=None, *args, **kwargs):
        import torch
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


    @classmethod
    def calculate_loss(cls, prediction, gt):
        loss_fn = torch.nn.CrossEntropyLoss()
        
        loss = loss_fn(prediction, gt.to(prediction.device))
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
                token_remap:bool = False,
                logit_remap:bool = False,
                verbose:bool = False,
                **kwargs):

        tokens = {
            'input_ids': input_ids,
        }
        if token_remap:
            tokens = self.token_remap(input_ids, std_tokenizer=self.tokenizer)  # remap to server tokenizer

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
            output_dict['hidden_states'] = model_output.hidden_states[-1][:,-output_length:,:]
            hidden_dim = output_dict['hidden_states'].size(-1)
            if isinstance(hidden_dim_bounds, int):
                hidden_dim_bounds = [0, hidden_dim_bounds]
            
            hidden_dim_bounds = hidden_dim_bounds if hidden_dim_bounds else [0, hidden_dim+1]
            output_dict['hidden_states'] = output_dict['hidden_states'][:, :, hidden_dim_bounds[0]:hidden_dim_bounds[1]]
            


        return output_dict


    default_device = 'cpu'
    @property
    def device(self) -> str:
        # deepspeed has .module.device to access device
        if hasattr(self, '_device'):
            self._device = self.default_device
        return self._device
    
    def set_device(self, device, str) -> str:
        self._device = self.resolve_device(device)
        self.to(device)
        return self._device


    def set_model(self, model_name:str, device:str = None, autocast:bool = False, **extra_model_kwargs):
        from transformers import  AutoModelForCausalLM, AutoModel, AutoConfig

        self.model_name = self.shortcuts.get(model_name, model_name)
        # model_config = AutoConfig.from_pretrained(self.model_name)
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, 
                                            **extra_model_kwargs)        
        
        self.model_config = json.loads(self.model.config.to_json_string())
        
        device = self.resolve_device(device=device)

        self.model = self.model.to(device)
        
        
        self.autocast = autocast
        if self.autocast:
            self.model = self.model.half()
            
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
        
        
        sample['topk'] = 4096
        pred = self.forward(**sample, no_grad=False)
        pred['logits'] = decode_topk(pred['topk'], vocab_size=self.model.config.vocab_size)
        logits =  pred['logits']
        targets = targets[:,-logits.shape[1]:]
        pred = logits.reshape(-1, logits.size(-1))
        loss = self.calculate_loss(prediction=logits.reshape(-1, logits.size(-1)), 
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
        module_tag = self.resolve_module_tag(tag)
        path = self.resolve_path('pretrained/'+module_tag)
        self.model.save_pretrained(path, *args, **kwargs)
        self.tokenizer.save_pretrained(path, *args, **kwargs)
        
    def save(self, tag:str = None, trainable_only:bool = True):
        path = self.resolve_path(tag)
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
        """
        
        """
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
        


if __name__ == "__main__":
    
    TransformerModel.run()


