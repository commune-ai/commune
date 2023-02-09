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
class TransformerModel( nn.Module, commune.Module):
    shortcuts =  {
        'gptj': 'EleutherAI/gpt-j-6B',
        'gpt2.7b': 'EleutherAI/gpt-neo-2.7B',
        'gpt125m': 'EleutherAI/gpt-neo-125M',
        'gptjt': 'togethercomputer/GPT-JT-6B-v1',
        'gptneox20b': 'EleutherAI/gpt-neox-20b'
         }

    def __init__(self,
                # model_name: str="EleutherAI/gpt-j-6B",
                model_name: str="gptneox20b",
                tokenizer:Union[str, 'tokenizer'] = None,
                optimizer: torch.optim  = None,
                metrics: Dict[str, 'Metric'] = None,
                device='cuda',
                tag = None,
                load = True,
                finetune : dict = dict(num_layers=10),
                **model_kwargs
                ):
        
        
        self.tag = tag 
        
        nn.Module.__init__(self)
        
        # set model and tokenizer
        self.set_model(model_name=model_name,device=device, **model_kwargs)

        # set tokenizer to model name (HF only) if tokenizer == None
        self.set_tokenizer(tokenizer=tokenizer if tokenizer != None else self.model_name)
        
        self.set_optimizer(optimizer=optimizer)
        
        self.set_metrics(metrics=metrics)
        
        self.set_stats()
        
        
        if load:
            self.load()
        
        self.set_fine_tuning_params(**finetune)
        
        
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
    

    def forward(self, *args,no_grad=True, **kwargs):
        # import ipdb; ipdb.set_trace()
        if no_grad:
            with torch.no_grad():
                result = self.local_forward(*args,**kwargs)
        else:
            result = self.local_forward(*args,**kwargs)
        # import ipdb; ipdb.set_trace()
        return result


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

        if verbose:
            print('INPUT_STATISTICS: ',tensor_info_dict(input_dict))

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

        if verbose:
            print('OUTPUT_STATISTICS: ',tensor_info_dict(output_dict))

        return output_dict


    @property
    def device(self):
        # deepspeed has .module.device to access device
        return self.model.device

    def set_model(self, model_name:str, device:str = 'cuda', **extra_model_kwargs):
        from transformers import  AutoModelForCausalLM, AutoModel, AutoConfig


        self.autocast = extra_model_kwargs.get('autocast', False)
        self.model_name = self.shortcuts.get(model_name, model_name)
        # model_config = AutoConfig.from_pretrained(self.model_name)
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, 
                                            **extra_model_kwargs)        
        self.model_config = self.model.config
        print('model_name', self.model_name)
        # self.model = self.model.to(device)
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

    def getattr(self, k):
        return getattr(self,  k)

    @property
    def __config_file__(self):
        return self.__file__.replace('.py', '.yaml')

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
        
        
        

    def set_fine_tuning_params(self) -> Tuple[bool, str]:
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
                    if self.config.finetune.num_layers > len(child):
                        logger.warning(f'Number of finetune layers was set higher then the layers avaliable {len(child)}')
                        return None
                    return (name + '.' +str(len(child) - self.config.finetune.num_layers))
                
            for name, child in reverted_child_list:    
                name_ = find_last_layer(child)
                if name_ != None:
                    return (name+'.'+ name_)

            return None     

        if self.config.finetune.layer_name == None:
            last_layer_name = find_last_layer(self.model)
        else:
            last_layer_name = self.config.neuron.finetune.layer_name

        reached_last_layer = False

        # set the non-last layer parameters not to require grads
        if (self.config.finetune.all) or (last_layer_name == None):
            return False, last_layer_name

        logger.success(f'Set to finetune layer {last_layer_name} and onwards')
        
        for name, param in self.model.named_parameters():
            if last_layer_name in name or reached_last_layer == True:
                param.requires_grad = True
                reached_last_layer = True
            else:
                param.requires_grad = False

        if reached_last_layer == False:
            if self.config.finetune.all:
                logger.warning('Set to finetune the whole model, this will significantly increase the memory usage.')
            else:
                logger.warning(f'Cannot identify the last layer of the model with name {last_layer_name}, setting to finetune on all of the parameters.')

        return reached_last_layer, last_layer_name
 
    def learn(self, num_batches=10, dataset='dataset.huggingface', load:bool=True, save:bool=True, tag:str = None):
        self.tag = tag if tag else self.tag
        # Module.start('dataset.bittensor')
        
        if isinstance(dataset, str):
            dataset =  self.connect(dataset)

        t = commune.timer()
        
        if load:
            self.load()
        
        total_loss = 0 
        
        for i in range(num_batches):
            sample =dataset.forward(fn='sample')
            samples_per_seconds = i/t.seconds
            loss = self.learn_step(sample=sample)
            
            print(f'({i}/{num_batches}) Samples/s: {samples_per_seconds} Loss: {loss}')

            self.stats.loss = ( (self.stats.steps* self.stats.loss ) + loss) / (self.stats.steps + 1)
            self.stats.steps += 1
        if save:
            self.save()
            
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
    

    def save_pretrained(self, path:str, *args, **kwargs):
        # Save the model and tokenizer
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
            'stats': dict(self.stats)
        }
    
        torch.save(state_dict, path)
        
        return path
    
    def load(self):
        path = self.resolve_path(self.module_tag)
        
        if not os.path.exists(path):
            return
        state_dict  = torch.load( path)
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.set_stats(state_dict['stats'])
        


    def set_fine_tuning_params(self, num_layers:int=1, layer_name:str = None, all:bool = False) -> Tuple[bool, str]:
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
    def experiment(cls, trial='trial_2', model_name='EleutherAI/gpt-j-6B', ):
        model = cls( tag=trial, model_name='EleutherAI/gpt-j-6B')
        # print('BROOO')
        # model = model.connect('model.transformer::EleutherAI_gpt-j-6B')
        # print(model.put_json('EleutherAI_gpt-neo-125M_bro', ))
        for i in range(100):
            output = model.learn(num_batches=100, save=True, load=False, dataset='dataset.bittensor')
        print(output)

    @classmethod
    def sandbox(cls ):
        # model = cls(model_name='gpt125m')
        model = cls.connect('model.transformer::gptj')
        dataset = cls.connect('dataset.bittensor')
        sample = dataset(fn='sample') 
        t = commune.timer()
        pred = model(fn='forward', kwargs=dict(autocast=True, no_grad=True, topk=4096, output_logits=False, **sample))
        print(pred['topk'].shape, pred.keys())
        print(t.seconds)
        # print(pred)
        
        
    @classmethod
    def remote_train(cls, 
                    model:str='gptj',
                    trial:str = '2', 
                    num_batches:int = 200,
                    num_epochs:int = 50, 
                    dataset:str= 'dataset.bittensor', **kwargs):
        model = cls.connect(f'model.transformer::{model}:{trial}')
        dataset = cls.connect(dataset)
    
        best_loss = 10e10
        for epoch in range(num_epochs):
            total_epoch_loss = 0
            epoch_loss = 0
            for i in range(num_batches):
                sample = dataset(fn='sample')
                loss = model(fn='learn_step', kwargs=dict(output_length=10, **sample))
                try:
                    total_epoch_loss += loss
                except:
                    continue
                epoch_loss = total_epoch_loss/(i+1)
                info_str = f'Batch {i}/{num_batches} Epoch {epoch}/{num_epochs} CE: {loss} Epoch Loss: {epoch_loss} Best Loss: {best_loss}'
                print(info_str)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                try:
                    model(fn='save', kwargs=dict(tag=trial), timeout=100)
                except TypeError:
                    continue




    @classmethod
    def local_train(cls, 
                    model:str='gpt125m',
                    trial:str = 'demo', 
                    num_batches:int = 200,
                    num_epochs:int = 200, 
                    dataset:str= 'dataset.bittensor', **kwargs):
        model = cls(model_name=model)
        dataset = cls.connect(dataset)
        
        print(model)
        best_loss = 10e10
        for epoch in range(num_epochs):
            total_epoch_loss = 0
            epoch_loss = 0
            for i in range(num_batches):
                sample = dataset(fn='sample')
                loss = model.learn_step(output_length=10, **sample)
                try:
                    total_epoch_loss += loss
                except:
                    continue
                epoch_loss = total_epoch_loss/(i+1)
                info_str = f'Batch {i}/{num_batches} Epoch {epoch}/{num_epochs} CE: {loss} Epoch Loss: {epoch_loss} Best Loss: {best_loss}'
                print(info_str)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                try:
                    model.save(tag=trial)
                except TypeError:
                    continue

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



if __name__ == "__main__":
    # print('FUCK')
    TransformerModel.run()
    # ModelServer().run()
    # TransformerModel.experiment()


