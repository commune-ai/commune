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
import json
from munch import Munch
import argparse
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

# import torch
import commune
# commune.utils
from torch import nn
    
"""
Examples 



"""
class OPT( nn.Module, commune.Module):
    shortcuts =  {
        'gptj': 'EleutherAI/gpt-j-6b',
        'gpt2.7b': 'EleutherAI/gpt-neo-2.7B',
        'gpt125m': 'EleutherAI/gpt-neo-125M',
        'gptjt': 'togethercomputer/GPT-JT-6B-v1',
        'gpt20b': 'EleutherAI/gpt-neox-20b',
        'opt': 'facebook/opt-13b',
         } 
    def __init__(self,
                # model_name: str="EleutherAI/gpt-j-6b",
                model_name: str='opt',
                checkpoint_path: str = None,
                max_memory: Union[Dict[int, str], int, float] = 100,
                no_split_module_classes=None,
                tokenizer:Union[str, 'tokenizer'] = None,
                optimizer: torch.optim  = None,
                metrics: Dict[str, 'Metric'] = None,
                override_device_map = None,
                device='cpu',
                tag = None,
                finetune : dict = dict(num_layers=10),
                **model_kwargs
                ):
        
        

        
        nn.Module.__init__(self)
        
        
        
        device = commune.resolve_device(device)
        
        # set model and tokenizer
        

        self.model_name = self.shortcuts.get(model_name, model_name)
        self.override_device_map = override_device_map if override_device_map else {}
        self.no_split_module_classes = no_split_module_classes if no_split_module_classes else []
        if self.model_name == 'EleutherAI/gpt-neox-20b':
            self.no_split_module_classes =  ["GPTNeoXLayer"]
            self.override_device_map = {'gpt_neox.embed_in': device}
            
        self.model_config = AutoConfig.from_pretrained(self.model_name)
        self.model_config.use_cache = False
        self.tag = tag
        self.model_device = device

        self.checkpoint_path = checkpoint_path if checkpoint_path else self.default_checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            commune.print(f'Creating weights path at {self.checkpoint_path}', 'purple')
            AutoModelForCausalLM.from_config(self.model_config).save_pretrained(self.checkpoint_path)
            
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(self.model_config)

            
        self.max_memory = self.resolve_max_memory(max_memory)
        
        commune.print(self.max_memory, 'green')
        
        self.device_map = infer_auto_device_map(
            self.model, 
            no_split_module_classes= self.no_split_module_classes,
            dtype=torch.bfloat16, #note: succeeds with float16 as well.
            max_memory = self.max_memory,
            )    
                
        self.device_map.update(self.override_device_map)
        
        commune.print(self.device_map, 'green')

        load_checkpoint_and_dispatch(
            self.model,
            self.checkpoint_path,
            device_map=self.device_map,
            offload_folder=None,
            offload_state_dict=False,
            dtype="bfloat16"
        )
            
        self.set_metrics(metrics=metrics)
        # convert model to bfloat16
        self.model_config = commune.dict2munch(json.loads(self.model_config.to_json_string()))

        # self.tokenizer = self.set_tokenizer(tokenizer if tokenizer else self.model_name)


        
    @property
    def default_checkpoint_path(self):
        return f"{self.tmp_dir()}/checkpoints/{self.module_tag}"

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
                attention_mask: torch.Tensor= None, 
                topk:int=None, 
                output_hidden_states:bool=False, 
                output_logits:bool = False,
                verbose:bool = False,
                output_length:int = 10,
                max_length: int = 256,
                device = None,
                **kwargs):

        # tokenizer the text if text is provided 
            
        # if input_ids is not provided, tokenize the text


        device = device if device else self.device
        
    
        if not isinstance(input_ids, torch.Tensor):
            if isinstance(input_ids, str):
                input_ids = [input_ids]
            assert isinstance(input_ids, list) and isinstance(input_ids[0], str)
            input_ids = self.tokenize(input_ids,device=device, max_length=max_length)
 

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
                input_dict[k] = input_dict[k][:,-max_length:].to(device)
            
            commune.print('TOKENIZE_' + k, 'purple')

        # if verbose:
        #     print('INPUT_STATISTICS: ',tensor_info_dict(input_dict))

        
        model_output = self.model(**input_dict)
        
        output_length = output_length if output_length else model_output.logits.size(1)
        model_output.logits = model_output.logits.to(device)

        output_dict = {}
        if topk:
            topk_tensor = self.encode_topk(model_output.logits[:,-output_length:,:], topk=topk)
            output_dict['topk']=topk_tensor.to(device)
            

        if output_logits:
            output_dict['logits']=model_output.logits[:,-output_length:,:]

        if output_hidden_states:
            output_dict['hidden_states'] = model_output.hidden_states[-1][:,-output_length:, :]

        # if verbose:
        #     print('OUTPUT_STATISTICS: ',tensor_info_dict(output_dict))

        for k,v in output_dict.items():
            output_dict[k] = v.to(device)


        return output_dict


    @property
    def device(self):
        # deepspeed has .module.device to access device
        # if str(model_device) == 'meta':
        #     model_device = 'cuda'
        return self.model_device


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
        
        tokenizer.padding_side = "left"
        # self.prep_tokenizer(tokenizer=self.tokenizer)

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

    def path2shortcut(self, path:str):
        return {v:k for k,v in self.shortcuts.items()}.get(path, path)
    @property
    def __config_file__(self):
        return self.__file__.replace('.py', '.yaml')

    def tokenize(self, text: str = 'Whadup',
                 input_ids_only:bool = True,
                 max_length=128, 
                 padding='max_length', 
                 truncation=True,
                 device: str=None) -> torch.Tensor:
        """ Returns tokenized text as torch tensor. """
        

        
        
        device = device if device != None else self.device
        tokenizer_output = self.tokenizer(text, 
                                          max_length=max_length, 
                                          padding=padding, 
                                          truncation=truncation,
                                          return_tensors="pt")
        
        
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
        
        
        

    def set_finetune(self) -> Tuple[bool, str]:
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

    @property
    def module_tag(self): 
        return self.resolve_module_tag()
    
    def resolve_module_tag(self, tag=None):
        tag = tag if tag else self.tag
        module_tag = self.path2shortcut(self.model_name).replace("/", "::")
        if tag:
            module_tag +=  f'::{tag}'
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
    def experiment(cls, trial='trial_2', model_name='EleutherAI/gpt-j-6b', ):
        model = cls( tag=trial, model_name='EleutherAI/gpt-j-6b')
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

    # def serve
    @classmethod
    def test(cls):
        self = cls()
        t=commune.timer()
        input_ids = self.tokenize(['broooo whadup']*32, device='cuda')
        print(t.seconds, input_ids.shape)
        
        
    @classmethod
    def resolve_max_memory(cls, max_memory: Union[Dict[int, str], int], buffer_memory:int=10) -> Dict[int, str]:
        
        if isinstance(max_memory, int):
            max_memory = cls.infer_max_memory(total_memory=max_memory, buffer_memory=buffer_memory)
        elif isinstance(max_memory, dict):
            max_memory = {int(k):v for k,v in max_memory.items()}
        else:
            raise ValueError(f'max_memory must be an int or dict, got {type(max_memory)}')
        
        max_memory = {int(k):v for k,v in max_memory.items()}

        gpu_ids = commune.gpus()
        for k,v in max_memory.items():
            assert isinstance(k, int), f'gpu_id must be an int, got {k}'
            assert isinstance(v, str), f'max_memory must be a string, got {v}'
            assert k in gpu_ids, f'gpu_id {k} not found in {gpu_ids}'
        
        return max_memory
    @classmethod
    def infer_max_memory(cls, total_memory:int= None, buffer_memory:int=10) -> Dict[int, str]:
        """ Returns a dictionary of gpu_id to max memory for each gpu.
        Args:
            total_memory (int, optional): Total memory to allocate. Defaults to None.
            buffer_memory (int, optional): Buffer memory to leave on each gpu. Defaults to 10.
        
        Returns 
            Dict[int, str]: Dictionary of gpu_id to max memory for each gpu.
        """
        total_memory = total_memory or cls.free_gpu_memory()
        gpu_info_map = commune.gpu_map()
        most_available_gpu_tuples = sorted(gpu_info_map.items(), key=lambda x: x[1]['free'] , reverse=True)

        
        
        leftover_memory = total_memory
        max_memory = {}
        for gpu_id, gpu_info in most_available_gpu_tuples:
            if leftover_memory == 0:
                break
            free_gpu_memory = int(gpu_info['free']) - buffer_memory
            if leftover_memory > free_gpu_memory:
                leftover_memory -= free_gpu_memory
                max_memory[gpu_id] = f"{free_gpu_memory}GiB"
            elif leftover_memory <= free_gpu_memory :
                max_memory[gpu_id] = f"{leftover_memory}GiB"
                leftover_memory = 0

        return max_memory



    @classmethod
    def prep_tokenizer(cls,tokenizer, std_tokenizer=None):
            
        tokenizer.padding_side = "left"  # Generative default expects most recent token on right-hand side with padding on left. https://github.com/huggingface/transformers/pull/10552
        # tokenizer.add_prefix_space = False
        # tokenizer.add_special_tokens({'bos_token': "[BOS]"}) # A special token representing the beginning of a sentence.
        # tokenizer.add_special_tokens({'eos_token': "[EOS]"}) # A special token representing the end of a sentence.
        # tokenizer.add_special_tokens({'unk_token': "[UNK]"}) # A special token representing an out-of-vocabulary token.
        # tokenizer.add_special_tokens({'sep_token': "[SEP]"}) # A special token separating two different sentences in the same input (used by BERT for instance)
        # tokenizer.add_special_tokens({'pad_token': "[PAD]"}) # A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by attention mechanisms or loss computation.
        # tokenizer.add_special_tokens({'cls_token': "[CLS]"}) # A special token representing the class of the input (used by BERT for instance).
        # tokenizer.add_special_tokens({'mask_token': "[MASK]"}) # A special token representing a masked token (used by masked-language modeling pretraining objectives, like BERT).
        # additional_special_tokens = [
        #     "<s>NOTUSED",  # Used by BARThez
        #     "</s>NOTUSED", # Used by BARThez
        #     "<eop>", # Used by MarianMT
        #     "<eod>", # Used by MarianMT
        #     "<formula>", # Used by Transformer XL
        #     "<mask_1>" # Used by Pegasus
        #     "<special0>", # Used by XLM
        #     "<special1>", # Used by XLM
        #     "<special2>", # Used by XLM
        #     "<special3>", # Used by XLM
        #     "<special4>", # Used by XLM
        #     "<special5>", # Used by XLM
        #     "<special6>", # Used by XLM
        #     "<special7>", # Used by XLM
        #     "<special8>", # Used by XLM
        #     "<special9>", # Used by XLM
        # ]
        # tokenizer.additional_special_tokens = additional_special_tokens

        # Define PAD Token = EOS Token (GPT2 generate convention, when PAD Token is None)
        # https://github.com/huggingface/transformers/blob/49c8c67fb815a277405f84dea4a66353e19fb347/tests/models/gpt2/test_modeling_gpt2.py#L532
        import bittensor
        std_tokenizer = std_tokenizer or bittensor.tokenizer()
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        set_vocab_len(tokenizer)
        set_whitespace_preserving(tokenizer)

        if std_tokenizer is not None:
            set_std_token_phrases(tokenizer, std_tokenizer)

        return tokenizer

if __name__ == "__main__":
    # print('FUCK')

    # GPTNeoX()
    
    # GPTNeoX.launch(kwargs=dict(max_memory={0: "60GiB", 2: "60GiB" }))
    OPT().serve()
    # TransformerModel.experiment()


