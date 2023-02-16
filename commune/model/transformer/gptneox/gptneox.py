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
from huggingface_hub import hf_hub_download
from commune.utils.tokenizer import prep_tokenizer, get_translation_map, translate_logits_to_probs_std, \
    translate_special_token_text, pad_offsets, topk_token_phrases, compact_topk_token_phrases
    
    

import commune
commune.new_event_loop()
import bittensor
# import torch

# commune.utils
from torch import nn
    
"""
Examples 



"""
class GPTNeoX( nn.Module, commune.Module):
    shortcuts =  {
        'gptj': 'EleutherAI/gpt-j-6B',
        'gpt2.7b': 'EleutherAI/gpt-neo-2.7B',
        'gpt125m': 'EleutherAI/gpt-neo-125M',
        'gptjt': 'togethercomputer/GPT-JT-6B-v1',
        'gpt20b': 'EleutherAI/gpt-neox-20b',
        'gptneox': 'EleutherAI/gpt-neox-20b'
         } 
    def __init__(self,
                # model_name: str="EleutherAI/gpt-j-6B",
                model_name: str='gptneox',
                checkpoint_path: str = None,
                max_memory: Union[Dict[int, str], int, float] = 100,
                max_per_gpu: int = 50,
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
        
        

        self.tag = tag

        self.set_model(model_name=model_name, override_device_map=override_device_map, 
                       no_split_module_classes=no_split_module_classes, 
                       device=device, 
                       max_memory=max_memory,
                       max_per_gpu=max_per_gpu)
        
        self.tokenizer = self.set_tokenizer(tokenizer if tokenizer else self.model_name)
        self.set_metrics(metrics=metrics)

        

    def set_model(self, model_name:str, 
                  override_device_map: Dict[int, str], 
                  no_split_module_classes:List[str],
                  max_per_gpu:int,
                  max_memory: Union[Dict[int, str], int, float],
                  device:str):
                
        device = self.resolve_device(device)
        self.model_name = self.shortcuts.get(model_name, model_name)
        self.override_device_map = override_device_map if override_device_map else {}
        self.no_split_module_classes = no_split_module_classes if no_split_module_classes else []
        self.model_config = AutoConfig.from_pretrained(self.model_name)
        self.model_config.use_cache = False 
        self.model_device = device
        
        if  os.path.exists(self.checkpoint_path):
            commune.log(f'Found weights path at {self.checkpoint_path}', 'green')
        else:
            commune.log(f'Creating new weights path at {self.checkpoint_path}', 'purple')
            AutoModelForCausalLM.from_config(self.model_config).save_pretrained(self.checkpoint_path)
            

        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(self.model_config)


        self.max_memory = self.resolve_max_memory(max_memory, max_per_gpu=max_per_gpu)

        commune.log(f'max_memory: {self.max_memory}', 'yellow')

        if self.model_name == 'EleutherAI/gpt-neox-20b':
            self.no_split_module_classes =  ["GPTNeoXLayer"]
            self.override_device_map = {'gpt_neox.embed_in': device}


        self.device_map = infer_auto_device_map(
            self.model, 
            no_split_module_classes= self.no_split_module_classes,
            dtype=torch.bfloat16, #note: succeeds with float16 as well.
            max_memory = self.max_memory,
            )    
                
        self.device_map.update(self.override_device_map)
        

        load_checkpoint_and_dispatch(
            self.model,
            self.checkpoint_path,
            device_map=self.device_map,
            offload_folder=None,
            offload_state_dict=False,
            dtype="bfloat16"
        )
            
        # convert model to bfloat16
        self.model_config = commune.dict2munch(json.loads(self.model_config.to_json_string()))



        
    @property
    def checkpoint_path(self):
        return f"{self.tmp_dir()}/checkpoints/{self.module_tag}"

    def calculate_loss(self, pediction, gt):
        loss =  self.metrics['cross_entropy'](pediction, gt)
        return loss

    def set_metrics(self, metrics=None):
        self.metrics = {}
        if metrics == None:
            self.metrics['cross_entropy'] =  torch.nn.CrossEntropyLoss()
        return metrics


    def forward(self, *args,no_grad=True, autocast:bool=False, **kwargs):
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
    
        # if verbose:
        #     print('INPUT_STATISTICS: ',tensor_info_dict(input_dict))
        
        tokens['input_ids'] = tokens['input_ids'].to(self.device)

        model_output = self.model(input_ids=tokens['input_ids'],
                                  output_hidden_states=True)
        
        # sometime we dont care about the begginning of the sequence
        
        output_length = output_length if output_length else model_output.logits.size(1)
        model_output.logits = model_output.logits.to(self.device)
        model_output.logits = model_output.logits[:,-output_length:,:]
        
        # remap back to original tokens if token_remap is True
        if logit_remap:
            pre_logits = model_output.logits
            probs_std = translate_logits_to_probs_std(pre_logits,
                                                        tokens['offset_mapping'], tokens['offset_mapping_std'],
                                                        self.tokenizer, self.std_tokenizer,
                                                        self.split_map_cache,
                                                        self.to_translation_map, 
                                                        self.from_translation_map,
                                                        tokens['input_ids'], input_ids)
            probs_std = probs_std
            logits_std = torch.log(probs_std + 1e-40)            
            model_output.logits = logits_std.to(self.device)
        
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
        # if str(model_device) == 'meta':
        #     model_device = 'cuda'
        return self.model_device



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
        
        commune.log(self.tokenizer, 'purple')
        
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
    def path2shortcut(self, path:str):
        return {v:k for k,v in self.shortcuts.items()}.get(path, path)

    def tokenize(self, text: str = 'Whadup',device:str = None, **kwargs) -> torch.Tensor:
        """ Returns tokenized text as torch tensor. """
        
        device = device if device != None else self.device
        tokenizer_output = self.tokenizer(text, **kwargs)
        
        return tokenizer_output.input_ids.to(device)

    def set_stats(self, stats:dict=None): 
        if stats == None:
            stats =  dict(
                steps = 0,
                loss = 0,
            )
        self.stats = Munch(stats)
        
        
    @property
    def module_tag(self): 
        return self.resolve_module_tag()
    
    def resolve_module_tag(self, tag=None):
        tag = tag if tag else self.tag
        module_tag = self.path2shortcut(self.model_name).replace("/", "::")
        if tag:
            module_tag +=  f'::{tag}'
        return module_tag
    
   
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
    def resolve_max_memory(cls, max_memory: Union[Dict[int, str], int], buffer_memory:int=10, max_per_gpu:int=50) -> Dict[int, str]:
        
        if isinstance(max_memory, int):
            max_memory = cls.infer_max_memory(total_memory=max_memory, 
                                              buffer_memory=buffer_memory, 
                                              max_per_gpu=max_per_gpu)
            
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
    def infer_max_memory(cls, total_memory:int= None, buffer_memory:int=10, max_per_gpu:int=50) -> Dict[int, str]:
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
            free_gpu_memory = min(max_per_gpu, free_gpu_memory)
            if leftover_memory > free_gpu_memory:
                leftover_memory -= free_gpu_memory
                max_memory[gpu_id] = f"{free_gpu_memory}GiB"
            elif leftover_memory <= free_gpu_memory :
                max_memory[gpu_id] = f"{leftover_memory}GiB"
                leftover_memory = 0

        return max_memory

    

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

if __name__ == "__main__":
    # print('FUCK')

    GPTNeoX().run()
    
    # GPTNeoX.launch(kwargs=dict(max_memory={0: "60GiB", 2: "60GiB" }))
    # print(GPTNeoX().to_translation_map)
    # TransformerModel.experiment()


