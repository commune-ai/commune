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
# commune.new_event_loop()
from commune.metric import MetricMap
from commune.utils.tokenizer import get_translation_map, translate_logits_to_probs_std, \
    translate_special_token_text, pad_offsets, topk_token_phrases, compact_topk_token_phrases, \
        encode_topk, decode_topk
 
"""
Examples 



"""
class Model( nn.Module, commune.Module):

    def __init__(self,
                 config = None,
                ):
        
        
        nn.Module.__init__(self) 
        self.set_config(config)
        
        
    def register_params(self, locals_dict: dict) -> None :
        
        if not hasattr(self, 'params'):
            self.params = {}
        for k, v in locals_dict.items():
            if k not in ['self', 'kwargs', 'args']:
                if v != None:
                    self.params[k] = deepcopy(locals_dict[k])
                
        
    def set_optimizer(self, optimizer:Union[Dict, 'Optimizer']=None):
        if isinstance(optimizer, dict):
            module_path = optimizer.pop('module', 'torch.optim.Adam')
            optimizer_kwargs = optimizer.get('params', optimizer.get('kwargs', optimizer))
        
        elif optimizer == None:
            module_path = 'torch.optim.Adam'
            optimizer_kwargs = {'lr': 0.02}
            
        
        else:
            raise NotImplementedError(optimizer)
        

        optimizer_class = self.import_object(module_path) 

        self.optimizer = optimizer_class(self.parameters(), **optimizer_kwargs)
        
        self.config['optimizer'] = {
            'module': module_path,
            **optimizer_kwargs,
        }
    def forward(self,  **kwargs) -> Union[Dict, torch.Tensor]:
        # import ipdb; ipdb.set_trace()
        no_grad = kwargs.pop('no_grad', True)
        autocast = kwargs.pop('autocast', True)
        empty_cache = kwargs.pop('empty_cache', True)
        #should the model learn from the input in this forward pass
        train = kwargs['train'] = kwargs.get('train', False)

        # set the model to train mode
        if train == True:
            no_grad = False
            
            
        if no_grad:
            with torch.no_grad():
                if autocast: 
                    with torch.cuda.amp.autocast():
                        result = self.local_forward(**kwargs)
                else:
                    result = self.local_forward(**kwargs)
        else:
            if autocast:
                with torch.cuda.amp.autocast():
                    result = self.local_forward(**kwargs)
            else:
                result = self.local_forward(**kwargs)
        
        
        if empty_cache:
            torch.cuda.empty_cache()
        return result

    def local_forward(self, **kwargs):
        raise NotImplementedError
    @property
    def device(self):
        # deepspeed has .module.device to access device
        if 'device' not in  self.config:
            self.set_device(device=None)
            
        return self.config['device']

    def set_device(self, device:str = None, resolve_device: bool = True):
        '''
        Sets the device for the model and returns the device
        '''
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if resolve_device:
            device = self.resolve_device(device)
        self.to(device)
        self.config['device'] = device
        return device
    



    def save(self, tag:str = None, keys:List[str]=None, trainable_only:bool = True, verbose:bool = True):
        tag = tag if tag else self.tag
        path = self.resolve_path(tag)

        model_state_dict = self.state_dict()
        
        if trainable_only:
            model_state_dict = {k:v for k,v in model_state_dict.items() if v.requires_grad} 
    
        
        os.makedirs(path, exist_ok=True)
        state_dict = {
            'model': model_state_dict,
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            }
        
        keys = state_dict.keys() if keys == None else keys
        
        for k in keys:
            assert k in state_dict, f'{k} not found in the state_dict'
        
        
        for k in keys:
            object_path = os.path.join(path, f'{k}.pt')
            torch.save(state_dict[k], object_path)
        
            if verbose:
                logger.success(f'Saving {k} to {object_path}')

        return path
    
    def load(self, tag=None, keys:List[str] = None, map_location: str = None, resolve_path:bool = True):
        map_location = map_location if map_location else self.device
        tag = tag if tag != None else self.tag
        path = self.resolve_path(tag) if resolve_path else tag
        import glob
        if not os.path.exists(path):
            return 
        path_list = glob.glob(os.path.join(path, '*.pt'))
        loaded_state_dict = {}
        for path in path_list:
            key = os.path.basename(path).replace('.pt', '')
            if not os.path.exists(path):
                logger.warning('No saved model found at {path}')
                return
            loaded_state_dict[key] = torch.load( path)
        
        state_dict = self.state_dict()
        

        
        # we want to save the base layers in case we want to change the layers

        # set the params and stats
        if 'config' in loaded_state_dict:
            self.set_config(config=loaded_state_dict['config'])
            self.set_params(**self.config)
            
        if 'model' in loaded_state_dict:
            if  hasattr(self, 'base_state_dict'):
                state_dict.update(self.base_state_dict)
                 
            else:
                self.base_state_dict = {}
                for k in loaded_state_dict['model'].keys():
                    self.base_state_dict[k] = state_dict[k].clone()
            
            state_dict.update(loaded_state_dict['model'])
            self.load_state_dict(state_dict)
        
        if 'optimizer' in state_dict:
            self.optimizer.load_state_dict(loaded_state_dict['optimizer'])
        
        
    def set_tag(self, tag):

        if tag == None:
            if hasattr(self, 'tag'):
                return self.tag
            else:
                tag = 'base'
                
        self.tag = tag
        # self.load(tag)
        
        
    def set_finetune(self, finetune ) -> Tuple[bool, str]:
        r''' Set to tune only the parameter of the last layer
            Returns: 
                reached_last_layer (:type:`bool`):
                    If we have set partial of the model to requires grad.
                
                last_layer_name (:type:`string`):
                    The name of the last layer that user specified or we found.
                    None if the user did not specify and we couldnt find it. 
        '''
        default_kwargs = dict(num_layers=1, layer_name = None, all = False)
        
        num_layers = finetune.get('num_layers', default_kwargs['num_layers'])
        layer_name = finetune.get('layer_name', default_kwargs['layer_name'])
        all = finetune.get('all', default_kwargs['all'])

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
            last_layer_name = find_last_layer(self)
        else:
            last_layer_name = layer_name

        reached_last_layer = False

        # set the non-last layer parameters not to require grads
        if (all) or (last_layer_name == None):
            return False, last_layer_name

        logger.success(f'Set to finetune layer {last_layer_name} and onwards')
        
        for name, param in self.named_parameters():
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
    def resolve_device(cls, device:str = None) -> str:
        return commune.resolve_device(device=device)
    
    def set_config(self, config) -> None:
        self.config = deepcopy(config)
        self.config.pop('self',None)
        self.config.pop('kwargs', None)
        self.config.pop('args', None)
    
    def set_stats(self, stats: dict):
        if stats == None:
            if hasattr(self, 'stats'):
                stats = self.stats
            else:
                stats = {}
        self.stats = stats
        self.config['stats'] = self.stats


if __name__ == "__main__":
    
    Model.run()
    # TransformerModel.test()


