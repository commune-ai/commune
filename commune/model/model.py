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
import glob

    
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
                 **kwargs
                ):
        
        
        self.init_model(model)
        # sets to self.config (with kwargs injected)
        config = self.set_config(config, kwargs=kwargs)
        self.set_model(config)

    @property
    def tag(self):
        if self.config.get('tag', None) == None:
            self.config['tag'] = 'base'
        return self.config['tag']
        

    @tag.setter
    def tag(self, tag):
        self.config['tag'] = tag

    def init_model(self):
        nn.Module.__init__(self) 
    @classmethod
    def train_fleet(cls, *args, **kwargs):
        return cls.module('model.transformer').train_fleet(*args, **kwargs)
         
    @classmethod
    def shortcuts(cls, *args, **kwargs):
        return cls.module('model.transformer').shortcuts
    
    @property
    def stats(self):
        return self.config.get('stats', {})

    @stats.setter
    def stats(self, stats):
        assert isinstance(stats, dict)
        self.config['stats'] = stats
        self.save(keys=['config'])
    
    def reset_stats(self):
        self.stats = {}
    
    def set_stats(self, stats: dict = None,):
        if stats == None:
            stats = {}
        assert isinstance(stats, dict)
        self.stats = stats
        
    @classmethod
    def learn(cls, *args, **kwargs):
        return cls.module('model.transformer').learn(*args, **kwargs)
        
    def set_optimizer(self, optimizer:dict=None):
        
        if optimizer == None:
            optimizer  = dict(
                module='torch.optim.Adam',
                lr=1e-5
            )
        if isinstance(optimizer, dict):
            module_path = optimizer.pop('module', 'torch.optim.Adam')
            optimizer_kwargs = optimizer.get('params', optimizer.get('kwargs', optimizer))

        else:
            raise NotImplementedError(optimizer)
        

        optimizer_class = self.import_object(module_path) 

        self.optimizer = optimizer_class(self.parameters(), **optimizer_kwargs)
        
        self.config['optimizer'] = {
            'module': module_path,
            **optimizer_kwargs,
        }
        
        
    def set_lr(self, lr:float):
        assert lr > 0, f'lr must be greater than 0, got {lr}'
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.config['optimizer']['lr'] = lr
    set_learning_rate = set_lr
        
    def forward(self,  **kwargs) -> Union[Dict, torch.Tensor]:
        # import ipdb; ipdb.set_trace()
        no_grad = kwargs.pop('no_grad', True)
        autocast = kwargs.pop('autocast', True)
        empty_cache = kwargs.pop('empty_cache', True)
        #should the model learn from the input in this forward pass
        train = kwargs['train'] = kwargs.get('train', False)

        # set the model to train mode
        if train:
            no_grad = False
            if self.training == False:
                self.train()
                self.training = True
        else:
            if self.training == True:
                self.eval()
            no_grad = True
            
            
        if no_grad:
            with torch.no_grad():
                if autocast: 
                    with torch.cuda.amp.autocast():
                        result = self._forward(**kwargs)
                else:
                    result = self._forward(**kwargs)
        else:
            if autocast:
                with torch.cuda.amp.autocast():
                    result = self._forward(**kwargs)
            else:
                result = self._forward(**kwargs)
        
        
        if empty_cache:
            torch.cuda.empty_cache()
        return result


    
    # def process_forward_locals(self, locals):
    #     kwargs = self.locals2kwargs(locals)
        
    #     # import ipdb; ipdb.set_trace()
    #     no_grad = kwargs.pop('no_grad', True)
    #     autocast = kwargs.pop('autocast', True)
    #     empty_cache = kwargs.pop('empty_cache', True)
    #     train = kwargs['train'] = kwargs.get('train', False)

    #     # set the model to train mode
    #     if train:
    #         no_grad = False
    #         if self.training == False:
    #             self.train()
    #             self.training = True
    #     else:
    #         no_grad = True
            
    #     if no_grad == True:
    #         # need to set no_grad to false to run forward ,or it will recurse  forever
    #         kwargs['no_grad'] = False
    #         with torch.no_grad():
    #             return self.forward(**kwargs)
    #     if autocast == True:
    #         kwargs['autocast'] = False
    #         with torch.cuda.amp.autocast():
    #             return self.forward(**kwargs)
            
       

    def _forward(self, **kwargs):
        raise NotImplementedError
    @property
    def device(self):
        # deepspeed has .module.device to access device
        if 'device' not in  self.config:
            self.set_device(device=None)
            
        return self.config['device']
    @device.setter
    def device(self, device):
        # deepspeed has .module.device to access device
        if self.is_number(device):
            device = f'cuda:{device}'
        self.set_device(device)
            
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
    



    def resolve_tag(self, tag):
        if tag == None:
            tag = self.tag
        return tag

    def save(self, 
             tag:str = None,  
             trainable_only:bool = True,
             verbose:bool = False,
             keys = None):
        tag = self.resolve_tag(tag)
        path = self.resolve_state_path(tag)

        model_state_dict = self.state_dict()
        
        if trainable_only:
            model_state_dict = {k:v for k,v in model_state_dict.items() if v.requires_grad} 
    
        
        os.makedirs(path, exist_ok=True)
        state_dict = {
            'model': model_state_dict,
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            }
        
        if keys == None:
            keys = list(state_dict.keys())
        else:
            assert isinstance(keys, list), f'keys must be a list, got {keys}'
            assert all([isinstance(k, str) for k in keys]), f'keys must be a list of strings, got {keys}'
            assert all([k in state_dict for k in keys]), f'keys must be a list of strings, got {keys}'
            keys = keys
        state_path_dict = {}
        for k in keys:
            state_path_dict[k] = os.path.join(path, f'{k}.pt')
        self.config['state_path_dict']= {**self.config.get('state_path_dict', {}), **state_path_dict}
        
        for k in keys:
            torch.save(state_dict[k], state_path_dict[k])        

        return path
    
    
    def check_config(self, config, ensure_keys=[]):
        for k in ensure_keys:
            assert config[k] == self.config[k], f'{k} in config {config[k]} does not match {k} in model {self.config[k]}'
    @classmethod
    def ls_tags(self):
        return self.ls()
    
    @classmethod
    def tags(cls):
        return cls.ls(return_full_path=False)
    def refresh(self, tag = None, verbose:bool = True, keys=['config']) -> Dict[str, Any]:
        tag = tag if tag != None else self.tag
        path = self.resolve_path(tag)
        self.rm_json(path)
        return path
    
    @classmethod
    def get_stats(cls, tag=None):
        if tag == None:
            tag = cls.tags()[0]
        return cls.get_json(cls.resolve_path(tag)+'/config.json').get('stats', {})
    

    @classmethod
    def get_stats_table(cls, tag=None):
        stats = cls.get_stats(tag)
        return pd.DataFrame(stats).T

    
    def resolve_state_path(self, tag=None):
        tag = tag if tag != None else self.tag
        path = self.resolve_path(tag)
        return path
    
    def reset_params(self):
        self.load_state_dict(self.og_state_dict['model'])
        self.optimizer.load_state_dict(self.og_state_dict['optimizer'])
    
    
    def load(self, tag=None, 
             keys:List[str] = None, 
             map_location: str = None,
             **kwargs):
        if not hasattr(self, 'load_cnt'):
            self.load_cnt = 0
            
        self.load_cnt += 1
        
        map_location = map_location if map_location else self.device
        tag = tag if tag != None else self.tag
        path = self.resolve_state_path(tag)
        if not os.path.exists(path):
            self.print(f'Couldnt find {path}')
            return 
        
        path_list = glob.glob(os.path.join(path, '*.pt'))
        loaded_state_dict = {}
        
        # load the keys (model, optimizer, config) into a dict
        for path in path_list:
            key = os.path.basename(path).replace('.pt', '')
            if not os.path.exists(path):
                self.print('No saved model found at {path}')
                return
            loaded_state_dict[key] = torch.load(path)
        
        if 'config' in loaded_state_dict:
            config = loaded_state_dict['config']
            self.check_config(config)
            self.set_config(config)
            # DO WE WANT TO REBUILD THE MODEL WHEN WE LOAD A CONFIG WITH SLIGHTLY DIFFERENT PARAMS
            
        if self.load_cnt == 1:
            # save the original state dict to get the vanilla model
            self.og_state['model'] = {k:states_dict[k] for k,v in loaded_state_dict['model'].keys() if v.requires_grad}
            self.og_state['optimizer'] = self.optimizer.state_dict()
            self.og_state['config'] = self.copy(self.config)
            
        states_dict = self.state_dict()
        if 'model' in loaded_state_dict:
            self.print('Loading model')
            self.update_state_dict(loaded_state_dict['model'])
    
        if 'optimizer' in loaded_state_dict:
            self.print('Loading optimizer')
            self.og_optimizer_state_dict = self.optimizer.state_dict()
            self.optimizer.load_state_dict(loaded_state_dict['optimizer'])
        
    def update_state_dict(self, state_dict:dict):
        assert isinstance(state_dict, dict), f'state_dict must be a dict, got {type(state_dict)}'
        state_dict = self.state_dict()
        state_dict.update(state_dict)
        self.load_state_dict(state_dict)
        
        
        
    def get_state_dict(self, keys=None):

        assert isinstance(state_dict, dict), f'state_dict must be a dict, got {type(state_dict)}'
        state_dict = self.state_dict()
        if keys == None:
            keys = state_dict.keys()
        state_dict.update(state_dict)
        self.load_state_dict(state_dict)
        

    def set_finetune(self, finetune) -> Tuple[bool, str]:
        r''' Set to tune only the parameter of the last layer
            Returns: 
                reached_last_layer (:type:`bool`):
                    If we have set partial of the model to requires grad.
                
                last_layer_name (:type:`string`):
                    The name of the last layer that user specified or we found.
                    None if the user did not specify and we couldnt find it. 
        '''
        self.config['finetune'] = num_layers = finetune
        all = False
        layer_name = None

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
                        self.print(f'Number of finetune layers was set higher then the layers avaliable {len(child)}')
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

        self.print(f'Set to finetune layer {last_layer_name} and onwards')
        
        for name, param in self.named_parameters():
            if last_layer_name in name or reached_last_layer == True:
                param.requires_grad = True
                
                reached_last_layer = True
            else:
                param.requires_grad = False

        if reached_last_layer == False:
            if all:
                self.print('Set to finetune the whole model, this will significantly increase the memory usage.')
            else:
                self.print(f'Cannot identify the last layer of the model with name {last_layer_name}, setting to finetune on all of the parameters.')

        self.print(self.num_params(trainable=True), 'trainable parameters')
        self.print(self.num_params(trainable=False), 'untrainable parameters')
        return reached_last_layer, last_layer_name
    
    
    @classmethod
    def resolve_device(cls, device:str = None) -> str:
        return commune.resolve_device(device=device)



        
    def num_params(self, trainable:bool = True) -> int:
        total_params = 0
        
        for name, param in self.named_parameters():
            if trainable:
                if param.requires_grad:
                    total_params += param.numel()
            else:
                total_params += param.numel()
                
        return total_params

    # @classmethod
    # def deploy(cls, *args, **kwargs):
    #     return cls.base_model().deploy(*args, **kwargs)


    @classmethod
    def deploy_fleet(cls, *args, **kwargs):
        return cls.base_model().deploy_fleet(*args, **kwargs)
    fleet = deploy_fleet
    
    @classmethod
    def base_model(cls):
        return cls.module('model.transformer')
    
    @classmethod
    def train_fleet(cls, *args, **kwargs):
        return cls.base_model().train_fleet(*args, **kwargs)
    
    @classmethod
    def test(cls, *args, **kwargs):
        return cls.base_model().test(*args, **kwargs)
    # train = test

    @classmethod
    def sandbox(cls, *args,**kwargs):
        self = cls(*args,**kwargs)
        print(self.config)


    @classmethod
    def quantize(cls,
                 model:str,
                 dynamic_q_layer : set = {torch.nn.Linear}, 
                 dtype=torch.qint8, **kwargs):
        self = torch.ao.quantization.quantize_dynamic( model,  # the original model
        dynamic_q_layer,  # a set of layers to dynamically quantize
        dtype=torch.qint8)
        return self
    


if __name__ == "__main__":
    
    Model.run()
    # TransformerModel.test()


