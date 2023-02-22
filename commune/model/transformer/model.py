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
import bittensor
from commune.utils.tokenizer import get_translation_map, translate_logits_to_probs_std, \
    translate_special_token_text, pad_offsets, topk_token_phrases, compact_topk_token_phrases, \
        encode_topk, decode_topk
 
"""
Examples 



"""
class TransformerModel( nn.Module, commune.Module):
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
                tokenizer:Union[str, 'tokenizer'] = None,
                optimizer: torch.optim  = None,
                metrics: Dict[str, 'Metric'] = None,
                device: str='cuda',
                load: bool = True,
                autocast: bool = False,
                finetune : dict = dict(num_layers=4),
                stats: dict = None, 
                topk: int = 4096,
                
                **kwargs
                ):
        
        
        self.tag = tag if tag else model_name
        self.topk = topk
        
        print('BROOO')
        
        
        
        nn.Module.__init__(self)
        
        # set model and tokenizer

        self.set_model(model_name=model_name,device=device, autocast=autocast, **kwargs)

        # set tokenizer to model name (HF only) if tokenizer == None
        self.set_tokenizer(tokenizer=tokenizer if tokenizer != None else self.model_name)
        
        self.set_optimizer(optimizer=optimizer)
        
        self.metrics = MetricMap(metrics=metrics)
        
        self.set_stats( **(stats if stats else {'tag': self.tag}))
        
        
        if load:
            self.load(tag)
        
        self.set_fine_tuning_params(**finetune)
        
        
    def set_metric(self, key:str, value:float, **kwargs):
        return self.metrics.set_metric(key=key, value=value, **kwargs)
        
    def get_metric(self, key:str, value:float, **kwargs) -> float:
        return self.metrics.get_metric(key=key, value=value, **kwargs)
    
    def get_metrics(self)-> Dict:
        return self.metrics.value_map
        

    def set_optimizer(self, optimizer:Dict=None):
        
        if isinstance(optimizer, dict):
            module_path = optimizer.pop('module', 'torch.optim.Adam')
            assert module_name != None, f'Please specify a valid optimizer ex: torch.optim.Adam'
            optimizer_class = self.import_object(module_path) 
            optimizer_kwargs = optimizer.get('kwargs', optimizer)
            optimizer_args = optimizer.get('args', [])
            self.optimizer_class(*optimizer_args,**optimizer_kwargs)
                
        elif optimizer == None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.00002)
        
        else:
            raise NotImplementedError(optimizer)
        
        
        return self.optimizer


    @classmethod
    def calculate_loss( cls, pred:torch.Tensor,
                       gt:torch.Tensor = None,
                       input: torch.Tensor=None , 
                       return_value: bool = False,
                       *args, **kwargs) -> torch.Tensor:
        '''
        Calculate the loss for the model.
        '''
        if input != None:
            gt = input[:, -(pred.shape[1]-1):].flatten()
            pred = pred[:, :pred.shape[1]-1]
            
        if len(pred.shape) == 3:
            pred = pred.reshape(-1, pred.shape[-1])
        
        assert gt.shape == pred.shape[:1], f'gt.shape: {gt.shape} pred.shape: {pred.shape}'

        loss_fn = torch.nn.CrossEntropyLoss( *args, **kwargs)
        loss =  loss_fn(pred, gt.to(pred.device))
        if return_value:
            return loss.item()
        return loss

    

    def forward(self, *args,  no_grad:bool=True, autocast: bool = True,**kwargs,):
        # import ipdb; ipdb.set_trace()

        #should the model learn from the input in this forward pass
        learn = kwargs['learn'] = kwargs.get('learn', True)

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
                
                topk:int=4096, 
                output_topk:bool = True,
                output_hidden_states:bool=True,
                hidden_state_index: int = -1,
                hidden_dim_bounds: List =  None,
                output_logits:bool = True,
                output_length:int = 10,
                token_remap:bool = False,
                logit_remap:bool = False,
                verbose:bool = False,
                return_keys:List[str] = None,
                learn: bool = False,
                client_mode: bool =  False,
                **kwargs):

        # if isinstance(input_ids, str) or ((isinstance(input_ids, list) and isinstance(input_ids[0], str))):
        #     input_ids = self.tokenize(input_ids)
        #     token_remap = False
        # transformers.set_seed(0)
        # transformers.enable_full_determinism(0)
        # remap the tokens if token_remap is True
  
        # enable client mode for more efficient tensors to be sent over the wire
        if client_mode == True:
            output_logits = False
            output_hidden_states = False
            output_topk = True
        tokens = {
            'input_ids': input_ids,
        }
        if token_remap:
            tokens = self.token_remap(input_ids, std_tokenizer=self.tokenizer)  # remap to server tokenizer
    
        # if verbose:
        #     print('INPUT_STATISTICS: ',tensor_info_dict(input_dict))
        
        tokens['input_ids'] = tokens['input_ids'].to(self.device)

        print(tokens['input_ids'].device, self.device)
        model_output = self.model(input_ids=tokens['input_ids'],
                                  output_hidden_states=True)
        
        
        
        if learn:
            self.optimizer.zero_grad()
            
        
        # sometime we dont care about the begginning of the sequence
        
        output_length = output_length if output_length else model_output.logits.size(1)
        model_output.logits = model_output.logits[:,-output_length:,:]
        
        output_dict = {}
        
             
        # remap back to original tokens if token_remap is True
        if logit_remap:
            self.logit_remap(logits = model_output.logits, input_ids=input_ids)
           
        
        if isinstance(return_keys, list):
            for key in return_keys:
                assert key in ['logits','hidden_states','input_ids'], f'Invalid return key {key}'
                # set the output_{key} to True
                locals()[f'output_{key}'] = True

        
        if output_topk:
            topk_tensor = self.encode_topk(model_output.logits, topk=topk)
            output_dict['topk']=topk_tensor
            
        if output_logits:
            output_dict['logits']=model_output.logits

        if output_hidden_states:
            output_dict['hidden_states'] = model_output.hidden_states[hidden_state_index] 
            output_dict['hidden_states'] = output_dict['hidden_states'][:,-output_length:,:]
            hidden_dim = output_dict['hidden_states'].size(-1)
            if isinstance(hidden_dim_bounds, int):
                hidden_dim_bounds = [0, hidden_dim_bounds]
            
            hidden_dim_bounds = hidden_dim_bounds if hidden_dim_bounds else [0, hidden_dim+1]
            output_dict['hidden_states'] = output_dict['hidden_states'][:, :, hidden_dim_bounds[0]:hidden_dim_bounds[1]]
            
            
        
        
        


        return output_dict

    def loss_fct(self, logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
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
        if not hasattr(self, '_device'):
            if hasattr(self, 'model'):
                assert hasattr(self.model, 'device'), 'model has no device attribute'
                self._device = self.model.device
            else:
                self._device = self.set_device(device)
                
            
        return self._device

    def set_device(self, device:str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = self.resolve_device(device)
        self._device = device
        self.to(device)
        return self._device
    def set_model(self, model_name:str, device:str = None, autocast:bool = False, **extra_model_kwargs):
        from transformers import  AutoModelForCausalLM, AutoModel, AutoConfig

        self.model_name = self.shortcuts.get(model_name, model_name)
        # config = AutoConfig.from_pretrained(self.model_name)
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, 
                                            **extra_model_kwargs)        
        
        # convert config to config
        self.config = json.loads(self.model.config.to_json_string())
        
        self.set_device(device=device)
        
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

    @classmethod
    def test_model(cls, batch_size=8, sequence_length=256, model_name='EleutherAI/gpt-neox-20b'):
        self = cls(serve=False, model_name=model_name)
        example = ["My name is Philipp and I"]*batch_size
        input_ids = self.tokenizer(example,return_tensors="pt", max_length=sequence_length, padding='max_length').input_ids.to(self.device)
        
        print('TESTING LOGITS OUTPUT')
        logits = self.forward(input_ids, output_hidden_states=True, topk=None,verbose=True)
        
        print('TESTING TOPK OUTPUT')
        logits = self.forward(input_ids, output_hidden_states=True, topk=None,verbose=True)
    

    def learn_step(self, **sample):
        '''
        {
            '
        }
        '''
        save = sample.pop('save', False)
        load = sample.pop('load', False)
        tag = sample.pop('tag', None)
        
        if load:
            self.load(tag)
        sample['no_grad'] = False
        
        
        original_kwargs = {}
        original_kwargs['output_logits'] = sample.get('output_logits', True)
        # we need the logits and we need to 
        if  original_kwargs['output_logits'] == False:
            sample['output_logits'] = True 
        
        self.optimizer.zero_grad()
        model_output = self.forward(**sample)
        loss = self.calculate_loss(pred=model_output['logits'], input=sample['input_ids'])   
        loss.backward()
        self.optimizer.step()
        
        self.set_metric('loss', loss.item())
        self.stats['learn_steps'] =  self.stats.get('learn_steps', 0) + 1

        
        
        if not original_kwargs['output_logits']:
            del model_output['logits']
            
        model_output['metrics'] = self.get_metrics()
        model_output['stats'] = self.stats
        
        if save:
            self.save(tag)
        
        return Munch(model_output)
    

    def set_stats(self, **stats) -> Dict[str, Any]: 
        if not hasattr(self, 'stats'):
            self.stats = {'tag': self.tag}
        
        self.stats.update(stats)
        return self.stats
        
    def set_stat(self, key:str, value:Any) -> Dict[str, Any]: 
        if not hasattr(self, 'stats'):
            self.stats = {'tag': self.tag}
        self.stats[key] = value
        return value
    
    def get_stat(self, key:str,default_value:Any= None) -> Any: 
        if not hasattr(self, 'stats'):
            self.stats = {'tag': self.tag}
        return self.stats.get(key, default_value)
    
    
    
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
        module_tag = self.resolve_module_tag(tag=tag)
        path = self.resolve_path(module_tag)
        model_state_dict = self.model.state_dict()
        
        if trainable_only:
            model_state_dict = {k:v for k,v in model_state_dict.items() if v.requires_grad} 
    
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state_dict = {
            'model': model_state_dict,
            'optimizer': self.optimizer.state_dict(),
            'stats': self.stats,
            'metrics': self.metrics.to_dict(),
            'config': self.config
        }
        
        logger.success(f'Saving path {path}')
        
    
        torch.save(state_dict, path)
        
        return path
    
    def load(self, tag=None):
        module_tag = self.resolve_module_tag(tag=tag)
        path = self.resolve_path(module_tag)
        if not os.path.exists(path):
            logger.warning('No saved model found at {path}')
            return
        loaded_state  = torch.load( path)
        state_dict = self.model.state_dict()
        
        
        for k,v in loaded_state['model'].items():
            assert k in state_dict
            state_dict[k] = v
        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(loaded_state['optimizer'])
        self.metrics = MetricMap.from_dict(loaded_state.get('metrics', {}))
        self.set_stats(**loaded_state['stats'])
        

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
    def run_train(cls, 
                    model:str='gpt125m',
                    tag:str = 'demo', 
                    num_batches:int = 10000,
                    window_size:int = 50,
                    backoff_window_size:int = 25,
                    max_iters_since_best:int = 100,
                    dataset:str= 'dataset::bittensor',
                    **kwargs
                    ):
        model = cls(model_name=model,tag=tag, load=True,  **kwargs)
        dataset = commune.connect(dataset)
    
        best_loss = 10e10
        stats = model.get_stats()
        stats['best_loss'] = stats.get('loss', best_loss)
        
        if stats['best_loss'] < 0.1:
            stats['best_loss'] = 10e10
        
        commune.log(f'Loaded {stats} from {tag}', 'yellow')

        
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
                commune.log(f'Failed to get sample {fail_count} times', 'red')
                continue
            
            
            loss = model.learn_step(**sample)
            
            # update the metric_window
            metric.update(loss)
            window_loss  = metric.value
        
            if verbose:
                info_str = f'Batch {i}/{num_batches} CE: {loss} : {window_loss} Best Loss: {best_loss}'
                commune.log(info_str, 'purple')
                
            if window_loss < best_loss and i > window_size and iters_since_best > backoff_window_size:
                best_loss = window_loss
                model.set_stats(loss=best_loss)
                commune.log(f'Best Stats: {model.get_stats()} ', 'green')
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
        output = model.learn_step(**sample, save=True)
        print(output)

        # output['logits'] = decode_topk(output['topk'])
        
        # print(cls.calculate_loss(output['logits'].reshape(-1, output['logits'].shape[-1]), targets[:, -output_length:].flatten()))
        


if __name__ == "__main__":
    
    TransformerModel.run()
    # TransformerModel.test()


