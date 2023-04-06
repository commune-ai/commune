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
from commune.utils.torch import tensor_dict_info
from commune.utils.tokenizer import decode_topk, get_translation_map
import streamlit as st
# logger = logger.opt(colors=True)
import commune
import os
# import torch
# commune.utils
from torch import nn
from torch import Tensor
from commune.model.ensemble.ensemble_blocks import AdapterModel
from commune.model import Model
"""
Examples 



"""

import nest_asyncio
nest_asyncio.apply()


class EnsembleModel( Model):

    def __init__(self,
                models: List[str] = None,
                # models: List[str] = ['model::gpt125m'],
                tokenizer: 'tokenizer' = 'gptj',
                optimizer:  'torch.optimizer' = None,
                sample_fraction: float = 1.0,
                metrics: Dict= None,
                load: bool = True,
                device: str = 'cuda',
                tag: str = None,
                stats: Dict = None,
                ):
        Model.__init__(self, config = locals())

        self.loop = self.new_event_loop()
        # set model and tokenizer
        self.set_tag(tag)
        self.set_models(models)
        self.set_tokenizer(tokenizer)
        # self.set_optimizer(optimizer)
        self.set_device(device)
        self.set_stats(stats)
        if load:
            self.load()
        


    def default_models(self):
        return [m for m in commune.servers() if m.startswith('model')][:100]
    @classmethod
    def test(cls, topk=512, output_length=10, num_batches = 10):
        
        model = cls()
        dataset = commune.connect('dataset::bittensor')
        model = model.to('cuda')

        for i in range(num_batches):
            sample = dataset.sample(sequence_length=256)
            t = commune.timer()
            stats = model.forward(**sample)
            stats['stats'].pop('peer_loss_per_sentence')
            print(t.seconds, stats['stats']['best_peer_loss_per_token'].mean().item())
            
        # output['logits'] = decode_topk(output['topk'])
        
        # print(cls.calculate_loss(output['logits'].reshape(-1, output['logits'].shape[-1]), targets[:, -output_length:].flatten()))
     
    @classmethod
    def calculate_loss( cls, logits, input_ids,  **kwargs) -> torch.Tensor:
        '''
        Calculate the loss for the model.
        '''
        pred = logits
        gt = input_ids[:, -(pred.shape[1]-1):].flatten()
        pred = pred[:, :pred.shape[1]-1]
            
        if len(pred.shape) == 3:
            pred = pred.reshape(-1, pred.shape[-1])
        
        assert gt.shape == pred.shape[:1], f'gt.shape: {gt.shape} pred.shape: {pred.shape}'

        loss_fn = torch.nn.CrossEntropyLoss(**kwargs)
        loss =  loss_fn(pred, gt.to(pred.device))
        return loss


    
    async def async_model_forward(self, model, *args, **kwargs):
        return self.models[model].forward(*args, **kwargs, asyncio_future=True, timeout=5)
        
    def aggregate(self, 
                  x: List[torch.Tensor], 
                  *args, **kwargs) -> Dict[str, torch.Tensor]:
        
        
        if isinstance(x, list):
            x = torch.stack(x, dim=0)
        x = torch.sum(x, dim=0)
        x = torch.softmax(x, dim=-1)
        x = torch.log(x + 1e-10)
        
        return x
    @property
    def model_names(self) -> List[str]:
        return list(self.models.keys())

    def forward(self, *args, 
                output_length=10, 
                topk=512, 
                return_topk_only=True, 
                sample_fraction: float = 1.0,
                **kwargs):
        
        
        kwargs.update(dict(
            return_keys=['topk'],
            output_length=output_length,
            token_remap = False , 
            logit_remap = False,
            topk=topk
        ))
        
        
        jobs = []
        import random 
        
        selected_models = random.sample(list(self.models.keys()),  int(sample_fraction * len(self.models)))
        
        for model in selected_models:
            job = self.models[model].forward(*args, **kwargs, asyncio_future=True, timeout=4)
            jobs.append(job) 
            

        
        peer_outputs =  self.loop.run_until_complete(asyncio.gather(*jobs))
        
        peer_outputs = [peer_output for peer_output in peer_outputs if 'topk' in peer_output]
        max_token_index = 50400
    
        model_names = selected_models
        for model_i, peer_output in enumerate(peer_outputs):
            print(model_i, peer_output['topk'].shape, 'BROOO')
            if 'topk'  in peer_output:
                peer_output['logits'] = decode_topk(peer_output['topk'], vocab_size=int(max_token_index+1), topk= kwargs['topk'])
                peer_outputs[model_i] = peer_output
        
        output_dict = dict(
            # peer_logits = torch.stack([x['logits'] for x in peer_outputs], dim=0),
            
            logits = None,
            
            peer_logits = [],
            stats = dict(
                    peer_loss_per_token = [],
                    peer_loss_per_sentence = [],
                    peer_loss = [])
            
        )
        
        # calculate score per token and sentence
        for model_i, peer_output in enumerate(peer_outputs):
            peer_loss_per_token = self.calculate_loss(logits=peer_output['logits'], input_ids=kwargs['input_ids'], reduction='none')
            peer_loss_per_token = peer_loss_per_token.reshape(peer_loss_per_token.shape[0], -1)
            output_dict['stats']['peer_loss_per_token'].append(peer_loss_per_token)
            output_dict['stats']['peer_loss_per_sentence'].append(peer_loss_per_token.mean(dim=-1))
            output_dict['stats']['peer_loss'].append(peer_loss_per_token.mean())
            
            
            
            
            
        output_dict['stats']['peer_loss'] = torch.stack(output_dict['stats']['peer_loss'], dim=0)
        output_dict['stats']['best_peer_id'] = torch.argmin(output_dict['stats']['peer_loss'], dim=0)
        output_dict['stats']['best_peer_loss'] = torch.index_select(output_dict['stats']['peer_loss'],  index = output_dict['stats']['best_peer_id'] , dim=0)
        output_dict['stats']['best_peer_loss_per_token'] = torch.index_select(torch.stack(output_dict['stats']['peer_loss_per_token'], dim=0), index = output_dict['stats']['best_peer_id'] , dim=0)
        output_dict['stats']['peer_loss_per_sentence'] = torch.stack(output_dict['stats']['peer_loss_per_sentence'], dim=0)

        output_dict['stats']['best_peer_loss_per_sentence'] = torch.index_select(output_dict['stats']['peer_loss_per_sentence'], index = output_dict['stats']['best_peer_id'] , dim=0)
        prior_routing_scores = output_dict['stats']['peer_loss_per_sentence'] 
        prior_routing_scores_std = (prior_routing_scores.std(0)[None,...] + 1E-10)
        if len(peer_outputs) == 1:
            prior_routing_scores_std = 1
        else:
            prior_routing_scores_std = (prior_routing_scores.std(0)[None,...] + 1E-10)

        prior_routing_scores =  (prior_routing_scores - prior_routing_scores.max(0).values[None, ...])/prior_routing_scores_std
        prior_routing_scores = prior_routing_scores / prior_routing_scores.sum(0)[None, :]
            
        
        # calculate the routing scores
        for model_i, peer_output in enumerate(peer_outputs):
            pred = peer_output['logits']
            peer_score = prior_routing_scores[model_i, ...][:].to(self.device) #+ peer_output['routing_score'][model_i])/2
            self.print(pred.shape, prior_routing_scores[model_i].shape, 'BROOO')
            output_dict['peer_logits'] += [torch.einsum('ijk,i -> ijk', pred , prior_routing_scores[model_i])]
        
        
        output_dict['logits'] = (self.aggregate(output_dict['peer_logits']).to('cuda') )
        
        ensemble_loss_per_token = self.calculate_loss(logits=output_dict['logits'], input_ids=kwargs['input_ids'], reduction='none')
        # output_dict['stats']['ensemble_loss_per_token'] = ensemble_loss_per_token.reshape(-1, output_dict['logits'].shape[1]-1)
        
        output_dict['stats']['ensemble_loss_per_token'] = ensemble_loss_per_token

        output_dict['stats']['ensemble_loss_per_sentence'] = ensemble_loss_per_token.mean(-1)
        output_dict['stats']['ensemble_loss'] = output_dict['stats']['ensemble_loss_per_sentence'].mean()
        output_dict['stats']['ensemble_loss_per_token'] = ensemble_loss_per_token

        return Munch(output_dict)
    


    @property
    def device(self) -> str:
        return self._device

    def set_models(self, models: Union[List, Dict]=None):
        self.model_name = 'ensemble'
        self.models = {} 
        if models is None:
            models = self.default_models()
        connect_model_jobs = [self.async_connect(model, loop=self.loop) for model in models]
        model_clients = asyncio.run(asyncio.gather(*connect_model_jobs))
        for model, client in zip(models, model_clients):
            self.models[model] = client
 
        return self.models
    

    def list_models(self):
        return list(self.models.keys())

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    shortcuts = {
        'gptj': 'EleutherAI/gpt-j-6b',
    }

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

    def tokenize(self, text: str = 'Whadup', input_ids_only:bool = True, device: str=None) -> torch.Tensor:
        """ Returns tokenized text as torch tensor. """
        device = device if device != None else self.device
        tokenizer_output = self.tokenizer(text, return_tensors='pt')
        if input_ids_only:
            return tokenizer_output.input_ids.to(self.device)
        return self.tokenizer(text, return_tensors='pt').input_ids.to(self.device)

    @classmethod
    def get_dataset(cls, dataset: str = 'dataset.text.bittensor', device: str=None, refresh:bool = True) -> torch.utils.data.Dataset:
        """ Returns a torch dataset. """
        if not cls.server_exists(dataset) or refresh:
            commune.launch('dataset.text.bittensor', name=dataset)
        
        return commune.connect(dataset, wait_for_server=True)

    def set_tokenizer(self, tokenizer:Union[str, 'tokenizer', None]):
        import bittensor
        self.std_tokenizer = bittensor.tokenizer()
        
        if isinstance(tokenizer, str):
            from transformers import AutoTokenizer
            tokenizer = self.shortcuts.get(tokenizer, tokenizer)
            self.config['tokenizer'] = tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast= True)
            except ValueError:
                print('resorting ot use_fast = False')
                tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
            except OSError:
                print('resorting ot use_fast = False')
                tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        elif tokenizer == None:
            tokenizer = self.std_tokenizer
            
        self.tokenizer = tokenizer
        
        

        from commune.utils.tokenizer import prep_tokenizer
        self.tokenizer = prep_tokenizer(self.tokenizer, self.std_tokenizer)
        
        self.to_translation_map = get_translation_map(self.tokenizer, self.std_tokenizer)
        self.from_translation_map = get_translation_map(self.std_tokenizer, self.tokenizer)
        self.split_map_cache = {}

        return self.tokenizer


    @classmethod
    def test_neuron(cls, tokenizer='bittensor', num_batches=10, dataset='dataset::bittensor', batch_size=32, sequence_length=12, topk=4096, **model_kwargs):
        from commune.block.bittensor.neuron.miner import neuron
        from bittensor.utils.tokenizer_utils import phrase_cross_entropy, topk_token_phrases, prep_tokenizer
        self = cls( tokenizer=tokenizer)
        self.to('cuda')
        nucleus = neuron(model=self).model
        nucleus.model.train()
        nucleus.model.eval()
        nucleus.model = nucleus.model.half()
        nucleus.model.config.hidden_size
        nucleus.model.config.pad_token_id
        nucleus.model.config.eos_token_id
        nucleus.model.named_parameters()
        state_dict = nucleus.model.state_dict()
        nucleus.model.load_state_dict(state_dict)
        
        dataset = commune.connect(dataset)
        sample = dataset.sample()
        
        for i in range(num_batches):
            sample = dataset.sample(batch_size=32, sequence_length=256)
            target = sample['input_ids'][:, -1:] 
            inputs_x = sample['input_ids'][:, :-1] 
            t = commune.timer()
            message, _model_output, topk_tensor = nucleus.encode_forward_causallmnext(inputs_x, topk=topk)
            loss_tuple = phrase_cross_entropy(topk_tensor=topk_tensor, target_phrases=target)
            commune.print(f'Loss : {loss_tuple[0].item()} Time: {t.seconds}', 'cyan')
 
    @classmethod
    def run_neuron(cls, tokenizer='bittensor'):
        from commune.block.bittensor.neuron.miner import neuron
        self = cls( tokenizer=tokenizer)
        n = neuron(model=self)  
        n.run()
 

    def set_device(self, device:str = None):
        if device == None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._device = torch.device(device)
        return self.device
    
    @classmethod
    def sandbox(cls):
        
        self = cls()
        # commune.launch(module = 'dataset.text.bittensor', name='dataset.text.bittensor')
        
        dataset = commune.connect('dataset.text.bittensor')
        sample = dataset.sample()
        self.forward(**sample)
        # st.write(dataset.sample())
if __name__ == "__main__":
    
    EnsembleModel.sandbox()
    # EnsembleModel.run_neuron()
    # EnsembleModel.test_neuron()
    # print('FUCK')f
    # TransformerModel('gptj', tag='demo', load=True).save_pretrained()
    
    # TransformerModel.run()
    # TransformerModel.experiment()


