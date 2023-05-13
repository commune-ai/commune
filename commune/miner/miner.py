# import nest_asyncio
# nest_asyncio.apply()
import commune
commune.new_event_loop()
import bittensor
import streamlit as st
import torch
from typing import Dict, List, Union, Any
import random
from copy import deepcopy
import asyncio
from munch import Munch
from bittensor.utils.tokenizer_utils import prep_tokenizer, get_translation_map, translate_logits_to_probs_std, \
    translate_special_token_text, pad_offsets, topk_token_phrases, compact_topk_token_phrases
import time
from torch import nn
import os


class Miner(commune.Module, nn.Module):


    
    
    @classmethod
    def mine(cls, 
               wallet='ensemble.Hot5',
               model_name:str=os.path.expanduser('~/models/gpt-j-6B-vR'),
               network = 'finney',
               netuid=3,
               port = None,
               device = None,
               prometheus_port = None,
               debug = True,
               no_set_weights = True,
               remote:bool = False,
               tag=None,
               sleep_interval = 2,
               autocast = True,
               ):
        
        
        if tag == None:
            tag = f'{wallet}::{network}::{netuid}'
        if remote:
            kwargs = cls.locals2kwargs(locals())
            kwargs['remote'] = False
            return cls.remote_fn(fn='mine',name=f'miner::{tag}',  kwargs=kwargs)
            
        if port == None:
            port = cls.free_port()
        assert not cls.port_used(port), f'Port {port} is already in use.'
  
        
        config = bittensor.neurons.core_server.neuron.config()
        
        # model things
        config.neuron.no_set_weights = no_set_weights
        config.neuron.model_name = model_name
        
        if device is None:
            device = cls.most_free_gpu()
        
        assert torch.cuda.is_available(), 'No CUDA device available.'
        config.neuron.device = f'cuda:{device}'
        config.neuron.autocast = autocast
        
        # axon port
        port = port  if port is not None else cls.free_port()
        config.axon.port = port
        assert not cls.port_used(config.axon.port), f'Port {config.axon.port} is already in use.'
        
        # prometheus port
        config.prometheus.port =  port + 1 if prometheus_port is None else prometheus_port
        while cls.port_used(config.prometheus.port):
            config.prometheus.port += 1
            
            
        config.axon.prometheus.port = config.prometheus.port
        config.netuid = netuid
        config.logging.debug = debug


        # network
        subtensor = bittensor.subtensor(network=network)
        bittensor.utils.version_checking()
    
        # wallet
        coldkey, hotkey = wallet.split('.')
        wallet = bittensor.wallet(name=coldkey, hotkey=hotkey)
        
        cls.print('Config: ', config)
        # wait for registration
        while not wallet.is_registered(subtensor= subtensor, netuid=  netuid):
            time.sleep(sleep_interval)
            cls.print(f'Pending Registration {wallet} Waiting {sleep_interval}s ...')
            
        cls.print(f'Wallet {wallet} is registered on {network}')
             
             
        
        bittensor.neurons.core_server.neuron(
               wallet=wallet,
               subtensor=subtensor,
               config=config,
               netuid=netuid).run()


        
if __name__ == '__main__':
    Miner.run()

        


