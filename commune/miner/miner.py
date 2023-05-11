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

from torch import nn



class Miner(commune.Module, nn.Module):


    
    
    @classmethod
    def start(cls, 
               model_name:str='gptjvr',
               wallet='collective.0',
               network = 'finney',
               netuid=3,
               port = 9269,
               prometheus_port = 8269,
               debug = True,
               no_set_weights = True,
               remote:bool = False,
               tag=None,
               sleep_interval = 2
               ):
        
        
        if tag == None:
            tag = f'{wallet}::{network}::{netuid}'
        if remote:
            kwargs = cls.locals2kwargs(locals())
            kwargs['remote'] = False
            return cls.remote_fn(fn='miner',name=f'miner::{tag}',  kwargs=kwargs)
            

        assert not cls.port_used(port), f'Port {port} is already in use.'
  
        
        config = bittensor.neurons.core_server.neuron.config()
        config.neuron.no_set_weights = no_set_weights
        config.neuron.model_name = model_name
        config.axon.port = port  if port is not None else cls.free_port()
        config.prometheus.port = config.axon.prometheus['port'] = prometheus_port if prometheus_port is not None else cls.free_port()
        config.netuid = netuid
        config.logging.debug = debug
        config.neuron.pretrained = False
        

        subtensor = bittensor.subtensor(network=network)
        bittensor.utils.version_checking()
    
        coldkey, hotkey = wallet.split('.')
        wallet = bittensor.wallet(name=coldkey, hotkey=hotkey)
        
        import time
        
        while not wallet.is_registered(subtensor= subtensor, netuid=  netuid):
            time.sleep(sleep_interval)
            cls.print(f'Pending Registration {wallet} Waiting {sleep_interval}s ...')
            
        cls.print(f'Wallet {wallet} is registered on {network}')
             
             
        bittensor.neurons.core_server.neuron(model=server, 
               wallet=wallet,
               subtensor=subtensor,
               config=config,
               netuid=netuid).run()


     
if __name__ == '__main__':
    Miner.run()

        
