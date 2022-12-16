
#!/usr/bin/env python3
##################
##### Import #####
##################
import ray
import torch
import concurrent.futures
import time
import psutil
import random
import sys, os
import argparse
from tqdm import tqdm
import asyncio
import numpy as np
import sys
import pandas as pd
from typing import Optional, Any, Union, Dict, List
from torch.utils.data.dataloader import DataLoader
from copy import deepcopy
import threading
import time
import queue
from munch import Munch
import inspect
from glob import glob
from importlib import import_module
import shutil

asyncio.set_event_loop(asyncio.new_event_loop())
import streamlit as st
import bittensor
sys.path[0] = os.getenv('PWD')

from commune.utils import *


class AxonSandbox:
    def __init__(self, 
                subtensor: Union[Dict[str,Any], bittensor.subtensor]=None,
                wallet: Union[Dict[str,Any], bittensor.wallet] = None,
                receptor_pool: Union[Dict[str,Any], bittensor.receptor_pool] = None,
                load:bool = True,
                config=None):

        # Avoid deadlocks when usinng tokenizer.

        self.config = self.load_config(config=config)

        self.subtensor = self.set_subtensor(subtensor)
        self.wallet = self.set_wallet(wallet)
        self.receptor_pool =self.set_receptor_pool(receptor_pool=None)
        self.axon = bittensor.axon_general(ip='0.0.0.0',wallet=self.wallet)
        self.axon.start()
    def set_subtensor(self, subtensor=None):

        if subtensor == None:
            subtensor = bittensor.subtensor( config = self.config.bittensor )
            graph = bittensor.metagraph( subtensor = subtensor )
            graph.load()
            self.subtensor = subtensor
            self.graph = graph
        if self.sync_delay > self.config.get('delay_threshold', 100):
            self.graph.sync()
            self.graph.save()

        return self.subtensor
    
    
    @property
    def sample_probability(self):
        # sample inversly proportional
        max_sample_size =self.max_sample_size
        sample_size_array = self.sample_size_array
        pre_normalized = (max_sample_size - sample_size_array)**2 +  1e-10
        normalized = pre_normalized / np.sum(pre_normalized)

        return normalized

    @property
    def sampleidx2size(self):
        x = {k:len(v) for k,v in self.sampleidx2result.items()}
        return x

    @property
    def sample_size_array(self):
        return np.array(list(self.sampleidx2size.values()))
    @property
    def total_sample_size(self):
        return int(np.sum(self.sample_size_array))

    @property
    def max_sample_size(self):
        return int(np.max(self.sample_size_array))

    @property
    def min_sample_size(self):
        return int(np.min(self.sample_size_array))

    @property
    def split(self):
        if 'split' not in self.config:
            self.config['split'] = 'train'
        return self.config['split']

    def set_tokenizer(self):
        tokenizer = self.launch(**self.config['tokenizer'])
        return tokenizer

    def set_receptor_pool(self, receptor_pool=None):
        rp_config = deepcopy(self.config['receptor_pool'])
        rp_config['kwargs']['wallet']=self.wallet

        if receptor_pool == None:
            receptor_pool = self.launch( **rp_config)  
        self.receptor_pool = receptor_pool

        return self.receptor_pool


    def set_wallet(self,wallet):
        wallet = wallet if wallet else self.config.wallet
        if isinstance(wallet, dict):
            self.wallet = bittensor.wallet(**wallet)
        elif isinstance(wallet, bittensor.wallet):
            self.wallet = wallet
        else:
            raise NotImplemented(f'{type(wallet)} type of wallet is not available')
    
        return self.wallet

 

    @staticmethod
    def str2synapse(synapse:str, *args, **kwargs):
        return getattr(bittensor.synapse, synapse)(*args, **kwargs)
    @property
    def available_synapses(self):
        return [f for f in dir(bittensor.synapse) if f.startswith('Text')]

    synapses = all_synapses=available_synapses
    @staticmethod
    def errorcode2name(code):
        code2name_map =  {k:f'{v}' for k,v in zip(bittensor.proto.ReturnCode.values(),bittensor.proto.ReturnCode.keys())}
        return code2name_map[code]

    @property
    def current_block(self):
        return self.subtensor.block
    
    @property
    def synced_block(self): 
        return self.graph.block.item()
    @property
    def sync_delay(self):
        return self.current_block - self.synced_block
    

    @classmethod
    def launch(cls, module:str, fn:Optional[str]=None ,kwargs:dict={}, args:list=[]):
        module_class = cls.import_object(module)
        module_kwargs = {**kwargs}
        module_args = [*args]
        
        if fn == None:
            module_object =  module_class(*module_args,**module_kwargs)
        else:
            module_init_fn = getattr(module_class,fn)
            module_object =  module_init_fn(*module_args, **module_kwargs)
        return module_object


    def resolve_synapse(self, synapse:str, *args,**kwarga):
        return getattr(bittensor.synapse, synapse)()

    @property
    def __file__(self):
        module_path =  inspect.getmodule(self).__file__
        return module_path

    @property
    def __config_file__(self):
        return self.__file__.replace('.py', '.yaml')

    def load_config(self, config:Optional[Union[str, dict]]=None):
        if config == None:
            config = load_yaml(self.__config_file__)
        elif isinstance(config, str):
            config =  load_yaml(config)
        elif isinstance(config, dict):
            config = config
        
        config = Munch(config)

        # Add bittensor in there.
        config.bittensor =  bittensor.config(parser = self.argparser())
        
        return config

    @classmethod
    def import_module(cls, import_path:str) -> 'Object':
        return import_module(import_path)

    @classmethod
    def import_object(cls, key:str)-> 'Object':
        module = '.'.join(key.split('.')[:-1])
        object_name = key.split('.')[-1]
        obj =  getattr(import_module(module), object_name)
        return obj


    def argparser(self) -> 'argparse.ArgumentParser':
        parser = argparse.ArgumentParser( 
            description=f"Bittensor Speed Test ",
            usage="python3 speed.py <command args>",
            add_help=True
        )
        bittensor.wallet.add_args(parser)
        bittensor.logging.add_args(parser)
        bittensor.subtensor.add_args(parser)
        return parser

    @property
    def module_path(self):
        local_path = self.__file__.replace(os.getenv('PWD'), '')
        local_path, local_path_ext = os.path.splitext(local_path)
        module_path ='.'.join([local_path.replace('/','.')[1:], self.__class__.__name__])
        return module_path


    @property
    def tmp_dir_path(self) -> str:
        '''
        The temporary directory path for storing any artifacts.

        Returns:
            tmp_dir_pat (str)
        '''

        # Remove the PWD.
        tmp_dir_path =  '/tmp' + self.__file__.replace(os.getenv('PWD'), '')

        # Remove the extension.
        tmp_dir_path, tmp_dir_path_ext = os.path.splitext(tmp_dir_path)
        return tmp_dir_path

    def resolve_path(self, path):
        if self.tmp_dir_path not in path:
            path = os.path.join(self.tmp_dir_path, path)
        path = ensure_path(path)
        return path

    async def async_save_json(self,path:str, data:Union[list, dict], use_tmp_dir:bool=True) -> 'NoneType':
        if use_tmp_dir:
            path = self.resolve_path(path)
        return await async_save_json(path, data)

    async def async_load_json(self,path:str, use_tmp_dir:bool=True, handle_error:bool=False) -> 'NoneType':
        if use_tmp_dir:
            path = self.resolve_path(path)

        try:
            data = await async_load_json(path)
        except FileNotFoundError as e:
            if handle_error:
                return None
            else:
                raise e
        
        return data

    def rm_json(self,path:str ,use_tmp_dir:bool=True) -> Union['NoneType', str]:
        
        if use_tmp_dir:
            path = os.path.join(self.tmp_dir_path, path)
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.isfile(path):
                path = ensure_json_path(path, ensure_extension=True, ensure_directory=False)
                os.remove(path)
            else:
                raise NotImplementedError(path)
            
            return path
        else:
            return None
    def glob_paths(self, query:Optional[str]=None) -> List[str]: 
        sample_paths = [f for f in glob(self.tmp_dir_path+'/**', recursive=True) if os.path.isfile(f)]
        if isinstance(query, str):
            sample_paths = [f for f in sample_paths if query in f]
        return sample_paths






if __name__ == '__main__':

    axon = AxonSandbox()
    st.write(axon.axon )
    
    # metrics = dict(
    #         total_bin_size = module.total_sample_size,
    #         min_bin_size = module.min_sample_size, 
    #         max_bin_size = module.max_sample_size,
    #         num_samples = len(module.sampleidx2result)
            
    #         )
    # metrics['mean_bin_size'] = metrics['total_bin_size'] / metrics['num_samples']

    # st.write(metrics)

        



    # for step in range(10):
    #     inputs = next(data)

    #     rpool = bittensor.receptor_pool(
    #         wallet=wallet, max_active_receptors=len(endpoints)
    #     )

    #     results = rpool.forward(endpoints, synapses=synapses, inputs=[inputs] * len(endpoints),
    #                             timeout=12)

    #     st.write(results)
    #     del rpool

    #     hidden_states = []
    #     causals = []
    #     codes = defaultdict(tuple)
    #     for peer in range(len(endpoints)):
    #         hidden_states.append(results[0][peer][0][:, -4:, :])
    #         causals.append(results[0][peer][1][:, -4:, :])
    #         codes[peer] = (results[1][peer][0], results[1][peer][1])
    