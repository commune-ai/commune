import streamlit as st
import torch
import os,sys
import asyncio
from transformers import AutoConfig
asyncio.set_event_loop(asyncio.new_event_loop())
import bittensor
from typing import List, Union, Optional, Dict
from munch import Munch
import commune as c
class Dashboard(c.Module):
    
    def __init__(self,
                wallet:Union[bittensor.wallet, str] = None,
                subtensor: Union[bittensor.subtensor, str] = 'local',
                ):
        
        self.set_subtensor(subtensor=subtensor)
        self.set_wallet(wallet=wallet)
        
    @property
    def network_options(self):
        return ['finney','nakamoto', 'local', 'nobunaga', '0.0.0.0:9944'] 
    
    @property
    def chain_endpoint_options(self):
        return ['0.0.0.0:9944']
        
    def set_subtensor(self, subtensor=None):
        if isinstance(subtensor, str):
            if subtensor in self.network_options:
                subtensor = bittensor.subtensor(network=subtensor)
            elif ':' in subtensor:
                subtensor = bittensor.subtensor(chain_endpoint=subtensor)
        
        self.subtensor = subtensor if subtensor else bittensor.subtensor()
        self.metagraph = bittensor.metagraph(subtensor=self.subtensor).load()
        
        return self.subtensor
        
    def set_wallet(self, wallet=None)-> bittensor.wallet:
        if isinstance(wallet, str):
            name, hotkey = wallet.split('.')
            wallet =bittensor.wallet(name=name, hotkey=hotkey)
        
        self.wallet = wallet if wallet else bittensor.wallet()
        return self.wallet
    
    @property
    def neuron(self):
        return self.wallet.get_neuron(subtensor=self.subtensor)
        
    
    def list_wallets(self, registered=True, unregistered=True):
        wallet_paths = self.list_wallet_paths()
        wallet_path = os.path.expanduser(self.wallet.config.wallet.path)
        
        wallets = [p.replace(wallet_path, '').replace('/hotkeys/','.') for p in wallet_paths]

        
        return wallets
            

       
    @property
    def network(self):
        return self.subtensor.network
    
    
    @property
    def is_registered(self):
        return self.wallet.is_registered(subtensor= self.subtensor)
       
    
    def sync(self):
        return self.metagraph.sync()
    
    
    # Streamlit Landing Page    
    selected_wallets = []
    def streamlit_sidebar(self):
        wallets_list = self.list_wallets()
        wallet = st.selectbox(f'Select Wallets ({wallets_list[0]})', wallets_list, 0)
        self.set_wallet(wallet)
        
        network_options = self.network_options
        network = st.selectbox(f'Select Network ({network_options[0]})', self.network_options, 0)
        self.set_subtensor(subtensor=network)
        
        sync_network = st.button('Sync the Network')
        if sync_network:
            self.sync()
             
    def streamlit_neuron_metrics(self, num_columns=3):
        with st.expander('Neuron Stats', True):
            cols = st.columns(num_columns)
            if self.is_registered:
                neuron = self.neuron
                for i, (k,v) in enumerate(neuron.__dict__.items()):
                    
                    if type(v) in [int, float]:
                        cols[i % num_columns].metric(label=k, value=v)
                st.write(neuron.__dict__)
            else:
                st.write(f'## {self.wallet} is not Registered on {self.subtensor.network}')

    @classmethod
    def streamlit(cls):
        c.set_page_config(layout="wide")
        self = cls(wallet='fish.1', subtensor='nobunaga')

        with st.sidebar:
            self.streamlit_sidebar()
            
            
        st.write(f'# BITTENSOR DASHBOARD {self.network}')

        self.streamlit_neuron_metrics()
        
        

if __name__ == "__main__":
    Dashboard.streamlit()


    



