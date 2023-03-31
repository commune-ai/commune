import streamlit as st
import torch
import os,sys
import asyncio
from transformers import AutoConfig
asyncio.set_event_loop(asyncio.new_event_loop())

import bittensor
import commune
from typing import List, Union, Optional, Dict
from munch import Munch

class BittensorModule(commune.Module):
    
    def __init__(self,

                wallet:Union[bittensor.wallet, str] = None,
                subtensor: Union[bittensor.subtensor, str] = 'finney',
                register: bool = False
                ):
        
        self.set_subtensor(subtensor=subtensor)
        self.set_wallet(wallet=wallet)
        
        
    @property
    def network_options(self):
        network_options = ['finney','nakamoto', 'nobunaga', '0.0.0.0:9944'] 
        if os.getenv('SUBTENSOR', None) is not None:
            network_options.append(os.getenv('SUBTENSOR'))
            
        return network_options
        
        
        
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
        self.wallet = self.get_wallet(wallet)
        return self.wallet
    
    def get_wallet(self, wallet=None):
        if isinstance(wallet, str):
            if len(wallet.split('.')) == 2:
                name, hotkey = wallet.split('.')
                wallet =bittensor.wallet(name=name, hotkey=hotkey)
            elif len(wallet.split('.')) == 1:
                wallet = bittensor.wallet(name=wallet)
            else:
                raise NotImplementedError(wallet)
        if not hasattr(self, 'wallet'):
            self.wallet = bittensor.wallet()
            
        wallet = wallet if wallet else self.wallet
        return wallet 
    
    def get_neuron(self, wallet=None):
        wallet = self.get_wallet(wallet)
        return wallet.get_neuron(subtensor=self.subtensor)
    
    
    def get_port(self, wallet=None):
        return self.get_neuron(wallet=wallet).port

    def get_info(self, wallet=None, key=None):
        return self.get_neuron(wallet=wallet).port
    
    
    @property
    def neuron(self):
        return self.wallet.get_neuron(subtensor=self.subtensor)
        
    
    def list_wallet_paths(self, registered=False):
        wallet_path = os.path.expanduser(self.wallet.config.wallet.path)
        wallet_list = []
        import glob
        st.write(wallet_path)
        return glob.glob(wallet_path+'/*/hotkeys/*')
    
    def list_wallets(self, registered=True, unregistered=True, output_wallet:bool = True):
        wallet_paths = self.list_wallet_paths()
        wallet_path = os.path.expanduser(self.wallet.config.wallet.path)
        wallets = [p.replace(wallet_path, '').replace('/hotkeys/','.') for p in wallet_paths]

        if output_wallet:
            wallets = [self.get_wallet(w) for w in wallets]
        return wallets
    
    @property
    def default_network(self):
        return self.network_options[0]
    
    @property
    def default_wallet(self):
        return self.list_wallets()[0]
            
    selected_wallets = []
    def streamlit_sidebar(self):
        wallet = st.selectbox(f'Select Wallets ({self.default_wallet})', wallets_list, 0)
        self.set_wallet(wallet)
        
        network_options = self.network_options
        network = st.selectbox(f'Select Network ({self.default_network})', self.network_options, 0)
        self.set_subtensor(subtensor=network)
        
        sync_network = st.button('Sync the Network')
        if sync_network:
            self.sync()
       
    @property
    def network(self):
        return self.subtensor.network
    
    
    @property
    def is_registered(self):
        return self.wallet.is_registered(subtensor= self.subtensor)
        
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
    
    def sync(self):
        return self.metagraph.sync()
    
    @classmethod
    def dashboard(cls):
        st.set_page_config(layout="wide")
        self = cls(wallet='fish', subtensor='nobunaga')

        with st.sidebar:
            self.streamlit_sidebar()
            
        st.write(f'# BITTENSOR DASHBOARD {self.network}')
        # wallets = self.list_wallets(output_wallet=True)
        
        commune.print(commune.run_command('pm2 status'), 'yellow')
        st.write(commune.run_command('pm2 status').split('\n'))
        # st.write(wallets[0].__dict__)
        
        # self.register()
        # st.write(self.run_miner('fish', '100'))

        # self.streamlit_neuron_metrics()
        
    def run_miner(self, 
                coldkey='fish',
                hotkey='1', 
                port=None,
                subtensor = "194.163.191.101:9944",
                interpreter='python3',
                refresh: bool = False):
        
        name = f'miner_{coldkey}_{hotkey}'
        
        wallet = self.get_wallet(f'{coldkey}.{hotkey}')
        neuron = self.get_neuron(wallet)
        
     
        
        try:
            import cubit
        except ImportError:
            commune.run_command('pip install https://github.com/opentensor/cubit/releases/download/v1.1.2/cubit-1.1.2-cp310-cp310-linux_x86_64.whl')
        if port == None:
            port = neuron.port
    
        
        if refresh:
            commune.pm2_kill(name)
            
        
        assert commune.port_used(port) == False, f'Port {port} is already in use'
        command_str = f"pm2 start commune/model/client/model.py --name {name} --time --interpreter {interpreter} --  --logging.debug  --subtensor.chain_endpoint {subtensor} --wallet.name {coldkey} --wallet.hotkey {hotkey} --axon.port {port}"
        # return commune.run_command(command_str)
        st.write(command_str)
          
          
          
    
    def ensure_env(self):

        try:
            import bittensor
        except ImportError:
            commune.run_command('pip install bittensor')
            
        return cubit
    
    
        try:
            import cubit
        except ImportError:
            commune.run_command('pip install https://github.com/opentensor/cubit/releases/download/v1.1.2/cubit-1.1.2-cp310-cp310-linux_x86_64.whl')
            
    

    @property
    def default_subnet(self):
        return 3
        

    def register ( 
            self, 
            wallet = None,
            netuid = None,
            subtensor: 'bittensor.Subtensor' = None, 
            wait_for_inclusion: bool = False,
            wait_for_finalization: bool = True,
            prompt: bool = False,
            max_allowed_attempts: int = 3,
            cuda: bool = True,
            dev_id: Union[int, List[int]] = None,
            TPB: int = 256,
            num_processes: Optional[int] = None,
            update_interval: Optional[int] = 50_000,
            output_in_place: bool = True,
            log_verbose: bool = True,
        ) -> 'bittensor.Wallet':
        """ Registers the wallet to chain.
        Args:
            subtensor( 'bittensor.Subtensor' ):
                Bittensor subtensor connection. Overrides with defaults if None.
            wait_for_inclusion (bool):
                If set, waits for the extrinsic to enter a block before returning true, 
                or returns false if the extrinsic fails to enter the block within the timeout.   
            wait_for_finalization (bool):
                If set, waits for the extrinsic to be finalized on the chain before returning true,
                or returns false if the extrinsic fails to be finalized within the timeout.
            prompt (bool):
                If true, the call waits for confirmation from the user before proceeding.
            max_allowed_attempts (int):
                Maximum number of attempts to register the wallet.
            cuda (bool):
                If true, the wallet should be registered on the cuda device.
            dev_id (int):
                The cuda device id.
            TPB (int):
                The number of threads per block (cuda).
            num_processes (int):
                The number of processes to use to register.
            update_interval (int):
                The number of nonces to solve between updates.
            output_in_place (bool):
                If true, the registration output is printed in-place.
            log_verbose (bool):
                If true, the registration output is more verbose.
        Returns:
            success (bool):
                flag is true if extrinsic was finalized or uncluded in the block. 
                If we did not wait for finalization / inclusion, the response is true.
        """
        # Get chain connection.
        if subtensor == None: subtensor = self.subtensor
        
        netuid = netuid if netuid is not None else self.default_subnet
        dev_id = dev_id if dev_id is not None else self.gpus()
        wallet = wallet if wallet is not None else self.wallet
        
        subtensor.register(
            wallet = wallet,
            netuid = netuid,
            wait_for_inclusion = wait_for_inclusion,
            wait_for_finalization = wait_for_finalization,
            prompt=prompt, max_allowed_attempts=max_allowed_attempts,
            output_in_place = output_in_place,
            cuda=cuda,
            dev_id=dev_id,
            TPB=TPB,
            num_processes=num_processes,
            update_interval=update_interval,
            log_verbose=log_verbose,
        )
        
        return self
  
  
    @classmethod
    def register_loop(cls, *args, **kwargs):
        import commune
        # commune.new_event_loop()
        self = cls(*args, **kwargs)
        wallets = self.list_wallets()
        for wallet in wallets:
            # print(wallet)
            self.set_wallet(wallet)
            self.register(dev_id=commune.gpus())
            
    @classmethod
    def create_wallets(cls, 
                       coldkeys: List[str] = [f'ensemble_{i}' for i in range(3)],
                       hotkeys : List[str] = [f'{i}' for i in range(10)],
                       coldkey_use_password:bool = False, 
                       hotkey_use_password:bool = False
                       ):
        
        
        for ck in coldkeys:
            for hk in hotkeys:
                bittensor.wallet(name=ck, hotkey=hk).create(coldkey_use_password=coldkey_use_password, 
                                                            hotkey_use_password=hotkey_use_password)     
                
    @classmethod
    def create_wallet(cls, 
                       coldkey: str = [f'ensemble_{i}' for i in range(3)],
                       hotkey : str = [f'{i}' for i in range(10)],
                       coldkey_use_password:bool = False, 
                       hotkey_use_password:bool = False
                       ) :
        return  bittensor.wallet(name=ck, hotkey=hk).create(coldkey_use_password=coldkey_use_password, 
                                                            hotkey_use_password=hotkey_use_password)     
            
            
            
    @classmethod
    def register_wallet(
                        cls, 
                        dev_id: Union[int, List[int]] = None, 
                        wallet='ensemble_0.0',
                        **kwargs
                        ):
        cls(wallet=wallet).register(dev_id=dev_id, **kwargs)
    
                        
    @classmethod  
    def sandbox(cls):
        
        gpus = commune.gpus()
        processes_per_gpus = 2
        
        for i in range(processes_per_gpus):
            for dev_id in gpus:
                cls.launch(fn='register_wallet', name=f'reg.{i}.gpu{dev_id}', kwargs=dict(dev_id=dev_id), mode='pm2')
                
        
        # print(cls.launch(f'register_{1}'))
        # self = cls(wallet=None)
        # self.create_wallets()
        # # st.write(dir(self.subtensor))
        # st.write(self.register(dev_id=0))
if __name__ == "__main__":
    BittensorModule.run()


    



