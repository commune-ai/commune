
import torch
import os,sys
import asyncio
from transformers import AutoConfig
import commune as c
c.new_event_loop()

import bittensor
from typing import List, Union, Optional, Dict
from munch import Munch
import time
import streamlit as st

class BittensorModule(c.Module):
    wallets_path = os.path.expanduser('~/.bittensor/wallets/')
    
    def __init__(self,

                wallet:Union[bittensor.wallet, str] = None,
                network: Union[bittensor.subtensor, str] = 'finney',
                netuid: int = 3,
                ):
        
        self.set_subtensor(subtensor=network)
        self.set_netuid(netuid=netuid)
        
    @classmethod
    def network_options(cls):
        network_options = ['finney', 'test', 'local'] 

            
        return network_options
    
    
    def set_netuid(self, netuid: int = None):
        assert isinstance(netuid, int)
        self.netuid = netuid
        return self.netuid
    
    network2endpoint = {
        'test': 'wss://test.finney.opentensor.ai:443',
        'local': 'ws://0.0.0.0:9944',
        'finney': 'wss://entrypoint-finney.opentensor.ai:443',
    }
    @classmethod
    def get_endpoint(cls, network:str):
        return cls.network2endpoint.get(network, None)
       
       
    @classmethod
    def is_endpoint(cls, endpoint):
        # TODO: check if endpoint is valid, can be limited to just checking if it is a string
        return bool(':' in endpoint and cls.is_number(endpoint.split(':')[-1]))
      
    @classmethod
    def get_subtensor(cls, subtensor:Union[str, bittensor.subtensor]='finney') -> bittensor.subtensor:
        
        if  subtensor == None:
            subtensor = bittensor.subtensor()
        elif isinstance(subtensor, str):
            if cls.is_endpoint(subtensor):
                subtensor = bittensor.subtensor(chain_endpoint=subtensor)
            elif subtensor in cls.network2endpoint: 
                endpoint = cls.network2endpoint[subtensor]
                subtensor = bittensor.subtensor(chain_endpoint=endpoint)
            else:
                raise NotImplementedError(subtensor)
            
        elif isinstance(subtensor, type(None)):
            subtensor = bittensor.subtensor()
        elif isinstance(subtensor, bittensor.Subtensor):
            subtensor = subtensor
        else:
            raise NotImplementedError(subtensor)
        return subtensor
    
    
    def set_subtensor(self, subtensor=None):
         
        self.subtensor = self.get_subtensor(subtensor)
        self.metagraph = bittensor.metagraph(subtensor=self.subtensor).load()
        
        return self.subtensor
        
    def set_wallet(self, wallet=None)-> bittensor.Wallet:
        ''' Sets the wallet for the module.'''
        self.wallet = self.get_wallet(wallet)
        self.wallet.create(False, False)
        return self.wallet
    
    @classmethod
    def get_wallet(cls, wallet:Union[str, bittensor.wallet]='ensemble.1') -> bittensor.wallet:
        
        if isinstance(wallet, str):
            if len(wallet.split('.')) == 2:
                name, hotkey = wallet.split('.')
            elif len(wallet.split('.')) == 1:
                name = wallet
                hotkey = cls.hotkeys(name)[0]
            else:
                raise NotImplementedError(wallet)
                
            wallet =bittensor.wallet(name=name, hotkey=hotkey)
        elif isinstance(wallet, bittensor.Wallet):
            wallet = wallet
        else:
            raise NotImplementedError(wallet)

        return wallet 
    def resolve_subtensor(self, subtensor: 'Subtensor' = None) -> 'Subtensor':
        if isinstance(subtensor, str):
            subtensor = self.get_subtensor(subtensor)
        if subtensor is None:
            subtensor = self.subtensor
        return subtensor
    

    def resolve_netuid(self, netuid: int = None) -> int:
        if netuid is None:
            netuid = self.netuid
        return netuid
    default_netuid = 3
    @classmethod
    def get_netuid(cls, netuid: int = None) -> int:
        if netuid is None:
            netuid = cls.default_netuid
        return netuid
    
    
    @classmethod
    def get_neuron(cls, wallet=None, netuid: int = None, subtensor=None):
        wallet = cls.get_wallet(wallet)
        netuid = cls.get_netuid(netuid)
        subtensor = cls.get_subtensor(subtensor)
        neuron_info = wallet.get_neuron(subtensor=subtensor, netuid=netuid)
        if neuron_info is None:
            neuron_info = cls.munch({'axon_info': {}, 'prometheus_info': {}})
            
        return neuron_info
    
    @classmethod
    def wallet2neuron(cls, *args, **kwargs):
        kwargs['registered'] = True
        wallet2neuron = {}
        
        async def async_get_neuron(w):
            neuron_info = cls.get_neuron(wallet=w)
            return neuron_info
        
        wallets = cls.wallets(*args, **kwargs)
        jobs = [async_get_neuron(w) for w in wallets]
        neurons = cls.gather(jobs)
        
        wallet2neuron = {w:n for w, n in zip(wallets, neurons)}
            
            
        return wallet2neuron
    
    
    @classmethod
    def wallet2axon(cls, *args, **kwargs):

        wallet2neuron = cls.wallet2neuron(*args, **kwargs)
        wallet2axon = {w:n.axon_info for w, n in wallet2neuron.items()}
            
            
        return wallet2axon
    
    w2a = wallet2axon
    
    @classmethod
    def wallet2port(cls, *args, **kwargs):

        wallet2neuron = cls.wallet2neuron(*args, **kwargs)
        wallet2port = {w: n.axon_info.port for w, n in wallet2neuron.items()}
            
            
        return wallet2port
    
    w2p = wallet2port
    
    
    @classmethod
    def wallet2stats(cls, *args, **kwargs):
        kwargs['registered'] = True
        wallet2neuron = {}
        for w  in cls.wallets(*args, **kwargs):
            wallet2neuron[w] = cls.get_neuron(wallet=w)
            
        return wallet2neuron
    
    w2n = wallet2neuron
            
    
    get_neuron = get_neuron
    
    @classmethod
    def get_axon(cls, wallet=None, netuid: int = None, subtensor=None):
        neuron_info = cls.get_neuron(wallet=wallet, netuid=netuid, subtensor=subtensor)
        axon_info = neuron_info.axon_info
        return axon_info
    
    @classmethod
    def get_prometheus(cls, wallet=None, netuid: int = None, subtensor=None):
        subtensor = cls.get_subtensor(subtensor)
        neuron_info= cls.get_neuron(wallet=wallet, netuid=netuid)
            
        prometheus_info = neuron_info.prometheus_info
        return prometheus_info

    
    
    @property
    def neuron_info(self):
        return self.get_neuron(subtensor=self.subtensor, netuid=self.netuid, wallet=self.wallet)
    
    @property
    def axon_info(self):
        return self.get_axon(subtensor=self.subtensor, netuid=self.netuid, wallet=self.wallet)
        
    @property
    def prometheus_info(self):
        return self.get_prometheus(subtensor=self.subtensor, netuid=self.netuid, wallet=self.wallet)
        
    # def get_axon_port(self, wallet=None, netuid: int = None):
    #     netuid = self.resolve_netuid(netuid)
    #     return self.get_neuron(wallet=wallet, netuid=netuid ).axon_info.port

    # def get_prometheus_port(self, wallet=None, netuid: int = None):
    #     netuid = self.resolve_netuid(netuid)
    #     return self.get_neuron(wallet=wallet, netuid=netuid ).axon_info.port

    
    
    @classmethod
    def walk(cls, path:str) -> List[str]:
        import os
        path_list = []
        for root, dirs, files in os.walk(path):
            if len(dirs) == 0 and len(files) > 0:
                for f in files:
                    path_list.append(os.path.join(root, f))
        return path_list
    @classmethod
    def wallet_paths(cls):
        wallet_list =  cls.ls(cls.wallets_path, recursive=True)
        sorted(wallet_list)
        return wallet_list
    
    @classmethod
    def wallets(cls, search = None, registered=False, subtensor='finney', netuid:int=None):
        wallets = []
        if registered:
            subtensor = cls.get_subtensor(subtensor)
            netuid = cls.get_netuid(netuid)
        for c in cls.coldkeys():
            for h in cls.hotkeys(c):
                wallet = f'{c}.{h}'
                if registered:
                    if not cls.is_registered(wallet, subtensor=subtensor, netuid=netuid):
                        continue
                    
                
                if search is not None:
                    if search not in wallet:
                        continue
                    
                wallets.append(wallet)
                
        wallets = sorted(wallets)
        return wallets
    
    @classmethod
    def registered_wallets(cls, search=None,  subtensor='finney', netuid:int=None):
        wallets =  cls.wallets(search=search,registered=True, subtensor=subtensor, netuid=netuid)
        return wallets

    reged = registered_wallets
    @classmethod
    def unregistered_wallets(cls, search=None,  subtensor='finney', netuid:int=None):
        wallets =  cls.wallets(search=search,registered=False, subtensor=subtensor, netuid=netuid)
        registered_wallets = cls.registered_wallets(search=search, subtensor=subtensor, netuid=netuid)
        unregistered_wallets = [w for w in wallets if w not in registered_wallets]
        return unregistered_wallets
    
    unreged = unregistered_wallets
    
    
    @classmethod
    def wallet2path(cls, search = None):
        wallets = cls.wallets()
        wallet2path = {}
        for wallet in wallets:
            ck, hk = wallet.split('.')
            if search is not None:
                if search not in wallet:
                    continue
                    
            wallet2path[wallet] = os.path.join(cls.wallets_path, ck, 'hotkeys', hk)

        
        return wallet2path
    
    @classmethod
    def get_wallet_path(self, wallet=None):
        if wallet is None:
            wallet = self.wallet
        return self.wallet2path()[wallet]
    
    @classmethod
    def rm_wallet(cls, wallet):
        wallet2path = cls.wallet2path()
        assert wallet in wallet2path, f'Wallet {wallet} not found in {wallet2path.keys()}'
        cls.rm(wallet2path[wallet])
     
        return cls.wallets()
    
    @classmethod
    def rm_coldkey(cls, coldkey):
        
        assert coldkey in cls.coldkeys(), f'Coldkey {coldkey} not found in {cls.coldkeys()}'
        coldkey_path = os.path.join(cls.wallets_path, coldkey)
        assert os.path.exists(coldkey_path), f'Coldkey path {coldkey_path} does not exist'
        return cls.rm(coldkey_path)
    
    @classmethod
    def hotkeys(cls, wallet='default'):
        coldkeys = cls.coldkeys()
        assert wallet in coldkeys, f'Wallet {wallet} not found in {coldkeys}'
        return  [os.path.basename(p) for p in cls.ls(os.path.join(cls.wallets_path, wallet, 'hotkeys'))]
        
    @classmethod
    def coldkeys(cls, wallet='default'):
        
        return  [os.path.basename(p)for p in cls.ls(cls.wallets_path)]

        
    def coldkey_exists(cls, wallet='default'):
        return [os.path.basename(p)for p in cls.ls(cls.wallets_path)]
    
    @classmethod
    def list_wallets(cls, registered=True, unregistered=True, output_wallet:bool = True):
        wallet_paths = cls.wallet_paths()
        wallets = [p.replace(cls.wallets_path, '').replace('/hotkeys/','.') for p in wallet_paths]

        if output_wallet:
            wallets = [cls.get_wallet(w) for w in wallets]
            
        return wallets
    
    @classmethod
    def wallet_exists(cls, wallet:str):
        wallets = cls.wallets()
        return bool(wallet and wallets)
    
    @classmethod
    def hotkey_exists(cls, coldkey:str, hotkey:str) -> bool:
        hotkeys = cls.hotkeys(coldkey)
        return bool(hotkey in hotkeys)
    
    @classmethod
    def coldkey_exists(cls, coldkey:str) -> bool:
        coldkeys = cls.coldkeys()
        return bool(coldkey in coldkeys)
    
    
    
    @property
    def default_network(self):
        return self.network_options()[0]
    
    @property
    def default_wallet(self):
        return self.list_wallets()[0]
              
    @property
    def network(self):
        return self.subtensor.network
    @classmethod
    def is_registered(cls, wallet = None, netuid: int = None, subtensor: 'Subtensor' = None):
        netuid = cls.get_netuid(netuid)
        wallet = cls.get_wallet(wallet)
        subtensor = cls.get_subtensor(subtensor)
        return wallet.is_registered(subtensor= subtensor, netuid=  netuid)

    @property
    def registered(self):
        return self.is_registered(wallet=self.wallet, netuid=self.netuid, subtensor=self.subtensor)
    def sync(self, netuid=None):
        netuid = self.resolve_netuid(netuid)
        return self.metagraph.sync(netuid=netuid)
    
    def wait_until_registered(self, netuid: int = None, wallet: 'Wallet'=None, interval:int=60):
        seconds_waited = 0
        # loop until registered.
        while not self.is_registered( netuid=netuid, wallet=wallet, subtensor=self.subtensor):
            # sleep then sync
            self.print(f'Waiting for registering {seconds_waited} seconds', color='purple')
            self.sleep(interval)
            seconds_waited += interval
            self.sync(netuid=netuid)

            
    # @classmethod
    # def dashboard(cls):
        
    #     st.set_page_config(layout="wide")
    #     self = cls(wallet='collective.0', network='finney')

    #     with st.sidebar:
    #         self.streamlit_sidebar()
            
    #     st.write(f'# BITTENSOR DASHBOARD {self.network}')
    #     wallets = self.list_wallets(output_wallet=True)
        
    #     st.write(wallets[0].__dict__)
        
    #     # self.register()
    #     # st.write(self.run_miner('fish', '100'))

    #     # self.streamlit_neuron_metrics()
    

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
            c.run_command('pip install https://github.com/opentensor/cubit/releases/download/v1.1.2/cubit-1.1.2-cp310-cp310-linux_x86_64.whl')
        if port == None:
            port = neuron.port
    
        
        if refresh:
            c.pm2_kill(name)
            
        
        assert c.port_used(port) == False, f'Port {port} is already in use'
        command_str = f"pm2 start c/model/client/model.py --name {name} --time --interpreter {interpreter} --  --logging.debug  --subtensor.chain_endpoint {subtensor} --wallet.name {coldkey} --wallet.hotkey {hotkey} --axon.port {port}"
        # return c.run_command(command_str)
        st.write(command_str)
          
          
          
    
    def ensure_env(self):

        try:
            import bittensor
        except ImportError:
            c.run_command('pip install bittensor')
            
        return cubit
    
    
        try:
            import cubit
        except ImportError:
            c.run_command('pip install https://github.com/opentensor/cubit/releases/download/v1.1.2/cubit-1.1.2-cp310-cp310-linux_x86_64.whl')
            
    

    @property
    def default_subnet(self):
        return 3
        
    @classmethod
    def resolve_dev_id(cls, dev_id: Union[int, List[int]] = None):
        if dev_id is None:
            dev_id = c.gpus()
            
        return dev_id
    
    def resolve_wallet(self, wallet=None):
        if isinstance(wallet, str):
            wallet = self.get_wallet(wallet)
        if wallet is None:
            wallet = self.wallet
        return wallet


    def resolve_wallet_name(self, wallet=None):
        if isinstance(wallet, str):
            wallet = self.get_wallet(wallet)
        if wallet is None:
            wallet = self.wallet
        wallet_name = f'{wallet.name}.{wallet.hotkey_str}'
        return wallet

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
            remote: bool = False,
             
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
        if cuda:
            assert self.cuda_available()
        # Get chain connection.
        subtensor = self.resolve_subtensor(subtensor)
        netuid = self.resolve_netuid(netuid)
        dev_id = self.resolve_dev_id(dev_id)
        wallet = self.resolve_wallet(wallet)
        
        
        self.print(f'Registering wallet: {wallet.name}::{wallet.hotkey} on {netuid}', 'yellow')
        
        register_kwargs = dict(
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
                            wallet=wallet
                        )
        if remote:
            self.launch(fn='register_wallet', 
                        name = f'register::{wallet.name}::{wallet.hotkey}',
                        kwargs=register_kwargs)
            
        else:
            subtensor.register(**register_kwargs)
        
        return self
  
  
    @classmethod
    def register_loop(cls, *args, **kwargs):
        # c.new_event_loop()
        self = cls(*args, **kwargs)
        wallets = self.list_wallets()
        for wallet in wallets:
            # print(wallet)
            self.set_wallet(wallet)
            self.register(dev_id=c.gpus())
            
            
    @classmethod
    def create_wallets_from_dict(cls, 
                                 wallets: Dict,
                                 overwrite: bool = True):
        
        '''
        wallet_dict = {
            'coldkey1': { 'hotkeys': {'hk1': 'mnemonic', 'hk2': 'mnemonic2'}},
        '''
        wallets = {}
        for coldkey_name, hotkey_dict in wallet_dict.items():
            bittensor.wallet(name=coldkey_name).create_from_mnemonic(coldkey, overwrite=overwrite)
            wallets = {coldkey_name: {}}
            for hotkey_name, mnemonic in hotkey_dict.items():
                wallet = bittensor.wallet(name=coldkey_name, hotkey=hotkey_name).regenerate_hotkey(mnemonic=mnemonic, overwrite=overwrite)
                wallets[coldkey_name] = wallet
    @classmethod
    def create_wallets(cls, 
                       wallets: Union[List[str], Dict] = [f'ensemble.{i}' for i in range(3)],
                       coldkey_use_password:bool = False, 
                       hotkey_use_password:bool = False
                       ):
        
        if isinstance(wallets, list):
            for wallet in wallets:
                cls.get_wallet(wallet)
                cls.create_wallet(coldkey=ck, hotkey=hk, coldkey_use_password=coldkey_use_password, hotkey_use_password=hotkey_use_password)   
           
           
    @classmethod
    def add_key_fleet(cls, name='ensemble',
                      hotkeys=[i+1 for i in range(8)] , 
                      use_password: bool=False,
                      overwrite:bool = True):
        
        cls.add_coldkey(name=name, use_password=use_password, overwrite=overwrite)
        
        for hotkey in hotkeys:
            cls.add_hotkey(coldkey=name, hotkey=hotkey, use_password=use_password, overwrite=overwrite)

        

            
    @classmethod 
    def add_coldkey (cls,name,
                       mnemonic:str = None,
                       use_password=False,
                       overwrite:bool = True) :
        
        if not overwrite:
            assert not cls.coldkey_exists(name), f'Wallet {name} already exists.'
        wallet = bittensor.wallet(name=name)
        if mnemonic is None:
            wallet.create_new_coldkey(use_password=use_password, overwrite=overwrite)
        else:
            wallet.regenerate_coldkey(mnemonic=mnemonic, use_password=use_password, overwrite=overwrite)
        return wallet
    
            
    @classmethod 
    def add_coldkeypub (cls,name = 'default',
                       ss58_address:str = None,
                       use_password=False,
                       overwrite:bool = True) :
        
        wallet = bittensor.wallet(name=name)
        wallet.regenerate_coldkeypub(ss58_address=ss58_address, use_password=use_password, overwrite=overwrite)
        return name


    @classmethod
    def new_coldkey( cls, name,
                           n_words:int = 12,
                           use_password: bool = False,
                           overwrite:bool = False) -> 'Wallet':  
        
        if not overwrite:
            assert not cls.coldkey_exists(name), f'Wallet {name} already exists.'
        
        wallet = bittensor.wallet(name=name)
        wallet.create_new_coldkey(n_words=n_words, use_password=use_password, overwrite=overwrite)
        
        
    @classmethod
    def new_hotkey( cls, name :str,
                        hotkey:str,
                        n_words:int = 12,
                        overwrite:bool = False,
                        use_password:bool = False) -> 'Wallet': 
        hotkey = str(hotkey) 
        assert cls.coldkey_exists(name), f'Wallet {name} does not exist.'
        if not overwrite:
            assert not cls.hotkey_exists(name, hotkey), f'Hotkey {hotkey} already exists.'
        
        wallet = bittensor.wallet(name=name, hotkey=hotkey)
        wallet.create_new_hotkey(n_words=n_words, use_password=use_password, overwrite=overwrite)
        
        

    @classmethod 
    def add_hotkey (cls, coldkey,
                        hotkey,
                       mnemonic:str = None,
                       use_password=False,
                       overwrite:bool = True) :
        hotkey= str(hotkey)
        coldkey= str(coldkey)
        assert coldkey in cls.coldkeys()

        wallet = bittensor.wallet(name=coldkey, hotkey=hotkey)
        if mnemonic is None:
            wallet.create_new_hotkey(use_password=use_password, overwrite=overwrite)
        else:
            wallet.regenerate_hotkey(mnemonic=mnemonic, use_password=use_password, overwrite=overwrite)
        return wallet
    
    @classmethod 
    def regen_hotkey (cls,name = 'default.default',
                       mnemonic:str = None,
                       use_password=False,
                       overwrite:bool = True) :
        
        
        assert len(name.split('.')) == 2, 'name must be of the form coldkey.hotkey'
        wallet = bittensor.wallet(name=name, hotkey=hotkey)
        
        wallet.regenerate_coldkey(mnemonic=mnemonic, use_password=use_password, overwrite=overwrite)
            
        return wallet
    
          
    @classmethod
    def add_wallet(cls, 
                      wallet: str = 'default.default',
                       coldkey : str = None,
                       mnemonic: str= None,
                       use_password:bool = False, 
                       overwrite : bool = True,
                       ) :
        if len(wallet.split('.')) == 2:
           coldkey, hotkey = wallet.split('.')
        else:
            raise ValueError('wallet must be of the form coldkey.hotkey')
           
        assert isinstance(hotkey, str), 'hotkey must be a string (or None)'
        assert isinstance(coldkey, str), 'coldkey must be a string'
        
        wallet = bittensor.wallet(name=coldkey, hotkey=hotkey)
        if coldkey:
            wallet.create_from_(mnemonic_ck, use_password=use_password, overwrite=overwrite)
        if mnemonic:
            return wallet.regenerate_hotkey(mnemonic=mnemonic, use_password=hotkey_use_password, overwrite=overwrite)
        else:
            return  wallet.create(coldkey_use_password=coldkey_use_password, hotkey_use_password=hotkey_use_password)     
                 
    @classmethod
    def register_wallet(
                        cls, 
                        wallet='default.default',
                        network: str = 'test',
                        netuid: Union[int, List[int]] = 3,
                        dev_id: Union[int, List[int]] = None, 
                        create: bool = True,                        
                        **kwargs
                        ):

        self = cls(wallet=wallet,netuid=netuid, network=network)
        # self.sync()
        self.register(dev_id=dev_id, **kwargs)

    @classmethod  
    def sandbox(cls):
        
        processes_per_gpus = 2
        for i in range(processes_per_gpus):
            for dev_id in c.gpus():
                cls.launch(fn='register_wallet', name=f'reg.{i}.gpu{dev_id}', kwargs=dict(dev_id=dev_id), mode='pm2')
        
        # print(cls.launch(f'register_{1}'))
        # self = cls(wallet=None)
        # self.create_wallets()
        # # st.write(dir(self.subtensor))
        # st.write(self.register(dev_id=0))
        
    # Streamlit Landing Page    
    selected_wallets = []
    def streamlit_sidebar(self):

        wallets_list = self.list_wallets(output_wallet=False)
        
        wallet = st.selectbox(f'Select Wallets ({wallets_list[0]})', wallets_list, 0)
        self.set_wallet(wallet)
        network_options = self.network_options()
        network = st.selectbox(f'Select Network ({network_options[0]})', network_options, 0)
        self.set_subtensor(subtensor=network)
        
        sync_network = st.button('Sync the Network')
        if sync_network:
            self.sync()
            
            st.write(self.wallet)
            

        st.metric(label='Balance', value=int(self.balance)/1e9)


    @staticmethod
    def display_metrics_dict(metrics:dict, num_columns=3):
        if metrics == None:
            return
        if not isinstance(metrics, dict):
            metrics = metrics.__dict__
        
        cols = st.columns(num_columns)

            
        for i, (k,v) in enumerate(metrics.items()):
            
            if type(v) in [int, float]:
                cols[i % num_columns].metric(label=k, value=v)
                
    default_model_name = os.path.expanduser('~/models/gpt-j-6B-vR')
    def streamlit_neuron_metrics(self, num_columns=3):
        if not self.registered:
            st.write(f'## {self.wallet} is not Registered on {self.subtensor.network}')
            self.button['register'] = st.button('Register')
            self.button['burned_register'] = st.button('Burn Register')


        
            if self.button['register']:
                self.register_wallet()
            if self.button['burned_register']:
                self.burned_register()
                
            neuron_info = self.get_neuron(wallet=self.wallet)
            axon_info = neuron_info.axon_info
            prometheus_info = axon_info.get('prometheus_info', {})
            # with st.expander('Miner', True):
                
            #     self.resolve_wallet_name(wallet)
            #     miner_kwargs = dict()
                
                
            #     axon_port = neuron_info.get('axon_info', {}).get('port', None)
            #     if axon_port == None:
            #         axon_port = self.free_port()
            #     miner_kwargs['axon_port'] = st.number_input('Axon Port', value=axon_port)
                
                
                
            #     prometheus_port = prometheus_info.get('port', None)
            #     if prometheus_port == None:
            #         prometheus_port = axon_port + 1
            #         while self.port_used(prometheus_port):
            #             prometheus_port = prometheus_port + 1
                        
                
                
            #     miner_kwargs['prometheus_port'] = st.number_input('Prometheus Port', value=prometheus_port)
            #     miner_kwargs['device'] = st.number_input('Device', self.most_free_gpu() )
            #     assert miner_kwargs['device'] in c.gpus(), f'gpu {miner_kwargs["device"]} is not available'
            #     miner_kwargs['model_name'] = st.text_input('model_name', self.default_model_name )
            #     miner_kwargs['remote'] = st.checkbox('remote', False)
            
            #     self.button['mine'] = st.button('Start Miner')

            #     if self.button['mine']:
            #         self.mine(**miner_kwargs)
            
            return  
        
        neuron_info = self.get_neuron(self.wallet)
        with st.expander('Neuron Stats', False):
            self.display_metrics_dict(neuron_info)


        with st.expander('Axon Stats', False):
            self.display_metrics_dict(neuron_info.axon_info)

        with st.expander('Prometheus Stats', False):
            self.display_metrics_dict(neuron_info.prometheus_info)

    @classmethod
    def dashboard(cls):
        st.set_page_config(layout="wide")
        self = cls( )
        self.button = {}
        with st.sidebar:
            self.streamlit_sidebar()
                    
            
        
        self.streamlit_neuron_metrics()
    @property
    def balance(self):
        return self.wallet.balance 
    
    
    @classmethod
    def burned_register (
            cls,
            wallet: 'bittensor.Wallet' = None,
            netuid: int = None,
            wait_for_inclusion: bool = True,
            wait_for_finalization: bool = True,
            prompt: bool = False,
            subtensor = None,
            max_fee = 2.0,
            wait_for_fee = True
        ):
        wallet = cls.get_wallet(wallet)
        netuid = cls.get_netuid(netuid)
        subtensor = cls.get_subtensor(subtensor)
        fee = cls.burn_fee(subtensor=subtensor)
        if wait_for_fee:
            while fee > max_fee:
                cls.print(f'fee {fee} is too high, max_fee is {max_fee}')
                time.sleep(1)
                fee = cls.burn_fee(subtensor=subtensor)
        assert fee < max_fee, f'fee {fee} is too high, max_fee is {max_fee}'
        subtensor.burned_register(
            wallet = wallet,
            netuid = netuid,
            wait_for_inclusion = wait_for_inclusion,
            wait_for_finalization = wait_for_finalization,
            prompt = prompt
        )
        
    burn_reg = burned_register
    
    @classmethod
    def burned_register_many(cls, *wallets, **kwargs):
        for wallet in wallets:
            cls.burned_register(wallet=wallet, **kwargs)
            
    burn_reg_many = burned_register_many
        
    @classmethod
    def burned_register_coldkey(cls, coldkey, max_wallets = None,  sleep_interval=3, **kwargs):
        
        wallets = cls.unregistered(coldkey)
        if max_wallets == None:
            max_wallets = cls.num_gpus()
        
        # if max_wallets == None:
        wallets = wallets[:max_wallets]
        for wallet in wallets:
            assert cls.wallet_exists(wallet), f'wallet {wallet} does not exist'
            cls.print(f'burned_register {wallet}')
            cls.burned_register(wallet=wallet, **kwargs)
            cls.sleep(sleep_interval)
    
    burn_reg_ck = burned_register_coldkey
    
    @classmethod
    def transfer(cls, 
                 wallet,
                dest:str,
                amount: Union[float, bittensor.Balance] , 
                wait_for_inclusion: bool = False,
                wait_for_finalization: bool = True,
                subtensor: 'bittensor.Subtensor' = None,
                prompt: bool = False,):
        wallet = cls.get_wallet(wallet)
        balance = cls.get_balance(wallet)
        assert balance >= amount, f'balance {balance} is less than amount {amount}'
        wallet.transfer( 
            dest=dest,
            amount=amount, 
            wait_for_inclusion = wait_for_inclusion,
            wait_for_finalization= wait_for_finalization,
            subtensor = subtensor,
            prompt = prompt)
    
    @classmethod
    def get_balance(self, wallet):
        wallet = self.get_wallet(wallet)
        return wallet.balance
    
    @classmethod
    def address(cls, wallet):
        wallet = cls.get_wallet(wallet)
        return wallet.coldkey.ss58_address
        
    @classmethod
    def score(cls, wallet='collective.0'):
        cmd = f"grep Loss ~/.pm2/logs/{wallet}.log"+ " | awk -F\| {'print $10'} | awk {'print $2'} | awk '{for(i=1;i<=NF;i++) {sum[i] += $i; sumsq[i] += ($i)^2}} END {for (i=1;i<=NF;i++) {printf \"%f +/- %f \", sum[i]/NR, sqrt((sumsq[i]-sum[i]^2/NR)/NR)}}'"
        print(cmd)
        return cls.cmd(cmd)
    
    default_model_name = os.path.expanduser('~/models/gpt-j-6B-vR')

    @classmethod
    def mine(cls, 
               wallet='ensemble.Hot5',
               model_name:str= None,
               network = 'finney',
               netuid=3,
               port = None,
               device = None,
               prometheus_port = None,
               debug = True,
               no_set_weights = True,
               remote:bool = True,
               tag=None,
               sleep_interval = 2,
               autocast = True,
               burn_reg = False,
               max_fee = 2.0,
               ):
        
        
        if tag == None:
            tag = f'{wallet}::{network}::{netuid}'
        if remote:
            kwargs = cls.locals2kwargs(locals())
            kwargs['remote'] = False
            return cls.remote_fn(fn='mine',name=f'miner::{tag}',  kwargs=kwargs)
            
        if port == None:
            port = cls.free_port()
        while cls.port_used(port):
            port = port + 1
            
            
        if model_name == None:
            model_name = cls.default_model_name
        assert not cls.port_used(port), f'Port {port} is already in use.'
  
        config = bittensor.neurons.core_server.neuron.config()
        
        # model things
        config.neuron.no_set_weights = no_set_weights
        config.neuron.model_name = model_name
        
        # device setting 
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
            
            
        if model_name == None:
            model_name = cls.default_model_name
        # network
        subtensor = bittensor.subtensor(network=network)
        bittensor.utils.version_checking()
    
        # wallet
        coldkey, hotkey = wallet.split('.')
        wallet = bittensor.wallet(name=coldkey, hotkey=hotkey)
        
        
        wallet_registered = wallet.is_registered(subtensor= subtensor, netuid=  netuid)
            
        while not wallet.is_registered(subtensor= subtensor, netuid=  netuid):
            
            if burn_reg:
                cls.burned_register(
                    wallet = wallet,
                    netuid = netuid,
                    wait_for_inclusion = False,
                    wait_for_finalization = True,
                    prompt = False,
                    max_fee = max_fee,
                    subtensor = subtensor,
                )
            time.sleep(sleep_interval)
            cls.print(f'Pending Registration {wallet} Waiting {sleep_interval}s ...')
            
        cls.print(f'Wallet {wallet} is registered on {network}')
             
        # enseure ports are free
        while cls.port_used(config.axon.port):
            config.axon.port += 1             
        while cls.port_used(config.prometheus.port):
            config.prometheus.port += 1
            
        
        
        bittensor.neurons.core_server.neuron(
               wallet=wallet,
               subtensor=subtensor,
               config=config,
               netuid=netuid).run()


    @classmethod
    def ensure_registration(cls, 
                            
                            wallet, 
                            burn_reg = False,
                            subtensor = None, 
                            netuid = None, 
                            sleep_interval=2):
            # wait for registration
            while not cls.is_registered(wallet, subtensor=subtensor, netuid=netuid):
                # burn registration
                
                if burn_reg:
                    cls.burned_register(
                        wallet = wallet,
                        netuid = netuid,
                        wait_for_inclusion = False,
                        wait_for_finalization = True,
                        prompt = False,
                        subtensor = subtensor,
                        max_fee = max_fee,
                    )
                    
                c.sleep(sleep_interval)
                
                cls.print(f'Pending Registration {wallet} Waiting 2s ...')
                
        

    @classmethod
    def miner_fleet(cls, name='ensemble', 
                    hotkeys=[i+1 for i in range(8)],
                    remote=True,
                    netuid=3,
                    network='finney',
                    refresh: bool = False,
                    burned_register=False, 
                    wait_for_register=True,
                    max_fee=2.0): 
        
        
        wallets = [f'{name}.{hotkey}' for hotkey in hotkeys]
        
        gpus = cls.gpus()
        subtensor = cls.get_subtensor(network)
        
        axon_ports = []
        for i, wallet in enumerate(wallets):
            
            assert cls.wallet_exists(wallet), f'Wallet {wallet} does not exist.'
            if wait_for_register:
                # wait for registration
                while not cls.is_registered(wallet, subtensor=subtensor, netuid=netuid):
                    # burn registration
                    
                    if burned_register:
                        cls.burned_register(
                            wallet = wallet,
                            netuid = netuid,
                            wait_for_inclusion = False,
                            wait_for_finalization = True,
                            prompt = False,
                            subtensor = subtensor,
                            max_fee = max_fee,
                        )
                    time.sleep(2)
                    cls.print(f'Pending Registration {wallet} Waiting 2s ...')
                    
            device = i
            assert device < len(gpus), f'Not enough GPUs. Only {len(gpus)} available.'
            tag = f'{wallet}::{network}::{netuid}'
            miner_name = f'miner::{tag}'
            axon_port = cls.free_port()
            while cls.port_used(axon_port) or axon_port in axon_ports:
                axon_port += 1
            axon_ports.append(axon_port)
            prometheus_port = axon_port - 1000
            if miner_name in cls.miners() and not refresh:
                cls.print(f'{miner_name} is already running. Skipping ...')
                continue
            else:
                cls.print(f'Deploying -> Miner: {miner_name} Device: {device} Axon_port: {axon_port}, Prom_port: {prometheus_port}')
                continue
                cls.mine(wallet=wallet,
                         remote=remote, 
                         tag=tag, 
                         device=device, 
                        port=axon_port,
                        prometheus_port = prometheus_port,
                         burned_register=burned_register)
            
            # self.mine(wallet=wallet, remote=remote, tag=tag)
            
    @classmethod
    def miners(cls, prefix='miner'):
        return cls.pm2_list(prefix)    
    @classmethod
    def kill_miners(cls, prefix='miner'):
        return cls.kill(prefix)    


    @classmethod
    def block(cls, subtensor='finney'):
        return cls.get_subtensor(subtensor).get_current_block()
    
    @classmethod
    def burn_fee(cls, subtensor='finney'):
        subtensor = cls.get_subtensor(subtensor)
        return subtensor.query_subtensor('Burn', None, [3]).value/1e9

    
    @classmethod
    def mlogs(cls, wallet, name='miner', network='finney', netuid=3):
        return c.logs(f'miner::{wallet}::{network}::{netuid}')


    @classmethod
    def unstake_all(cls, coldkey):
        
        assert coldkey in cls.coldkeys()
        hotkeys = cls.hotkeys(wallet)
        for hotkey in hotkeys:
            wallet = f'{wallet}.{hotkey}'
            balance = cls.get_balance(wallet)
            cls.unstake(wallet=wallet, 
                        amount=balance,
                        wait_for_inclusion=wait_for_inclusion,
                        wait_for_finalization=wait_for_finalization,
                        prompt=prompt, 
                        subtensor=subtensor)
            

    @classmethod
    def unstake (
        cls,
        wallet,
        amount: float = None, 
        wait_for_inclusion:bool = True, 
        wait_for_finalization:bool = False,
        prompt: bool = False,
        subtensor: 'bittensor.subtensor' = None,
    ) -> bool:
        """ Removes stake into the wallet coldkey from the specified hotkey uid."""
        subtensor = cls.get_subtensor(subtensor)

        
        wallet = cls.get_wallet(wallet)
        wallet.hotkey.ss58_address
        return subtensor.unstake( wallet=wallet, 
                                 hotkey_ss58=wallet.hotkey.ss58_address, 
                                 amount=amount,
                                 wait_for_inclusion=wait_for_inclusion,
                                 wait_for_finalization=wait_for_finalization, 
                                 prompt=prompt )


    @classmethod
    def sandbox(cls):
        self = cls(network='local')
        cls.pritn(self.reged(subtensor='local'))
        
        
    @classmethod
    def coldkey_map(cls, coldkey):
        
        hotkeys = cls.hotkeys(coldkey)
        wallets = [cls.get_wallet(w) for w in cls.wallets(coldkey)]
        
        coldkey_map = {
            'coldkey': wallets[0].coldkey.ss58_address,
            'hotkeys': {w.hotkey_str: w.hotkey.mnemonic for w in wallets},
        }
        
        return coldkey_map
    @classmethod
    def local_node(cls):
        return cls.cmd('sudo docker-compose up', cwd=f'{cls.repo_path}/subspace')
if __name__ == "__main__":
    BittensorModule.run()



