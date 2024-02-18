
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
    default_config =  c.get_config('bittensor')
    default_coldkey = default_config['coldkey']
    default_network = default_config['network']
    wallets_path = os.path.expanduser('~/.bittensor/wallets/')
    default_model = default_config['model']
    default_netuid = default_config['netuid']
    network2endpoint = default_config['network2endpoint'] 
    default_pool_address = default_config['pool_address']
    chain_repo = f'{c.repo_path}/repos/subtensor'

    def __init__(self,
                wallet:Union[bittensor.wallet, str] = None,
                network: Union[bittensor.subtensor, str] =default_network,
                netuid: int = default_netuid,
                config = None,
                ):
        self.set_config(config)
        self.set_subtensor(network=network, netuid=netuid)
        self.set_wallet(wallet)
        

    @classmethod
    def networks(cls):
        networks = list(cls.network2endpoint.keys())
        return networks
    

    @classmethod
    def get_endpoint(cls, network:str):
        return cls.network2endpoint.get(network, None)
       
       
    @classmethod
    def is_endpoint(cls, endpoint):
        # TODO: check if endpoint is valid, can be limited to just checking if it is a string
        return bool(':' in endpoint and cls.is_number(endpoint.split(':')[-1]))
      
    @classmethod
    def get_subtensor(cls, network:Union[str, bittensor.subtensor]='local') -> bittensor.subtensor:
        if network == None:
            network = cls.default_network
        if isinstance(network, str):
            endpoint = cls.network2endpoint[network]
            subtensor = bittensor.subtensor(chain_endpoint=endpoint)
        else:
            subtensor = network
        
        return subtensor
    
    
    @classmethod
    def get_metagraph(cls,
                      netuid = default_netuid, 
                      network=default_network, 
                      subtensor = None,
                      sync:bool = False,
                      load:bool = True,
                      save:bool = False,
                      block:bool = None):
        
        if subtensor == None:
            subtensor = cls.get_subtensor(network=network)
            
        netuid = cls.get_netuid(netuid)
    
        try:
            metagraph = bittensor.metagraph(subtensor=subtensor, netuid=netuid)
        except TypeError as e:
            metagraph = bittensor.metagraph(netuid=netuid)


        if save:
            load = False
        if load:
            try:
                metagraph.load()
                save= sync = False
            except FileNotFoundError as e:
                c.print(e, color='red')
                save = sync = True
            
            
        
        if sync:
            metagraph.sync( block=block)
            
        if save:
            metagraph.save()
        return metagraph
    
    meta = get_metagraph
    
    @classmethod
    def set_default(cls,**kwargs):
        config = cls.config()
        for key, value in kwargs.items():
            if key in config:
                assert isinstance(value, type(config[key])), f'Expected {key} to be of type {type(config[key])}, got {type(value)}'
                config[key] = value
                
        cls.save_config(config)
            
            
    
    def set_subtensor(self, network:str=None, netuid:int=None):
         
        self.subtensor = self.get_subtensor(network)
        self.metagraph = self.get_metagraph(subtensor=self.subtensor, netuid=netuid)

            
        
        return self.subtensor
        
    def set_wallet(self, wallet=None)-> bittensor.Wallet:
        ''' Sets the wallet for the module.'''
        self.wallet = self.get_wallet(wallet)
        return self.wallet
    
    @classmethod
    def get_wallet(cls, wallet:Union[str, bittensor.wallet]='ensemble.1') -> bittensor.wallet:
        if wallet is None:
            wallet =cls.default_coldkey
        if isinstance(wallet, str):
            if len(wallet.split('.')) == 2:
                name, hotkey = wallet.split('.')
                wallet =bittensor.wallet(name=name, hotkey=hotkey)
            elif len(wallet.split('.')) == 1:
                name = wallet
                wallet =bittensor.wallet(name=name)
            else:
                raise NotImplementedError(wallet)
                
            
        elif isinstance(wallet, bittensor.Wallet):
            wallet = wallet
        else:
            raise NotImplementedError(wallet)

        return wallet 
    def resolve_subtensor(self, subtensor: 'Subtensor' = None) -> 'Subtensor':
        if isinstance(subtensor, str):
            subtensor = self.get_subtensor(network=subtensor)
        if subtensor is None:
            subtensor = self.subtensor
        return subtensor
    

    def resolve_netuid(self, netuid: int = None) -> int:
        if netuid is None:
            netuid = self.netuid
        return netuid
    @classmethod
    def get_netuid(cls, netuid: int = None) -> int:
        if netuid is None:
            netuid = cls.default_netuid
        return netuid
    
    _neurons_cache = {}
    
    
    
    
    @classmethod
    def get_neurons(cls,
                    netuid: int = default_netuid, 
                    cache:bool = True,
                     subtensor: 'Subtensor' = None,
                     key: str = None,
                     index: List[int] = None,
                     **kwargs
                     ) -> List['Neuron']:
        neurons = None
        if cache:
            neurons =   cls._neurons_cache.get(netuid, None)
        
        if neurons is None:
            neurons = cls.get_metagraph(subtensor=subtensor, netuid=netuid, **kwargs).neurons
        
        if cache:
            cls._neurons_cache[netuid] = neurons
            
        if key is not None:
            if isinstance(key, list):
                neurons = [{k:getattr(n, k) for k in key} for n in neurons]
            elif isinstance(key, str):
                neurons = [getattr(n, key) for n in neurons]
            else:
                raise NotImplemented
                
            
            
        if index != None:
            if isinstance(index, int):
                if index < 0:
                    index = len(neurons) + index
                neurons = neurons[index]
                
            if isinstance(index,list) and len(index) == 2:
                neurons = neurons[index[0]:index[1]]
            
        return neurons
    
    neurons = get_neurons

    @classmethod
    def hotkey2stats(cls, *args, key=['emission', 'incentive','dividends'],  **kwargs):
        key = list(set(key + ['hotkey']))
        neurons = cls.neurons(*args,key=key,**kwargs)
        return {n['hotkey']: n for n in neurons}

    
    @classmethod
    def get_neuron(cls, wallet=None, netuid: int = None, subtensor=None):
        wallet = cls.get_wallet(wallet)
        netuid = cls.get_netuid(netuid)
        neuron_info = cls.munch({'axon_info': {}, 'prometheus_info': {}})

        c.print(f'Getting neuron info for {wallet.hotkey.ss58_address} on netuid {netuid}', color='green')
        neurons = cls.get_neurons(netuid=netuid, subtensor=subtensor)
        for neuron in neurons:
            if wallet.hotkey.ss58_address == neuron.hotkey:
                neuron_info =  neuron
        return neuron_info
    
    @classmethod
    def miner_stats(cls, wallet=None, netuid: int = None, subtensor=None):
        wallet = cls.get_wallet(wallet)
        netuid = cls.get_netuid(netuid)
        subtensor = cls.get_subtensor(subtensor)
        neuron_info = wallet.get_neuron(subtensor=subtensor, netuid=netuid)
        neuron_stats = {}
        
        for k, v in neuron_info.__dict__.items():
            if type(v) in [int, float, str]:
                neuron_stats[k] = v
            
        return neuron_stats
    
    # def whitelist(self):
    #     return ['miners', 'wallets', 'check_miners', 'reged','unreged', 'stats', 'mems','servers', 'add_server', 'top_neurons']
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
    def stats(cls, *args, sigdigs=3,keys=['emission', 'hotkey'], **kwargs):
        import pandas as pd
        wallet2neuron = cls.neurons(keys=keys, **kwargs)
        rows = []
        df = pd.DataFrame(rows)
        df = df.set_index('wallet')
        return df
    
    @classmethod
    def get_stake(cls, hotkey, coldkey = default_coldkey, **kwargs):
        if hotkey in cls.wallets():
            wallet = hotkey
        else:
            wallet = f'{coldkey}.{hotkey}'
        wallet = cls.get_wallet(wallet)
        neuron = cls.get_neuron(wallet=wallet, **kwargs)
        
        return float(neuron.stake)
    
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
    def zero_emission_miners(cls, coldkey=default_coldkey, **kwargs):
        miners = []
        wallet2stats = cls.wallet2stats(coldkey=coldkey, **kwargs)
        for k, stats in wallet2stats.items():

            if stats['emission'] == 0:
                miners.append(k)

        return miners
        
    @classmethod
    def stats(cls, **kwargs):
        wallet2stats =cls.wallet2stats(**kwargs)
        
        rows = []
        for w_name, w_stats in wallet2stats.items():
            w_stats['name'] = w_name
            rows.append(w_stats)
        
        df = c.df(rows)
        return df

    
    @classmethod
    def wallet2stats(cls, coldkey=default_coldkey, netuid=None , key=['emission', 'incentive']):
        wallet2hotkey = cls.wallet2hotkey(coldkey=coldkey, netuid=netuid)
        hotkey2stats = cls.hotkey2stats(key=key, netuid=netuid)

        wallet2stats = {}
        for w,hk in wallet2hotkey.items():
            if hk in hotkey2stats:
                wallet2stats[w] = hotkey2stats[hk]
                
            
        return wallet2stats
    
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
    def wallets(cls,
                search = None,
                registered=False, 
                unregistered=False,
                mode = 'name',
                coldkey = default_coldkey,
                subtensor=default_network, 
                netuid:int=default_netuid,
                registered_hotkeys = None
                ):
                
        if unregistered:
            registered = False
        elif registered:
            unregistered = False
        wallets = []

        if coldkey == None:
            coldkeys = cls.coldkeys()
        elif isinstance(coldkey, str):
            coldkeys = [coldkey]

        for ck in coldkeys:
            for hk in cls.hotkeys(ck):
                wallet = f'{ck}.{hk}'
                
                if search is not None:
                    if not wallet.startswith(search):
                        continue
                    
                wallet_obj = cls.get_wallet(wallet)
                if registered or unregistered:
                    if registered_hotkeys == None:
                        registered_hotkeys = cls.get_neurons(subtensor=subtensor, netuid=netuid, key='hotkey')

                    if registered and wallet_obj.hotkey.ss58_address not in registered_hotkeys:
                            continue
                    if unregistered and wallet_obj.hotkey.ss58_address in registered_hotkeys:
                            continue
                    
                if mode in ['obj', 'object']:
                    wallets.append(wallet_obj)
                elif mode in ['name', 'path']:
                    wallets.append(wallet)
                elif mode in ['address', 'addy','hk']:
                    wallets.append(wallet_obj.hotkey.ss58_address)
                else:
                    raise ValueError(f'Invalid mode: {mode}')
                    
                
        return wallets
    
    
    
    keys = wallets



    @classmethod
    def registered_wallets(cls, search=None,  subtensor=default_network, netuid:int=None):
        wallets =  cls.wallets(search=search,registered=True, subtensor=subtensor, netuid=netuid)
        return wallets

    keys = wallets
    @classmethod
    def registered_hotkeys(cls, coldkey=default_coldkey,  subtensor=default_network, netuid:int=None):
        hks =  [w.split('.')[-1] for w in cls.wallets(search=coldkey,registered=True, subtensor=subtensor, netuid=netuid)]
        return hks

    reged = registered_wallets
    @classmethod
    def unregistered_wallets(cls, search=None,  subtensor=default_network, netuid:int=None):
        wallets =  cls.wallets(search=search,unregistered=True, subtensor=subtensor, netuid=netuid)
        return wallets
    
    unreged = unregistered_wallets
    
    @classmethod
    def wallet2hotkey(cls, *args, **kwargs):
        kwargs['mode'] = 'object'
        wallets = cls.wallets(*args, **kwargs)

        return {w.name+'.'+w.hotkey_str: w.hotkey.ss58_address for w in wallets}
    
    
    @classmethod
    def hotkey2miner(cls, *args, **kwargs):
        kwargs['mode'] = 'object'
        wallet2hotkey = cls.wallet2hotkeys()


        return {w.name+'.'+w.hotkey_str: w.hotkey.ss58_address for w in wallets}

    @classmethod
    def unregistered_hotkeys(cls, coldkey=default_coldkey,  subtensor=default_network, netuid:int=None):
        return [w.split('.')[-1] for w in cls.unregistered_wallets(search=coldkey, subtensor=subtensor, netuid=netuid)]
    unreged_hotkeys = unreged_hks = unregistered_hotkeys
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
    def get_wallet_path(cls, wallet):
        ck, hk = wallet.split('.')
        return  os.path.join(cls.wallets_path, ck, 'hotkeys', hk)
    
    @classmethod
    def rm_wallet(cls, wallet):
        wallet2path = cls.wallet2path()
        assert wallet in wallet2path, f'Wallet {wallet} not found in {wallet2path.keys()}'
        cls.rm(wallet2path[wallet])
     
        return {'wallets': cls.wallets(), 'msg': f'wallet {wallet} removed'}
    
    @classmethod
    def rename_coldkey(cls, coldkey1, coldkey2):
        coldkey1_path = cls.coldkey_dir_path(coldkey1)
        cls.print(coldkey1_path)
        assert os.path.isdir(coldkey1_path)
        coldkey2_path = os.path.dirname(coldkey1_path) + '/'+ coldkey2
       
        cls.print(f'moving {coldkey1} ({coldkey1_path}) -> {coldkey2} ({coldkey2_path})')
        cls.mv(coldkey1_path,coldkey2_path)
    
    @classmethod
    def rename_wallet(cls, wallet1, wallet2):
        wallet1_path = cls.get_wallet_path(wallet1)
        wallet2_path = cls.get_wallet_path(wallet2)
        cls.print(f'Renaming {wallet1} to {wallet2}')

        
        cls.mv(wallet1_path, wallet2_path)
        return [wallet1, wallet2]
    
    @classmethod
    def coldkey_path(cls, coldkey):
        coldkey_path = os.path.join(cls.wallets_path, coldkey)
        return coldkey_path + '/coldkey'
    @classmethod
    def hotkey_path(cls, hotkey, coldkey=default_coldkey):
        coldkey_path = os.path.join(cls.wallets_path, coldkey)
        return coldkey_path + '/hotkeys/' + str(hotkey)
    
    
    @classmethod
    def coldkey_dir_path(cls, coldkey):
        return os.path.dirname(cls.coldkey_path(coldkey))
    
    
    get_coldkey_path = coldkey_path
    @classmethod
    def coldkeypub_path(cls, coldkey):
        coldkey_path = os.path.join(cls.wallets_path, coldkey)
        return coldkey_path + '/coldkeypub.txt'
    
    def rm_wallets(cls, *wallets, **kwargs):
        for w in wallets:
            cls.rm_wallet(w, **kwargs)
            
        return cls.wallets()
    @classmethod
    def wallet_path(cls, wallet):
        return cls.wallet2path().get(wallet)
    
    @classmethod
    def rm_coldkey(cls,*coldkeys):
        
        coldkeys_removed = []
        for coldkey in coldkeys:
            coldkey = str(coldkey)
        
            if coldkey in cls.coldkeys():
                coldkey_path = cls.coldkey_dir_path(coldkey)
                cls.rm(coldkey_path)
                coldkeys_removed.append(coldkey)
            else:
                cls.print(f'Coldkey {coldkey} not found in {cls.coldkeys()}')
        
        return {'msg': f'Coldkeys removed {coldkeys_removed}', 'coldkeys': cls.coldkeys()}


    @classmethod
    def hotkeys(cls, coldkey=default_coldkey):
        hotkeys =   [os.path.basename(p) for p in cls.ls(os.path.join(cls.wallets_path, coldkey, 'hotkeys'))]
        
        
        if all([c.is_number(hk) for hk in hotkeys]):
            hotkeys = [int(hk) for hk in hotkeys]
            hotkeys = sorted(hotkeys)
            hotkeys = [str(hk) for hk in hotkeys]
        else:
            hotkeys = sorted(hotkeys)

        return hotkeys
        
    @classmethod
    def coldkeys(cls, wallet='default'):
        
        return  [os.path.basename(p)for p in cls.ls(cls.wallets_path)]

        
    @classmethod
    def coldkey_exists(cls, wallet='default'):
        return os.path.exists(cls.get_coldkey_path(wallet))
    
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
        return bool(wallet in wallets)
    
    @classmethod
    def hotkey_exists(cls, coldkey:str, hotkey:str) -> bool:
        hotkeys = cls.hotkeys(coldkey)
        return bool(hotkey in hotkeys)
    
    @classmethod
    def coldkey_exists(cls, coldkey:str) -> bool:
        coldkeys = cls.coldkeys()
        return bool(coldkey in coldkeys)
    
    
    
    # @property
    # def default_network(self):
    #     return self.networks()[0]
    

    @property
    def network(self):
        return self.subtensor.network
    
    @classmethod
    def is_registered(cls, wallet = None, netuid: int = default_netuid, subtensor: 'Subtensor' = default_network):
        
        netuid = cls.get_netuid(netuid)
        wallet = cls.get_wallet(wallet)
        hotkeys = cls.get_neurons( netuid=netuid, subtensor=subtensor, key='hotkey')
        return bool(wallet.hotkey.ss58_address in hotkeys)
        

    @property
    def registered(self):
        return self.is_registered(wallet=self.wallet, netuid=self.netuid, subtensor=self.subtensor)
    
    @classmethod
    def sync(cls, subtensor= None, netuid: int = None, block =  None):
        c.print('Syncing...')
        return cls.get_metagraph(netuid=netuid,
                                  subtensor=subtensor,
                                  block=block,
                                  sync=True,
                                  save=True)
        
    save = sync
    @classmethod
    def metagraph_staleness(cls, metagraph=None,
                            subtensor=None, netuid=None):
        if metagraph is None:
            metagraph = cls.get_metagraph(netuid=netuid,
                        subtensor=subtensor,
                        load=True)
        subtensor = cls.get_subtensor(subtensor)
            
        current_block = subtensor
        block_staleness = current_block - metagraph.block
        return block_staleness.item()
    
    @classmethod
    def sync_loop(cls,subtensor= 'local', 
                  netuid: int = None,
                  block =  None, 
                  remote=False,
                  max_metagraph_staleness=10):
        if remote:
            kwargs = c.locals2kwargs(locals())
            kwargs['remote'] = False
            return cls.remote_fn(fn='sync_loop', kwargs=kwargs)
        metagraph_block = 0
        subtensor = cls.get_subtensor(subtensor)
        netuid = cls.get_netuid(netuid)
        prev_metagraph_staleness = 0
        while True:
            metagraph_staleness = cls.metagraph_staleness(subtensor=subtensor, netuid=netuid)
            if metagraph_staleness > max_metagraph_staleness:
                c.print(f'Block staleness {metagraph_staleness} > {max_metagraph_staleness} for {subtensor} {netuid}. Syncing')
                metagraph = cls.get_metagraph(netuid=netuid,
                                        subtensor=subtensor,
                                        sync=True,
                                        save=True)
            else:
                if prev_metagraph_staleness != metagraph_staleness:
                    prev_metagraph_staleness = metagraph_staleness
                    c.print(f'Block staleness {metagraph_staleness} < {max_metagraph_staleness} for {subtensor} {netuid}. Sleeping')
                
            
        
        
    
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
        
    #     c.set_page_config(layout="wide")
    #     self = cls(wallet='collective.0', network=default_network)

    #     with st.sidebar:
    #         self.streamlit_sidebar()
            
    #     st.write(f'# BITTENSOR DASHBOARD {self.network}')
    #     wallets = self.list_wallets(output_wallet=True)
        
    #     st.write(wallets[0].__dict__)
        
    #     # self.register()
    #     # st.write(self.run_miner('fish', '100'))

    #     # self.streamlit_neuron_metrics()
     
          
    
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
           
    #################
    #### Staking ####
    #################
    @classmethod
    def stake(
        cls, 
        hotkey:str,
        coldkey = default_coldkey,
        hotkey_ss58: Optional[str] = None,
        amount: Union['Balance', float] = None, 
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        prompt: bool = False,
        subtensor = None
    ) -> bool:
        """ Adds the specified amount of stake to passed hotkey uid. """
        wallet = cls.get_wallet(f'{coldkey}.{hotkey}')
        subtensor = cls.get_subtensor(subtensor)
        return subtensor.add_stake( 
            wallet = wallet,
            hotkey_ss58 = hotkey_ss58, 
            amount = amount, 
            wait_for_inclusion = wait_for_inclusion,
            wait_for_finalization = wait_for_finalization, 
            prompt = prompt
        )

    @classmethod
    def add_keys(cls, name=default_coldkey,
                      coldkey_ss58: Optional[str] = None,
                      hotkeys=None , 
                      n = 100,
                      use_password: bool=False,
                      overwrite:bool = False):
        if coldkey_ss58:
            cls.add_coldkeypub(name=name, ss58_address=coldkey_ss58_address, use_password=use_password, overwrite=overwrite)
        cls.add_coldkey(name=name, use_password=use_password, overwrite=overwrite)
        hotkeys = hotkeys if hotkeys!=None else list(range(n))
        for hotkey in hotkeys:
            cls.add_hotkey(coldkey=name, hotkey=hotkey, use_password=use_password, overwrite=overwrite)
            
        return {'msg': f'Added {len(hotkeys)} hotkeys to {name}'}

    @classmethod
    def switch_version(cls, version='4.0.1'):
        version = str(version) 
        if str(version) == '4':
            version = '4.0.1'
        elif str(version) == '5':
            version = '5.1.0'
        c.cmd(f'pip install bittensor=={version}', verbose=True)

    @classmethod
    def setup(cls, network='local'):
        if network == 'local':
            cls.local_node()
        cls.add_keys()
        cls.fleet(network=network)
            
    @classmethod 
    def add_coldkey (cls,
                    name,
                       mnemonic:str = None,
                       use_password=False,
                       overwrite:bool = False) :
        
        if cls.coldkey_exists(name) and not overwrite:
            cls.print(f'Coldkey {name} already exists', color='yellow')
            return name
        wallet = bittensor.wallet(name=name)
        if not overwrite:
            if cls.coldkey_exists(name):
                return wallet
        
        if mnemonic is None:
            wallet.create_new_coldkey(use_password=use_password, overwrite=overwrite)
        else:
            wallet.regenerate_coldkey(mnemonic=mnemonic, use_password=use_password, overwrite=overwrite)
        return {'msg': f'Coldkey {name} created', 'success': True}
            
    @classmethod 
    def add_coldkeypub (cls,name = 'default',
                       ss58_address:str = None,
                       use_password=False,
                       overwrite:bool = True) :
        
        wallet = bittensor.wallet(name=name)
        wallet.regenerate_coldkeypub(ss58_address=ss58_address, overwrite=overwrite)
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
    def add_hotkeys(cls, coldkey=default_coldkey,hotkeys:list = 10,  **kwargs):
        if isinstance(hotkeys, int):
            hotkeys =  list(range(hotkeys))
        assert isinstance(hotkeys, list), f'hotkeys must be a list or int, got {type(hotkeys)}'
        
        for hk in hotkeys:
            cls.add_hotkey(coldkey=coldkey, hotkey=hk, **kwargs)  
        

    @classmethod 
    def add_hotkey (cls,
                        coldkey = default_coldkey,
                       hotkey = None,
                       mnemonic:str = None,
                       use_password=False,
                       overwrite:bool = False) :
        hotkey= str(hotkey)
        coldkey= str(coldkey)
        assert coldkey in cls.coldkeys()
        wallet = f'{coldkey}.{hotkey}'
        if cls.wallet_exists(wallet):
            if not overwrite:
                cls.print(f'Wallet {wallet} already exists.', color='yellow')
                return wallet
        wallet = bittensor.wallet(name=coldkey, hotkey=hotkey)
        if mnemonic is None:
            wallet.create_new_hotkey(use_password=use_password, overwrite=overwrite)
        else:
            wallet.regenerate_hotkey(mnemonic=mnemonic, use_password=use_password, overwrite=overwrite)
        return wallet
    
    @classmethod 
    def regen_hotkey (cls,
                      hotkey:str,
                      coldkey:str =default_coldkey,
                       mnemonic:str = None,
                       use_password=False,
                       overwrite:bool = True) :
        
        
        assert len(name.split('.')) == 2, 'name must be of the form coldkey.hotkey'
        wallet = bittensor.wallet(name=coldkey, hotkey=hotkey)
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
    def register_wallet_params(cls, wallet_name:str, params:dict):
        registered_info = cls.get('registered_info', {})
        registered_info[wallet_name] = params
        cls.put('registered_info', registered_info)   
        
    @classmethod
    def unregister_wallet_params(cls, wallet_name:str):
        registered_info = cls.get('registered_info', {})
        if wallet_name in registered_info:
            registered_info.pop(wallet_name)
        cls.put('registered_info', registered_info)  
        
    @classmethod
    def registered_wallet_params(cls):
        return cls.get('registered_info', {})
        
    @classmethod
    def register_wallet(
                        cls, 
                        wallet='default.default',
                        subtensor: str =default_network,
                        netuid: Union[int, List[int]] = default_netuid,
                        dev_id: Union[int, List[int]] = None, 
                        create: bool = True,                        
                        **kwargs
                        ):
        params = c.locals2kwargs(locals())
        
        
        self = cls(wallet=wallet,netuid=netuid, subtensor=subtensor)
        # self.sync()
        wallet_name = c.copy(wallet)
        cls.register_wallet_params(wallet_name=wallet_name, params=params)
        try:
            self.register(dev_id=dev_id, **kwargs)
        except Exception as e:
            c.print(e, color='red')
        finally:
            cls.unregister_wallet_params(wallet_name=wallet_name)
    


        
    # Streamlit Landing Page    
    selected_wallets = []
    def streamlit_sidebar(self):

        wallets_list = self.list_wallets(output_wallet=False)
        
        wallet = st.selectbox(f'Select Wallets ({wallets_list[0]})', wallets_list, 0)
        self.set_wallet(wallet)
        networks = self.networks()
        network = st.selectbox(f'Select Network ({networks[0]})', networks, 0)
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
        c.set_page_config(layout="wide")
        self = cls( )
        self.button = {}
        with st.sidebar:
            self.streamlit_sidebar()
                    
            
        
        self.streamlit_neuron_metrics()
    @classmethod
    def balance(cls, wallet=default_coldkey):
        wallet = cls.get_wallet(wallet)
        return wallet.balance 
    
    
    @classmethod
    def burned_register (
            cls,
            wallet: 'bittensor.Wallet' = None,
            netuid: int = None,
            wait_for_inclusion: bool = True,
            wait_for_finalization: bool = True,
            prompt: bool = False,
            subtensor = None,
            max_fee = 1.0,
            wait_for_fee = True
        ):
        wallet = cls.get_wallet(wallet)
        netuid = cls.get_netuid(netuid)
        subtensor = cls.get_subtensor(subtensor)
        fee = cls.burn_fee(subtensor=subtensor)
        while fee >= max_fee:
            cls.print(f'fee {fee} is too high, max_fee is {max_fee}')
            time.sleep(1)
            fee = cls.burn_fee(subtensor=subtensor)
            if cls.is_registered(wallet=wallet, netuid=netuid, subtensor=subtensor):
                cls.print(f'wallet {wallet} is already registered on {netuid}')
                return True
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
    def burned_register_coldkey(cls, coldkey = default_coldkey,
                                sleep_interval=3,
                                **kwargs):
        
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
                dest:str,
                amount: Union[float, bittensor.Balance], 
                wallet = default_coldkey,
                wait_for_inclusion: bool = False,
                wait_for_finalization: bool = True,
                subtensor: 'bittensor.Subtensor' = None,
                prompt: bool = False,
                min_balance= 0.1,
                gas_fee: bool = 0.0001):
        wallet = cls.get_wallet(wallet)
        balance = cls.get_balance(wallet)
        balance = balance - gas_fee
        if balance < min_balance:
            cls.print(f'Not Enough Balance for Transfer --> Balance ({balance}) < min balance ({min_balance})', color='red')
            return None
        else:
            cls.print(f'Enough Balance for Transfer --> Balance ({balance}) > min balance ({min_balance})')
        
        print(f'balance {balance} amount {amount}')
        if amount == -1:
            amount = balance - gas_fee
            
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
        return float(wallet.balance)
    
    
    @classmethod
    def address(cls, wallet = default_coldkey):
        wallet = cls.get_wallet(wallet)
        return wallet.coldkeypub.ss58_address
    ss58 = address
    @classmethod
    def score(cls, wallet='collective.0'):
        cmd = f"grep Loss ~/.pm2/logs/{wallet}.log"+ " | awk -F\| {'print $10'} | awk {'print $2'} | awk '{for(i=1;i<=NF;i++) {sum[i] += $i; sumsq[i] += ($i)^2}} END {for (i=1;i<=NF;i++) {printf \"%f +/- %f \", sum[i]/NR, sqrt((sumsq[i]-sum[i]^2/NR)/NR)}}'"
        print(cmd)
        return cls.cmd(cmd)
    

    @classmethod
    def neuron_class(cls,  model=default_model, netuid=default_netuid):
        if netuid in [1, 11]:
            neuron_path = cls.getc('neurons').get(model)
            neuron_class = c.import_object(neuron_path)
        else: 
            raise ValueError(f'netuid {netuid} not supported')
        return neuron_class



    @classmethod
    def neuron(cls, *args, mode=None, netuid=3, **kwargs):
        
        if netuid == 3:
            neuron =  cls.module('bittensor.miner.neuron')(*args, **kwargs)
        elif netuid in [1,11]:
            neuron = cls.import_object('commune.bittensor.neurons.neurons.text.prompting')(*args, **kwargs)
            
        return neuron

    @classmethod
    def mine_many(cls, *hotkeys, coldkey=default_coldkey, **kwargs):
        for hk in hotkeys:
            cls.mine(wallet=f'{coldkey}.{hk}', **kwargs)



    @classmethod
    def get_miner_name(cls, wallet:str, network:str, netuid:int) -> str:
        network = cls.resolve_network(network)
        name = f'miner::{wallet}::{network}::{netuid}'

        return name
    

    @classmethod
    def mine(cls, 
               wallet='alice.1',
               network =default_network,
               netuid=default_netuid,
               model = default_model,
               port = None,
               prometheus_port:int = None,
               device:int = None,
               debug = True,
               no_set_weights = True,
               remote:bool = True,
               sleep_interval:int = 2,
               autocast:bool = True,
               burned_register:bool = False,
               logging:bool = True,
               max_fee:int = 2.0,
               refresh=True,
               miner_name:str = None,
               refresh_ports:bool = False,
               vpermit_required: bool = False,
               min_allowed_stake: float = 15000
               ):
        
        
        kwargs = cls.locals2kwargs(locals())
        # resolve the name of the remote function
        if remote:
            if miner_name == None:
                miner_name = cls.get_miner_name(wallet=wallet, 
                                                      network=network,
                                                      netuid=netuid)
            
            kwargs['remote'] = False
            return cls.remote_fn(fn='mine',name=miner_name,  kwargs=kwargs, refresh=refresh)

            
        neuron_class = cls.neuron_class(netuid=netuid, model=model)
        config = neuron_class.config()
        config.merge(bittensor.BaseMinerNeuron.config())
        config.neuron.blacklist.vpermit_required = vpermit_required
        config.neuron.blacklist.min_allowed_stake = min_allowed_stake
        # model things
        config.neuron.no_set_weights = no_set_weights
        config.netuid = netuid 
        
        # network
        subtensor = bittensor.subtensor(network=network)
        bittensor.utils.version_checking()


        
        # wallet
        coldkey, hotkey = wallet.split('.')
        config.wallet.name = coldkey
        config.wallet.hotkey = hotkey
        wallet = bittensor.wallet(config=config)
        
        if cls.is_registered(wallet=wallet, subtensor=subtensor, netuid=netuid):
            cls.print(f'wallet {wallet} is already registered')
            neuron = cls.get_neuron(wallet=wallet, subtensor=subtensor, netuid=netuid)
            if not refresh_ports:
                port = neuron.axon_info.port
            config.wallet.reregister = False
        else:
            cls.ensure_registration(wallet=wallet, 
                                    subtensor=subtensor, 
                                    netuid=netuid,
                                    max_fee=max_fee,
                                    burned_register=burned_register, 
                                    sleep_interval=sleep_interval,
                                    display_kwargs=kwargs)
        config.logging.debug = debug       
        
        wallet = bittensor.wallet(config=config)
        c.print(f'wallet {wallet} is registered')
   
        config.axon.port = cls.resolve_port(port)
        c.print(f'using port {config.axon.port}')
        neuron_class(config=config).run()

    @classmethod
    def validator_neuron(cls, mode='core', modality='text.prompting'):
        return c.import_object(f'commune.bittensor.neurons.{modality}.validators.{mode}.neuron.neuron')
    @classmethod
    def validator(cls,
               wallet=f'{default_coldkey}.vali',
               network =default_network,
               netuid=1,
               device = None,
               debug = True,
               remote:bool = True,
               tag=None,
               sleep_interval = 2,
               autocast = True,
               burned_register = False,
               logging:bool = True,
               max_fee = 2.0,
               modality='text.prompting', 
               mode = 'relay'
               ):
        kwargs = cls.locals2kwargs(locals())
        c.print(kwargs)
        if tag == None:
            if network in ['local', 'finney']:
                tag = f'{wallet}::finney::{netuid}'
            else:
                tag = f'{wallet}::{network}::{netuid}'
            kwargs['tag'] = tag
        if remote:
            kwargs['remote'] = False
            return cls.remote_fn(fn='validator',name=f'validator::{tag}',  kwargs=kwargs)
        
        subtensor = bittensor.subtensor(network=network)    
        coldkey, hotkey = wallet.split('.')
        wallet = bittensor.wallet(name=coldkey, hotkey=hotkey)
        bittensor.utils.version_checking()

        cls.ensure_registration(wallet=wallet, 
                                subtensor=subtensor, 
                                netuid=netuid,
                                max_fee=max_fee,
                                burned_register=burned_register, 
                                sleep_interval=sleep_interval,
                                display_kwargs=kwargs)
        
        validator_neuron = cls.validator_neuron(mode=mode, modality=modality)
        config = validator_neuron.config()
            
        device = cls.most_free_gpu() if device == None else device
        if not str(device).startswith('cuda:'):
            device = f'cuda:{device}'
        config.neuron.device = device
        
        validator_neuron(config=config, 
                            wallet=wallet,
                            subtensor=subtensor,
                            netuid=netuid).run()

    @classmethod
    def ensure_registration(cls, 
                            wallet, 
                            subtensor =default_network, 
                            burned_register = False,
                            netuid = 3, 
                            max_fee = 2.0,
                            sleep_interval=60,
                            display_kwargs=None):
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
                    
                c.sleep(sleep_interval)
                
                cls.print(f'Pending Registration {wallet} Waiting 2s ...')
                if display_kwargs:
                    cls.print(display_kwargs)
            cls.print(f'{wallet} is registered on {subtensor} {netuid}!')
    @classmethod
    def burn_reg_unreged(cls, time_sleep= 10, **kwargs):
        for w in cls.unreged():
            cls.burned_register(w, **kwargs)
        
    @classmethod
    def miner_exists(cls, wallet:str, network:str=default_network, netuid:int=default_netuid):
        miners = cls.miners(network=network, netuid=netuid)
        return wallet in miners

    @classmethod
    def fleet(cls,  
            name:str=default_coldkey, 
            netuid:int= default_netuid,
            network:str=default_network,
            model : str = default_model,
            refresh: bool = True,
            burned_register:bool=False, 
            ensure_registration:bool=False,
            device:str = 'cpu',
            max_fee:float=3.5,
            refresh_ports:bool = False,
            hotkeys:List[str] = None,
            remote: bool = False,
            reged : bool = False,
            n:int = 1000):

        if remote:
            kwargs = c.localswkwargs(locals())
            kwargs['remote'] = False
            cls.remote_fn('fleet', kwargs=kwargs)

        if reged:
            wallets = cls.reged(name)
        else:
            if hotkeys == None:
                wallets = [f'{name}.{h}' for h in cls.hotkeys(name)]
            else:
                wallets  = [f'{name}.{h}' for h in hotkeys]
            
                
        subtensor = cls.get_subtensor(network)

        avoid_ports = []
        miners = cls.miners(netuid=netuid)
        
        for i, wallet in enumerate(wallets):
            miner_name = cls.get_miner_name(wallet=wallet, network=network, netuid=netuid)
            if cls.miner_exists(wallet, network=network, netuid=netuid) and not refresh:
                cls.print(f'{miner_name} is already running. Skipping ...')
                continue
            
            if miner_name in miners and not refresh:
                cls.print(f'{miner_name} is already running. Skipping ...')
                continue
            
            
            assert cls.wallet_exists(wallet), f'Wallet {wallet} does not exist.'
            
            if cls.is_registered(wallet, subtensor=subtensor, netuid=netuid):
                cls.print(f'{wallet} is already registered on {subtensor} {netuid}!')
                neuron = cls.get_neuron(wallet=wallet, subtensor=subtensor, netuid=netuid)

                if refresh_ports:
                    axon_port = cls.free_port(reserve=False, avoid_ports=avoid_ports)
                    avoid_ports.append(axon_port)
                    prometheus_port = axon_port - 1000
                else:
                    axon_port = neuron.axon_info.port
                    prometheus_port = neuron.prometheus_info.port
            else:
                # ensure registration
                if ensure_registration:
                    cls.ensure_registration(wallet,
                                            subtensor=subtensor, 
                                            netuid=netuid,
                                            burned_register=burned_register,
                                            max_fee=max_fee)
                    burned_register = False # only burn register for first wallet
                axon_port = cls.free_port(reserve=False, avoid_ports=avoid_ports)
                avoid_ports.append(axon_port)
                prometheus_port = axon_port - 1000
                
            
            avoid_ports += [axon_port, prometheus_port]
            avoid_ports = list(set(avoid_ports)) # avoid duplicates, though htat shouldnt matter
                

            cls.print(f'Deploying -> Miner: {miner_name} Device: {device} Axon_port: {axon_port}, Prom_port: {prometheus_port}')
            cls.mine(wallet=wallet,
                        remote=True, 
                        model=model,
                        netuid=netuid,
                        device=device, 
                        refresh=refresh,
                        port=axon_port,
                        network=network,
                        miner_name = miner_name,
                        prometheus_port = prometheus_port,
                        burned_register=burned_register,
                        refresh_ports=refresh_ports,
                        max_fee=max_fee)
            
            n -= 1 
            if n <= 0:
                cls.print('Max miners reached')
                break
        
    @classmethod
    def miners(cls,
                wallet:str=None,
                unreged:bool=False, 
                reged:bool=False,
                netuid:int=default_netuid,
                network:str =default_network,
                prefix:str='miner'):
        kwargs =  c.locals2kwargs(locals())
        return list(cls.wallet2miner( **kwargs).values())
    
    @classmethod
    def validators(cls, *args, **kwargs):
        return list(cls.wallet2validator(*args, **kwargs).keys())
        
    @classmethod
    def version(cls):
        return c.version('bittensor')
    @classmethod
    def wallet2validator(cls, 
                         wallet=None,
                         unreged=False, 
                         reged=False,
                         netuid=default_netuid,
                         network =default_network,
                         prefix='validator'):
        network = cls.resolve_network(network)
        wallet2miner = {}
        if unreged:
            filter_wallets = cls.unreged()
        elif reged:
            filter_wallets = cls.reged()
        else:
            filter_wallets = []
            
        for m in cls.pm2_list(prefix):
            
            wallet_name = m.split('::')[1]
            if netuid != None and m.split('::')[-1] != str(netuid):
                continue
            if len(filter_wallets) > 0 and wallet_name not in filter_wallets:
                continue
            wallet2miner[wallet_name] = m
            
        if wallet in wallet2miner:
            return wallet2miner[wallet]
        return wallet2miner
     
     
    @classmethod
    def resolve_network(cls, network):
        if network in ['local']:
            network = 'finney'
        return network
    
    @classmethod
    def wallet2miner(cls, 
                         wallet=None,
                         unreged=False, 
                         reged=False,
                         netuid=default_netuid,
                         network =default_network,
                         prefix='miner'):
        wallet2miner = {}
        network = cls.resolve_network(network)
        if unreged:
            filter_wallets = cls.unreged()
        elif reged:
            filter_wallets = cls.reged()
        else:
            filter_wallets = []
            
            
        for m in cls.pm2_list(prefix):
            
            wallet_name = m.split('::')[1]
            network_name = m.split('::')[2]
            if netuid != None and m.split('::')[-1] != str(netuid):
                continue
            if len(filter_wallets) > 0 and wallet_name not in filter_wallets:
                continue
            if network != None and network != network_name:
                continue
            wallet2miner[wallet_name] = m
            
        if wallet in wallet2miner:
            return wallet2miner[wallet]
        return wallet2miner
     
  
    w2m = wallet2miner
    @classmethod
    def get_miner(cls, wallet):
        return cls.wallet2miner(wallet)
    @classmethod
    def kill_miners(cls, prefix='miner'):
        return c.kill(prefix)    

    @classmethod
    def kill(cls, *wallet):
        w2m = cls.wallet2miner()
        for w in wallet:
            if w in w2m:
                cls.print(f'Killing {w}')
                c.kill(w2m[w])
            else:
                cls.print(f'Miner {w} not found.')

    @classmethod
    def restart(cls, wallet):
        return c.restart(cls.w2m(wallet))
    @classmethod
    def block(cls, subtensor=default_network):
        subtensor = cls.get_subtensor(subtensor)
        return subtensor.get_current_block()
    
    @classmethod
    def burn_fee(cls, subtensor='finney', netuid=default_netuid):
        subtensor = cls.get_subtensor(subtensor)
        return subtensor.query_subtensor('Burn', None, [netuid]).value/1e9
    @classmethod
    def query_map(cls, key='Uids', subtensor='finney', netuid=default_netuid):
        subtensor = cls.get_subtensor(subtensor)
        return subtensor.query_map_subtensor(key, None, [netuid]).records



    @classmethod
    def key2uid(cls, netuid:int=default_netuid):
        uids = cls.query_map('Uids', netuid=netuid)[:10]
        keys = {k.value:v.value for k,v in uids}
        return uids


    @classmethod
    def logs(cls, wallet,  **kwargs):
        return c.logs(cls.wallet2miner(**kwargs).get(wallet), mode='local', start_line=-10, end_line=-1)

    @classmethod
    async def async_logs(cls, wallet, network=default_network, netuid=3):
        processes = c.pm2ls(wallet)
        logs_dict = {}
        for p in processes:
            if any([p.startswith(k) for k in ['miner', 'validator'] ]):
                logs_dict[p.split('::')[0]] = c.logs(p, mode='local')
            
        if len(logs_dict) == 1:
            return list(logs_dict.values())[0]
            
        return logs_dict

    @classmethod
    def miner2logs(cls, 
                    network=default_network,
                    netuid=1, 
                    verbose:bool = True):
        
        miners = cls.miners()
        miner2logs = {}
        for miner in miners:
            miner2logs[miner] = c.pm2_logs(miner, end_line=200, mode='local')
        
        
        if verbose:
            for miner, logs in miner2logs.items():
                pad = 100*'-'
                color = cls.random_color()
                cls.print(pad,f'\n{miner}\n', pad, color=color)
                cls.print( logs, '\n\n', color=color)
            
        # return miner2logs


    check_miners = miner2logs

    @classmethod
    def unstake_coldkey(cls, 
                        coldkey = default_coldkey,
                        wait_for_inclusion = True,
                        wait_for_finalization = False,
                        prompt = False,
                        subtensor = None, 
                        min_stake = 0.1
                        ):

        for wallet in cls.wallets(coldkey, registered=True):
            cls.print(f'Unstaking {wallet} ...')
            stake = cls.get_stake(wallet)
            if stake >= min_stake:
                cls.print(f'Unstaking {wallet} Stake/MinStake ({stake}>{min_stake})')
                amount_unstaked = cls.unstake(wallet=wallet, 
                                wait_for_inclusion=True,
                                wait_for_finalization=wait_for_finalization,
                                prompt=prompt, 
                                subtensor=subtensor)
            else:
                cls.print(f'Not enough stake {stake} to unstake {wallet}, min_stake: {min_stake}')
                
    unstake_ck = unstake_coldkey
    
    
    @classmethod
    def set_pool_address(cls, pool_address):
        cls.putc('pool_address', pool_address)
        cls.print(f'Set pool address to {pool_address}')
        
    @classmethod
    def set_coldkey(cls, coldkey):
        return cls.putc('coldkey', coldkey)
        
    @classmethod
    def pool_address(cls):
        return cls.getc('pool_address', cls.default_pool_address)
    
    @classmethod
    def unstake2pool(cls,
                     pool_address:str = None,
                     coldkey:str = default_coldkey,
                     loops = 20,
                     transfer: bool = True,
                     min_balance: float = 0.1,
                     min_stake: float = 0.1,
                     remote = True,
                     sleep = 1,
                     
                     ):
        
        if remote:
            kwargs = cls.locals2kwargs(locals())
            kwargs['remote'] = False
            return cls.remote_fn(fn='unstake2pool',name=f'bt::unstake2pool',  kwargs=kwargs)
        
        if pool_address == None:
            pool_address = cls.pool_address()
        for i in range(loops):
            
            
            cls.print(f'-YOOO- Unstaking {coldkey}')
            

            cls.unstake_coldkey(coldkey=coldkey, min_stake=min_stake) # unstake all wallets
                                
            if pool_address == cls.address(coldkey):
                cls.print(f'Coldkey {coldkey} is equal to {pool_address}, skipping transfer')
            else:
                cls.transfer(dest=pool_address, amount=-1, wallet=coldkey, min_balance=min_balance)

                
            cls.sleep(sleep)
        
            
        
        
        

    @classmethod
    def unstake(
        cls,
        wallet , 
        amount: float = None ,
        wait_for_inclusion:bool = True, 
        wait_for_finalization:bool = False,
        prompt: bool = False,
        subtensor: 'bittensor.subtensor' = None,
    ) -> bool:
        """ Removes stake into the wallet coldkey from the specified hotkey uid."""
        subtensor = cls.get_subtensor(subtensor)
        wallet = cls.get_wallet(wallet)
        
        return subtensor.unstake( wallet=wallet, 
                                 hotkey_ss58=wallet.hotkey.ss58_address, 
                                 amount=amount,
                                 wait_for_inclusion=wait_for_inclusion,
                                 wait_for_finalization=wait_for_finalization, 
                                 prompt=prompt )
        
    @classmethod
    def sand(cls, ratio=1.0, model=default_model ):
        reged = cls.reged()
        reged = reged[:int(len(reged)*ratio)]
        for wallet in reged:
            cls.mine(wallet, model=model)
        
    @classmethod
    def allinone(cls, overwrite_keys=False, refresh_miners=False, refresh_servers= False):
        cls.add_keys(overwrite=overwrite_keys) # add keys job
        cls.add_servers(refresh=refresh_servers) # add servers job
        cls.fleet(refresh=refresh_miners) # fleet job
        cls.unstake2pool() # unstake2pool job
    @classmethod
    def mems(cls,
                     coldkey:str=default_coldkey, 
                     reged : bool = False,
                     unreged:bool = False,
                     miners_only:bool = False,
                     no_miners: bool = False,
                     path:str = None,
                     network:str = default_network,
                     netuid:int=default_netuid):
        if reged:
            hotkeys = cls.registered_hotkeys(coldkey, netuid=netuid)
            unreged = False
        elif unreged:
            hotkeys = cls.unregistered_hotkeys(coldkey, netuid=netuid) 
        else:
            hotkeys =  cls.hotkeys(coldkey)
        
        wallets = [cls.wallet_json(f'{coldkey}.{hotkey}' ) for hotkey in hotkeys]
        wallets = [w for w in wallets if w != None]


        hotkey_map = {hotkeys[i]: w['secretPhrase'] for i, w in enumerate(wallets)}
        
        coldkey_json = cls.coldkeypub_json(coldkey)
        
        if 'ss58Address' not in coldkey_json:
            coldkey_json = cls.coldkey_json(coldkey)
            
        
        coldkey_info = [f"btcli regen_coldkeypub --ss58 {coldkey_json['ss58Address']} --wallet.name {coldkey}"]

            
        if miners_only or no_miners:
            miners = cls.miners(netuid=netuid, network=network, reged=reged)
            c.print()

            if no_miners:
                assert miners_only == False
            if  miners_only:
                assert no_miners == False

        
            
        template = 'btcli regen_hotkey --wallet.name {coldkey} --wallet.hotkey {hotkey} --mnemonic {mnemonic}'
        for hk, hk_mnemonic in hotkey_map.items():
            wallet = f'{coldkey}.{hk}'
            
            if miners_only:
                if wallet not in miners :
                    continue
                
            if no_miners:
                if wallet in miners :
                    continue
            info = template.format(mnemonic=hk_mnemonic, coldkey=coldkey, hotkey=hk)
            
            coldkey_info.append(info)
            
        coldkey_info_text = '\n'.join(coldkey_info)
        if path is not None:
            cls.put_text(path, coldkey_info_text)
        # return coldkey_info
        
        return coldkey_info_text
    
    
    @classmethod
    def wallet_json(cls, wallet):
        path = cls.get_wallet_path(wallet)
        return cls.get_json(path)
    
    
    @classmethod
    def coldkey_json(cls, coldkey=default_coldkey):
        path = cls.coldkey_path(coldkey)
        coldkey_json = cls.get_json(path, None)
        if coldkey_json is None:
            coldkey_json = cls.coldkeypub_json(coldkey)
        return coldkey_json
    
    @classmethod
    def hotkey_json(cls, hotkey, coldkey=default_coldkey):
        path = cls.hotkey_path(hotkey, coldkey)
        coldkey_json = cls.get_json(path, {})
        return coldkey_json
    
    

    @classmethod
    def coldkeypub_json(cls, coldkey):
        path = cls.coldkeypub_path(coldkey)
        return cls.get_json(path)
    
    
    @classmethod
    def servers_online(cls):
        return 
    
    servers = servers_online
    @classmethod
    def servers(cls, **kwargs):

        return c.servers('server')
    
    @classmethod
    def wallet_json(cls, wallet):
        path = cls.get_wallet_path(wallet)
        return cls.get_json(path)

    
                
    @classmethod
    def get_top_uids(cls, k:int=100, 
                     netuid=None,
                     subtensor=None,
                     metagraph=None, 
                     return_dict=True,
                     **kwargs):

        if metagraph == None:
            metagraph = cls.get_metagraph(netuid=netuid, subtensor=subtensor, **kwargs)
        
        sorted_indices = torch.argsort(metagraph.incentive, descending=True)
        top_uids = sorted_indices[:k]
        if return_dict:
            
            top_uids = {uid: metagraph.incentive[uid].item() for i, uid in enumerate(top_uids.tolist())}
        
        return top_uids
    
    def top_uids(self,k=10):
        self.get_top_uids(metagraph=self.metagraph,k=k)
    def incentive(self ):
        return self.metagraph.incentive.data  
    def uids(self):
        return self.metagraph.uids.data
    def dividends(self):
        return self.metagraph.dividends.data
    
    @classmethod
    def start_node(cls, mode='docker', sudo=True):
        if mode == 'docker':
            return cls.cmd('docker-compose up -d', cwd=cls.chain_repo, verbose=True, sudo=sudo)
        else:
            raise NotImplementedError('Only docker mode is supported at this time.')
    
    @classmethod
    def node_logs(cls, mode='docker', sudo=True):
        if mode == 'docker':
            return cls.cmd('docker ps', cwd=cls.chain_repo, verbose=False, sudo=sudo)
        else:
            raise NotImplementedError('Only docker mode is supported at this time.')
    


    @classmethod
    def redeploy_zero_miners(cls, netuid=11, coldkey=default_coldkey, **kwargs):
        zero_miners = cls.zero_emission_miners(coldkey=coldkey, netuid=netuid)
        hotkeys = [int(m.split('.')[-1]) for m in zero_miners]
        cls.fleet(
                   hotkeys=hotkeys, 
                  refresh=True, 
                  refresh_ports=True, 
                  netuid=netuid, 

                  **kwargs)


        

    @classmethod
    def num_miners(cls,**kwargs):
        return len(cls.miners(**kwargs))
    
    n_miners = num_miners

    @classmethod
    def watchdog(cls, 
                 sync_interval:int=5,
                 print_interval:int = 10,
                 remote:bool=True):
        if remote:
            kwargs = c.locals2kwargs(locals())
            kwargs['remote'] = False
            return cls.remote_fn('watchdog', kwargs=kwargs)
            
        self = cls()
        time_start = c.time()
        time_elapsed = 0
        counts = { 'save':0}
        while True:
            time_elapsed = c.time() - time_start
            
            if time_elapsed % sync_interval == 0:
                self.sync()
                counts['sync'] += 1
            if time_elapsed % print_interval == 0:
                c.log(f"Watchdog: {time_elapsed} seconds elapsed COUNTS ->S {counts}")
            

    _reged_wallets = None
       
    def talk(self, 
             prompt:str = 'what is the whether', 
             role:str='assistant',  
            timeout:int=4, 
            n:int=10, 
            trials:int=3,
            n_jobs:int = 2,
            **kwargs):
        assert trials > 0, 'trials must be greater than 0'
        if self._reged_wallets == None:
            reged = self.reged()
            self._reged_wallets = [self.get_wallet(r) for r in reged]
        assert len(self._reged_wallets) > 0, 'No registered wallets found'
        wallet = c.choice(self._reged_wallets)
        d = bittensor.text_prompting_pool(keypair=wallet.hotkey, metagraph=self.metagraph)
        uids = c.shuffle(list(range(self.metagraph.n)))[:n]
        
        jobs = [d.async_forward(roles=[role], messages=[prompt], timeout=timeout, uids=uids) for i in range(n_jobs)]
        response = []
        for r in c.gather(jobs):
            response.extend(r)
        
        success_responses = [r.completion.strip() for r in response if r.return_code == 1]
        if len(success_responses) == 0:
            c.print(f'No successful responses for prompt {prompt} role {role} timeout {timeout} n {n}')
            return self.talk(prompt=prompt, role=role, timeout=timeout, n=n, trials=trials-1, **kwargs)
        return success_responses[0]


    @classmethod
    def model_fleet(cls, n=20):
        free_ports = c.free_ports(n=n)
        for i in range(n):
            cls.serve( name=f'model.bt.{i}', port=free_ports[i])





