
import scalecodec
from retry import retry
from typing import List, Dict, Union, Optional, Tuple
from substrateinterface import SubstrateInterface
from typing import List, Dict, Union, Optional, Tuple
from commune.utils.network import ip_to_int, int_to_ip
from rich.prompt import Confirm
from commune.modules.subspace.balance import Balance
from commune.modules.subspace.utils import (U16_MAX,  is_valid_address_or_public_key, )
from commune.modules.subspace.chain_data import (ModuleInfo, custom_rpc_type_registry)
import streamlit as st
import json
import os
import commune as c


class Subspace(c.Module):
    """
    Handles interactions with the subspace chain.
    """
    whitelist = ['modules']
    fmt = 'j'
    git_url = 'https://github.com/commune-ai/subspace.git'
    default_config = c.get_config('subspace', to_munch=False)
    token_decimals = default_config['token_decimals']
    network = default_config['network']
    chain = network
    libpath = chain_path = c.libpath + '/subspace'
    spec_path = f"{chain_path}/specs"
    netuid = default_config['netuid']

    def __init__( 
        self, 
        **kwargs,
    ):
        # c.print(self.config())
        self.set_config(kwargs=kwargs)

        
    def set_network(self, 
                network:str = network,
                url : str = None,
                websocket:str=None, 
                ss58_format:int=42, 
                type_registry:dict=None, 
                type_registry_preset='substrate-node-template',
                cache_region=None, 
                runtime_config=None, 
                ws_options=None, 
                auto_discover=True, 
                auto_reconnect=True, 
                verbose:bool=False,
                max_trials:int = 10,
                parallel_calls:bool=1,
                **kwargs):

        '''
        A specialized class in interfacing with a Substrate node.

        Parameters
       A specialized class in interfacing with a Substrate node.

        Parameters
        url : the URL to the substrate node, either in format <https://127.0.0.1:9933> or wss://127.0.0.1:9944
        
        ss58_format : The address type which account IDs will be SS58-encoded to Substrate addresses. Defaults to 42, for Kusama the address type is 2
        
        type_registry : A dict containing the custom type registry in format: {'types': {'customType': 'u32'},..}
        
        type_registry_preset : The name of the predefined type registry shipped with the SCALE-codec, e.g. kusama
        
        cache_region : a Dogpile cache region as a central store for the metadata cache
        
        use_remote_preset : When True preset is downloaded from Github master, otherwise use files from local installed scalecodec package
        
        ws_options : dict of options to pass to the websocket-client create_connection function
        : dict of options to pass to the websocket-client create_connection function
                
        '''
        from substrateinterface import SubstrateInterface
        
        if network == None:
            network = self.config.network
        if network == 'local':
            self.config.local = True
        chain = c.module('subspace.chain')
        trials = 0
        while trials < max_trials :
            trials += 1
            if url == None:
                url = chain.resolve_node_url(url=url, chain=network, local=self.config.local)
            
            self.url = url
            url = url.replace(c.ip(), '0.0.0.0')
            
            kwargs.update(url=url, 
                        websocket=websocket, 
                        ss58_format=ss58_format, 
                        type_registry=type_registry, 
                        type_registry_preset=type_registry_preset, 
                        cache_region=cache_region, 
                        runtime_config=runtime_config, 
                        ws_options=ws_options, 
                        auto_discover=auto_discover, 
                        auto_reconnect=auto_reconnect)
            try:
                self.substrate= SubstrateInterface(**kwargs)
                break
            except Exception as e:
                self.config.local = False
                url = None
        if trials == max_trials:
            c.print(f'Could not connect to {url}')
            return {'success': False, 'message': f'Could not connect to {url}'}

        response = {'network': network, 'url': url, 'success': True}

        return response

    def __repr__(self) -> str:
        return f'<Subspace: network={self.network}>'
    def __str__(self) -> str:

        return f'<Subspace: network={self.network}>'
    
    def shortyaddy(self, address, first_chars=4):
        return address[:first_chars] + '...' 


    def rank_modules(self,search=None, k='stake', n=10, modules=None, reverse=True, names=False, **kwargs):
        modules = self.modules(search=search, **kwargs) if modules == None else modules
        modules = sorted(modules, key=lambda x: x[k], reverse=reverse)
        if names:
            return [m['name'] for m in modules]
        if n != None:
            modules = modules[:n]
        return modules[:n]
    
    def top_modules(self,search=None, k='stake', n=10, modules=None, **kwargs):
        top_modules = self.rank_modules(search=search, k=k, n=n, modules=modules, reverse=True, **kwargs)
        return top_modules[:n]
    def top_module_keys(self,search=None, k='dividends ', n=10, modules=None, **kwargs):
        top_modules = self.rank_modules(search=search, k=k, n=n, modules=modules, reverse=True, **kwargs)
        return [m['key'] for m in top_modules[:n]]
    

    best = best_modules = top_modules
    
    def bottom_modules(self,search=None, k='stake', n=None, modules=None, **kwargs):
        bottom_modules = self.rank_modules(search=search, k=k, n=n, modules=modules, reverse=False, **kwargs)
        return bottom_modules[:n]
    
    worst = worst_modules = bottom_modules


    def key2tokens(self, network = None, fmt=fmt, decimals=2):
        key2tokens = {}
        key2balance = self.key2balance(network=network, fmt=fmt, decimals=decimals)
        for key, balance in key2balance.items():
            if key not in key2tokens:
                key2tokens[key] = 0
            key2tokens[key] += balance
            
        for netuid in self.netuids():
            key2stake = self.key2stake(network=network, fmt=fmt, netuid=netuid, decimals=decimals)

            for key, stake in key2stake.items():
                if key not in key2tokens:
                    key2tokens[key] = 0
                key2tokens[key] += stake
            
        return key2tokens



    key2value = key2tokens    
    def names2uids(self, names: List[str] = None, **kwargs ) -> Union['torch.tensor', list]:
        # queries updated network state
        names = names or []
        name2uid = self.name2uid(**kwargs)
        uids = []
        for name in names:
            if name in name2uid:
                uids += [name2uid[name]]
        return uids
    
    def get_netuid_for_subnet(self, network: str = None) -> int:
        return {'commune': 0}.get(network, 0)


    def get_existential_deposit(
        self,
        block: Optional[int] = None,
        fmt = 'nano'
    ) -> Optional[Balance]:
        """ Returns the existential deposit for the chain. """
        result = self.query_constant(
            module_name='Balances',
            constant_name='ExistentialDeposit',
            block = block,
        )
        
        if result is None:
            return None
        
        return self.format_amount( result, fmt = fmt )
        

    def wasm_file_path(self):
        wasm_file_path = self.libpath + '/target/release/wbuild/node-subspace-runtime/node_subspace_runtime.compact.compressed.wasm'
        return wasm_file_path

    def stake_from(self, netuid = 0, block=None, update=False, network=network):
        return {k: list(map(list,v)) for k,v in self.query_map('StakeFrom', netuid, block=block, update=update, network=network)}
    
    def delegation_fee(self, netuid = 0, block=None, network=None, update=False):
        return {k:v for k,v in self.query_map('DelegationFee', netuid, block=block ,update=update, network=network)}
    
    def stake_to(self, netuid = None, network=None, block=None, update=False):
        network = self.resolve_network(network)
        netuid  = self.resolve_netuid(netuid)
        return {k: list(map(list,v)) for k,v in self.query_map('StakeTo', netuid, block=block, update=update)}

    def query(self, name:str,  params = None, block=None,  network: str = network, module:str='SubspaceModule', update=False, netuid=None):
        
        """
        query a subspace storage function with params and block.
        """
        
        if params == None:
            params = []
        else:
            if not isinstance(params, list):
                params = [params]
        params_str = ','.join([str(p) for p in params])
        cache_path = f'query/{name}_params_{params_str}'
        if not update:
            value = self.get(cache_path, None)
            if value != None:
                return value
        

        if not isinstance(params, list):
            params = [params]
        network = self.resolve_network(network)

        with self.substrate as substrate:
            response =  substrate.query(
                module=module,
                storage_function = name,
                block_hash = None if block == None else substrate.get_block_hash(block), 
                params = params
            )
            
        value =  response.value

        self.put(cache_path, value)
        return value

    def query_constant( self, 
                        constant_name: str, 
                       module_name: str = 'SubspaceModule', 
                       block: Optional[int] = None ,
                       network: str = None) -> Optional[object]:
        """ 
        Gets a constant from subspace with
        module_name, constant_name, and block. 
        """

        network = self.resolve_network(network)

        with self.substrate as substrate:
            value =  substrate.query(
                module=module_name,
                storage_function=constant_name,
                block_hash = None if block == None else substrate.get_block_hash(block)
            )
            
        return value
    
    


    def query_map(self, name: str, 
                  params: list = None,
                  block: Optional[int] = None, 
                  network:str = 'main',
                  page_size=1000,
                  max_results=100000,
                  module='SubspaceModule',
                  update: bool = True
                  ) -> Optional[object]:
        """ Queries subspace map storage with params and block. """

        if params == None:
            params = []
        if params == None:
            params = []
        if params != None and not isinstance(params, list):
            params = [params]
        
        params_str = '-'.join([str(p) for p in params])
        path = f'cache/network.{name}_params_{params_str}.json'
        if not update:
            value = self.get(path, None)
            if value != None:
                return value
        
        network = self.resolve_network(network)

        with self.substrate as substrate:
            block_hash = None if block == None else substrate.get_block_hash(block)
            qmap =  substrate.query_map(
                module=module,
                storage_function = name,
                params = params,
                page_size = page_size,
                max_results = max_results,
                block_hash =block_hash
            )

        qmap = [[k.value,v.value] for k,v  in qmap]
        self.put(path, qmap)
                
        return qmap

    def runtime_spec_version(self, network:str = 'main'):
        # Get the runtime version
        self.resolve_network(network=network)
        c.print(self.substrate.runtime_config.__dict__)
        runtime_version = self.query_constant(module_name='System', constant_name='SpVersionRuntimeVersion')
        return runtime_version
        
        
    def stale_modules(self, *args, **kwargs):
        modules = self.my_modules(*args, **kwargs)
        servers = c.servers(network='local')
        servers = c.shuffle(servers)
        return [m['name'] for m in modules if m['name'] not in servers]

    #####################################
    #### Hyper parameter calls. ####
    #####################################

    """ Returns network ImmunityPeriod hyper parameter """
    def immunity_period (self, netuid: int = None, block: Optional[int] = None, network :str = None ) -> Optional[int]:
        netuid = self.resolve_netuid( netuid )
        return self.query("ImmunityPeriod",params=netuid, block=block )


    """ Returns network MinAllowedWeights hyper parameter """
    def min_allowed_weights (self, netuid: int = None, block: Optional[int] = None ) -> Optional[int]:
        netuid = self.resolve_netuid( netuid )
        return self.query("MinAllowedWeights", params=[netuid], block=block)
    """ Returns network MinAllowedWeights hyper parameter """
    def max_allowed_weights (self, netuid: int = None, block: Optional[int] = None, **kwargs ) -> Optional[int]:
        netuid = self.resolve_netuid( netuid )
        return self.query("MaxAllowedWeights", params=[netuid], block=block, **kwargs)

    """ Returns network SubnetN hyper parameter """
    def n(self, network = network , netuid: int = None, block: Optional[int] = None ) -> int:
        self.resolve_network(network)
        netuid = self.resolve_netuid( netuid )
        return self.query('N', params=[netuid], block=block )

    """ Returns network MaxAllowedUids hyper parameter """
    def max_allowed_uids (self, netuid: int = None, block: Optional[int] = None, **kwargs ) -> Optional[int]:
        netuid = self.resolve_netuid( netuid )
        return self.query('MaxAllowedUids', netuid, block=block , **kwargs)

    """ Returns network Tempo hyper parameter """
    def tempo (self, netuid: int = None, block: Optional[int] = None) -> int:
        netuid = self.resolve_netuid( netuid )
        return self.query('Tempo', params=[netuid], block=block)

    ##########################
    #### Account functions ###
    ##########################
    
    """ Returns network Tempo hyper parameter """
    def stakes(self, netuid: int = None, block: Optional[int] = None, fmt:str='nano', max_staleness = 100,network=None, update=False, **kwargs) -> int:
        stakes =  self.query_map('Stake', netuid , update=update, **kwargs)
        return {k: self.format_amount(v, fmt=fmt) for k,v in stakes}

    """ Returns the stake under a coldkey - hotkey pairing """
    
    
    
    def resolve_key_ss58(self, key:str, network='main', netuid:int=0):
        if key == None:
            key = c.get_key(key)

        if isinstance(key, str):
            if c.valid_ss58_address(key):
                return key
            else:
                if c.key_exists( key ):
                    key = c.get_key( key )
                    key_address = key.ss58_address
                else:
                    name2key = self.name2key()
                    assert key in name2key, f"Invalid Key {key} as it should have ss58_address attribute."
                    if key in name2key:
                        key_address = name2key[key]
                    else:
   
                        raise Exception(f"Invalid Key {key} as it should have ss58_address attribute.")   
        # if the key has an attribute then its a key
        elif hasattr(key, 'ss58_address'):
            key_address = key.ss58_address
        c.print(key, 'key')
        assert c.valid_ss58_address(key_address), f"Invalid Key {key_address} as it should have ss58_address attribute."
        return key_address

    def subnet2modules(self, network:str='main', **kwargs):
        subnet2modules = {}
        self.resolve_network(network)

        for netuid in self.netuids():
            c.print(f'Getting modules for SubNetwork {netuid}')
            subnet2modules[netuid] = self.my_modules(netuid=netuid, **kwargs)

        return subnet2modules
    
    def module2netuids(self, network:str='main', **kwargs):
        subnet2modules = self.subnet2modules(network=network, **kwargs)
        module2netuids = {}
        for netuid, modules in subnet2modules.items():
            for module in modules:
                if module['name'] not in module2netuids:
                    module2netuids[module['name']] = []
                module2netuids[module['name']] += [netuid]
        return module2netuids
    @classmethod
    def from_nano(cls,x):
        return x / (10**cls.token_decimals)
    to_token = from_nano
    @classmethod
    def to_nanos(cls,x):
        """
        Converts a token amount to nanos
        """
        return x * (10**cls.token_decimals)
    from_token = to_nanos

    @classmethod
    def format_amount(cls, x, fmt='nano', decimals = None):
        if fmt in ['nano', 'n']:
            x =  x
        elif fmt in ['token', 'unit', 'j', 'J']:
            x = cls.to_token(x)
        
        if decimals != None:
            x = c.round_decimals(x, decimals=decimals)

        return x
    
    def get_stake( self, key_ss58: str, block: Optional[int] = None, netuid:int = None , fmt='j', update=False ) -> Optional['Balance']:
        
        key_ss58 = self.resolve_key_ss58( key_ss58 )
        netuid = self.resolve_netuid( netuid )
        stake = self.query( 'Stake',params=[netuid, key_ss58], block=block , update=update)
        return self.format_amount(stake, fmt=fmt)


    def get_staked_modules(self, key : str , netuid=None, **kwargs) -> Optional['Balance']:
        modules = self.modules(netuid=netuid, **kwargs)
        key_address = self.resolve_key_ss58( key )
        staked_modules = {}
        for module in modules:
            for k,v in module['stake_from']:
                if k == key_address:
                    staked_modules[module['name']] = v

        return staked_modules
        

    def get_stake_to( self, key: str = None, module_key=None, block: Optional[int] = None, netuid:int = None , fmt='j' , names:bool = True, network=None, **kwargs) -> Optional['Balance']:
        network = self.resolve_network(network)
        key_address = self.resolve_key_ss58( key )
        netuid = self.resolve_netuid( netuid )
        stake_to =  {k: self.format_amount(v, fmt=fmt) for k, v in self.query( 'StakeTo', params=[netuid, key_address], block=block, **kwargs )}

        if module_key != None:
            module_key = self.resolve_key_ss58( module_key )
            stake_to : int ={ k:v for k, v in stake_to}.get(module_key, 0)

        if names:
            key2name = self.key2name(netuid=netuid)
            stake_to = {key2name[k]:v for k,v in stake_to.items()}
        return stake_to
    get_staketo = get_stake_to
    
    def get_value(self, key):
        value = self.get_balance(key)
        for netuid in self.netuids():
            stake_to = self.get_stake_to(key)
            value += sum(stake_to.values())
        return value

    

    def get_stakers( self, key: str, block: Optional[int] = None, netuid:int = None , fmt='j' ) -> Optional['Balance']:
        stake_from = self.get_stake_from(key=key, block=block, netuid=netuid, fmt=fmt)
        key2module = self.key2module(netuid=netuid)
        return {key2module[k]['name'] : v for k,v in stake_from}
        
    def get_stake_from( self, key: str, from_key=None, block: Optional[int] = None, netuid:int = None, fmt='j', update=True  ) -> Optional['Balance']:
        key = self.resolve_key_ss58( key )
        netuid = self.resolve_netuid( netuid )
        state_from =  [(k, self.format_amount(v, fmt=fmt)) for k, v in self.query( 'StakeFrom', block=block, params=[netuid, key], update=update )]
 
        if from_key is not None:
            from_key = self.resolve_key_ss58( from_key )
            state_from ={ k:v for k, v in state_from}.get(from_key, 0)

        return state_from
    
    
    def get_total_stake_from( self, key: str, from_key=None, block: Optional[int] = None, netuid:int = None, fmt='j', update=True  ) -> Optional['Balance']:
        stake_from = self.get_stake_from(key=key, from_key=from_key, block=block, netuid=netuid, fmt=fmt, update=update)
        return sum([v for k,v in stake_from])
    
    get_stakefrom = get_stake_from 


    ###########################
    #### Global Parameters ####
    ###########################

    @property
    def block(self, network:str=None, trials=100) -> int:
        return self.get_block(network=network)


    
   
    @classmethod
    def archived_blocks(cls, network:str=network, reverse:bool = True) -> List[int]:
        # returns a list of archived blocks 
        
        blocks =  [f.split('.B')[-1].split('.json')[0] for f in cls.glob(f'archive/{network}/state.B*')]
        blocks = [int(b) for b in blocks]
        sorted_blocks = sorted(blocks, reverse=reverse)
        return sorted_blocks

    @classmethod
    def oldest_archive_path(cls, network:str=network) -> str:
        oldest_archive_block = cls.oldest_archive_block(network=network)
        assert oldest_archive_block != None, f"No archives found for network {network}"
        return cls.resolve_path(f'state_dict/{network}/state.B{oldest_archive_block}.json')
    @classmethod
    def newest_archive_block(cls, network:str=network) -> str:
        blocks = cls.archived_blocks(network=network, reverse=True)
        return blocks[0]
    @classmethod
    def newest_archive_path(cls, network:str=network) -> str:
        oldest_archive_block = cls.newest_archive_block(network=network)
        return cls.resolve_path(f'archive/{network}/state.B{oldest_archive_block}.json')
    @classmethod
    def oldest_archive_block(cls, network:str=network) -> str:
        blocks = cls.archived_blocks(network=network, reverse=True)
        if len(blocks) == 0:
            return None
        return blocks[-1]

    @classmethod
    def loop(cls, 
                network = network,
                netuid:int = 0,
                interval = 60,
                modules = ['model'], 
                sleep:float=10,
                remote:bool=True, **kwargs):
        if remote:
            kwargs = c.locals2kwargs(locals())
            kwargs['remote'] = False
            return cls.remote_fn('loop', kwargs=kwargs)
        subspace = cls(network=network)
        last_update_time = 0
        while True:
            lag = cls.lag()
            c.print({'lag': lag, 'interval': interval, 'last_update_time': last_update_time, 'sleep': sleep, 'block': subspace.block, 'network': network})
            if lag > interval:
                c.print(f'Updating SubNetwork {netuid} at block {subspace.block}')
                c.print(subspace.sync())
                last_update_time = lag
            c.sleep(sleep)


    state_dict_cache = {}
    def state_dict(self,
                    network=network, 
                    key: Union[str, list]=None, 
                    inlcude_weights:bool=False, 
                    update:bool=False, 
                    verbose:bool=False, 
                    netuids: List[int] = None,
                    parallel:bool=True,
                    save:bool = True,
                    timeout = 10,
                    **kwargs):
        
        # cache and update are mutually exclusive 
        if  update == False:
            c.print('Loading state_dict from cache', verbose=verbose)
            state_dict = self.latest_archive(network=network)
            if len(state_dict) > 0:
                self.state_dict_cache = state_dict

        if len(self.state_dict_cache) == 0 :

            # get the latest block
            block = self.block
            netuids = self.netuids(network=network, block=block, update=True)
            c.print(f'Getting state_dict for {netuids} at block {block}', verbose=verbose)

            subnets = [self.subnet(netuid=netuid, network=network, block=block, update=True, fmt='nano') for netuid in netuids]

            c.print(f'Getting modules for {netuids} at block {block}', verbose=verbose)
        
            state_dict = {'subnets': subnets, 
                        'modules': [self.modules(netuid=netuid, network=network, include_weights=inlcude_weights, block=block, update=True, parallel=parallel) for netuid in netuids],
                        'stake_to': [self.stake_to(network=network, block=block, update=True, netuid=netuid) for netuid in netuids],
                        'balances': self.balances(network=network, block=block, update=True),
                        'block': block,
                        'network': network,
                        }

            if save:

                path = f'state_dict/{network}.block-{block}-time-{int(c.time())}'
                c.print(f'Saving state_dict to {path}')
                
                self.put(path, state_dict) # put it in storage
                self.state_dict_cache = state_dict # update it in memory
            
        
        state_dict = c.copy(self.state_dict_cache)
        
        # if key is a string
        if key in state_dict:
            return state_dict[key]
    
        # if key is a list of keys
        if isinstance(key,list):
            return {k:state_dict[k] for k in key}
        return state_dict
    @classmethod
    def ls_archives(cls, network=network):
        if network == None:
            network = cls.network 
        return [f for f in cls.ls(f'state_dict') if os.path.basename(f).startswith(network)]

    
    @classmethod
    def block2archive(cls, network=network):
        paths = cls.ls_archives(network=network)

        block2archive = {int(p.split('-')[-1].split('-time')[0]):p for p in paths if p.endswith('.json') and f'{network}.block-' in p}
        return block2archive
    @classmethod
    def time2archive(cls, network=network):
        paths = cls.ls_archives(network=network)

        block2archive = {int(p.split('time-')[-1].split('.json')[0]):p for p in paths if p.endswith('.json') and f'time-' in p}
        return block2archive

    @classmethod
    def datetime2archive(cls,search=None, network=network):
        time2archive = cls.time2archive(network=network)
        datetime2archive = {c.time2datetime(time):archive for time,archive in time2archive.items()}
        # sort by datetime
        # 
        datetime2archive = {k:v for k,v in sorted(datetime2archive.items(), key=lambda x: x[0])}
        if search != None:
            datetime2archive = {k:v for k,v in datetime2archive.items() if search in k}
        return datetime2archive



    @classmethod
    def latest_archive_path(cls, network=network):
        latest_archive_time = cls.latest_archive_time(network=network)
    
        if latest_archive_time == None:
            return None
        time2archive = cls.time2archive(network=network)
        return time2archive[latest_archive_time]

    @classmethod
    def latest_archive_time(cls, network=network):
        time2archive = cls.time2archive(network=network)
        if len(time2archive) == 0:
            return None
        latest_time = max(time2archive.keys())
        return latest_time

    @classmethod
    def lag(cls, network:str = network):
        return c.timestamp() - cls.latest_archive_time(network=network) 

    @classmethod
    def latest_archive_datetime(cls, network=network):
        latest_archive_time = cls.latest_archive_time(network=network)
        assert latest_archive_time != None, f"No archives found for network {network}"
        return c.time2datetime(latest_archive_time)

    @classmethod
    def archive_staleness(self, network=network):
        return c.time() - self.latest_archive_time(network=network)

    @classmethod
    def latest_archive(cls, network=network):
        path = cls.latest_archive_path(network=network)
        if path == None:
            return {}
        return cls.get(path, {})
    
    def sync(self, network=None, remote:bool=True, local:bool=True, save:bool=True, **kwargs):

        network = self.resolve_network(network)
        response = self.state_dict(update=True, network=network, parallel=True)
        self.get_namespace(update=True)
        return {'success': True, 'block': response['block']}

    def sync_loop(self, interval=60, network=None, remote:bool=True, local:bool=True, save:bool=True):
        start_time = 0
        while True:
            current_time = c.timestamp()
            elapsed = current_time - start_time
            if elapsed > interval:
                c.print('SYNCING AND UPDATING THE SERVERS_INFO')
                c.print(c.infos(update=True, network='local'))
                response = self.sync(network=network, remote=remote, local=local, save=save)
                c.print(response)
                start_time = current_time
            c.sleep(interval)

    def subnet_exists(self, subnet:str, network=None) -> bool:
        subnets = self.subnets(network=network)
        return bool(subnet in subnets)

    def subnet_states(self, *args, **kwargs):
        subnet_states = []
        for netuid in self.netuids():
            subnet_state = self.subnet(*args,  netuid=netuid, **kwargs)
            subnet_states.append(subnet_state)
        return subnet_states

    def total_stake(self, network=network, block: Optional[int] = None, netuid:int=None, fmt='j', update=False) -> 'Balance':
        self.resolve_network(network)
        netuid = self.resolve_netuid(netuid)
        return self.format_amount(self.query( "TotalStake", params=[netuid], block=block, network=network , update=update), fmt=fmt)

    def total_balance(self, network=network, block: Optional[int] = None, fmt='j') -> 'Balance':
        return sum(list(self.balances(network=network, block=block, fmt=fmt).values()))

    def total_supply(self, network=network, block: Optional[int] = None, fmt='j', update=False) -> 'Balance':
        state = self.state_dict(network=network, block=block, update=update)
        total_balance = sum(list(state['balances'].values()))
        total_stake = sum([sum([v[1] for v in stake_to]) for k,stake_to in state['stake_to'][0].items()])
        return self.format_amount(total_balance + total_stake, fmt=fmt)
    
    mcap = market_cap = total_supply
            
        
    def subnet_params(self, 
                    netuid=netuid,
                    network = network,
                    block : Optional[int] = None,
                    update = False,
                    trials = 3,
                    timeout = 10,
                    parallel = True,
                    fmt:str='j') -> list:

        netuid = self.resolve_netuid(netuid)
        path = f'cache/network.subnet_params.{netuid}.json'

        if not update:
            value = self.get(path, None)
            if value != None:
                return value
            
        c.print(f'Getting subnet params for SubNetwork {netuid}')
        
        network = self.resolve_network(network)
        kwargs = dict( netuid = netuid , block=block, update=update)

        if parallel:
            async def query(**kwargs ):
                return self.query(**kwargs)
        else:
            query = self.query
        name2job = {
                'tempo': [query, dict(name='Tempo')],
                'immunity_period': [query, dict(name='ImmunityPeriod')],
                'min_allowed_weights': [query, dict(name='MinAllowedWeights')],
                'max_allowed_weights': [query, dict(name='MaxAllowedWeights')],
                'max_allowed_uids': [query, dict(name='MaxAllowedUids')],
                'min_stake': [query, dict(name='MinStake')],
                'founder': [query, dict(name='Founder')], 
                'founder_share': [query, dict(name='FounderShare')],
                'incentive_ratio': [query, dict(name='IncentiveRatio')],
                'trust_ratio': [query, dict(name='TrustRatio')],
                'vote_threshold': [query, dict(name='VoteThresholdSubnet')],
                'vote_mode': [query, dict(name='VoteModeSubnet')],
                'self_vote': [query, dict(name='SelfVote')],
                'name': [query, dict(name='SubnetNames')]
            }
        name2result = {}

        for i in range(1): 
            for k,(fn, fn_kwargs) in name2job.items():
                remote_kwargs = dict(**fn_kwargs, **kwargs)
                if parallel:
                    name2result[k] = query(**remote_kwargs)
                else:
                    name2result[k] = query(**remote_kwargs)

            if parallel:
                futures = [v for k,v in name2result.items()]
                results = c.wait(futures, timeout=timeout)
                name2result = {k:v for k,v in zip(name2result.keys(), results)}
        
            for name, result in name2result.items():
                if not c.is_error(result):
                    name2result[name] = result
                    name2job.pop(name)


        subnet = {k:name2result[k] for k in name2result}

        for k in ['min_stake']:
            subnet[k] = self.format_amount(subnet[k], fmt=fmt)
        self.put(path, subnet)

        c.print(f'Got subnet params for SubNetwork {netuid}')
        return subnet
    
    subnet = subnet_params
            

    def get_total_subnets( self, block: Optional[int] = None ) -> int:
        return self.query( 'TotalSubnets', block=block )      
    
    def get_emission_value_by_subnet( self, netuid: int = None, block: Optional[int] = None ) -> Optional[float]:
        netuid = self.resolve_netuid( netuid )
        return Balance.from_nano( self.query( 'EmissionValues', block=block, params=[ netuid ] ) )

    def is_registered( self, key: str, netuid: int = None, block: Optional[int] = None) -> bool:
        netuid = self.resolve_netuid( netuid )
        name2key = self.name2key(netuid=netuid)
        if key in name2key:
            key = name2key[key]
        if not c.valid_ss58_address(key):
            return False
        is_reged =  bool(self.query('Uids', block=block, params=[ netuid, key ]))
        return is_reged

    def get_uid_for_key_on_subnet( self, key_ss58: str, netuid: int, block: Optional[int] = None) -> int:
        return self.query( 'Uids', block=block, params=[ netuid, key_ss58 ] )  


    def register_subnets( self, *subnets, module='vali', **kwargs ) -> Optional['Balance']:
        if len(subnets) == 1:
            subnets = subnets[0]
        subnets = list(subnets)
        assert isinstance(subnets, list), f"Subnets must be a list. Got {subnets}"
        
        responses = []
        for subnet in subnets:
            tag = subnet
            response = c.register(module=module, tag=tag, subnet=subnet , **kwargs)
            c.print(response)
            responses.append(response)

        return responses
        

    def total_emission( self, netuid: int = 0, block: Optional[int] = None, fmt:str = 'j', **kwargs ) -> Optional[float]:
        total_emission =  sum(self.emission(netuid=netuid, block=block, **kwargs))
        return self.format_amount(total_emission, fmt=fmt)


    def regblock(self, netuid: int = None, block: Optional[int] = None, update=False ) -> Optional[float]:
        netuid = self.resolve_netuid( netuid )
        return {k:v for k,v  in self.query_map('RegistrationBlock',params=netuid, block=block, update=update ) }

    def age(self, netuid: int = None) -> Optional[float]:
        netuid = self.resolve_netuid( netuid )
        regblock = self.regblock(netuid=netuid)
        block = self.block
        age = {}
        for k,v in regblock.items():
            age[k] = block - v
        return age

     
    def global_params(self, network: str = network, netuid: int = 0 ) -> Optional[float]:

        """
        max_name_length: Option<u16>,
		max_allowed_subnets: Option<u16>,
		max_allowed_modules: Option<u16>,
		max_registrations_per_block: Option<u16>,
		unit_emission: Option<u64>,
		tx_rate_limit: Option<u64>,
        
        """
        self.resolve_network(network)
        netuid = self.resolve_netuid(netuid)
        global_params = {}
        global_params['max_name_length'] = self.query_constant( 'MaxNameLength')
        global_params['max_allowed_subnets'] = self.query_constant( 'MaxAllowedSubnets')
        global_params['max_allowed_modules'] = self.query_constant( 'MaxAllowedModules' )
        global_params['max_registrations_per_block'] = self.query_constant( 'MaxRegistrationsPerBlock' )
        global_params['unit_emission'] = self.query_constant( 'UnitEmission' )
        global_params['tx_rate_limit'] = self.query_constant( 'TxRateLimit' )
        global_params['vote_threshold'] = self.query_constant( 'GlobalVoteThreshold' )
        global_params['vote_mode'] = self.query_constant( 'VoteModeGlobal' )
        global_params['max_proposals'] = self.query_constant( 'MaxProposals' )

        for k,v in global_params.items():
            global_params[k] = v.value
        return global_params



    def get_balance(self, key: str = None , block: int = None, fmt='j', network=None, update=True) -> Balance:
        r""" Returns the token balance for the passed ss58_address address
        Args:
            address (Substrate address format, default = 42):
                ss58 chain address.
        Return:
            balance (bittensor.utils.balance.Balance):
                account balance
        """
        key_ss58 = self.resolve_key_ss58( key )
        result = 0 
        if not update:
            balances = self.balances(network=network, block=block, update=update, fmt=fmt)
            result =  balances.get(key_ss58, result)
        else:
            self.resolve_network(network)
            try:
                @retry(delay=2, tries=3, backoff=2, max_delay=4)
                def make_substrate_call_with_retry():
                    with self.substrate as substrate:
                        return substrate.query(
                            module='System',
                            storage_function='Account',
                            params=[key_ss58],
                            block_hash = None if block == None else substrate.get_block_hash( block )
                        )
                result = make_substrate_call_with_retry()
            except scalecodec.exceptions.RemainingScaleBytesNotEmptyException:
                c.critical("Your key it legacy formatted, you need to run btcli stake --ammount 0 to reformat it." )

        return  self.format_amount(result['data']['free'].value , fmt=fmt)
        
    balance =  get_balance

    def balances(self,fmt:str = 'n', network:str = network, block: int = None, n = None, update=False ) -> Dict[str, Balance]:
        path = f'cache/balances.{network}.block-{block}'
        balances = {}
        if not update:
            balances = self.get(path, {})

        if len(balances) == 0:
            network = self.resolve_network(network)
            with self.substrate as substrate:
                result = substrate.query_map(
                    module='System',
                    storage_function='Account',
                    block_hash = None if block == None else substrate.get_block_hash( block )
                )
            balances =  {r[0].value:r[1]['data']['free'].value for r in result}
            self.put(path, balances)

        for k, v in balances.items():
            balances[k] = self.format_amount(v, fmt=fmt)
        # sort by decending balance
        balances = {k:v for k,v in sorted(balances.items(), key=lambda x: x[1], reverse=True)}
        if isinstance(n, int):
            balances = {k:v for k,v in list(balances.items())[:n]}


        return balances
    
    
    def resolve_network(self, network: Optional[int] = None) -> int:
        if  not hasattr(self, 'substrate'):
            self.set_network(network)

        if network == None:
            network = self.network
        
        return network
    
    def resolve_subnet(self, subnet: Optional[int] = None) -> int:
        if isinstance(subnet, int):
            assert subnet in self.netuids()
            subnet = self.netuid2subnet(netuid=subnet)
        subnets = self.subnets()
        assert subnet in subnets, f"Subnet {subnet} not found in {subnets} for chain {self.network}"
        return subnet


    def subnets(self, **kwargs) -> Dict[int, str]:
        subnets = [s['name'] for s in self.subnet_states(**kwargs)]
        return subnets
    
    def netuids(self, network=None, update=False, block=None) -> Dict[int, str]:
        return sorted(list(self.subnet_namespace(network=network, update=update, block=block).values()))

    def subnet_names(self, network=network , update=False, block=None, **kwargs) -> Dict[str, str]:
        records = self.query_map('SubnetNames', update=update, network=network, block=block, **kwargs)
        return {k:v for k,v in records}

    def subnet2netuid(self, network=network, **kwargs ) -> Dict[str, str]:
        records = self.query_map('SubnetNames', **kwargs)
        return {v:k for k,v in records}

    subnet_namespace = subnet2netuid
    
    def netuid2subnet(self, netuid = None, network=network):
        subnet2netuid = {v:k for k,v in self.subnet2netuid(network=network).items()}
        if netuid != None:
            return subnet2netuid.get(netuid, None)
        return subnet2netuid
    def subnet2netuid(self,subnet:str = None):
        subnet2netuid = self.subnet_namespace
        if subnet != None:
            return subnet2netuid.get(subnet, None)
        return subnet2netuid
        

    def resolve_netuid(self, netuid: int = None, network=network) -> int:
        '''
        Resolves a netuid to a subnet name.
        '''
        if netuid == None:
            # If the netuid is not specified, use the default.
            netuid = 0

        if isinstance(netuid, str):
            # If the netuid is a subnet name, resolve it to a netuid.
            netuid = int(self.subnet_namespace(network=network).get(netuid))
        elif isinstance(netuid, int):
            if netuid == 0: 
                return netuid
            # If the netuid is an integer, ensure it is valid.
            assert netuid in self.netuids(), f"Netuid {netuid} not found in {self.netuids()} for chain {self.chain}"
            
        assert isinstance(netuid, int), "netuid must be an integer"
        return netuid
    
    resolve_net = resolve_subnet = resolve_netuid


    def key2name(self, key: str = None, netuid: int = None) -> str:
        modules = self.keys()
        key2name =  { m['key']: m['name']for m in modules}
        if key != None:
            return key2name[key]
        
    def name2uid(self,search:str=None, netuid: int = None, network: str = None) -> int:
        uid2name = self.uid2name(netuid=netuid, network=network)
        return {v:k for k,v in uid2name.items()}

    
        
    def name2key(self, search:str=None, network=network, netuid: int = None, update=False ) -> Dict[str, str]:
        # netuid = self.resolve_netuid(netuid)
        self.resolve_network(network)
        names = self.names(netuid=netuid, update=update)
        keys = self.keys(netuid=netuid, update=update)
        name2key =  { n: k for n, k in zip(names, keys)}
        if search != None:
            name2key = {k:v for k,v in name2key.items() if search in k}
            if len(name2key) == 1:
                return list(name2key.values())[0]
        return name2key





    def key2name(self,search=None, netuid: int = None, network=network, update=False) -> Dict[str, str]:
        return {v:k for k,v in self.name2key(search=search, netuid=netuid, network=network, update=update).items()}
        
    def is_unique_name(self, name: str, netuid=None):
        return bool(name not in self.get_namespace(netuid=netuid))



    def name2inc(self, name: str = None, netuid: int = netuid, nonzero_only:bool=True) -> int:
        name2uid = self.name2uid(name=name, netuid=netuid)
        incentives = self.incentive(netuid=netuid)
        name2inc = { k: incentives[uid] for k,uid in name2uid.items() }

        if name != None:
            return name2inc[name]
        
        name2inc = dict(sorted(name2inc.items(), key=lambda x: x[1], reverse=True))


        return name2inc



    def top_valis(self, netuid: int = netuid, n:int = 10, **kwargs) -> Dict[str, str]:
        name2div = self.name2div(name=None, netuid=netuid, **kwargs)
        name2div = dict(sorted(name2div.items(), key=lambda x: x[1], reverse=True))
        return list(name2div.keys())[:n]

    def name2div(self, name: str = None, netuid: int = netuid, nonzero_only: bool = True) -> int:
        name2uid = self.name2uid(name=name, netuid=netuid)
        dividends = self.dividends(netuid=netuid)
        name2div = { k: dividends[uid] for k,uid in name2uid.items() }
    
        if nonzero_only:
            name2div = {k:v for k,v in name2div.items() if v != 0}

        name2div = dict(sorted(name2div.items(), key=lambda x: x[1], reverse=True))
        if name != None:
            return name2div[name]
        return name2div

    @property
    def block_time(self):
        return self.config.block_time
    
    def epoch_time(self, netuid=None, network=None):
        return self.subnet(netuid=netuid, network=network)['tempo']*self.block_time

    def blocks_per_day(self, netuid=None, network=None):
        return 24*60*60/self.block_time
    

    def epochs_per_day(self, netuid=None, network=None):
        return 24*60*60/self.epoch_time(netuid=netuid, network=network)
    
    def emission_per_epoch(self, netuid=None, network=None):
        return self.subnet(netuid=netuid, network=network)['emission']*self.epoch_time(netuid=netuid, network=network)

    def get_block(self, network=None, block_hash=None): 
        self.resolve_network(network)
        return self.substrate.get_block( block_hash=block_hash)['header']['number']

    def seconds_per_epoch(self, netuid=None, network=None):
        self.resolve_network(network)
        netuid =self.resolve_netuid(netuid)
        return self.block_time * self.subnet(netuid=netuid)['tempo']

    
    def get_module(self, name:str = None, key=None, netuid=None, **kwargs) -> ModuleInfo:
        if key != None:
            module = self.key2module(key=key, netuid=netuid, **kwargs)
        if name != None:
            module = self.name2module(name=name, netuid=netuid, **kwargs)
            
        return module

    @property
    def null_module(self):
        return {'name': None, 'key': None, 'uid': None, 'address': None, 'stake': 0, 'balance': 0, 'emission': 0, 'incentive': 0, 'dividends': 0, 'stake_to': {}, 'stake_from': {}, 'weight': []}
        
        
    def name2module(self, name:str = None, netuid: int = None, **kwargs) -> ModuleInfo:
        modules = self.modules(netuid=netuid, **kwargs)
        name2module = { m['name']: m for m in modules }
        default = {}
        if name != None:
            return name2module.get(name, self.null_module)
        return name2module
        
        
        
        
        
    def key2module(self, key: str = None, netuid: int = None, default: dict =None, **kwargs) -> Dict[str, str]:
        modules = self.modules(netuid=netuid, **kwargs)
        key2module =  { m['key']: m for m in modules }
        
        if key != None:
            key_ss58 = self.resolve_key_ss58(key)
            return  key2module.get(key_ss58, default if default != None else {})
        return key2module
        
    def module2key(self, module: str = None, **kwargs) -> Dict[str, str]:
        modules = self.modules(**kwargs)
        module2key =  { m['name']: m['key'] for m in modules }
        
        if module != None:
            return module2key[module]
        return module2key
    

    def module2stake(self,*args, **kwargs) -> Dict[str, str]:
        
        module2stake =  { m['name']: m['stake'] for m in self.modules(*args, **kwargs) }
        
        return module2stake

    
    def modules(self,
                search=None,
                network = 'main',
                netuid: int = 0,
                block: Optional[int] = None,
                fmt='nano', 
                keys : List[str] = ['uid2key', 'addresses', 'names', 'emission', 
                    'incentive', 'dividends', 'regblock', 'last_update', 
                    'stake_from', 'delegation_fee', 'trust'],
                update: bool = False,
                include_weights = False,
                df = False,
                parallel:bool = False ,
                timeout:int=200, 
                include_balances = False, 
                
                ) -> Dict[str, ModuleInfo]:
        import inspect
        if netuid == None:
            netuid = 0

        cache_path = f'modules/{network}.{netuid}'

        modules = []
        if not update :
            modules = self.get(cache_path, [])

        if len(modules) == 0:

            network = self.resolve_network(network)
            netuid = self.resolve_netuid(netuid)
            block = self.block if block == None else block
 
            
            if include_balances:
                keys += ['balances']
            if include_weights:
                keys += ['weights']
            if parallel:
                state = {}

                async def async_get_chain_data(key:str, network:str='main', block:int=None, netuid:int=0):
                    try:
                        results =  getattr(self, key)(netuid=netuid, block=block, update=True)
                    except Exception as e:
                        c.print(f"Failed to get {key} for netuid {netuid} at block {block}")
                        c.print(e)
                        results = None
                    return results

                while len(state) < len(keys):
                    futures = []
                    remaining_keys = [k for k in keys if k not in state]
                    for key in remaining_keys:
                        future = async_get_chain_data(key=key, netuid=netuid, block=block, network=network)
                        futures.append(future)
                    # remove completed futures
                    results = c.gather(futures, timeout=timeout)
                    for key, result in zip(remaining_keys, results):
                        if result == None:
                            continue
                        if c.is_error(result):
                            continue
                        state[key] = result
            else: 
                state = {}

                for key in c.tqdm(keys):
                    func = getattr(self, key)
                    args = inspect.getfullargspec(func).args

                    kwargs = {}
                    if 'netuid' in args:
                        kwargs['netuid'] = netuid
                    if 'block' in args:
                        kwargs['block'] = block

                    state[key] = func(**kwargs)


            for uid, key in state['uid2key'].items():

                module= {
                    'uid': uid,
                    'address': state['addresses'][uid],
                    'name': state['names'][uid],
                    'key': key,
                    'emission': state['emission'][uid],
                    'incentive': state['incentive'][uid],
                    'trust': state['trust'][uid] if len(state['trust']) > 0 else 0,
                    'dividends': state['dividends'][uid],
                    'stake_from': state['stake_from'].get(key, []),
                    'regblock': state['regblock'].get(uid, 0),
                    'last_update': state['last_update'][uid],
                    'delegation_fee': state['delegation_fee'].get(key, 20)
                }

                module['stake'] = sum([v for k,v in module['stake_from']])
                
                if include_weights:
                    if hasattr(state['weights'][uid], 'value'):
                        
                        module['weight'] = state['weights'][uid]
                    elif isinstance(state['weights'][uid], list):
                        module['weight'] =state['weights'][uid]
                    else: 
                        raise Exception(f"Invalid weight for module {uid}")

                if include_balances:
                    module['balance'] = state['balances'].get(key, 0)
                    
                modules.append(module)
            c.print(f'Saved state for netuid:{netuid} at block {block} at {cache_path}')
            self.put(cache_path, modules)

        if len(modules) > 0:
            keys = list(modules[0].keys())
            if isinstance(keys, str):
                keys = [keys]
            keys = list(set(keys))
            for i, module in enumerate(modules):
                modules[i] ={k: module[k] for k in keys}

                for k in ['emission', 'stake']:
                    module[k] = self.format_amount(module[k], fmt=fmt)

                for k in ['incentive', 'dividends']:
                    module[k] = module[k] / (U16_MAX)
                
                module['stake_from']= [(k, self.format_amount(v, fmt=fmt))  for k, v in module['stake_from']]
      
                if include_balances:
                    module['balance'] = self.format_amount(module['balance'], fmt=fmt)

                modules[i] = module
        if search != None:
            modules = [m for m in modules if search in m['name']]

        if df:
            modules = c.df(modules)

        return modules
    

    def name2uid(self, *args, **kwargs):
        uid2name = self.query_map('Name', *args, **kwargs)
        return {v:k for k,v in uid2name}
    

    def min_stake(self, netuid: int = None, network: str = None, fmt:str='j', registration=True, **kwargs) -> int:
        
        self.resolve_network(network)
        netuid = self.resolve_netuid(netuid)
        min_stake = self.query('MinStake', params=[netuid], network=network, **kwargs)
        min_stake = self.format_amount(min_stake, fmt=fmt)
        if registration:
            registrations_per_block = self.registrations_per_block(netuid=netuid)
            max_registrations_per_block = self.max_registrations_per_block(netuid=netuid)
            
            # 2 to the power of the number of registrations per block over the max registrations per block
            # this is the factor by which the min stake is multiplied, to avoid ddos attacks
            min_stake_factor = 2 **(registrations_per_block // max_registrations_per_block)
            return min_stake * min_stake_factor
        return min_stake


    def registrations_per_block(self, network: str = network, fmt:str='j', **kwargs) -> int:
        return self.query('RegistrationsPerBlock', params=[], network=network, **kwargs)
    regsperblock = registrations_per_block
    
    def max_registrations_per_block(self, network: str = network, fmt:str='j', **kwargs) -> int:
        return self.query('MaxRegistrationsPerBlock', params=[], network=network, **kwargs)
 
    def keys(self, netuid = None, **kwargs):
        return list(self.uid2key(netuid=netuid, **kwargs).values())
    def uids(self, netuid = None, **kwargs):
        return list(self.uid2key(netuid=netuid, **kwargs).keys())

    def uid2key(self, uid=None, netuid = None, update=False, **kwargs):
        netuid = self.resolve_netuid(netuid)
        uid2key = {v[0]: v[1] for v in self.query_map('Keys', params=[netuid], update=update, **kwargs)}
        # sort by uid
        if uid != None:
            return uid2key[uid]
        uids = list(uid2key.keys())
        uid2key = {uid: uid2key[uid] for uid in sorted(uids)}
        return uid2key

    def uid2name(self, netuid: int = None, update=False,  **kwargs) -> List[str]:
        netuid = self.resolve_netuid(netuid)
        names = {v[0]: v[1] for v in self.query_map('Name', params=[netuid], update=update,**kwargs)}
        names = {k: names[k] for k in sorted(names)}
        return names
    
    def names(self, netuid: int = None, update=False, **kwargs) -> List[str]:
        return list(self.uid2name(netuid=netuid, update=update, **kwargs).values())

    def addresses(self, netuid: int = None, update=False, **kwargs) -> List[str]:
        netuid = self.resolve_netuid(netuid)
        names = {v[0]: v[1] for v in self.query_map('Address', params=[netuid], update=update, **kwargs)}
        names = list({k: names[k] for k in sorted(names)}.values())
        return names

    def namespace(self, search=None, netuid: int = netuid, update:bool = False, timeout=10, local=False, **kwargs) -> Dict[str, str]:
        namespace = {}  

        names = self.names(netuid=netuid, update=update, **kwargs)
        addresses = self.addresses(netuid=netuid, update=update, **kwargs)
        namespace = {n: a for n, a in zip(names, addresses)}

        if search != None:
            namespace = {k:v for k,v in namespace.items() if search in k}

        if local:
            ip = c.ip()
            namespace = {k:v for k,v in namespace.items() if ip in str(v)}

        return namespace

    
    def registered_keys(self, netuid = None, **kwargs):
        keys = self.keys(netuid=netuid, **kwargs)
        address2key = c.address2key()
        registered_keys = []
        for k_addr in keys:
            if k_addr in address2key:
                registered_keys += [address2key[k_addr]]
        return registered_keys


    
    def weights(self,  netuid = None, **kwargs) -> list:
        netuid = self.resolve_netuid(netuid)
        subnet_weights =  self.query('Weights', params=netuid, **kwargs)
        weights = {uid:list(map(list, w)) for uid, w in subnet_weights if w != None and uid != None}
        uids = self.uids(netuid=netuid, **kwargs)
        weights = {uid: weights[uid] if uid in weights else [] for uid in uids}
        return {uid: w for uid, w in weights.items()}

    def get_weights(self,uid, netuid = None,  **kwargs) -> list:
        netuid = self.resolve_netuid(netuid)
        return self.query('Weights', params=[netuid, uid], **kwargs)

    def num_voters(self, netuid = None, **kwargs) -> list:
        weights = self.weights(netuid=netuid, **kwargs)
        return len({k:v for k,v in weights.items() if len(v) > 0})
            
    def regprefix(self, prefix, netuid = None, network=None, **kwargs):
        network = self.resolve_network(network)
        netuid = self.resolve_netuid(netuid)
        c.servers(network=network, prefix=prefix)
        
    def emission(self, netuid = netuid, network=None, nonzero=False, **kwargs):
        emissions = [v for v in self.query('Emission', params=[netuid], network=network, **kwargs)]
        if nonzero:
            emissions = [e for e in emissions if e > 0]
        return emissions
        
    def nonzero_emission(self, netuid = netuid, network=None, **kwargs):
        emission = self.emission(netuid=netuid, network=network, **kwargs)
        nonzero_emission =[e for e in emission if e > 0]
        return len(nonzero_emission)

    def incentive(self, netuid = netuid, block=None,   network=network, nonzero:bool=False, update:bool = False,  **kwargs):
        incentive = [v for v in self.query('Incentive', params=netuid, network=network, block=block, update=update, **kwargs)]

        if nonzero:
            incentive = {uid:i for uid, i in enumerate(incentive) if i > 0}
        return incentive
    
    def proposals(self, netuid = netuid, block=None,   network=network, nonzero:bool=False, update:bool = False,  **kwargs):
        proposals = [v for v in self.query_map('Proposals', network=network, block=block, update=update, **kwargs)]
        return proposals
        
    def trust(self, netuid = netuid, network=None, nonzero=False, update=False, **kwargs):
        trust = [v for v in self.query('Trust', params=netuid, network=network, update=update, **kwargs)]
        if nonzero:
            trust = [t for t in trust if t > 0]
        return trust
    def last_update(self, netuid = netuid, block=None,   network=network, update=False, **kwargs):
        return [v for v in self.query('LastUpdate', params=[netuid], network=network, block=block,  update=update, **kwargs)]
        
    def dividends(self, netuid = netuid, network=None, nonzero=False,  update=False, **kwargs):
        dividends =  [v for v in self.query('Dividends', params=netuid, network=network,  update=update,  **kwargs)]
        if nonzero:
            dividends = {i:d for i,d in enumerate(dividends) if d > 0}
        return dividends

    def registration_blocks(self, netuid: int = 0, nonzeros_only:bool = True,  update=False, **kwargs):
        registration_blocks = self.query_map('RegistrationBlock', netuid, update=update, **kwargs)
        registration_blocks = {k:v for k, v in registration_blocks if k != None and v != None}
        # filter based on key of registration_blocks
        registration_blocks = {uid:regblock for uid, regblock in sorted(list(registration_blocks.items()), key=lambda v: v[0])}
        registration_blocks =  list(registration_blocks.values())
        if nonzeros_only:
            registration_blocks = [r for r in registration_blocks if r > 0]
        return registration_blocks

    regblocks = registration_blocks


    def key2uid(self, key = None, network:str=  None ,netuid: int = None, **kwargs):
        key2uid =  {v:k for k,v in self.uid2key(network=network, netuid=netuid, **kwargs).items()}
        if key != None:
            key_ss58 = self.resolve_key_ss58(key)
            return key2uid[key_ss58]
        return key2uid
        

    def get_archive_blockchain_archives(self, netuid=netuid, network:str=network, **kwargs) -> List[str]:

        datetime2archive =  self.datetime2archive(network=network, **kwargs) 
        break_points = []
        last_block = 10e9
        blockchain_id = 0
        get_archive_blockchain_ids = []
        for dt, archive_path in enumerate(datetime2archive):
            
            archive_block = int(archive_path.split('block-')[-1].split('-')[0])
            if archive_block < last_block :
                break_points += [archive_block]
                blockchain_id += 1
            last_block = archive_block
            get_archive_blockchain_ids += [{'blockchain_id': blockchain_id, 'archive_path': archive_path, 'block': archive_block}]

            c.print(archive_block, archive_path)

        return get_archive_blockchain_ids


    def get_archive_blockchain_info(self, netuid=netuid, network:str=network, **kwargs) -> List[str]:

        datetime2archive =  self.datetime2archive(network=network, **kwargs) 
        break_points = []
        last_block = 10e9
        blockchain_id = 0
        get_archive_blockchain_info = []
        for i, (dt, archive_path) in enumerate(datetime2archive.items()):
            c.print(archive_path)
            archive_block = int(archive_path.split('block-')[-1].split('-time')[0])
            
            c.print(archive_block < last_block, archive_block, last_block)
            if archive_block < last_block :
                break_points += [archive_block]
                blockchain_id += 1
                blockchain_info = {'blockchain_id': blockchain_id, 'archive_path': archive_path, 'block': archive_block, 'earliest_block': archive_block}
                get_archive_blockchain_info.append(blockchain_info)
                c.print(archive_block, archive_path)
            last_block = archive_block
            if len(break_points) == 0:
                continue


        return get_archive_blockchain_info


    @classmethod
    def most_recent_archives(cls,):
        archives = cls.search_archives()
        return archives
    
    @classmethod
    def num_archives(cls, *args, **kwargs):
        return len(cls.datetime2archive(*args, **kwargs))

    @classmethod
    def search_archives(cls, 
                    lookback_hours : int = 10,
                    end_time :str = 'now', 
                    start_time: Optional[Union[int, str]] = None, 
                    netuid=0, 
                    n = 1000,
                    **kwargs):


        if end_time == 'now':
            end_time = c.time()
        elif isinstance(end_time, str):
            c.print(end_time)
            
            end_time = c.datetime2time(end_time)
        elif isinstance(end_time, int):
            pass
        else:
            raise Exception(f'Invalid end_time {end_time}')
            end_time = c.time2datetime(end_time)



        if start_time == None:
            start_time = end_time - lookback_hours*3600
            start_time = c.time2datetime(start_time)

        if isinstance(start_time, int) or isinstance(start_time, float):
            start_time = c.time2datetime(start_time)
        
        if isinstance(end_time, int) or isinstance(end_time, float):
            end_time = c.time2datetime(end_time)
        

        assert end_time > start_time, f'end_time {end_time} must be greater than start_time {start_time}'
        datetime2archive = cls.datetime2archive()
        datetime2archive= {k: v for k,v in datetime2archive.items() if k >= start_time and k <= end_time}
        c.print(len(datetime2archive))
        factor = len(datetime2archive)//n
        if factor == 0:
            factor = 1
        archives = []

        c.print('Searching archives from', start_time, 'to', end_time)

        cnt = 0
        for i, (archive_dt, archive_path) in enumerate(datetime2archive.items()):
            if i % factor != 0:
                continue
            archive_block = int(archive_path.split('block-')[-1].split('-time')[0])
            archive = c.get(archive_path)
            total_balances = sum([b for b in archive['balances'].values()])
            # st.write(archive['modules']netuid[:3])
            total_stake = sum([sum([_[1]for _ in m['stake_from']]) for m in archive['modules'][netuid]])
            subnet = archive['subnets'][netuid]
            row = {
                    'block': archive_block,  
                    'total_stake': total_stake*1e-9,
                    'total_balance': total_balances*1e-9, 
                    'market_cap': (total_stake+total_balances)*1e-9 , 
                    'dt': archive_dt, 
                    'block': archive['block'], 
                    'path': archive_path, 
                    'mcap_per_block': 0,
                }
            
            if len(archives) > 0:
                denominator = ((row['block']//subnet['tempo']) - (archives[-1]['block']//subnet['tempo']))*subnet['tempo']
                if denominator > 0:
                    row['mcap_per_block'] = (row['market_cap'] - archives[-1]['market_cap'])/denominator

            archives += [row]
            
        return archives

    @classmethod
    def archive_history(cls, *args, 
                     network=network, 
                     netuid= 0 , update=True,  **kwargs):
        path = f'history/{network}.{netuid}.json'

        archive_history = []
        if not update:
            archive_history = cls.get(path, [])
        if len(archive_history) == 0:
            archive_history =  cls.search_archives(*args,network=network, netuid=netuid, **kwargs)
            cls.put(path, archive_history)
            
        
        return archive_history
        

        

        
        
        

    @classmethod
    def dashboard(cls):
        import plotly.express as px


        self = cls()

        modules = self.modules(fmt='j')
        
        num_searches = self.get('dashboard/search', [])
        search = st.text_input('search', '') 

        modules = [m for m in modules if search in m['name'] or search in m['key'] or search in m['address']]
        df = c.df(modules)

        df['ip'] = df['address'].apply(lambda x: x.split(':')[0])
        df['module'] = df['name'].apply(lambda x: x.split('::')[0])

        cols = ['name', 'emission', 'incentive', 'dividends', 'stake', 'delegation_fee', 'trust',  'ip', 'last_update', 'module']
        with st.expander('columns', expanded=False):
            selected_cols = st.multiselect('Select columns', cols, default=cols)
            df = df[selected_cols]
        df = df.sort_values('emission', ascending=False)
        df = df[cols]

        st.write(df)


        with st.expander('histogram', expanded=False):
            col = st.selectbox('Select column', cols)
            fig = px.histogram(df, x=col)
            st.plotly_chart(fig)

        with st.expander('scatter', expanded=False):
            x = st.selectbox('Select x', cols)
            y = st.selectbox('Select y', cols)
            fig = px.scatter(df, x=x, y=y)
            st.plotly_chart(fig)

        with st.expander('pie', expanded=False):
            treemap = st.checkbox('treemap', False)
            if treemap:
                value = st.selectbox('Select column', cols, key='treemap')
                name = st.selectbox('Select name', ['module', 'ip'], key='pie_name')

                fig = px.treemap(df, path=[name], values=value)
                st.plotly_chart(fig)
            else:
                    
                col = st.selectbox('Select column', cols, key='pie')

                name = st.selectbox('Select name', ['module', 'ip'], key='pie_name')
                fig = px.pie(df, values=col, names=name)
                st.plotly_chart(fig)

        
        


    @classmethod
    def st_search_archives(cls,
                        start_time = '2023-09-08 04:00:00', 
                        end_time = '2023-09-08 04:30:00'):
        start_time = st.text_input('start_time', start_time)
        end_time = st.text_input('end_time', end_time)
        df = cls.search_archives(end_time=end_time, start_time=start_time)

        
        st.write(df)



    def key_usage_path(self, key:str):
        key_ss58 = self.resolve_key_ss58(key)
        return f'key_usage/{key_ss58}'

    def key_used(self, key:str):
        return self.exists(self.key_usage_path(key))
    
    def use_key(self, key:str):
        return self.put(self.key_usage_path(key), c.time())
    
    def unuse_key(self, key:str):
        return self.rm(self.key_usage_path(key))
    
    def test_key_usage(self):
        key_path = 'test_key_usage'
        c.add_key(key_path)
        self.use_key(key_path)
        assert self.key_used(key_path)
        self.unuse_key(key_path)
        assert not self.key_used(key_path)
        c.rm_key('test_key_usage')
        assert not c.key_exists(key_path)
        return {'success': True, 'msg': f'Tested key usage for {key_path}'}
        

    def get_nonce(self, key:str=None, network=None, **kwargs):
        key_ss58 = self.resolve_key_ss58(key)
        self.resolve_network(network)   
        return self.substrate.get_account_nonce(key_ss58)

    history_path = f'history'


    @classmethod
    def convert_snapshot(cls, from_version=1, to_version=2, network=network):
        
        if from_version == 1 and to_version == 2:
            factor = 1_000 / 42 # convert to new supply
            path = f'{cls.snapshot_path}/{network}.json'
            snapshot = c.get_json(path)
            snapshot['balances'] = {k: int(v*factor) for k,v in snapshot['balances'].items()}
            for netuid in range(len(snapshot['subnets'])):
                for j, (key, stake_to_list) in enumerate(snapshot['stake_to'][netuid]):
                    c.print(stake_to_list)
                    for k in range(len(stake_to_list)):
                        snapshot['stake_to'][netuid][j][1][k][1] = int(stake_to_list[k][1]*factor)
            snapshot['version'] = to_version
            c.put_json(path, snapshot)
            return {'success': True, 'msg': f'Converted snapshot from {from_version} to {to_version}'}

        else:
            raise Exception(f'Invalid conversion from {from_version} to {to_version}')

    @classmethod
    def check(cls, netuid=0):
        self = cls()

        # c.print(len(self.modules()))
        c.print(len(self.query_map('Keys', netuid)), 'keys')
        c.print(len(self.query_map('Name', netuid)), 'names')
        c.print(len(self.query_map('Address', netuid)), 'address')
        c.print(len(self.incentive()), 'incentive')
        c.print(len(self.uids()), 'uids')
        c.print(len(self.stakes()), 'stake')
        c.print(len(self.query_map('Emission')[0][1]), 'emission')
        c.print(len(self.query_map('Weights', netuid)), 'weights')


    


    @classmethod
    def pull(cls, rpull:bool = False):

        if len(cls.ls(cls.libpath)) < 5:
            c.rm(cls.libpath)
        c.pull(cwd=cls.libpath)
        if rpull:
            cls.rpull()


    @classmethod
    def push(cls, rpull:bool=False, image:bool = False ):
        c.push(cwd=cls.libpath)
        if image:
            cls.push_image()
        if rpull:
            cls.rpull()

    @classmethod
    def rpull(cls, timeout=10):
        # pull from the remote server
        c.rcmd('c s pull', verbose=True, timeout=timeout)
        c.rcmd('c s pull_image', verbose=True, timeout=timeout)

    @classmethod
    def status(cls):
        return c.status(cwd=cls.libpath)


    def storage_functions(self, network=network, block_hash = None):
        self.resolve_network(network)
        return self.substrate.get_metadata_storage_functions( block_hash=block_hash)
    storage_fns = storage_functions
        

    def storage_names(self,  search=None, network=network, block_hash = None):
        self.resolve_network(network)
        storage_names =  [f['storage_name'] for f in self.substrate.get_metadata_storage_functions( block_hash=block_hash)]
        if search != None:
            storage_names = [s for s in storage_names if search in s.lower()]
        return storage_names
    


    @classmethod
    def test(cls):
        s = c.module('subspace')()
        n = s.n()
        assert isinstance(n, int)
        assert n > 0

        market_cap = s.mcap()
        assert isinstance(market_cap, float), market_cap

        name2key = s.name2key()
        assert isinstance(name2key, dict)
        assert len(name2key) == n

        stats = s.stats(df=False)
        c.print(stats)
        assert isinstance(stats, list) 

    def check_storage(self, block_hash = None, network=network):
        self.resolve_network(network)
        return self.substrate.get_metadata_storage_functions( block_hash=block_hash)

    @classmethod
    def sand(cls): 
        node_keys =  cls.node_keys()
        spec = cls.spec()
        addy = c.root_key().ss58_address

        for i, (k, v) in enumerate(cls.datetime2archive('2023-10-17').items()):
            if i % 10 != 0:
                c.print(i, '/', len(v))
                continue
            state = c.get(v)
            c.print(state.keys())
            c.print(k, state['balances'].get(addy, 0))
        



    def test_balance(self, network:str = network, n:int = 10, timeout:int = 10, verbose:bool = False, min_amount = 10, key=None):
        key = c.get_key(key)

        balance = self.get_balance(network=network)
        assert balance > 0, f'balance must be greater than 0, not {balance}'
        balance = int(balance * 0.5)
        c.print(f'testing network {network} with {n} transfers of {balance} each')


    def test_commands(self, network:str = network, n:int = 10, timeout:int = 10, verbose:bool = False, min_amount = 10, key=None):
        key = c.get_key(key)

        key2 = c.get_key('test2')
        
        balance = self.get_balance(network=network)
        assert balance > 0, f'balance must be greater than 0, not {balance}'
        c.transfer(dest=key, amount=balance, timeout=timeout, verbose=verbose)
        balance = int(balance * 0.5)
        c.print(f'testing network {network} with {n} transfers of {balance} each')


    @classmethod
    def fix(cls):
        avoid_ports = []
        free_ports = c.free_ports(n=3, avoid_ports=avoid_ports)
        avoid_ports += free_ports

    def num_holders(self, **kwargs):
        balances = self.balances(**kwargs)
        return len(balances)

    def total_balance(self, **kwargs):
        balances = self.balances(**kwargs)
        return sum(balances.values())
    

    def sand(self, **kwargs):
        balances = self.my_balances(**kwargs)
        return sum(balances.values())
    
    """
    
    WALLET VIBES
    
    """

    
    def wallet(self, **kwargs):
        return c.module('subspace.wallet')(**kwargs)
    
    def register(self, *args, **kwargs):
        return self.wallet().register(*args,**kwargs)

    def stake(self, *args, **kwargs):
        return self.wallet().stake(*args,**kwargs)

    def stake_many(self, *args, **kwargs):
        return self.wallet().stake_many(*args,**kwargs)

    def unstake(self, *args, **kwargs):
        return self.wallet().unstake(*args,**kwargs)
    
    def unstake_many(self, *args, **kwargs):
        return self.wallet().unstake_many(*args,**kwargs)

    def add_profit_share(self, *args, **kwargs):
        return self.wallet().add_profit_share(*args,**kwargs)
    
    def vote(self, *args, **kwargs):
        return self.wallet().vote(*args,**kwargs)
    
    def transfer(self, *args, **kwargs):
        return self.wallet().transfer(*args,**kwargs)
    
    def transfer_many(self, *args, **kwargs):
        return self.wallet().transfer_many(*args,**kwargs)
    
    send = transfer

    

Subspace.run(__name__)
