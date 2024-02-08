
from retry import retry
from typing import *
from .balance import Balance
import json
import os
import commune as c
import requests 

U32_MAX = 4294967295
U16_MAX = 65535

class Subspace(c.Module):
    """
    Handles interactions with the subspace chain.
    """
    block_time = 8 # (seconds)
    whitelist = ['query', 'query_map']
    fmt = 'j'
    git_url = 'https://github.com/commune-ai/subspace.git'
    default_config = c.get_config('subspace', to_munch=False)
    token_decimals = default_config['token_decimals']
    network = default_config['network']
    chain = network
    libpath = chain_path = c.libpath + '/subspace'
    spec_path = f"{chain_path}/specs"
    netuid = default_config['netuid']
    local = default_config['local']
    
    features = ['Keys', 
                'StakeTo',
                'Name', 
                'Address',
                'Weights',
                'Emission', 
                'Incentive', 
                'Dividends', 
                'LastUpdate',
                'ProfitShares',
                'Proposals', 
                'Voter2Info',
                ]

    def __init__( 
        self, 
        **kwargs,
    ):
        config = self.set_config(kwargs=kwargs)
    connection_mode = 'ws'

    def resolve_url(self, url:str = None, network:str = network, mode=None , **kwargs):
        mode = mode or self.config.connection_mode
        network = 'network' or self.config.network
        if url == None:
            
            url_search_terms = [x.strip() for x in self.config.url_search.split(',')]
            is_match = lambda x: any([url in x for url in url_search_terms])
            urls = []
            for provider, mode2url in self.config.urls.items():
                if is_match(provider):
                    chain = c.module('subspace.chain')
                    if provider == 'commune':
                        url = chain.resolve_node_url(url=url, chain=network, mode=mode) 
                    elif provider == 'local':
                        url = chain.resolve_node_url(url=url, chain='local', mode=mode)
                    else:
                        url = mode2url[mode]

                    if isinstance(url, list):
                        urls += url
                    else:
                        urls += [url] 

            url = c.choice(urls)
        
        url = url.replace(c.ip(), '0.0.0.0')


        return url
    
    url2substrate = {}
    def get_substrate(self, 
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
                verbose:bool=True,
                max_trials:int = 10,
                cache:bool = True,
                mode = 'http',):
        from substrateinterface import SubstrateInterface

        url = self.resolve_url(url, mode=mode, network=network)
        
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

        if cache:
            if url in self.url2substrate:
                return self.url2substrate[url]


        substrate= SubstrateInterface(url=url, 
                    websocket=websocket, 
                    ss58_format=ss58_format, 
                    type_registry=type_registry, 
                    type_registry_preset=type_registry_preset, 
                    cache_region=cache_region, 
                    runtime_config=runtime_config, 
                    ws_options=ws_options, 
                    auto_discover=auto_discover, 
                    auto_reconnect=auto_reconnect)
        
        if cache:
            self.url2substrate[url] = substrate
        
        return substrate


    def set_network(self, 
                network:str = network,
                mode = 'http',
                url : str = None, **kwargs):
        self.url = url
        self.network= network or self.config.network
        self.substrate = self.get_substrate(network=network, url=url, mode=mode , **kwargs)
        return {'network': network, 'url': url}

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

    def stake_from(self, netuid = 0,
                    block=None, 
                    update=False,
                    network=network,
                    tuples = False,
                    fmt='nano'):
        stake_from =  {k: list(map(list,v)) for k,v in self.query_map('StakeFrom', block=block, update=update)[netuid].items()}
        stake_from = {k: list(map(lambda x : [x[0], self.format_amount(x[1], fmt=fmt)], v)) for k,v in stake_from.items()}
        if tuples:
            return stake_from
        return {k: {k2:v2 for k2,v2 in v} for k,v in stake_from.items()}
    
    def my_stake_from(self, netuid = 0, block=None, update=False, network=network, fmt='j'):
        stake_from_tuples = self.stake_from(netuid=netuid,
                                             block=block,
                                               update=update, 
                                               network=network, 
                                               tuples = True,
                                               fmt=fmt)
        address2key = c.address2key()
        stake_from_total = {}
        for module_key,staker_tuples in stake_from_tuples.items():
            for staker_key, stake in staker_tuples:
                if module_key in address2key:
                    stake_from_total[staker_key] = stake_from_total.get(staker_key, 0) + stake
        return stake_from_total   

    def my_stake_to(self, netuid = 0, block=None, update=True, network=network, fmt='j'):
        stake_to_tuples = self.stake_to(netuid=netuid, block=block, update=update, network=network, fmt=fmt)
        address2key = c.address2key()
        stake_to_total = {}
        for staker_key, module_tuples in stake_to_tuples.items():
            for module_key, stake in module_tuples:
                if module_key in address2key:
                    stake_to_total[staker_key] = stake_to_total.get(staker_key, 0) + stake
        return stake_to_total         
    
    
    def delegation_fee(self, netuid = 0, block=None, network=None, update=False, fmt='j'):
        delegation_fee = self.query_map('DelegationFee', block=block ,update=update, network=network)
        if netuid == 'all':
            return delegation_fee
        else:
            netuid = self.resolve_netuid(netuid)
            delegation_fee = {k: v for k,v in delegation_fee[netuid].items()}
            return delegation_fee

    def stake_to(self, netuid = 0, network=network, block=None, update=False, fmt='nano'):

        stake_to = self.query_map('StakeTo', block=block, update=update, network=network)
        
        if netuid == 'all':
            return stake_to
        else:
            netuid = self.resolve_netuid(netuid)
            stake_to = {k: v for k,v in stake_to[netuid].items()}
            return {k: list(map(lambda x : [x[0], self.format_amount(x[1], fmt=fmt)], v)) for k,v in stake_to.items()}

    
    def query(self, name:str,  
              params = None, 
              module:str='SubspaceModule',
              block=None,  
              network: str = network, 
              save= True,
            update=False):
        
        """
        query a subspace storage function with params and block.
        """
        network = self.resolve_network(network)
        path = f'query/{network}/{module}.{name}'
    
        params = params or []
        if not isinstance(params, list):
            params = [params]
            
        # we want to cache based on the params if there are any
        if len(params) > 0 :
            path = path + f'::params::' + '-'.join([str(p) for p in params])

        if not update:
            value = self.get(path, None)
            if value != None:
                return value
        substrate = self.get_substrate(network=network)
        response =  substrate.query(
            module=module,
            storage_function = name,
            block_hash = None if block == None else substrate.get_block_hash(block), 
            params = params
        )
        value =  response.value

        # if the value is a tuple then we want to convert it to a list
        if save:
            self.put(path, value)

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
        substrate = self.get_substrate(network=network)

        value =  substrate.query(
            module=module_name,
            storage_function=constant_name,
            block_hash = None if block == None else substrate.get_block_hash(block)
        )
            
        return value
    
    


    def query_map(self, name: str, 
                  params: list = None,
                  block: Optional[int] = None, 
                  network:str = network,
                  page_size=1000,
                  max_results=100000,
                  module='SubspaceModule',
                  update: bool = True,
                  return_dict:bool = True,
                  max_age = None, # max age in seconds
                  new_connection=False,
                  mode = 'http',
                  **kwargs
                  ) -> Optional[object]:
        """ Queries subspace map storage with params and block. """
        if name  == 'Account':
            module = 'System'

        network = self.resolve_network(network, new_connection=new_connection, mode=mode)
        path = f'query/{network}/{module}.{name}'

    
        # resolving the params
        params = params or []
        if not isinstance(params, list):
            params = [params]
        if len(params) > 0 :
            path = path + f'::params::' + '-'.join([str(p) for p in params])

        if not update:
            value = self.get(path, None, max_age=max_age)
            if value != None:
                return value
        
        network = self.resolve_network(network)

        # if the value is a tuple then we want to convert it to a list
        block = block or self.block
        substrate = self.get_substrate(network=network, mode=mode)
        block_hash = substrate.get_block_hash(block)
        qmap =  substrate.query_map(
            module=module,
            storage_function = name,
            params = params,
            page_size = page_size,
            max_results = max_results,
            block_hash =block_hash
        )

        new_qmap = {} if return_dict else []

        # number of records

        is_key_digit = []
        tqdm = c.tqdm(qmap, desc=f'Querying {name} map')

        for i, (k,v) in enumerate(qmap):
            tqdm.update(1)
            if not isinstance(k, tuple):
                k = [k]
            if type(k) in [tuple,list]:
                k = [_k.value for _k in k]
                if len(is_key_digit) == 0:
                    is_key_digit = [c.is_digit(_k) for _k in k]
                
                if len(is_key_digit) > 0:
                    for _i_k, _k in enumerate(k):
                        if is_key_digit[_i_k]:
                            _k = int(_k)
            if hasattr(v, 'value'):
                v = v.value
            if return_dict:
                c.dict_put(new_qmap, k, v)
            else:
                new_qmap.append([k,v])

        if return_dict:
            num_key_digits = sum([1 for _ in is_key_digit if _])
            
            if num_key_digits == 0:
                pass
            elif num_key_digits == 1:
                # this means that you have [uid] as keys
                newer_map = [None] * len(new_qmap)
                for k,v in new_qmap.items():
                    newer_map[int(k)] = v
                new_qmap = newer_map
            elif num_key_digits == 2: # for eights
                # this means that you have [netuid, uid] as keys
                newer_map = [None] * len(new_qmap)
                for k1,v1 in new_qmap.items():
                    v1_n = max(v1.keys()) + 1
                    tmp_map = [None] * v1_n
                    for k2,v2 in v1.items():
                        tmp_map[int(k2)] = v2
                    newer_map[int(k1)] = tmp_map
                new_qmap = newer_map
        
        self.put(path, new_qmap)

        return new_qmap

    def runtime_spec_version(self, network:str = 'main'):
        # Get the runtime version
        self.resolve_network(network=network)
        c.print(self.substrate.runtime_config.__dict__)
        runtime_version = self.query_constant(module_name='System', constant_name='SpVersionRuntimeVersion')
        return runtime_version
        
        
    #####################################
    #### Hyper parameter calls. ####
    #####################################

    """ Returns network SubnetN hyper parameter """
    def n(self, network = network , netuid: int = None, block: Optional[int] = None ) -> int:
        self.resolve_network(network)
        netuid = self.resolve_netuid( netuid )
        return self.query('N', params=[netuid], block=block , update=True)

    ##########################
    #### Account functions ###
    ##########################
    
    """ Returns network Tempo hyper parameter """
    def stakes(self, netuid: int = 0, block: Optional[int] = None, fmt:str='nano', max_staleness = 100,network=None, update=False, **kwargs) -> int:
        stakes =  self.query_map('Stake', update=update, **kwargs)[netuid]
        return {k: self.format_amount(v, fmt=fmt) for k,v in stakes.items()}

    """ Returns the stake under a coldkey - hotkey pairing """
    
    def resolve_key_ss58(self, key:str, network='main', netuid:int=0, resolve_name=True, **kwargs):
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
                    assert resolve_name, f"Invalid Key {key} as it should have ss58_address attribute."
                    name2key = self.name2key(network=network, netuid=netuid)
                    assert key in name2key, f"Invalid Key {key} as it should have ss58_address attribute."
                    if key in name2key:
                        key_address = name2key[key]
                    else:
   
                        raise Exception(f"Invalid Key {key} as it should have ss58_address attribute.")   
        # if the key has an attribute then its a key
        elif hasattr(key, 'ss58_address'):
            key_address = key.ss58_address
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
    def format_amount(cls, x, fmt='nano', decimals = None, format=None):
        fmt = format or fmt # format is an alias for fmt
        if fmt in ['nano', 'n']:
            x =  x
        elif fmt in ['token', 'unit', 'j', 'J']:
            x = cls.to_token(x)
        
        if decimals != None:
            x = c.round_decimals(x, decimals=decimals)

        return x
    
    def get_stake( self, key_ss58: str, block: Optional[int] = None, netuid:int = None , fmt='j', update=True ) -> Optional['Balance']:
        
        key_ss58 = self.resolve_key_ss58( key_ss58)
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
        

    def get_stake_to( self, key: str = None, module_key=None, block: Optional[int] = None, netuid:int = None , fmt='j' , network=None, update=True, **kwargs) -> Optional['Balance']:
        network = self.resolve_network(network)
        key_address = self.resolve_key_ss58( key )
        netuid = self.resolve_netuid( netuid )
        stake_to = self.query( 'StakeTo', params=[netuid, key_address], block=block, update=update)
        stake_to =  {k: self.format_amount(v, fmt=fmt) for k, v in stake_to}
        if module_key != None:
            module_key = self.resolve_key_ss58( module_key )
            stake_to ={ k:v for k, v in stake_to}.get(module_key, 0)
        return stake_to
    
    get_staketo = get_stake_to
    
    def get_value(self, key=None):
        key = self.resolve_key_ss58(key)
        value = self.get_balance(key)
        netuids = self.netuids()
        for netuid in netuids:
            stake_to = self.get_stake_to(key, netuid=netuid)
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
    def ls_archives(cls, network=network):
        if network == None:
            network = cls.network 
        return [f for f in cls.ls(f'state_dict') if os.path.basename(f).startswith(network)]

    
    @classmethod
    def block2archive(cls, network=network):
        paths = cls.ls_archives(network=network)

        block2archive = {int(p.split('-')[-1].split('-time')[0]):p for p in paths if p.endswith('.json') and f'{network}.block-' in p}
        return block2archive

    def latest_archive_block(self, network=network) -> int:
        latest_archive_path = self.latest_archive_path(network=network)
        block = int(latest_archive_path.split(f'.block-')[-1].split('-time')[0])
        return block


        

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
    def latest_archive(cls, network=network):
        path = cls.latest_archive_path(network=network)
        if path == None:
            return {}
        return cls.get(path, {})
    
 


    def light_sync(self, network=None, remote:bool=True, netuids=None, local:bool=True, save:bool=True, timeout=20, **kwargs):
        netuids = self.netuids(network=network, update=True) if netuids == None else netuids
        assert len(netuids) > 0, f"No netuids found for network {network}"
        stake_from_futures = []
        namespace_futures = []
        weight_futures = []
        for netuid in netuids:
            stake_from_futures += [c.asubmit(self.stake_from, netuid=netuid, network=network, update=True)]
            namespace_futures += [c.asubmit(self.namespace, netuid=netuid, network=network, update=True)]
            weight_futures += [c.asubmit(self.weights, netuid=netuid, network=network, update=True)]

        c.gather(stake_from_futures + namespace_futures + weight_futures, timeout=timeout)

        # c.print(namespace_list)
        return {'success': True, 'block': self.block}


    def loop(self, intervals = {'light': 5, 'full': 600}, network=None, remote:bool=True):
        if remote:
            return self.remote_fn('loop', kwargs=dict(intervals=intervals, network=network, remote=False))
        last_update = {k:0 for k in intervals.keys()}
        staleness = {k:0 for k in intervals.keys()}
        c.get_event_loop()

        while True:
            block = self.block
            timestamp = c.timestamp()
            staleness = {k:timestamp - last_update[k] for k in intervals.keys()}
            if staleness["full"] > intervals["full"]:
                request = {
                            'network': network, 
                           'block': block
                           }
                try:
                    self.sync(**request)
                except Exception as e:
                    c.print(e)
                    continue
                last_update['full'] = timestamp
            

    def subnet_exists(self, subnet:str, network=None) -> bool:
        subnets = self.subnets(network=network)
        return bool(subnet in subnets)

    def subnet_emission(self, netuid:str = 0, network=None, block=None, update=False, **kwargs):
        return sum(self.query( "Emission", block=block, params=[netuid], network=network , update=update))
    
    def unit_emission(self, network=None, block=None, update=False, **kwargs):
        return self.query_constant( "UnitEmission", block=block,network=network)

    def subnet_state(self,  network=None, block=None, update=False, netuid=0, fmt='j', **kwargs):
        netuid= self.resolve_netuid(netuid)
        subnet_state = self.subnet_params(network=network, block=block, update=update, **kwargs)
        subnet_state['emission'] = self.subnet_emission(network=network, block=block, update=update, **kwargs)
        subnet_state['blocks_per_day'] = ( 24*60*60 / (self.block_time))
        subnet_state['daily_emission'] = self.subnet_emission(network=network, block=block, update=update, **kwargs) * subnet_state['blocks_per_day']
        subnet_state['n'] = self.n(netuid=netuid)

        for k in ['min_stake', 'max_stake', 'emission', 'daily_emission']:
            subnet_state[k] = self.format_amount(subnet_state[k], fmt=fmt)

        return subnet_state

        

    def subnet_states(self, *args, **kwargs):
        subnet_states = []
        for netuid in self.netuids():
            subnet_state = self.subnet(*args,  netuid=netuid, **kwargs)
            subnet_states.append(subnet_state)
        return subnet_states

    def total_stake(self, network=network, block: Optional[int] = None, netuid:int='all', fmt='j', update=False) -> 'Balance':
        return sum([sum([sum(list(map(lambda x:x[1], v))) for v in vv.values()]) for vv in self.stake_to(network=network, block=block,update=update, netuid='all')])

    def total_balance(self, network=network, block: Optional[int] = None, fmt='j', update=False) -> 'Balance':
        return sum(list(self.balances(network=network, block=block, fmt=fmt).values()), update=update)

    def mcap(self, network=network, block: Optional[int] = None, fmt='j', update=False) -> 'Balance':
        total_balance = self.total_balance(network=network, block=block, update=update)
        total_stake = self.total_stake(network=network, block=block, update=update)
        return self.format_amount(total_stake + total_balance, fmt=fmt)
    
    market_cap = total_supply = mcap  
            
        
    @classmethod
    def feature2storage(cls, feature:str):
        storage = ''
        capitalize = True
        for i, x in enumerate(feature):
            if capitalize:
                x =  x.upper()
                capitalize = False

            if '_' in x:
                capitalize = True

            storage += x
        return storage

    
    def subnet_params(self, 
                    netuid=0,
                    network = network,
                    block : Optional[int] = None,
                    update = False,
                    timeout = 30,
                    fmt:str='j', 
                    rows:bool = True
                    ) -> list:

        name2feature  = {
                'tempo': "Tempo",
                'immunity_period': 'ImmunityPeriod',
                'min_allowed_weights': 'MinAllowedWeights',
                'max_allowed_weights': 'MaxAllowedWeights',
                'max_allowed_uids': 'MaxAllowedUids',
                'min_stake': 'MinStake',
                'founder': 'Founder', 
                'founder_share': 'FounderShare',
                'incentive_ratio': 'IncentiveRatio',
                'trust_ratio': 'TrustRatio',
                'vote_threshold': 'VoteThresholdSubnet',
                'vote_mode': 'VoteModeSubnet',
                'self_vote': 'SelfVote',
                'name': 'SubnetNames',
                'max_stake': 'MaxStake',
            }
        

        network = self.resolve_network(network)
        path = f'cache/{network}.subnet_params.json'
        subnet_params = None if update else self.get(path, None) 
    
        
        features = list(name2feature.keys())
        block = block or self.block

        if subnet_params == None:
            async def query(**kwargs ):
                return self.query_map(**kwargs)
            
            subnet_params = {}
            n = len(features)
            progress = c.tqdm(total=n, desc=f'Querying {n} features')
            while True:
                
                features_left = [f for f in features if f not in subnet_params]
                if len(features_left) == 0:
                    c.print(f'All features queried, {c.emoji("checkmark")}')
                    break

                name2job = {k:query(name=v, update=update, block=block) for k, v in name2feature.items()}
                jobs = list(name2job.values())
                results = c.wait(jobs, timeout=timeout)
                for i, feature in enumerate(features_left):
                    if c.is_error(results[i]):
                        c.print(f'Error querying {results[i]}')
                    else:
                        progress.update(1)
                        subnet_params[feature] = results[i]
            self.put(path, subnet_params)

        

        if netuid != None and netuid != 'all':
            netuid = self.resolve_netuid(netuid)
            new_subnet_params = {}
            for k,v in subnet_params.items():

                new_subnet_params[k] = v[netuid]
            subnet_params = new_subnet_params

            for k in ['min_stake', 'max_stake']:
                subnet_params[k] = self.format_amount(subnet_params[k], fmt=fmt)
        else:
            if rows:
                num_subnets = len(subnet_params['tempo'])
                subnets_param_rows = []
                for netuid in range(num_subnets):
                    subnets_param_row = {}
                    for k in subnet_params.keys():
                        c.print( subnet_params[k])
                        subnets_param_row[k] = subnet_params[k][netuid]
                    subnets_param_rows.append(subnets_param_row)
                subnet_params = subnets_param_rows                    
        return subnet_params
    
    subnet = subnet_params


    def subnet2params( self, network: int = None, block: Optional[int] = None ) -> Optional[float]:
        netuids = self.netuids(network=network)
        subnet2params = {}
        netuid2subnet = self.netuid2subnet()
        for netuid in netuids:
            subnet = netuid2subnet[netuid]
            subnet2params[subnet] = self.subnet_params(netuid=netuid, block=block)
        return subnet2params
    
    def subnet2emission( self, network: int = None, block: Optional[int] = None ) -> Optional[float]:
        subnet2emission = self.subnet2params(network=network, block=block)
        return subnet2emission

    

    def subnet2state( self, network: int = None, block: Optional[int] = None ) -> Optional[float]:
        subnet2state = self.subnet2params(network=network, block=block)

        return subnet2state
            

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


    def regblock(self, netuid: int = None, block: Optional[int] = None, network=network, update=False ) -> Optional[float]:
        netuid = self.resolve_netuid( netuid )
        return {k:v for k,v  in enumerate(self.query_map('RegistrationBlock',block=block, update=update )[netuid]) }

    def age(self, netuid: int = None) -> Optional[float]:
        netuid = self.resolve_netuid( netuid )
        regblock = self.regblock(netuid=netuid)
        block = self.block
        age = {}
        for k,v in regblock.items():
            age[k] = block - v
        return age

     
    def global_params(self, 
                      network: str = network,
                         timeout = 2,
                         update = False,
                         block : Optional[int] = None,
                         fmt = 'nanos'
                          ) -> Optional[float]:
        
        path = f'cache/{network}.global_params.json'
        global_params = None if update else self.get(path, None)

        if global_params == None:
            self.resolve_network(network)

            global_params = {}
            global_params['burn_rate'] =  'BurnRate' 
            global_params['max_name_length'] =  'MaxNameLength'
            global_params['max_allowed_modules'] =  'MaxAllowedModules' 
            global_params['max_allowed_subnets'] =  'MaxAllowedSubnets'
            global_params['max_proposals'] =  'MaxProposals'
            global_params['max_registrations_per_block'] =  'MaxRegistrationsPerBlock' 
            global_params['min_burn'] =  'MinBurn' 
            global_params['min_stake'] =  'MinStakeGlobal' 
            global_params['min_weight_stake'] =  'MinWeightStake'       
            global_params['unit_emission'] =  'UnitEmission' 
            global_params['tx_rate_limit'] =  'TxRateLimit' 
            global_params['vote_threshold'] =  'GlobalVoteThreshold' 
            global_params['vote_mode'] =  'VoteModeGlobal' 

            async def aquery_constant(f, **kwargs):
                return self.query_constant(f, **kwargs)
            
            for k,v in global_params.items():
                global_params[k] = aquery_constant(v, block=block )
            
            futures = list(global_params.values())
            results = c.wait(futures, timeout=timeout)
            global_params = dict(zip(global_params.keys(), results))

            for i,(k,v) in enumerate(global_params.items()):
                global_params[k] = v.value
            
            self.put(path, global_params)

        for k in ['min_stake', 'min_burn', 'unit_emission']:
            global_params[k] = self.format_amount(global_params[k], fmt=fmt)
        return global_params



    def balance(self,
                 key: str = None ,
                 block: int = None,
                   fmt='j',
                     network=None,
                       update=True, 
                          **kwargs) -> Optional['Balance']:
        
        r""" Returns the token balance for the passed ss58_address address
        Args:
            address (Substrate address format, default = 42):
                ss58 chain address.
        Return:
            balance (bittensor.utils.balance.Balance):
                account balance
        """
        key_ss58 = self.resolve_key_ss58( key )
        self.resolve_network(network)

        result = self.query(
                module='System',
                name='Account',
                params=[key_ss58],
                block = block,
                network=network,
                update=update
            )

        return  self.format_amount(result['data']['free'] , fmt=fmt)
        
    get_balance = balance 

    def get_account(self, key = None, network=None, update=True):
        self.resolve_network(network)
        key = self.resolve_key_ss58(key)
        account = self.substrate.query(
            module='System',
            storage_function='Account',
            params=[key],
        )
        return account
    
    def accounts(self, key = None, network=None, update=True, block=None):
        self.resolve_network(network)
        key = self.resolve_key_ss58(key)
        accounts = self.query_map(
            module='System',
            name='Account',
            update=update,
            block = block,
        )
        return accounts
    
    def balances(self,fmt:str = 'n', network:str = network, block: int = None, n = None, update=False ) -> Dict[str, Balance]:
        accounts = self.accounts(network=network, update=update, block=block)
        balances =  {k:v['data']['free'] for k,v in accounts.items()}
        balances = {k: self.format_amount(v, fmt=fmt) for k,v in balances.items()}
        return balances
    
    
    def resolve_network(self, network: Optional[int] = None, new_connection =False, mode='ws', **kwargs) -> int:
        if  not hasattr(self, 'substrate') or new_connection:
            self.set_network(network, **kwargs)

        if network == None:
            network = self.network
        
        return network
    
    def resolve_subnet(self, subnet: Optional[int] = None) -> int:
        if isinstance(subnet, int):
            assert subnet in self.netuids()
            subnet = self.netuid2subnet(netuid=subnet)
        subnets = self.subnets()
        assert subnet in subnets, f"Subnet {subnet} not found in {subnets}"
        return subnet


    def subnets(self, **kwargs) -> Dict[int, str]:
        subnets = [s['name'] for s in self.subnet_params(netuid='all', **kwargs)]
        return subnets
    
    def netuids(self, network=network, update=False, block=None) -> Dict[int, str]:
        return list(self.netuid2subnet(network=network, update=update, block=block).keys())

    def netuid2subnet(self, network=network , update=False, block=None, **kwargs) -> Dict[str, str]:
        records = self.query_map('SubnetNames', update=update, network=network, block=block, **kwargs)
        return {k:v for k,v in enumerate(records)}
    
    def subnet_names(self, network=network , update=False, block=None, **kwargs) -> Dict[str, str]:
        return [ v for k,v in self.netuid2subnet(network=network, update=update, block=block, **kwargs).items()]

    def subnet2netuid(self, subnet=None, network=network, update=False,  **kwargs ) -> Dict[str, str]:
        records = self.query_map('SubnetNames', network=network, update=update, **kwargs)
        subnet2netuid =  {netuid:name for netuid,name in enumerate(records) }
        if subnet != None:
            return subnet2netuid[subnet]
        return subnet2netuid

    subnet_namespace = subnet2netuid

    def resolve_netuid(self, netuid: int = None, network=network) -> int:
        '''
        Resolves a netuid to a subnet name.
        '''
        if netuid == None:
            # If the netuid is not specified, use the default.
            return 0

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
        
    def name2uid(self,name = None, search:str=None, netuid: int = None, network: str = None) -> int:
        uid2name = self.uid2name(netuid=netuid, network=network)
        name2uid =  {v:k for k,v in uid2name.items()}
        if name != None:
            return name2uid[name]
        if search != None:
            name2uid = {k:v for k,v in name2uid.items() if search in k}
            if len(name2uid) == 1:
                return list(name2uid.values())[0]
        return name2uid

    
        
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

    
    def get_module(self, key='vali',
                    netuid=0,
                    network='main',
                    fmt='j',
                    **kwargs) -> 'ModuleInfo':
        url = self.resolve_url(network=network, mode='http')
        key = self.resolve_key_ss58(key)
        start_time = c.time()
        module = requests.post(url, 
                            json={'id':1, 
                                  'jsonrpc':'2.0', 
                                  'method': 'subspace_getModuleInfo', 
                                  'params': [key, netuid]}).json()
        latency = c.time() - start_time
        c.print(f"Latency: {latency}")
        module = {**module['result']['stats'], **module['result']['params']}
        module['stake_from'] = [[k, self.format_amount(v, fmt=fmt)] for k,v in module['stake_from']]
        # convert list of u8 into a string 
        module['name'] = self.list2str(module['name'])
        module['address'] = self.list2str(module['address'])
        module['key'] = key
        module['stake'] = sum([v for k,v in module['stake_from']])
        module['dividends'] = module['dividends'] / (U16_MAX)
        module['incentive'] = module['incentive'] / (U16_MAX)
        module['emission'] = self.format_amount(module['emission'], fmt=fmt)


        return module
    
    def list2str(self, l):
        return ''.join([chr(x) for x in l]).strip()

    def get_modules(self, keys:list, timeout = 10) -> List['ModuleInfo']:

        modules = []
        futures = [c.submit(self.get_module, args=[key]) for key in keys]
        progress = c.tqdm(total=len(keys))
        for i, result in  enumerate(c.wait(futures, timeout=timeout, generator=True)):
            
            if isinstance(result, dict) and 'name' in result:
                modules += [result]
                progress.update(1)

            else:
                c.print(result.keys())
            
        return modules
        
    @property
    def null_module(self):
        return {'name': None, 'key': None, 'uid': None, 'address': None, 'stake': 0, 'balance': 0, 'emission': 0, 'incentive': 0, 'dividends': 0, 'stake_to': {}, 'stake_from': {}, 'weight': []}
        
        
    def name2module(self, name:str = None, netuid: int = None, **kwargs) -> 'ModuleInfo':
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

    @classmethod
    def get_feature(cls, feature, **kwargs):
        self = cls()
        return getattr(self, feature)(**kwargs)


    def format_module(self, module: 'ModuleInfo', fmt:str='j') -> 'ModuleInfo':
        for k in ['emission', 'stake']:
            module[k] = self.format_amount(module[k], fmt=fmt)
        for k in ['incentive', 'dividends']:
            module[k] = module[k] / (U16_MAX)
        module['stake_from']= [(k, self.format_amount(v, fmt=fmt))  for k, v in module['stake_from'].items()]
        return module
    def modules(self,
                search=None,
                network = network,
                netuid: int = 0,
                block: Optional[int] = None,
                fmt='nano', 
                features : List[str] = ['keys', 
                                        'addresses', 
                                        'names', 
                                        'emission', 
                                        'incentive', 
                                        'dividends', 
                                        'last_update', 
                                        'stake_from', 
                                        'delegation_fee',
                                        'trust', 
                                        'regblock', 
                                        'weights'],
                timeout = 100,
                update: bool = False,
                df = False,
                
                ) -> Dict[str, 'ModuleInfo']:
        
        modules = []

        if netuid in ['all']:
            kwargs = c.locals2kwargs(locals())
            netuids = self.netuids()
            for netuid in netuids:
                modules += self.modules(**kwargs, netuid=netuid)


        path = f'modules/{network}.{netuid}'
        if not update:
            modules = self.get(path, [])
        
        if len(modules) == 0:
            block = block or self.block
            state = {}
            is_success = lambda x: not c.is_error(x) and x != None
            while True :
                features_left = [f for f in features if f not in state]
                if len(features_left) == 0:
                    c.print('All features queried')
                    break

                c.print(f'Querying {features_left}')
                futures  = [c.submit(self.get_feature,kwargs= dict(feature=f, network=network, netuid=netuid, block=block, update=update) ) for f in features_left]
                for i, result in  enumerate(c.wait(futures, timeout=timeout)):
                    feature = features_left[i]
                    if is_success(result):
                        state[feature] = result
                    else:
                        c.print(f"Error fetching {feature}")
                
                
            for uid, key in enumerate(state['keys']):
                try:
                    module= {
                        'uid': uid,
                        'address': state['addresses'][uid],
                        'name': state['names'][uid] if uid < len(state['names']) else None,
                        'key': key,
                        'emission': state['emission'][uid],
                        'incentive': state['incentive'][uid],
                        'trust': state['trust'][uid] if len(state['trust']) > uid else 0,
                        'dividends': state['dividends'][uid],
                        'stake_from': state['stake_from'].get(key, {}),
                        'regblock': state['regblock'].get(uid, 0),
                        'last_update': state['last_update'][uid],
                        'delegation_fee': state['delegation_fee'].get(key, 20),
                    }

                    module['stake'] =  sum([v for k,v in module['stake_from'].items()])

                except Exception as e:
                    module = self.null_module
                if  len(state['weights']) > 0:
                    try:
                        module['weight'] = state['weights'][uid]
                    except Exception as e:
                        module['weight'] = []
                    
                modules.append(module)
            
                self.put(path, modules)

        if len(modules) > 0:
            keys = list(modules[0].keys())
            if isinstance(keys, str):
                keys = [keys]
            keys = list(set(keys))
            for i, module in enumerate(modules):
                modules[i] ={k: module[k] for k in keys}
                modules[i] = self.format_module(modules[i], fmt=fmt)
        if search != None:
            modules = [m for m in modules if search in m['name']]
        if df:
            modules = c.df(modules)

        return modules
    


    def min_stake(self, netuid: int = 0, network: str = None, fmt:str='j', registration=True, **kwargs) -> int:
        
        self.resolve_network(network)
        netuid = self.resolve_netuid(netuid)
        min_stake = self.query('MinStake', network=network, **kwargs)[netuid]
        return min_stake

    def registrations_per_block(self, network: str = network, fmt:str='j', **kwargs) -> int:
        return self.query('RegistrationsPerBlock', params=[], network=network, **kwargs)
    regsperblock = registrations_per_block
    
    def max_registrations_per_block(self, network: str = network, fmt:str='j', **kwargs) -> int:
        return self.query('MaxRegistrationsPerBlock', params=[], network=network, **kwargs)
 
    def uids(self, netuid = None, **kwargs):
        return list(self.uid2key(netuid=netuid, **kwargs).keys())

    def keys(self, uid=None, 
             netuid = None,
              update=False, 
             network=network, 
             return_dict = False,
             **kwargs):
        netuid = self.resolve_netuid(netuid)
        uid2key = {uid:k for uid,k in enumerate(self.query_map('Keys', update=update, network=network, **kwargs)[netuid])}
        # sort by uid
        if uid != None:
            return uid2key[uid]
        uids = list(uid2key.keys())
        uid2key = {uid: uid2key[uid] for uid in sorted(uids)}
        if return_dict:
            return uid2key
        return list(uid2key.values())
    def uid2key(self, uid=None, 
             netuid = None,
              update=False, 
             network=network, 
             return_dict = True,
             **kwargs):
        return self.keys(uid=uid, netuid=netuid, update=update, network=network, return_dict=return_dict, **kwargs)
    

    def uid2name(self, netuid: int = 0, update=False,  **kwargs) -> List[str]:
        netuid = self.resolve_netuid(netuid)
        names = {k: v for k,v in enumerate(self.query_map('Name', update=update,**kwargs)[netuid])}
        names = {k: names[k] for k in sorted(names)}
        return names
    
    def names(self, 
              netuid: int = 0, 
              update=False,
               return_dict = False,
                **kwargs) -> List[str]:
        netuid = self.resolve_netuid(netuid)
        names = self.query_map('Name', update=update,**kwargs)[netuid]
        if return_dict:
            names = {k: v for k,v in enumerate(names)}
            return {k: names[k] for k in sorted(names)}
        return names 

    def addresses(self, netuid: int = None, update=False, **kwargs) -> List[str]:
        netuid = self.resolve_netuid(netuid)
        addreses = self.query_map('Address', update=update, **kwargs)[netuid]
        return addreses

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


    
    def weights(self,  netuid = None, nonzero:bool = False, network=network, update=False, **kwargs) -> list:
        netuid = self.resolve_netuid(netuid)
        subnet_weights =  self.query_map('Weights', network=network, update=update, **kwargs)
        if len(subnet_weights) <= netuid:
            subnet_weights = []
        else:
            subnet_weights = subnet_weights[netuid]
        if nonzero:
            subnet_weights = {k:v for k,v in subnet_weights if len(v) > 0}
        return subnet_weights

    def save_weights(self, nonzero:bool = False, network=network,**kwargs) -> list:
        self.query_map('Weights',network=network, update=True, **kwargs)
        return {'success': True, 'msg': 'Saved weights'}

    def pending_deregistrations(self, netuid = 0, update=False, **kwargs):
        pending_deregistrations = self.query_map('PendingDeregisterUids',update=update,**kwargs)[netuid]
        return pending_deregistrations
    
    def num_pending_deregistrations(self, netuid = 0, **kwargs):
        pending_deregistrations = self.pending_deregistrations(netuid=netuid, **kwargs)
        return len(pending_deregistrations)
        
    def emission(self, netuid = 0, network=network, nonzero=False, update=False, **kwargs):
        emissions = self.query_map('Emission',network=network, update=update, **kwargs)
        if netuid != None and len(emissions) > netuid:
            emissions = emissions[netuid]
            if nonzero:
                emissions =[e for e in emissions if e > 0]

        return emissions
    

    def incentive(self, 
                  netuid = netuid, 
                  block=None,  
                  network=network, 
                  nonzero:bool=False, 
                  update:bool = False, 
                  return_dict=False, 
                  **kwargs):
        incentive = self.query_map('Incentive', network=network, block=block, update=update, **kwargs)[netuid]

        if nonzero:
            incentive = {uid:i for uid, i in enumerate(incentive) if i > 0}
            # sort by incentive
            incentive = {k:v for k,v in sorted(incentive.items(), key=lambda x: x[1], reverse=False)}
        else:
            if return_dict: 
                return {k:v for k,v in enumerate(incentive)}
        return incentive
    
    def proposals(self, netuid = netuid, block=None,   network=network, nonzero:bool=False, update:bool = False,  **kwargs):
        proposals = [v for v in self.query_map('Proposals', network=network, block=block, update=update, **kwargs)]
        return proposals
        
    def trust(self, netuid = None, network=network, nonzero=False, update=False, **kwargs):
        trust = self.query_map('Trust', network=network, update=update, **kwargs)
        if netuid != None and len(trust) > netuid:
            trust = trust[netuid]

        return trust
    
    def last_update(self, netuid = netuid, block=None, network=network, update=False, **kwargs):
        return self.query_map('LastUpdate', network=network, block=block,  update=update, **kwargs)[netuid]
        

    def dividends(self, netuid = netuid, network=network, nonzero=False,  update=False, return_dict=False, **kwargs):
        netuid = self.resolve_netuid(netuid)
        dividends =  self.query_map('Dividends', network=network,  update=update,  **kwargs)[netuid]
        if nonzero:
            dividends = {i: d for i,d in enumerate(dividends) if d > 0}
            dividends = {k:v for k,v in sorted(dividends.items(), key=lambda x: x[1], reverse=False)}
        else:
            if return_dict:
                return {k:v for k,v in enumerate(dividends)}
            
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


    def key2uid(self, key = None, network:str=  None ,netuid: int = 0, **kwargs):
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

    def keep_archives(self, loockback_hours=24, end_time='now'):
        all_archive_paths = self.ls_archives()
        kept_archives = self.search_archives(lookback_hours=loockback_hours, end_time=end_time)
        kept_archive_paths = [a['path'] for a in kept_archives]
        rm_archive_paths = [a for a in all_archive_paths if a not in kept_archive_paths]
        for archive_path in rm_archive_paths:
            c.print('Removing', archive_path)
            c.rm(archive_path)
        return kept_archive_paths

    @classmethod
    def search_archives(cls, 
                    lookback_hours : int = 24,
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
    def archive_history(cls, 
                     *args, 
                     network=network, 
                     netuid= 0 , 
                     update=True,  
                     **kwargs):
        
        path = f'history/{network}.{netuid}.json'

        archive_history = []
        if not update:
            archive_history = cls.get(path, [])
        if len(archive_history) == 0:
            archive_history =  cls.search_archives(*args,network=network, netuid=netuid, **kwargs)
            cls.put(path, archive_history)
            
        return archive_history
        
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



    def stats(self, 
              search = None,
              netuid=0,  
              network = network,
              df:bool=True, 
              update:bool = False , 
              local: bool = True,
              cols : list = ['name', 'emission','incentive', 'dividends', 'stake', 'last_update', 'serving'],
              sort_cols = ['name', 'serving',  'emission', 'stake'],
              fmt : str = 'j',
              include_total : bool = True,
              **kwargs
              ):

        modules = self.my_modules(netuid=netuid, update=update, network=network, fmt=fmt, **kwargs)
        stats = []

        local_key_addresses = list(c.key2address().values())
        servers = c.servers(network='local')
        for i, m in enumerate(modules):
            if m['key'] not in local_key_addresses :
                continue
            # sum the stake_from
            m['stake_from'] = sum([v for k,v in m['stake_from']][1:])
            # we want to round these values to make them look nice
            for k in ['emission', 'dividends', 'incentive', 'stake_from']:
                m[k] = c.round(m[k], sig=4)

            m['serving'] = bool(m['name'] in servers)
            stats.append(m)
        df_stats =  c.df(stats)
        if len(stats) > 0:
            df_stats = df_stats[cols]
            if 'last_update' in cols:
                df_stats['last_update'] = df_stats['last_update'].apply(lambda x: x)
            if 'emission' in cols:
                epochs_per_day = self.epochs_per_day(netuid=netuid, network=network)
                df_stats['emission'] = df_stats['emission'] * epochs_per_day
            sort_cols = [c for c in sort_cols if c in df_stats.columns]  
            df_stats.sort_values(by=sort_cols, ascending=False, inplace=True)
            if search is not None:
                df_stats = df_stats[df_stats['name'].str.contains(search, case=True)]

        if not df:
            return df_stats.to_dict('records')
        else:
            return df_stats




    def my_modules(self,
                   search:str=None,  
                   netuid:int=0, 
                   network=network, 
                   df:bool = True, 
                   timeout=30,
                   batch_size=30,
                   update = False,
                   fmt = 'j',
                   modules:List[int] = None, 
                   n = 100,
                   **kwargs):
        
        path = 'my_modules'
        if not update:
            modules = self.get(path, None)
            if modules != None:
                return modules
            
        
        futures = []
        modules = []

        my_keys = self.my_keys(netuid=netuid, network=network, update=update, **kwargs)
        modules = self.get_modules(my_keys[:n], timeout=timeout)
        self.put(path, modules)
        return modules

        # futures = []
        # path = f'my_modules/{network}.{netuid}'
        # if not update:
        #     modules = self.get(path, modules)

        # if modules == None:
        #     path = f'my_modules/{network}.{netuid}'
        #     modules = []
        #     for k in keys:
        #         kwargs = dict(key=k, netuid=netuid, network=network)
        #         futures += [c.submit(self.get_module, kwargs = kwargs, timeout=timeout)]
        #         if len(futures) >= batch_size:
        #             for future in c.as_completed(futures):
        #                 module = future.result()
        #                 futures.remove(future)
        #                 if not c.is_error(module):
        #                     modules += [module]
        #                 break

        #     for future in c.as_completed(futures, timeout=timeout):
        #         module = future.result()
        #         if not c.is_error(module):
        #             modules += [module]

        #     self.put(path, modules)
            
        # if search != None:
        #     modules = [m for m in modules if search in m['name'].lower()]
    
        return modules


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

    def state_dict(self , 
                   timeout=1000, 
                   network='main', 
                   update=False, 
                   features=features,
                   mode='http', 
                   save = False,
                   block=None):
        
        
        start_time = c.time()
        self.resolve_network(network)

        if save:
            update = True
        if not update:
            state_path = self.latest_archive_path() # get the latest archive path
            state_dict = c.get(state_path, None)
            if state_path != None:
                return state_dict

        block = block if block != None else self.block

        path = f'state_dict/{network}.block-{block}-time-{int(c.time())}'

        def fn_query(*args, **kwargs):
            self = Subspace(mode=mode)
            return self.query_map(*args,**kwargs)
        
        
        def get_feature(feature, **kwargs):
            self = Subspace(mode=mode)
            return getattr(self, feature)(**kwargs)


        feature2params = {}
        feature2params['balances'] = [get_feature, dict(feature='balances', update=update, block=block)]
        feature2params['subnets'] = [get_feature, dict(feature='subnet_params', update=update, block=block, netuid=None, timeout=timeout)]
        feature2params['global'] = [get_feature, dict(feature='global_params', update=update, block=block, timeout=timeout)]
        
        for f in features:
            feature2params[f] = [fn_query, dict(name=f, update=update, block=block)]
        num_features = len(feature2params)
        progress = c.tqdm(total=num_features)

        feature2result = {}
        state_dict = {'block': block}
        while len(feature2params) > 0:
            
            for feature, (fn, kwargs) in feature2params.items():
                if feature in feature2result:
                    continue
                feature2result[feature] = c.submit(fn, kwargs) 
            result2feature = {v:k for k,v in feature2result.items()}
            futures = list(feature2result.values())
            for future in c.as_completed(futures, timeout=timeout):
                feature = result2feature[future]
                result = future.result()
                feature2params.pop(feature, None)
                result2feature.pop(future, None)
                if c.is_error(result):
                    c.print('ERROR IN FEATURE', feature, result)
                    continue
                state_dict[feature] = result

                # verbose 
                msg = {
                    'features_left': list(feature2params.keys()),

                }
                c.print(msg)
                progress.update(1)
            
            feature2result = {}

        if save:
            self.put(path, state_dict)
            end_time = c.time()
            latency = end_time - start_time
            response = {"success": True,
                        "msg": f'Saving state_dict to {path}', 
                        'latency': latency, 
                        'block': state_dict['block']}

        
        return response  # put it in storage
    

    def sync(self,*args, **kwargs):
        return  self.state_dict(*args, save=True, update=True, **kwargs)

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
    
    
    """
    #########################################
                    CHAIN LAND
    #########################################
    
    """

    def chain(self, *args, **kwargs):
        return c.module('subspace.chain')(*args, **kwargs)
    
    def chain_config(self, *args, **kwargs):
        return self.chain(*args, **kwargs).config
    
    def chains(self, *args, **kwargs):
        return self.chain(*args, **kwargs).chains()

    """
    #########################################
                    CHAIN LAND
    #########################################
    
    """
    ##################
    #### Register ####
    ##################

    def register(
        self,
        name: str , # defaults to module.tage
        address : str = 'NA',
        stake : float = 0,
        subnet: str = 'commune',
        key : str  = None,
        module_key : str = None,
        network: str = network,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        existential_balance = 1,
        nonce=None,
        fmt = 'nano',


    ) -> bool:
        
        assert name != None, f"Module name must be provided"

        # resolve the subnet name
        if subnet == None:
            subnet = self.config.subnet

        network =self.resolve_network(network)

        if address:
            address = c.namespace(network='local').get(name, name)
            address = address.replace(c.default_ip,c.ip())
        if module_key == None:
            module_key = c.get_key(name).ss58_address

        key = self.resolve_key(key)

        # Validate address.
        subnet2netuid = self.subnet2netuid()
        # default to commune
        netuid = subnet2netuid.get(subnet, 0)
        
        if stake == None:
            min_stake = self.subnet_params(netuid=netuid, fmt='j')['min_stake']
            stake = min_stake + existential_balance

        stake = self.to_nanos(stake)

        params = { 
                    'network': subnet.encode('utf-8'),
                    'address': address.encode('utf-8'),
                    'name': name.encode('utf-8'),
                    'stake': stake,
                    'module_key': module_key,
                } 
        # create extrinsic call
        response = self.compose_call('register', params=params, key=key, wait_for_inclusion=wait_for_inclusion, wait_for_finalization=wait_for_finalization, nonce=nonce)
        return response

    reg = register

    ##################
    #### Transfer ####
    ##################
    def transfer(
        self,
        dest: str, 
        amount: float , 
        key: str = None,
        network : str = None,
        nonce= None,
        
    ) -> bool:
        
        key = self.resolve_key(key)
        network = self.resolve_network(network)
        dest = self.resolve_key_ss58(dest)
        account_balance = self.get_balance( key.ss58_address , fmt='j' )
        if amount > account_balance:
            return {'success': False, 'message': f'Insufficient balance: {account_balance}'}

        amount = self.to_nanos(amount) # convert to nano (10^9 nanos = 1 token)
        dest_balance = self.get_balance( dest , fmt='j')

        response = self.compose_call(
            module='Balances',
            fn='transfer',
            params={
                'dest': dest, 
                'value': amount
            },
            key=key,
            nonce = nonce
        )

        if response['success']:
            response.update(
                {
                'from': {
                    'address': key.ss58_address,
                    'old_balance': account_balance,
                    'new_balance': self.get_balance( key.ss58_address , fmt='j')
                } ,
                'to': {
                    'address': dest,
                    'old_balance': dest_balance,
                    'new_balance': self.get_balance( dest , fmt='j'),
                }, 
                }
            )
        
        return response


    send = transfer

    ##################
    #### Transfer ####
    ##################
    def add_profit_shares(
        self,
        keys: List[str], 
        shares: List[float] = None , 
        key: str = None,
        network : str = None,
    ) -> bool:
        

        name2key = self.name2key()
        key = self.resolve_key(key)
        network = self.resolve_network(network)
        assert len(keys) > 0, f"Must provide at least one key"
        assert all([c.valid_ss58_address(k) for k in keys]), f"All keys must be valid ss58 addresses"
        if shares == None:
            shares = [1 for _ in keys]
        
        assert len(keys) == len(shares), f"Length of keys {len(keys)} must be equal to length of shares {len(shares)}"

        response = self.compose_call(
            module='SubspaceModule',
            fn='add_profit_shares',
            params={
                'keys': keys, 
                'shares': shares
            },
            key=key
        )

        return response


    def switch_module(self, module:str, new_module:str, n=10, timeout=20):
        stats = c.stats(module, df=False)
        namespace = c.namespace(new_module, public=True)
        servers = list(namespace.keys())[:n]
        stats = stats[:len(servers)]

        kwargs_list = []

        for m in stats:
            if module in m['name']:
                if len(servers)> 0: 
                    server = servers.pop()
                    server_address = namespace.get(server)
                    kwargs_list += [{'module': m['name'], 'name': server, 'address': server_address}]

        results = c.wait([c.submit(c.update_module, kwargs=kwargs, timeout=timeout, return_future=True) for kwargs in kwargs_list])
        
        return results
                


    def update_module(
        self,
        module: str, # the module you want to change
        # params from here
        name: str = None,
        address: str = None,
        delegation_fee: float = None,
        netuid: int = None,
        network : str = network,
        nonce = None,
        tip: int = 0,


    ) -> bool:
        self.resolve_network(network)
        key = self.resolve_key(module)
        netuid = self.resolve_netuid(netuid)  
        module_info = self.get_module(module)
        c.print(module_info,  module)
        if module_info['key'] == None:
            return {'success': False, 'msg': 'not registered'}
        
        c.print(module_info)

        if name == None:
            name = module
    
        if address == None:
            namespace_local = c.namespace(network='local')
            address = namespace_local.get(name,  f'{c.ip()}:{c.free_port()}'  )
            address = address.replace(c.default_ip, c.ip())
        # Validate that the module is already registered with the same address
        if name == module_info['name'] and address == module_info['address']:
            c.print(f"{c.emoji('check_mark')} [green] [white]{module}[/white] Module already registered and is up to date[/green]:[bold white][/bold white]")
            return {'success': False, 'message': f'{module} already registered and is up to date with your changes'}
        
        # ENSURE DELEGATE FEE IS BETWEEN 0 AND 100

        params = {
            'netuid': netuid, # defaults to module.netuid
             # PARAMS #
            'name': name, # defaults to module.tage
            'address': address, # defaults to module.tage
            'delegation_fee': delegation_fee, # defaults to module.delegate_fee
        }

        for k in ['delegation_fee']:
            if params[k] == None:
                params[k] = module_info[k]

        # check delegation_bounds
        assert params[k] != None, f"Delegate fee must be provided"
        delegation_fee = params['delegation_fee']
        if delegation_fee < 1.0 and delegation_fee > 0:
            delegation_fee = delegation_fee * 100
        assert delegation_fee >= 0 and delegation_fee <= 100, f"Delegate fee must be between 0 and 100"



        reponse  = self.compose_call('update_module',params=params, key=key, nonce=nonce, tip=tip)

        return reponse



    #################
    #### UPDATE SUBNET ####
    #################
    def update_subnet(
        self,
        netuid: int = None,
        key: str = None,
        network = network,
        nonce = None,
        update= True,
        **params,
    ) -> bool:
            
        self.resolve_network(network)
        netuid = self.resolve_netuid(netuid)
        subnet_params = self.subnet_params( netuid=netuid , update=update, network=network, fmt='nanos')
        # infer the key if you have it
        if key == None:
            key2address = self.address2key()
            if subnet_params['founder'] not in key2address:
                return {'success': False, 'message': f"Subnet {netuid} not found in local namespace, please deploy it "}
            key = c.get_key(key2address.get(subnet_params['founder']))
            c.print(f'Using key: {key}')

        # remove the params that are the same as the module info
        params = {**subnet_params, **params}
        for k in ['name', 'vote_mode']:
            params[k] = params[k].encode('utf-8')
        params['netuid'] = netuid

        response = self.compose_call(fn='update_subnet',
                                     params=params, 
                                     key=key, 
                                     nonce=nonce)

        return response


    #################
    #### Serving ####
    #################
    def propose_subnet_update(
        self,
        netuid: int = None,
        key: str = None,
        network = 'main',
        nonce = None,
        **params,
    ) -> bool:

        self.resolve_network(network)
        netuid = self.resolve_netuid(netuid)
        c.print(f'Adding proposal to subnet {netuid}')
        subnet_params = self.subnet_params( netuid=netuid , update=True)
        # remove the params that are the same as the module info
        params = {**subnet_params, **params}
        for k in ['name', 'vote_mode']:
            params[k] = params[k].encode('utf-8')
        params['netuid'] = netuid

        response = self.compose_call(fn='add_subnet_proposal',
                                     params=params, 
                                     key=key, 
                                     nonce=nonce)


        return response



    #################
    #### Serving ####
    #################
    def vote_proposal(
        self,
        proposal_id: int = None,
        key: str = None,
        network = 'main',
        nonce = None,
        **params,

    ) -> bool:

        self.resolve_network(network)
        # remove the params that are the same as the module info
        params = {
            'proposal_id': proposal_id,
            'netuid': netuid,
        }

        response = self.compose_call(fn='add_subnet_proposal',
                                     params=params, 
                                     key=key, 
                                     nonce=nonce)


        return response



    #################
    #### Serving ####
    #################
    def update_global(
        self,
        key: str = None,
        network = 'main',
        **params,
    ) -> bool:

        key = self.resolve_key(key)
        network = self.resolve_network(network)
        global_params = self.global_params( )
        global_params.update(params)
        params = global_params
        for k,v in params.items():
            if isinstance(v, str):
                params[k] = v.encode('utf-8')

        # this is a sudo call
        response = self.compose_call(fn='update_global',
                                     params=params, 
                                     key=key, 
                                     sudo=True)

        return response





    #################
    #### set_code ####
    #################
    def set_code(
        self,
        wasm_file_path = None,
        key: str = None,
        network = network,
    ) -> bool:

        if wasm_file_path == None:
            wasm_file_path = self.wasm_file_path()

        assert os.path.exists(wasm_file_path), f'Wasm file not found at {wasm_file_path}'

        self.resolve_network(network)
        key = self.resolve_key(key)

        # Replace with the path to your compiled WASM file       
        with open(wasm_file_path, 'rb') as file:
            wasm_binary = file.read()
            wasm_hex = wasm_binary.hex()

        code = '0x' + wasm_hex

        # Construct the extrinsic
        response = self.compose_call(
            module='System',
            fn='set_code',
            params={
                'code': code.encode('utf-8')
            },
            unchecked_weight=True,
            sudo = True,
            key=key
        )

        return response

    
    def transfer_stake(
            self,
            new_module_key: str ,
            module_key: str ,
            amount: Union['Balance', float] = None, 
            key: str = None,
            netuid:int = None,
            wait_for_inclusion: bool = False,
            wait_for_finalization: bool = True,
            network:str = None,
            existential_deposit: float = 0.1,
            sync: bool = False
        ) -> bool:
        # STILL UNDER DEVELOPMENT, DO NOT USE
        network = self.resolve_network(network)
        netuid = self.resolve_netuid(netuid)
        key = c.get_key(key)

        c.print(f':satellite: Staking to: [bold white]SubNetwork {netuid}[/bold white] {amount} ...')
        # Flag to indicate if we are using the wallet's own hotkey.

        name2key = self.name2key(netuid=netuid)
        module_key = self.resolve_module_key(module_key=module_key, netuid=netuid, name2key=name2key)
        new_module_key = self.resolve_module_key(module_key=new_module_key, netuid=netuid, name2key=name2key)

        assert module_key != new_module_key, f"Module key {module_key} is the same as new_module_key {new_module_key}"
        assert module_key in name2key.values(), f"Module key {module_key} not found in SubNetwork {netuid}"
        assert new_module_key in name2key.values(), f"Module key {new_module_key} not found in SubNetwork {netuid}"

        stake = self.get_stakefrom( module_key, from_key=key.ss58_address , fmt='j', netuid=netuid)

        if amount == None:
            amount = stake
        
        amount = self.to_nanos(amount - existential_deposit)
        
        # Get current stake
        params={
                    'netuid': netuid,
                    'amount': int(amount),
                    'module_key': module_key

                    }

        balance = self.get_balance( key.ss58_address , fmt='j')

        response  = self.compose_call('transfer_stake',params=params, key=key)

        if response['success']:
            new_balance = self.get_balance(key.ss58_address , fmt='j')
            new_stake = self.get_stakefrom( module_key, from_key=key.ss58_address , fmt='j', netuid=netuid)
            msg = f"Staked {amount} from {key.ss58_address} to {module_key}"
            return {'success': True, 'msg':msg, 'balance': {'old': balance, 'new': new_balance}, 'stake': {'old': stake, 'new': new_stake}}
        else:
            return  {'success': False, 'msg':response.error_message}



    def stake(
            self,
            module: Optional[str] = None, # defaults to key if not provided
            amount: Union['Balance', float] = None, 
            key: str = None,  # defaults to first key
            netuid:int = None,
            network:str = None,
            existential_deposit: float = 0.01,
            update=False
        ) -> bool:
        """
        description: 
            Unstakes the specified amount from the module. 
            If no amount is specified, it unstakes all of the amount.
            If no module is specified, it unstakes from the most staked module.
        params:
            amount: float = None, # defaults to all
            module : str = None, # defaults to most staked module
            key : 'c.Key' = None,  # defaults to first key 
            netuid : Union[str, int] = 0, # defaults to module.netuid
            network: str= main, # defaults to main
        return: 
            response: dict
        
        """
        network = self.resolve_network(network)
        netuid = self.resolve_netuid(netuid)
        key = c.get_key(key)
        
        name2key = self.name2key(netuid=netuid, update=update)
        
        if module == None:
            module_key = list(name2key.values())[0]

        else:
            
            if module in name2key:
                module_key = name2key[module]
            else:
                module_key = module

        # Flag to indicate if we are using the wallet's own hotkey.
        old_balance = self.get_balance( key.ss58_address , fmt='j')
        if amount is None:
            amount = old_balance

        amount = int(self.to_nanos(amount - existential_deposit))
        assert amount > 0, f"Amount must be greater than 0 and greater than existential deposit {existential_deposit}"
        
        # Get current stake
        params={
                    'netuid': netuid,
                    'amount': amount,
                    'module_key': module_key
                    }

        response = self.compose_call('add_stake',params=params, key=key)
        return response



    def unstake(
            self,
            module : str = None, # defaults to most staked module
            amount: float =None, # defaults to all of the amount
            key : 'c.Key' = None,  # defaults to first key
            netuid : Union[str, int] = 0, # defaults to module.netuid
            network: str= None,
            **kwargs
        ) -> dict:
        """
        description: 
            Unstakes the specified amount from the module. 
            If no amount is specified, it unstakes all of the amount.
            If no module is specified, it unstakes from the most staked module.
        params:
            amount: float = None, # defaults to all
            module : str = None, # defaults to most staked module
            key : 'c.Key' = None,  # defaults to first key 
            netuid : Union[str, int] = 0, # defaults to module.netuid
            network: str= main, # defaults to main
        return: 
            response: dict
        
        """
        if isinstance(module, int):
            amount = module
            module = None
        network = self.resolve_network(network)
        key = c.get_key(key)
        netuid = self.resolve_netuid(netuid)
        # get most stake from the module
        stake_to = self.get_stake_to(netuid=netuid, names = False, fmt='nano', key=key)

        module_key = None
        if module == None:
            # find the largest staked module
            max_stake = 0
            for k,v in stake_to.items():
                if v > max_stake:
                    max_stake = v
                    module_key = k            
        else:
            key2name = self.key2name(netuid=netuid)
            name2key = {key2name[k]:k for k,v in key2name.items()}
            if module in name2key:
                module_key = name2key[module]
            else:
                module_key = module
        
        # we expected to switch the module to the module key
        assert c.valid_ss58_address(module_key), f"Module key {module_key} is not a valid ss58 address"
        assert module_key in stake_to, f"Module {module_key} not found in SubNetwork {netuid}"
        if amount == None:
            amount = stake_to[module_key]
        # convert to nanos
        params={
            'amount': int(self.to_nanos(amount)),
            'netuid': netuid,
            'module_key': module_key
            }
        response = self.compose_call(fn='remove_stake',params=params, key=key, **kwargs)

        return response
            
    
    
    

    def stake_many( self, 
                        modules:List[str] = None,
                        amounts:Union[List[str], float, int] = None,
                        key: str = None, 
                        netuid:int = 0,
                        min_balance = 100_000_000_000,
                        n:str = 100,
                        network: str = None) -> Optional['Balance']:
        network = self.resolve_network( network )
        netuid = self.resolve_netuid( netuid )
        key = self.resolve_key( key )

        if modules == None:
            my_modules = self.my_modules(netuid=netuid, network=network, update=False)
            modules = [m['key'] for m in my_modules if 'vali' in m['name']]

        modules = modules[:n] # only stake to the first n modules

        assert len(modules) > 0, f"No modules found with name {modules}"
        module_keys = modules
        if amounts == None:
            balance = self.get_balance(key=key, fmt='nanos') - min_balance
            amounts = [(balance // len(modules))] * len(modules) 
            assert sum(amounts) < balance, f'The total amount is {sum(amounts)} > {balance}'
        else:
            if isinstance(amounts, (float, int)): 
                amounts = [amounts] * len(modules)

            for i, amount in enumerate(amounts):
                amounts[i] = self.to_nanos(amount)

        assert len(modules) == len(amounts), f"Length of modules and amounts must be the same. Got {len(modules)} and {len(amounts)}."

        params = {
            "netuid": netuid,
            "module_keys": module_keys,
            "amounts": amounts
        }

        response = self.compose_call('add_stake_multiple', params=params, key=key)

        return response
                    


    def transfer_multiple( self, 
                        destinations:List[str],
                        amounts:Union[List[str], float, int],
                        key: str = None, 
                        netuid:int = 0,
                        n:str = 10,
                        local:bool = False,
                        network: str = None) -> Optional['Balance']:
        network = self.resolve_network( network )
        key = self.resolve_key( key )
        balance = self.get_balance(key=key, fmt='j')

        # name2key = self.name2key(netuid=netuid)


        
        key2address = c.key2address()
        name2key = self.name2key(netuid=netuid)

        if isinstance(destinations, str):
            local_destinations = [k for k,v in key2address.items() if destinations in k]
            if len(destinations) > 0:
                destinations = local_destinations
            else:
                destinations = [_k for _n, _k in name2key.items() if destinations in _n]

        assert len(destinations) > 0, f"No modules found with name {destinations}"
        destinations = destinations[:n] # only stake to the first n modules
        # resolve module keys
        for i, destination in enumerate(destinations):
            if destination in name2key:
                destinations[i] = name2key[destination]
            if destination in key2address:
                destinations[i] = key2address[destination]

        if isinstance(amounts, (float, int)): 
            amounts = [amounts] * len(destinations)

        assert len(destinations) == len(amounts), f"Length of modules and amounts must be the same. Got {len(modules)} and {len(amounts)}."
        assert all([c.valid_ss58_address(d) for d in destinations]), f"Invalid destination address {destinations}"



        total_amount = sum(amounts)
        assert total_amount < balance, f'The total amount is {total_amount} > {balance}'


        # convert the amounts to their interger amount (1e9)
        for i, amount in enumerate(amounts):
            amounts[i] = self.to_nanos(amount)

        assert len(destinations) == len(amounts), f"Length of modules and amounts must be the same. Got {len(modules)} and {len(amounts)}."

        params = {
            "netuid": netuid,
            "destinations": destinations,
            "amounts": amounts
        }

        response = self.compose_call('transfer_multiple', params=params, key=key)

        return response

    transfer_many = transfer_multiple


    def unstake_many( self, 
                        modules:Union[List[str], str] = None,
                        amounts:Union[List[str], float, int] = None,
                        key: str = None, 
                        netuid:int = 0,
                        network: str = None) -> Optional['Balance']:
        
        network = self.resolve_network( network )
        key = self.resolve_key( key )

        if modules == None or modules == 'all':
            stake_to = self.get_staketo(key=key, netuid=netuid, names=False, update=True, fmt='nanos') # name to amount
            module_keys = [k for k in stake_to.keys()]
            # RESOLVE AMOUNTS
            if amounts == None:
                amounts = [stake_to[m] for m in module_keys]

        else:
            stake_to = self.get_staketo(key=key, netuid=netuid, names=False, update=True, fmt='j') # name to amount
            name2key = self.name2key(netuid=netuid, update=True)

            module_keys = []
            for i, module in enumerate(modules):
                if c.valid_ss58_address(module):
                    module_keys += [module]
                else:
                    assert module in name2key, f"Invalid module {module} not found in SubNetwork {netuid}"
                    module_keys += [name2key[module]]
                
            # RESOLVE AMOUNTS
            if amounts == None:
                amounts = [stake_to[m] for m in module_keys]

            if isinstance(amounts, (float, int)): 
                amounts = [amounts] * len(module_keys)

            for i, amount in enumerate(amounts):
                amounts[i] = self.to_nanos(amount) 

        assert len(module_keys) == len(amounts), f"Length of modules and amounts must be the same. Got {len(module_keys)} and {len(amounts)}."

        params = {
            "netuid": netuid,
            "module_keys": module_keys,
            "amounts": amounts
        }

        response = self.compose_call('remove_stake_multiple', params=params, key=key)

        return response
                    




    def unstake_all( self, 
                        key: str = None, 
                        netuid = 0,
                        network = network,
                        min_stake: float = 256) -> Optional['Balance']:
        
        network = self.resolve_network( network )
        key = self.resolve_key( key )
    
        key_stake_to = self.get_stake_to(key=key, netuid=netuid, names=False, update=True, fmt='j') # name to amount
        c.print(key_stake_to)
        # params = {
        #     "netuid": netuid,
        #     "module_keys": module_keys,
        #     "amounts": amounts
        # }

        # response = self.compose_call('remove_stake_multiple', params=params, key=key)

        return key_stake_to
                    


    def my_servers(self, search=None,  **kwargs):
        servers = [m['name'] for m in self.my_modules(**kwargs)]
        if search != None:
            servers = [s for s in servers if search in s]
        return servers
    
    def my_modules_names(self, *args, **kwargs):
        my_modules = self.my_modules(*args, **kwargs)
        return [m['name'] for m in my_modules]

    def my_module_keys(self, *args,  **kwargs):
        modules = self.my_modules(*args, **kwargs)
        return [m['key'] for m in modules]

    def my_key2uid(self, *args, network=None, netuid=0, update=False, **kwargs):
        key2uid = self.key2uid(*args, network=network, netuid=netuid, **kwargs)
        key2address = c.key2address(update=update )
        key_addresses = list(key2address.values())
        my_key2uid = { k: v for k,v in key2uid.items() if k in key_addresses}
        return my_key2uid
    
    def staked_modules(self, key = None, netuid = 0, network = network, **kwargs):
        key = self.resolve_key(key)
        netuid = self.resolve_netuid(netuid)
        staked_modules = self.get_stake_to(key=key, netuid=netuid, names=True, update=True, **kwargs)
        keys = list(staked_modules.keys())

        modules = self.get_modules(keys)

        return modules

    
    
    
    def my_keys(self, *args, **kwargs):
        return list(self.my_key2uid(*args, **kwargs).keys())

    def vote(
        self,
        uids: Union['torch.LongTensor', list] = None,
        weights: Union['torch.FloatTensor', list] = None,
        netuid: int = None,
        key: 'c.key' = None,
        network = None,
        update=False,
        n = 10,
    ) -> bool:
        import torch
        network = self.resolve_network(network)
        netuid = self.resolve_netuid(netuid)
        key = self.resolve_key(key)
        
        subnet = self.subnet( netuid = netuid )
        min_allowed_weights = subnet['min_allowed_weights']
        max_allowed_weights = subnet['max_allowed_weights']

        # checking if the "uids" are passed as names -> strings
        if uids != None and all(isinstance(item, str) for item in uids):
            names2uid = self.names2uids(names=uids, netuid=netuid)
            for i, name in enumerate(uids):
                if name in names2uid:
                    uids[i] = names2uid[name]
                else:
                    c.print(f'Could not find {name} in network {netuid}')
                    return False

        if uids == None:
            # we want to vote for the nonzero dividedn
            uids = self.nonzero_dividend_uids(netuid=netuid, network=network, update=update)
            assert len(uids) > 0, f"No nonzero dividends found in network {netuid}"
            # shuffle the uids
            uids = c.shuffle(uids)
            
        if weights is None:
            weights = [1 for _ in uids]

  
        if len(uids) < min_allowed_weights:
            n = self.n(netuid=netuid)
            while len(uids) < min_allowed_weights:
                
                uid = c.choice(list(range(n)))
                if uid not in uids:
                    uids.append(uid)
                    weights.append(1)

        uid2weight = {uid: weight for uid, weight in zip(uids, weights)}

        uids = list(uid2weight.keys())
        weights = weights[:len(uids)]

        c.print(f'Voting for {len(uids)} uids in network {netuid} with {len(weights)} weights')

        
        if len(uids) == 0:
            return {'success': False, 'message': f'No uids found in network {netuid}'}
        
        assert len(uids) == len(weights), f"Length of uids {len(uids)} must be equal to length of weights {len(weights)}"


        uids = uids[:max_allowed_weights]
        weights = weights[:max_allowed_weights]

        # uids = [int(uid) for uid in uids]
        uid2weight = {uid: weight for uid, weight in zip(uids, weights)}
        uids = list(uid2weight.keys())
        weights = list(uid2weight.values())

        # sort the uids and weights
        uids = torch.tensor(uids)
        weights = torch.tensor(weights)
        indices = torch.argsort(weights, descending=True)
        uids = uids[indices]
        weights = weights[indices]
        c.print(weights)
        weight_sum = weights.sum()
        assert weight_sum > 0, f"Weight sum must be greater than 0. Got {weight_sum}"
        weights = weights / (weight_sum)
        U16_MAX = 2**16 - 1
        weights = weights * (U16_MAX)
        weights = list(map(lambda x : int(min(x, U16_MAX)), weights.tolist()))

        uids = list(map(int, uids.tolist()))

        params = {'uids': uids,
                  'weights': weights, 
                  'netuid': netuid}
        
        response = self.compose_call('set_weights',params = params , key=key)
            
        if response['success']:
            return {'success': True,  'num_weigts': len(uids), 'message': 'Set weights', 'key': key.ss58_address, 'netuid': netuid, 'network': network}
        
        return response

    set_weights = vote



    def register_servers(self, search=None, **kwargs):
        stakes = self.stakes()
        for m in c.servers(network='local'):
            try:
                key = c.get_key(m)
                if key.ss58_address in stakes:
                    self.update_module(module=m)
                else:
                    self.register(name=m)
            except Exception as e:
                c.print(e, color='red')

        
    reg_servers = register_servers
    def reged_servers(self, **kwargs):
        servers =  c.servers(network='local')



    def my_uids(self, *args, **kwargs):
        return list(self.my_key2uid(*args, **kwargs).values())
    
    
    
    def my_names(self, *args, **kwargs):
        my_modules = self.my_modules(*args, **kwargs)
        return [m['name'] for m in my_modules]
 


    def registered_servers(self, netuid = None, network = network,  **kwargs):
        netuid = self.resolve_netuid(netuid)
        network = self.resolve_network(network)
        servers = c.servers(network='local')
        registered_keys = []
        for s in servers:
            if self.is_registered(s, netuid=netuid):
                registered_keys += [s]
        return registered_keys
    reged = reged_servers = registered_servers

    def unregistered_servers(self, netuid = None, network = network,  **kwargs):
        netuid = self.resolve_netuid(netuid)
        network = self.resolve_network(network)
        network = self.resolve_network(network)
        servers = c.servers(network='local')
        unregistered_keys = []
        for s in servers:
            if not self.is_registered(s, netuid=netuid):
                unregistered_keys += [s]
        return unregistered_keys

    
    def check_reged(self, netuid = None, network = network,  **kwargs):
        reged = self.reged(netuid=netuid, network=network, **kwargs)
        jobs = []
        for module in reged:
            job = c.call(module=module, fn='info',  network='subspace', netuid=netuid, return_future=True)
            jobs += [job]

        results = dict(zip(reged, c.gather(jobs)))

        return results 

    unreged = unreged_servers = unregistered_servers
               
    
    def my_balances(self, search=None, min_value=1000, fmt='j', update=False, **kwargs):
        balances = self.balances(fmt=fmt, update=update, **kwargs)
        address2key = c.address2key(search)
        my_balances = {k:balances.get(k, 0) for k in address2key.keys()}

        # sort the balances
        my_balances = {k:my_balances[k] for k in sorted(my_balances.keys(), key=lambda x: my_balances[x], reverse=True)}
        if min_value != None:
            my_balances = {k:v for k,v in my_balances.items() if v >= min_value}
        return my_balances
    
    
    
    def launcher_key(self, search=None, min_value=1000, **kwargs):
        
        my_balances = self.my_balances(search=search, min_value=min_value, **kwargs)
        key_address =  c.choice(list(my_balances.keys()))
        key_name = c.address2key(key_address)
        return key_name
    
    def launcher_keys(self, search=None, min_value=1000, n=1000, **kwargs):
        my_balances = self.my_balances(search=search, min_value=min_value, **kwargs)
        key_addresses = list(my_balances.keys())[:n]
        address2key = c.address2key()
        return [address2key[k] for k in key_addresses]

    def stake_spread(self,  modules:list=None, key:str = None,ratio = 1.0, n:int=50):
        key = self.resolve_key(key)
        name2key = self.name2key()
        if modules == None:
            modules = self.top_valis(n=n)
        if isinstance(modules, str):
            modules = [k for k,v in name2key.items() if modules in k]

        modules = modules[:n]
        modules = c.shuffle(modules)

        name2key = {k:name2key[k] for k in modules if k in name2key}


        module_names = list(name2key.keys())
        module_keys = list(name2key.values())
        n = len(name2key)

        # get the balance, of the key
        balance = self.get_balance(key)
        assert balance > 0, f'balance must be greater than 0, not {balance}'
        assert ratio <= 1.0, f'ratio must be less than or equal to 1.0, not {ratio}'
        assert ratio > 0.0, f'ratio must be greater than or equal to 0.0, not {ratio}'

        balance = int(balance * ratio)
        assert balance > 0, f'balance must be greater than 0, not {balance}'
        stake_per_module = int(balance/n)


        c.print(f'staking {stake_per_module} per module for ({module_names}) modules')

        s = c.module('subspace')()

        s.stake_many(key=key, modules=module_keys, amounts=stake_per_module)

       
    def key2value(self, search=None, fmt='j', netuid=0, **kwargs):
        key2value = self.my_balance(search=search, fmt=fmt, netuid=netuid, **kwargs)
        for k,v in self.my_stake(search=search, fmt=fmt, netuid=netuid, **kwargs).items():
            key2value[k] += v
        return key2value

    def total_value(self, search=None, fmt='j', **kwargs):
        return sum(self.key2value(search=search, fmt=fmt, **kwargs).values())


    def my_stake(self, search=None, netuid = None, network = None, fmt=fmt,  decimals=2, block=None, update=False):
        mystaketo = self.my_stake_to(netuid=netuid, network=network, fmt=fmt, decimals=decimals, block=block, update=update)
        key2stake = {}
        for key, staketo_tuples in mystaketo.items():
            stake = sum([s for a, s in staketo_tuples])
            key2stake[key] = c.round_decimals(stake, decimals=decimals)
        if search != None:
            key2stake = {k:v for k,v in key2stake.items() if search in k}
        return key2stake
    


    def stake_top_modules(self,search=None, netuid=netuid, **kwargs):
        top_module_keys = self.top_module_keys(k='dividends')
        self.stake_many(modules=top_module_keys, netuid=netuid, **kwargs)
    
    def rank_my_modules(self,search=None, k='stake', n=10, **kwargs):
        modules = self.my_modules(search=search, **kwargs)
        ranked_modules = self.rank_modules(modules=modules, search=search, k=k, n=n, **kwargs)
        return modules[:n]


    mys =  mystake = key2stake =  my_stake



    def my_balance(self, search:str=None, update=False, network:str = 'main', fmt=fmt,  block=None, min_value:int = 0):

        balances = self.balances(network=network, fmt=fmt, block=block, update=update)
        my_balance = {}
        key2address = c.key2address()
        for key, address in key2address.items():
            if address in balances:
                my_balance[key] = balances[address]

        if search != None:
            my_balance = {k:v for k,v in my_balance.items() if search in k}
            
        my_balance = dict(sorted(my_balance.items(), key=lambda x: x[1], reverse=True))

        if min_value > 0:
            my_balance = {k:v for k,v in my_balance.items() if v > min_value}

        return my_balance
        
    key2balance = myb = mybal = my_balance

    def my_stake_to(self,search=None, netuid = 0, network = None, fmt=fmt,  decimals=2, block=None, update=False):
        staketo = self.stake_to(netuid=netuid, 
                                network=network, 
                                block=block, 
                                update=update)
        mystaketo = {}
        key2address = c.key2address()
        if netuid == 'all':
            netuids = list(range(len(staketo)))
        else:
            netuids = [netuid]
        
        for netuid in netuids:
            for key, address in key2address.items():
                if address in staketo:
                    mystaketo[key] = [[a, self.format_amount(s, fmt=fmt)] for a, s in staketo[address]]

        return mystaketo
    my_staketo = my_stake_to

    def my_value(
                 self, 
                 network = None,
                 update=False,
                 fmt='j'
                 ):
        return self.my_total_stake(network=network, update=update, fmt=fmt) + \
                    self.my_total_balance(network=network, update=update, fmt=fmt)
    
    my_supply   = my_value

    def nonzero_dividend_uids(self, netuid=netuid, network=network, **kwargs):
        dividends = self.dividends(netuid=netuid, network=network, nonzero=True, **kwargs)
        return list(dividends.keys())
    
    def nonzero_dividend_keys(self, netuid=netuid, network=network, update=False, **kwargs):
        nonzero_dividend_uids = self.nonzero_dividend_uids(netuid=netuid, network=network, update=update, **kwargs)
        uid2key = self.uid2key(netuid=netuid, network=network, update=update, **kwargs)
        return [uid2key[uid] for uid in nonzero_dividend_uids]

    def uid2dividend(self, netuid=netuid, network=network, update=False, nonzero=False,**kwargs):
        uid2dividend = {}
        dividends = self.dividends( netuid=netuid, network=network, update=update,nonzero=nonzero, return_dict=True,**kwargs)
        for uid, dividend in dividends.items():
            uid2dividend[uid] = dividend
        return uid2dividend

    def subnet2stake(self, network=None, update=False) -> dict:
        subnet2stake = {}
        for subnet_name in self.subnet_names(network=network):
            c.print(f'Getting stake for subnet {subnet_name}')
            subnet2stake[subnet_name] = self.my_total_stake(network=network, netuid=subnet_name , update=update)
        return subnet2stake

    def my_total_stake(self, network = None, fmt=fmt, update=False):
        return sum(self.staker2stake(network=network, fmt=fmt, update=update,  local=True).values())


    def staker2stake(self,  update=False, network=None, fmt='j', local=False):
        staker2netuid2stake = self.staker2netuid2stake(update=update, network=network, fmt=fmt, local=local)
        staker2stake = {}
        for staker, netuid2stake in staker2netuid2stake.items():
            if staker not in staker2stake:
                staker2stake[staker] = 0
            staker2stake[staker] +=  self.format_amount(sum(netuid2stake.values()),fmt=fmt)
        return staker2stake
    

    def staker2netuid2stake(self,  update=False, network=None, fmt='j', local=False):
        stake_to = self.query_map("StakeTo", update=update, network=network)
        staker2netuid2stake = {}
        for netuid , stake_to_subnet in enumerate(stake_to):
            for staker, stake_tuples in stake_to_subnet.items():
                staker2netuid2stake[staker] = staker2netuid2stake.get(staker, {})
                staker2netuid2stake[staker][netuid] = staker2netuid2stake[staker].get(netuid, [])
                staker2netuid2stake[staker][netuid] = sum(list(map(lambda x: x[-1], stake_tuples )))

        if local:
            address2key = c.address2key()
            staker2netuid2stake = {address:staker2netuid2stake.get(address,{}) for address in address2key.keys()}

        
        return staker2netuid2stake
    

 
    def my_total_balance(self, network = None, fmt=fmt, update=False):
        return sum(self.my_balance(network=network, fmt=fmt, update=update ).values())


    def check_valis(self, **kwargs):
        return self.check_servers(search='vali', **kwargs)
    
    def check_servers(self, search='vali',
                    wait_for_server=False, 
                    update:bool=False, 
                    min_lag=1000, 
                    key=None, 
                    min_stake:int =1000,  
                    network='local'):
        cols = ['name', 'registered', 'serving', 'address', 'last_update', 'stake', 'dividends']
        module_stats = self.stats(search=search, netuid=0, cols=cols, df=False, update=update)
        module2stats = {m['name']:m for m in module_stats}
        block = self.block


        response_batch = {}

        for module, stats in module2stats.items():
            if stats['stake'] > min_stake:
                # check if the module is serving
                lag = block - stats['last_update']
                should_validator_update = lag > min_lag and stats['dividends'] > 0
                if should_validator_update:
                    c.print(f"Vali {module} has not voted in {lag} blocks. Restarting...")
                if not c.server_exists(module) and should_validator_update:
                    response_batch[module] = c.serve(module)
                    c.print(response_batch[module])

    def compose_call(self,
                     fn:str, 
                    params:dict = None, 
                    key:str = None,
                    tip: int = 0, # tip can
                    module:str = 'SubspaceModule', 
                    wait_for_inclusion: bool = True,
                    wait_for_finalization: bool = True,
                    process_events : bool = True,
                    color: str = 'yellow',
                    verbose: bool = True,
                    save_history : bool = True,
                    sudo:bool  = False,
                    nonce: int = None,
                    remote_module: str = None,
                    unchecked_weight: bool = False,
                    network = network,
                    mode='ws',
                     **kwargs):

        """
        Composes a call to a Substrate chain.

        """
        key = self.resolve_key(key)
        network = self.resolve_network(network, mode=mode)

        if remote_module != None:
            kwargs = c.locals2kwargs(locals())
            return c.connect(remote_module).compose_call(**kwargs)

        params = {} if params == None else params
        if verbose:
            kwargs = c.locals2kwargs(locals())
            kwargs['verbose'] = False
            c.status(f":satellite: Calling [bold]{fn}[/bold]")
            return self.compose_call(**kwargs)

        start_time = c.datetime()
        ss58_address = key.ss58_address
        paths = {m: f'history/{self.network}/{ss58_address}/{m}/{start_time}.json' for m in ['complete', 'pending']}

        compose_kwargs = dict(
                call_module=module,
                call_function=fn,
                call_params=params,
        )

        c.print('compose_kwargs', compose_kwargs, color=color)
        tx_state = dict(status = 'pending',start_time=start_time, end_time=None)

        self.put_json(paths['pending'], tx_state)

        substrate = self.get_substrate(network=network, mode='ws')
        call = substrate.compose_call(**compose_kwargs)

        if sudo:
            call = substrate.compose_call(
                call_module='Sudo',
                call_function='sudo',
                call_params={
                    'call': call,
                }
            )
        if unchecked_weight:
            # uncheck the weights for set_code
            call = substrate.compose_call(
                call_module="Sudo",
                call_function="sudo_unchecked_weight",
                call_params={
                    "call": call,
                    'weight': (0,0)
                },
            )
        # get nonce 
        extrinsic = substrate.create_signed_extrinsic(call=call,keypair=key,nonce=nonce, tip=tip)

        response = substrate.submit_extrinsic(extrinsic=extrinsic,
                                                wait_for_inclusion=wait_for_inclusion, 
                                                wait_for_finalization=wait_for_finalization)

        if wait_for_finalization:
            if process_events:
                response.process_events()

            if response.is_success:
                response =  {'success': True, 'tx_hash': response.extrinsic_hash, 'msg': f'Called {module}.{fn} on {self.network} with key {key.ss58_address}'}
            else:
                response =  {'success': False, 'error': response.error_message, 'msg': f'Failed to call {module}.{fn} on {self.network} with key {key.ss58_address}'}
        else:
            response =  {'success': True, 'tx_hash': response.extrinsic_hash, 'msg': f'Called {module}.{fn} on {self.network} with key {key.ss58_address}'}
        

        tx_state['end_time'] = c.datetime()
        tx_state['status'] = 'completed'
        tx_state['response'] = response

        # remo 
        self.rm(paths['pending'])
        self.put_json(paths['complete'], tx_state)

        return response
            

    def tx_history(self, key:str=None, mode='complete',network=network, **kwargs):
        key_ss58 = self.resolve_key_ss58(key)
        assert mode in ['pending', 'complete']
        pending_path = f'history/{network}/{key_ss58}/{mode}'
        return self.glob(pending_path)
    
    def pending_txs(self, key:str=None, **kwargs):
        return self.tx_history(key=key, mode='pending', **kwargs)

    def complete_txs(self, key:str=None, **kwargs):
        return self.tx_history(key=key, mode='complete', **kwargs)

    def clean_tx_history(self):
        return self.ls(f'tx_history')

        
    def resolve_tx_dirpath(self, key:str=None, mode:'str([pending,complete])'='pending', network=network, **kwargs):
        key_ss58 = self.resolve_key_ss58(key)
        assert mode in ['pending', 'complete']
        pending_path = f'history/{network}/{key_ss58}/{mode}'
        return pending_path
    

    def resolve_key(self, key = None):
        if key == None:
            key = self.config.key
        if key == None:
            key = 'module'
        if isinstance(key, str):
            if c.key_exists( key ):
                key = c.get_key( key )
            else:
                raise ValueError(f"Key {key} not found in your keys, please make sure you have it")
        assert hasattr(key, 'ss58_address'), f"Invalid Key {key} as it should have ss58_address attribute."
        return key
        
    @classmethod
    def test_endpoint(cls, url=None):
        if url == None:
            url = c.choice(cls.urls())
        self = cls()
        c.print('testing url -> ', url, color='yellow' )

        try:
            self.set_network(url=url, max_trials=1)
            success = isinstance(self.block, int)
        except Exception as e:
            c.print(c.detailed_error(e))
            success = False

        c.print(f'success {url}-> ', success, color='yellow' )
        
        return success


    def stake_spread_top_valis(self):
        top_valis = self.top_valis()
        name2key = self.name2key()
        for vali in top_valis:
            key = name2key[vali]

    @classmethod
    def pull(cls, rpull:bool = False):
        if len(cls.ls(cls.libpath)) < 5:
            c.rm(cls.libpath)
        c.pull(cwd=cls.libpath)
        if rpull:
            cls.rpull()




    def dashboard(self, **kwargs):
        import streamlit as st
        return st.write(self.get_module())
    



Subspace.run(__name__)
