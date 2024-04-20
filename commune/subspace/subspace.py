
from retry import retry
from typing import *
import json
import os
import commune as c
import requests 
from substrateinterface import SubstrateInterface

U32_MAX = 2**32 - 1
U16_MAX = 2**16 - 1

class Subspace(c.Module):
    """
    Handles interactions with the subspace chain.
    """


    whitelist = ['query', 
                 'score',
                 'query_map', 
                 'get_module', 
                 'get_balance', 
                 'get_stake_to', 
                 'get_stake_from']

    module_features = ['key', 
                       'address', 
                       'name', 
                       'emission', 
                       'incentive', 
                       'dividends', 
                       'last_update', 
                       'stake_from', 
                       'weights',
                       'delegation_fee',
                       'trust', 
                       'regblock']

    subnet_features = [
                            "Tempo",
                           'ImmunityPeriod',
                            'MinAllowedWeights',
                           'MaxAllowedWeights',
                            'MaxAllowedUids',
                            'MinStake',
                            'Founder', 
                           'FounderShare',
                            'IncentiveRatio',
                            'TrustRatio',
                            'VoteModeSubnet',
                            'MaxWeightAge',
                            'MaxStake', 
                            'SubnetNames'
                            ]
    
    global_features = [  'BurnRate',
                         'MaxNameLength',
                            'MaxAllowedModules',
                            'MaxAllowedSubnets',
                            'MaxRegistrationsPerBlock', 
                            'MinBurn',
                            'MinStakeGlobal',
                            'MinWeightStake',
                            'UnitEmission',
    ] 



    
    module_features = [
                            'key', 
                            'name',
                            'address',
                            'emission',
                            'incentive', 
                            'dividends', 
                            'last_update', 
                            'stake_from', 
                            'delegation_fee']
    cost = 1
    block_time = 8 # (seconds)
    default_config = c.get_config('subspace', to_munch=False)
    token_decimals = 9
    network = default_config['network']
    chain = network
    libpath = chain_path = c.libpath + '/subspace'
    netuid = 0
    local = default_config['local']

    def __init__( 
        self, 
        **kwargs,
    ):
        self.set_config(kwargs=kwargs)

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
                network:str = 'main',
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
                trials:int = 10,
                cache:bool = True,
                mode = 'http',):
        
        network = network or self.config.network


        
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


        while trials > 0:
            try:
                url = self.resolve_url(url, mode=mode, network=network)

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
                break
            except Exception as e:
                trials = trials - 1
                if trials > 0:
                    raise e
        
        if cache:
            self.url2substrate[url] = substrate

        self.network = network
        self.url = url
        
        return substrate


    def set_network(self, 
                network:str = 'main',
                mode = 'http',
                trials = 10,
                url : str = None, **kwargs):
               
        self.substrate = self.get_substrate(network=network, url=url, mode=mode, trials=trials , **kwargs)
        response =  {'network': self.network, 'url': self.url}
        c.print(response)
        
        return response

    def __repr__(self) -> str:
        return f'<Subspace: network={self.network}>'
    def __str__(self) -> str:

        return f'<Subspace: network={self.network}>'



    def wasm_file_path(self):
        wasm_file_path = self.libpath + '/target/release/wbuild/node-subspace-runtime/node_subspace_runtime.compact.compressed.wasm'
        return wasm_file_path
    

    def my_stake_from(self, netuid = 0, block=None, update=False, network=network, fmt='j', max_age=1000 , **kwargs):
        stake_from_tuples = self.stake_from(netuid=netuid,
                                             block=block,
                                               update=update, 
                                            network=network, 
                                               tuples = True,
                                               fmt=fmt, max_age=max_age, **kwargs)

        address2key = c.address2key()
        stake_from_total = {}
        if netuid == 'all':
            for netuid, stake_from_tuples_subnet in stake_from_tuples.items():
                for module_key,staker_tuples in stake_from_tuples_subnet.items():
                    for staker_key, stake in staker_tuples:
                        if module_key in address2key:
                            stake_from_total[staker_key] = stake_from_total.get(staker_key, 0) + stake

        else:
            for module_key,staker_tuples in stake_from_tuples.items():
                for staker_key, stake in staker_tuples:
                    if module_key in address2key:
                        stake_from_total[staker_key] = stake_from_total.get(staker_key, 0) + stake

        
        for staker_address in address2key.keys():
            if staker_address in stake_from_total:
                stake_from_total[staker_address] = self.format_amount(stake_from_total[staker_address], fmt=fmt)
        return stake_from_total   

    
    def delegation_fee(self, netuid = 0, block=None, network=None, update=False, fmt='j'):
        delegation_fee = self.query_map('DelegationFee', netuid=netuid, block=block ,update=update, network=network)
        return delegation_fee

    def stake_to(self, netuid = 0, network=network, block=None,  max_age=1000, update=False, fmt='nano',**kwargs):
        stake_to = self.query_map('StakeTo', netuid=netuid, block=block, max_age=max_age, update=update, network=network, **kwargs)
        format_tuples = lambda x: [[_k, self.format_amount(_v, fmt=fmt)] for _k,_v in x]
        if netuid == 'all':
            stake_to = {netuid: {k: format_tuples(v) for k,v in stake_to[netuid].items()} for netuid in stake_to}
        else:
            stake_to = {k: format_tuples(v) for k,v in stake_to.items()}
    
        return stake_to
    
    
    def key2stake(self, netuid = 0,
                     block=None, 
                    update=False, 
                    names = False,
                    max_age = 1000,
                    network='main', fmt='j'):
        stake_to = self.stake_to(netuid=netuid, 
                                block=block, 
                                max_age=max_age,
                                update=update, 
                                network=network, 
                                fmt=fmt)
        address2key = c.address2key()
        stake_to_total = {}
        if netuid == 'all':
            stake_to_dict = stake_to
           
            for staker_address in address2key.keys():
                for netuid, stake_to in stake_to_dict.items(): 
                    if staker_address in stake_to:
                        stake_to_total[staker_address] = stake_to_total.get(staker_address, 0) + sum([v[1] for v in stake_to.get(staker_address)])
            c.print(stake_to_total)
        else:
            for staker_address in address2key.keys():
                if staker_address in stake_to:
                    stake_to_total[staker_address] = stake_to_total.get(staker_address, 0) + sum([v[1] for v in stake_to[staker_address]])
            # sort the dictionary by value
            stake_to_total = dict(sorted(stake_to_total.items(), key=lambda x: x[1], reverse=True))

        return stake_to_total
    my_stake_to = key2stake


    def empty_keys(self, network='main', block=None, update=False, max_age=1000, fmt='j'):
        key2address = c.key2address()
        key2value = self.key2value(network=network, block=block, update=update, max_age=max_age, fmt=fmt)
        empty_keys = []
        for key,key_address in key2address.items():
            key_value = key2value.get(key_address, 0)
            if key_value == 0:
                empty_keys.append(key)
               
        return empty_keys

    def key2value(self, netuid = 'all', block=None, update=False, max_age=1000, network='main', fmt='j', min_value=0, **kwargs):
        key2balance = self.key2balance(block=block, update=update, network=network, max_age=max_age, fmt=fmt)
        key2stake = self.key2stake(netuid=netuid, block=block, update=update, network=network, max_age=max_age, fmt=fmt)
        key2value = {}
        keys = set(list(key2balance.keys()) + list(key2stake.keys()))
        for key in keys:
            key2value[key] = key2balance.get(key, 0) + key2stake.get(key, 0)
        key2value = {k:v for k,v in key2value.items()}
        key2value = dict(sorted(key2value.items(), key=lambda x: x[1], reverse=True))
        return key2value
    
    def min_burn(self,  network='main', block=None, update=False, fmt='j'):
        min_burn = self.query('MinBurn', block=block, update=update, network=network)
        return self.format_amount(min_burn, fmt=fmt)



    def query(self, 
              name:str,  
              params = None, 
              module:str='SubspaceModule',
              block=None,  
              netuid = None,
              network: str = network, 
              save= True,
              max_age=1000,
              mode = 'http',
            update=False):
        
        """
        query a subspace storage function with params and block.
        """

        network = self.resolve_network(network)
        path = f'query/{network}/{module}.{name}'
    
        params = params or []
        if not isinstance(params, list):
            params = [params]
        if netuid != None and netuid != 'all':
            params = [netuid] + params
            
        # we want to cache based on the params if there are any
        if len(params) > 0 :
            path = path + f'::params::' + '-'.join([str(p) for p in params])

        value = self.get(path, None, max_age=max_age, update=update)
        if value != None:
            return value
        substrate = self.get_substrate(network=network, mode=mode)
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
    
    

    def query_map(self, name: str = 'StakeFrom', 
                  params: list = None,
                  block: Optional[int] = None, 
                  network:str = 'main',
                  netuid = None,
                  page_size=1000,
                  max_results=100000,
                  module='SubspaceModule',
                  update: bool = True,
                  max_age : str = 1000, # max age in seconds
                  mode = 'http',
                  **kwargs
                  ) -> Optional[object]:
        """ Queries subspace map storage with params and block. """
        # if all lowercase then we want to capitalize the first letter
        if name[0].islower():
            _splits = name.split('_')
            name = _splits[0].capitalize() + ''.join([s[0].capitalize() + s[1:] for s in _splits[1:]])
        if name  == 'Account':
            module = 'System'
        network = self.resolve_network(network, new_connection=False, mode=mode)

        path = f'query/{network}/{module}.{name}'
        # resolving the params
        params = params or []

        is_single_subnet = bool(netuid != 'all' and netuid != None)
        if is_single_subnet:
            params = [netuid] + params
        if not isinstance(params, list):
            params = [params]
        if len(params) > 0 :
            path = path + f'::params::' + '-'.join([str(p) for p in params])
        path = path+"::block::"
        paths = self.glob(path + '*')
        update = update or len(paths) == 0 or block != None
        if not update:
            last_path = sorted(paths, reverse=True)[0]
            value = self.get(last_path, None , max_age=max_age)
        else:
            value = None

        if value == None:
            # block = block or self.block
            path = path + f'{block}'
            network = self.resolve_network(network)
            # if the value is a tuple then we want to convert it to a list
    
            substrate = self.get_substrate(network=network, mode=mode)
            qmap =  substrate.query_map(
                module=module,
                storage_function = name,
                params = params,
                page_size = page_size,
                max_results = max_results,
                block_hash =substrate.get_block_hash(block)
            )

            new_qmap = {} 
            progress_bar = c.progress(qmap, desc=f'Querying {name} ma')
            for (k,v) in qmap:
                progress_bar.update(1)
                if not isinstance(k, tuple):
                    k = [k]
                if type(k) in [tuple,list]:
                    # this is a double map
                    k = [_k.value for _k in k]
                if hasattr(v, 'value'):
                    v = v.value
                    c.dict_put(new_qmap, k, v)

            self.put(path, new_qmap)
        
        else: 
            new_qmap = value

        def convert_dict_k_digit_to_int(d):
            is_int_bool = False
            for k,v in c.copy(d).items():
                if c.is_int(k):
                    is_int_bool = True
                    d[int(k)] = d.pop(k)
                    if isinstance(v, dict):
                        d[int(k)] = convert_dict_k_digit_to_int(v)
            if is_int_bool:
                # sort the dictionary by key
                d = dict(sorted(d.items()))
            
            return d
                    

        new_map = convert_dict_k_digit_to_int(new_qmap)

        return new_map
    
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
    def n(self,  netuid: int = 0, network = 'main' ,block: Optional[int] = None, update=True, **kwargs ) -> int:
        if netuid == 'all':
            return sum(self.query_map('N', block=block , update=update, network=network, **kwargs).values())
        else:
            return self.query( 'N', params=[netuid], block=block , update=update, network=network, **kwargs)

    ##########################
    #### Account functions ###
    
    """ Returns network Tempo hyper parameter """
    def stakes(self, netuid: int = 0, block: Optional[int] = None, fmt:str='nano', max_age = 100,network=None, update=False, **kwargs) -> int:
        stakes =  self.query_map('Stake', netuid=netuid, update=update, max_age=max_age, **kwargs)
        if netuid == 'all':
            subnet2stakes = c.copy(stakes)
            stakes = {}
            for netuid, subnet_stakes in subnet2stakes.items():
                for k,v in subnet_stakes.items():
                    stakes[k] = stakes.get(k, 0) + v
        
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

                    if key in name2key:
                        key_address = name2key[key]
                    else:
                        key_address = key 
        # if the key has an attribute then its a key
        elif hasattr(key, 'ss58_address'):
            key_address = key.ss58_address
        
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
    def format_amount(cls, x, fmt='nano', decimals = None, format=None, features=None, **kwargs):
        fmt = format or fmt # format is an alias for fmt

        if fmt in ['token', 'unit', 'j', 'J']:
            x = x / 10**9
        
        if decimals != None:
            x = c.round_decimals(x, decimals=decimals)
  

        return x
    
    def get_stake( self, key_ss58: str, block: Optional[int] = None, netuid:int = None , fmt='j', update=True ) -> Optional['Balance']:
        
        key_ss58 = self.resolve_key_ss58( key_ss58)
        netuid = self.resolve_netuid( netuid )
        stake = self.query( 'Stake',params=[netuid, key_ss58], block=block , update=update)
        return self.format_amount(stake, fmt=fmt)

    def all_balances(self, timeout=4):
        key2address = c.key2address()
        futures = []
        for key, address in key2address.items():
            future = c.submit(self.get_balance, kwargs={'key': address}, timeout=timeout)
            futures.append(future)
            c.print(f'{key}: {balance}')
        
        balances = c.wait(futures, timeout=timeout)
        return balances

    

    def all_key_info(self, netuid='all', timeout=10, update=False, **kwargs):
        my_keys = c.my_keys()


    def key_info(self, key:str = None, netuid='all', timeout=10, update=False, **kwargs):
        key_info = {
            'balance': c.get_balance(key=key, **kwargs),
            'stake_to': c.get_stake_to(key=key, netuid=netuid, **kwargs),
        }
        return key_info

    def my_total_stake_to( self, 
                     key: str = None, 
                     module_key=None,
                       block: Optional[int] = None, 
                       timeout=20,
                       names = False,
                        fmt='j' , network=None,
                          update=False,
                        max_age = 1000,
                         **kwargs) -> Optional['Balance']:
        kwargs['netuid'] = 'all'
        return sum(list(self.my_netuid2stake(key=key, module_key=module_key,
                                              block=block, timeout=timeout, names=names, fmt=fmt, 
                                 network=network, update=update, 
                                 max_age=max_age, **kwargs).values()))
        




    def staking_rewards( self, 
                     key: str = None, 
                     module_key=None,
                       block: Optional[int] = None, 
                       timeout=20,
                       period = 100, 
                       names = False,
                        fmt='j' , network=None, update=False,
                        max_age = 1000,
                         **kwargs) -> Optional['Balance']:

        block = int(block or self.block)
        block_yesterday = int(block - period)
        day_before_stake = self.my_total_stake_to(key=key, module_key=module_key, block=block_yesterday, timeout=timeout, names=names, fmt=fmt, network=network, update=update, max_age=max_age, **kwargs)
        day_after_stake = self.my_total_stake_to(key=key, module_key=module_key, block=block, timeout=timeout, names=names, fmt=fmt, network=network, update=update, max_age=max_age, **kwargs) 
        return (day_after_stake - day_before_stake)
    

    def clear_query_history(self):
        return self.rm('query')


    def my_netuid2stake( self, 
                     key: str = None, 
                     module_key=None,
                       block: Optional[int] = None, 
                       timeout=20,
                       names = False,
                        fmt='j' , network=None, update=False,
                        max_age = 1000,
                         **kwargs) -> Optional['Balance']:
        kwargs['netuid'] = 'all'
        return self.get_stake_to(key=key, module_key=module_key,  block=block, timeout=timeout, names=names, fmt=fmt, 
                                 network=network, update=update, 
                                 max_age=max_age, **kwargs)
        


    def get_stake_to( self, 
                     key: str = None, 
                     module_key=None,
                     netuid:int = 0 ,
                       block: Optional[int] = None, 
                       names = False,
                        fmt='j' , network=None, update=False,
                        max_age = 60,
                         **kwargs) -> Optional['Balance']:
        

        key_address = self.resolve_key_ss58( key )
        if netuid == 'all':
            netuid2stake_to = self.stake_to(key=key, module_key=module_key, 
                                                                block=block, 
                                                                netuid=netuid, fmt=fmt,
                                                                max_age=max_age,
                                                                  update=update, **kwargs)
            key2stake_to = {}
            for netuid, stake_to in netuid2stake_to.items():
                if key_address in stake_to:
                    key2stake_to[netuid] = {k:v for k, v in stake_to[key_address]}
            return key2stake_to
        

        netuid = self.resolve_netuid( netuid )
        stake_to = self.query( 'StakeTo', params=[netuid, key_address], block=block, update=update, network=network, max_age=max_age)
        stake_to =  {k: self.format_amount(v, fmt=fmt) for k, v in stake_to}
        if module_key != None:
            module_key = self.resolve_key_ss58( module_key )
            stake_to ={ k:v for k, v in stake_to.items()}.get(module_key, 0)
        if names:
            keys = list(stake_to.keys())
            modules = self.get_modules(keys, netuid=netuid, **kwargs)
            key2name = {m['key']: m['name'] for m in modules}

            stake_to = {key2name[k]: v for k,v in stake_to.items()}
        return stake_to
    
    
    def get_stake_total( self, 
                     key: str = None, 
                     module_key=None,
                     netuid:int = 'all' ,
                       block: Optional[int] = None, 
                       timeout=20,
                       names = False,
                        fmt='j' , network=None, update=True,
                         **kwargs) -> Optional['Balance']:
        stake_to = self.get_stake_to(key=key, module_key=module_key, netuid=netuid, block=block, timeout=timeout, names=names, fmt=fmt, network=network, update=update, **kwargs)
        if netuid == 'all':
            return sum([sum(list(x.values())) for x in stake_to])
        else:
            return sum(stake_to.values())
    
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



    def get_stake_from( self, key: str, from_key=None, block: Optional[int] = None, netuid:int = None, fmt='j', update=True  ) -> Optional['Balance']:
        key = self.resolve_key_ss58( key )
        netuid = self.resolve_netuid( netuid )
        stake_from = self.query( 'StakeFrom', params=[netuid, key], block=block,  update=update )
        state_from =  [(k, self.format_amount(v, fmt=fmt)) for k, v in stake_from ]
 
        if from_key != None:
            from_key = self.resolve_key_ss58( from_key )
            state_from ={ k:v for k, v in state_from}.get(from_key, 0)

        return state_from
    

    get_stakefrom = get_stake_from 


    ###########################
    #### Global Parameters ####
    ###########################

    @property
    def block(self, network:str=None) -> int:
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
        emissions = self.emission(block=block, update=update, network=network, netuid=netuid, **kwargs)
        if isinstance(emissions[0], list):
            emissions = [sum(e) for e in emissions]
        return sum(emissions)
    
    
    def unit_emission(self, network=None, block=None, update=False, **kwargs):
        return self.query_constant( "UnitEmission", block=block,network=network)

    def subnet_state(self,  netuid='all',  network='main', block=None, update=False, fmt='j', **kwargs):

        subnet_state = {
            'params': self.subnet_params(netuid=netuid, network=network, block=block, update=update, fmt=fmt, **kwargs),
            'modules': self.modules(netuid=netuid, network=network, block=block, update=update, fmt=fmt, **kwargs),
        }
        return subnet_state





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
    
    def my_subnet2netuid(self, key=None, block=None, update=False, **kwargs):
        address2key = c.address2key()
        subnet_params_list = self.subnet_params(block=block, update=update, netuid='all', **kwargs)
        subnet2netuid = {}
        for netuid, subnet_params in enumerate(subnet_params_list):
            if subnet_params['founder'] in address2key:
                subnet2netuid[subnet_params['name']] = netuid
        return subnet2netuid
    
    def my_subnets(self, key=None, block=None, update=False, **kwargs):
        return list(self.my_subnet2netuid(key=key, block=block, update=update, **kwargs).keys())


    @classmethod
    def feature2name(cls, feature='MinStake'):
        chunks = []
        for i, ch in enumerate(feature):
            if feature == 'SubnetNames':
                return 'name'
            if feature == 'MinStakeGlobal':
                return 'min_stake'
            if ch.isupper():
                if i == 0:
                    chunks += [ch.lower()]
                else:
                    chunks += [f'_{ch.lower()}']
            else:
                chunks += [ch]
        return ''.join(chunks)

    @classmethod
    def name2feature(cls, name='min_stake_fam'):
        chunks = name.split('_')
        return ''.join([c.capitalize() for c in chunks])


    def query_multi(self, params_batch , substrate=None, module='SubspaceModule', feature='SubnetNames', network='main'):
        substrate = substrate or self.get_substrate(network=network)

        # check if the params_batch is a list of lists
        for i,p in enumerate(params_batch):
            if isinstance(p, dict):
                p = [p.get('module', module), p.get('feature', feature), p.get('netuid', 0)]
            if len(p) == 1:
                p = [module, feature, p]
            assert len(p) == 3, f"[module, feature, netuid] should be of length 4. Got {p}"
            params_batch[i] = p
            
        assert isinstance(params_batch, list), f"params_batch should be a list of lists"
        multi_query = [substrate.create_storage_key(*p) for p in params_batch]
        results = substrate.query_multi(multi_query)
        return results

    def subnet_params(self, 
                    netuid=0,
                    network = 'main',
                    block : Optional[int] = None,
                    update = False,
                    timeout = 30,
                    max_age = 1000,
                    fmt:str='j', 
                    rows:bool = True,
                    features  = subnet_features,
                    value_features = ['min_stake', 'max_stake']
                        
                    ) -> list:  

        netuid = self.resolve_netuid(netuid)
        path = f'query/{network}/SubspaceModule.SubnetParams.{netuid}'          
        subnet_params = self.get(path, None, max_age=max_age, update=update)
        names = [self.feature2name(f) for f in features]
        name2feature = dict(zip(names, features))
        if subnet_params == None:
            subnet_params = {}
            multi_query = [("SubspaceModule", f, [0]) for f in name2feature.values()]
            subspace = self.get_substrate(network=network)
            results = self.query_multi(multi_query)
            for idx, (k, v) in enumerate(results):
                subnet_params[names[idx]] = v.value

            self.put(path, subnet_params)
        for k in value_features:
            if k in value_features:
                subnet_params[k] = self.format_amount(subnet_params[k], fmt=fmt)
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
        if not c.valid_ss58_address(key):
            key2addresss = c.key2address(netuid=netuid)
            if key in key2addresss:
                key = key2addresss[key]
        
        assert c.valid_ss58_address(key), f"Invalid key {key}"
        is_reged =  bool(self.query('Uids', block=block, params=[ netuid, key ]))
        return is_reged
    is_reg = is_registered

    def get_uid( self, key: str, netuid: int = 0, block: Optional[int] = None, update=False, **kwargs) -> int:
        return self.query( 'Uids', block=block, params=[ netuid, key ] , update=update, **kwargs)  

        

    def total_emission( self, netuid: int = 0, block: Optional[int] = None, fmt:str = 'j', **kwargs ) -> Optional[float]:
        total_emission =  sum(self.emission(netuid=netuid, block=block, **kwargs))
        return self.format_amount(total_emission, fmt=fmt)


    def regblock(self, netuid: int = 0, block: Optional[int] = None, network=network, update=False ) -> Optional[float]:
        regblock =  self.query_map('RegistrationBlock',block=block, update=update )
        if isinstance(netuid, int):
            regblock = regblock[netuid]
        return regblock

    def age(self, netuid: int = None) -> Optional[float]:
        netuid = self.resolve_netuid( netuid )
        regblock = self.regblock(netuid=netuid)
        block = self.block
        age = {}
        for k,v in regblock.items():
            age[k] = block - v
        return age
    
    

    def global_params(self, 
                    network = 'main',
                    block : Optional[int] = None,
                    update = False,
                    timeout = 30,
                    max_age = 10000,
                    fmt:str='j', 
                    rows:bool = True,
                    value_features = ['min_stake', 'min_burn', 'unit_emission', 'min_weight_stake'],
                    features  = global_features
                        
                    ) -> list:  

        path = f'query/{network}/SubspaceModule.GlobalParams'          
        subnet_params = self.get(path, None, max_age=max_age, update=update)
        names = [self.feature2name(f) for f in features]
        name2feature = dict(zip(names, features))
        if subnet_params == None:
            subnet_params = {}
            multi_query = [("SubspaceModule", f, []) for f in name2feature.values()]
            subspace = self.get_substrate(network=network)
            results = self.query_multi(multi_query)
            for idx, (k, v) in enumerate(results):

                subnet_params[names[idx]] = v.value

            self.put(path, subnet_params)
        for k in value_features:
            if k in value_features:
                subnet_params[k] = self.format_amount(subnet_params[k], fmt=fmt)
        return subnet_params


    def balance(self,
                 key: str = None ,
                 block: int = None,
                 fmt='j',
                 network=None,
                 max_age=0,
                 update=True) -> Optional['Balance']:
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
                update=update,
                max_age=max_age
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
    
    def accounts(self, key = None, network=None, update=True, block=None, max_age=100000, **kwargs):
        self.resolve_network(network)
        key = self.resolve_key_ss58(key)
        accounts = self.query_map(
            module='System',
            name='Account',
            update=update,
            block = block,
            max_age=max_age,
            **kwargs
        )
        return accounts
    
    def balances(self,fmt:str = 'n', network:str = network, block: int = None, n = None, update=False , **kwargs) -> Dict[str, 'Balance']:
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
        return self.subnet_names(**kwargs)
    
    def num_subnets(self, **kwargs) -> int:
        return len(self.subnets(**kwargs))
    
    def netuids(self, network=network, update=False, block=None) -> Dict[int, str]:
        return list(self.netuid2subnet(network=network, update=update, block=block).keys())

    def subnet_names(self, network=network , update=False, block=None, **kwargs) -> Dict[str, str]:
        records = self.query_map('SubnetNames', update=update, network=network, block=block, **kwargs)
        return list(records.values())
    
    netuid2subnet = subnet_names

    def subnet2netuid(self, subnet=None, network=network, update=False,  **kwargs ) -> Dict[str, str]:
        subnet2netuid =  {v:k for k,v in self.netuid2subnet(network=network, update=update, **kwargs).items()}
        if subnet != None:
            return subnet2netuid[subnet] if subnet in subnet2netuid else len(subnet2netuid)
        return subnet2netuid
    
    def netuid2subnet(self, netuid=None, network=network, update=False, block=None, **kwargs ) -> Dict[str, str]:
        netuid2subnet = self.query_map('SubnetNames', update=update, network=network, block=block, **kwargs)
        if netuid != None:
            return netuid2subnet[netuid]
        return netuid2subnet


    subnet_namespace = subnet2netuid

    def resolve_netuid(self, netuid: int = None, network=network, update=False) -> int:
        '''
        Resolves a netuid to a subnet name.
        '''
        if netuid == 'all':
            return netuid
        if netuid == None :
            # If the netuid is not specified, use the default.
            return 0
        
              
        if isinstance(netuid, str):
            subnet2netuid = self.subnet2netuid()
            assert netuid in subnet2netuid, f"Subnet {netuid} not found in {subnet2netuid}"
            return subnet2netuid[netuid]

        elif isinstance(netuid, int):
            if netuid == 0: 
                return netuid
            # If the netuid is an integer, ensure it is valid.
            
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

    @classmethod
    def get_feature(cls, key='names', network='main', netuid=0, update=False, max_age=1000, **kwargs):
        s = cls(network=network)
        return getattr(s, key)(netuid=netuid, update=update, max_age=max_age, **kwargs)
        
    def name2key(self, name:str=None, 
                 network=network, 
                 max_age=1000, 
                 timeout=30, 
                 netuid: int = 0, 
                 update=False, 
                 trials=3,
                 **kwargs ) -> Dict[str, str]:
        # netuid = self.resolve_netuid(netuid)
        self.resolve_network(network)
        names = c.submit(self.get_feature, args=['names'], kwargs={'netuid':netuid, 'update':update, 'max_age':max_age})
        keys = c.submit(self.get_feature, args=['keys'], kwargs={'netuid':netuid, 'update':update, 'max_age':max_age})
        names, keys = c.wait([names, keys], timeout=timeout)
        name2key = dict(zip(names, keys))
        if name != None:
            if name in name2key:
                return name2key[name]
            else:
                trials -= 1
                if trials == 0:
                    return None
                else:
                    return self.name2key(name=name, network=network, 
                                        timeout=timeout, netuid=netuid, update=True, 
                                        trials=trials, **kwargs)
                
        return name2key





    def key2name(self, key=None, netuid: int = None, network=network, update=False) -> Dict[str, str]:
        key2name =  {v:k for k,v in self.name2key(netuid=netuid, network=network, update=update).items()}
        if key != None:
            return key2name[key]
        return key2name
        
    def is_unique_name(self, name: str, netuid=None):
        return bool(name not in self.get_namespace(netuid=netuid))
    
    def epoch_time(self, netuid=0, network='main', update=False, **kwargs):
        return self.subnet_params(netuid=netuid, network=network)['tempo']*self.block_time

    def blocks_per_day(self, netuid=None, network=None):
        return 24*60*60/self.block_time
    

    def epochs_per_day(self, netuid=None, network=None):
        return 24*60*60/self.epoch_time(netuid=netuid, network=network)
    
    def emission_per_epoch(self, netuid=None, network=None):
        return self.subnet(netuid=netuid, network=network)['emission']*self.epoch_time(netuid=netuid, network=network)


    def get_block(self, network='main', block_hash=None, max_age=8): 
        network = network or 'main'
        path = f'cache/{network}.block'
        block = self.get(path, None, max_age=max_age)
        if block == None:
            self.resolve_network(network)
            block_header = self.substrate.get_block( block_hash=block_hash)['header']
            block = block_header['number']
            block_hash = block_header['hash']
            self.put(path, block)
        return block

    def block_hash(self, block = None, network='main'): 
        if block == None:
            block = self.block

        substrate = self.get_substrate(network=network)
        
        return substrate.get_block_hash(block)
    

    def seconds_per_epoch(self, netuid=None, network=None):
        self.resolve_network(network)
        netuid =self.resolve_netuid(netuid)
        return self.block_time * self.subnet(netuid=netuid)['tempo']

    
    def module_info(self, module='vali',
                    netuid=0,
                    network='main',
                    fmt='j',
                    method='subspace_getModuleInfo',
                    mode = 'http',
                    block = None,
                    lite = True, **kwargs ) -> 'ModuleInfo':
        url = self.resolve_url(network=network, mode=mode)
        module_key = module
        if not c.valid_ss58_address(module):
            module_key = self.name2key(name=module, network=network, **kwargs)
        json={'id':1, 'jsonrpc':'2.0',  'method': method, 'params': [module_key, netuid]}
        module = requests.post(url,  json=json).json()
        module = {**module['result']['stats'], **module['result']['params']}
        # convert list of u8 into a string Vector<u8> to a string
        module['name'] = self.vec82str(module['name'])
        module['address'] = self.vec82str(module['address'])
        module['dividends'] = module['dividends'] / (U16_MAX)
        module['incentive'] = module['incentive'] / (U16_MAX)
        module['stake_from'] = {k:self.format_amount(v, fmt=fmt) for k,v in module['stake_from']}
        module['stake'] = sum([v for k,v in module['stake_from'].items() ])
        module['emission'] = self.format_amount(module['emission'], fmt=fmt)
        module['key'] = module.pop('controller', None)
        module['vote_staleness'] = (block or self.block) - module['last_update']
        if lite :
            features = self.module_features + ['stake', 'vote_staleness']
            module = {f: module[f] for f in features}
        assert module['key'] == module_key, f"Key mismatch {module['key']} != {module_key}"
        return module


    minfo = get_module = module_info
    
    @staticmethod
    def vec82str(l:list):
        return ''.join([chr(x) for x in l]).strip()

    def get_modules(self, keys:list = None,
                         network='main',
                        netuid=0, 
                         timeout=20,
                         fmt='j',
                         block = None,
                         update = False,
                         batch_size = 8,
                           **kwargs) -> List['ModuleInfo']:
        netuid = self.resolve_netuid(netuid)
        block = block or self.block
        if netuid == 'all':
            futures = []
            all_keys = self.keys(update=update, netuid=netuid)
            for netuid in self.netuids():
                module = self.get_modules(keys=all_keys[netuid], netuid=netuid,   **kwargs)
                modules.append(module)
            return modules
        if keys == None:
            keys = self.keys(update=update, netuid=netuid)
        if len(keys) == 0:
            return []
        
        if len(keys) >= batch_size:
            key_batches = c.chunk(keys, chunk_size=batch_size)
            futures = []
            for key_batch in key_batches:
                f = c.submit(self.get_modules, kwargs=dict(keys=key_batch,
                                                        block=block, 
                                                        network=network, 
                                                        netuid=netuid, 
                                                        batch_size=len(keys) + 1,
                                                        timeout=timeout))
                futures += [f]
            module_batches = c.wait(futures, timeout=timeout)
            modules = c.copy([])
            name2module = {}
            for module_batch in module_batches:
                for m in module_batch:
                    if isinstance(m, dict) and 'name' in m:
                        name2module[m['name']] = m
                    
            modules = list(name2module.values())
            return modules

        progress_bar = c.tqdm(total=len(keys), desc=f'Querying {len(keys)} keys for modules')
        modules = []
        for key in keys:
            module = self.module_info(module=key, block=block, netuid=netuid, network=network, fmt=fmt, **kwargs)
            if isinstance(module, dict) and 'name' in module:
                modules.append(module)
                progress_bar.update(1)
            modules.append(module)
        
        return modules
    
        
    def my_modules(self, search=None, netuid=0, generator=False,  **kwargs):
        keys = self.my_keys(netuid=netuid, search=search)
        if netuid == 'all':
            modules = {}
            all_keys = keys 
            for netuid, keys in enumerate(all_keys):
                try:
                    modules[netuid]= self.get_modules(keys=keys, netuid=netuid, **kwargs)
                except Exception as e:
                    c.print(e)
            modules = {k: v for k,v in modules.items() if len(v) > 0 }
            return modules
        
        return self.get_modules(keys=keys, netuid=netuid, **kwargs)
    

    default_module ={
            'key': '5C5Yq15Gq8HmD6PmqEYd4VprQDnK3fp5BCwsvGfmCPDGQbjZ',
            'name': 'default_module',
            'address': '0.0.0.0:8888',
            'emission': 0,
            'incentive': 0,
            'dividends': 0,
            'last_update': 0,
            'stake_from': [],
            'delegation_fee': 20,
            'stake': 0
        }

    def format_module(self, module: 'ModuleInfo', fmt:str='j') -> 'ModuleInfo':
        for k in ['emission']:
            module[k] = self.format_amount(module[k], fmt=fmt)
        for k in ['incentive', 'dividends']:
            module[k] = module[k] / (U16_MAX)
        
        module['stake_from'] = {k: self.format_amount(v, fmt=fmt)  for k, v in module['stake_from']}
        return module
    
    def modules(self,
                search:str= None,
                network = 'main',
                netuid: int = 0,
                block: Optional[int] = None,
                fmt='nano', 
                features : List[str] = module_features,
                timeout = 100,
                max_age=1000,
                subnet = None,
                vector_features =['dividends', 'incentive', 'trust', 'last_update', 'emission'],
                **kwargs
                ) -> Dict[str, 'ModuleInfo']:
    

        name2feature = {
            'emission': 'Emission',
            'incentive': 'Incentive',
            'dividends': 'Dividends',
            'last_update': 'LastUpdate',
            'stake_from': 'StakeFrom',
            'delegation_fee': 'DelegationFee',
            'key': 'Keys',
            'name': 'Name',
            'address': 'Address',
        }

        name2default = {
            'delegation_fee': 20,
            'name': '',
            'key': '',

        }



        netuid = self.resolve_netuid(netuid or subnet)
        network = self.resolve_network(network)
        state = {}
        path = f'query/{network}/SubspaceModule.Modules:{netuid}'
        modules = self.get(path, None, max_age=max_age)
        if modules == None:

            progress = c.tqdm(total=len(features), desc=f'Querying {features}')
            future2key = {}
            def query(name, **kwargs):
                if name in vector_features:
                    fn = self.query_vector
                else:
                    fn = self.query_map
                name = name2feature.get(name, name)
                return fn(name=name, **kwargs)
            key2future = {}

            while not all([f in state for f in features ]):
                c.print(f'Querying {features}')
                for feature in features:
                    if feature in state or feature in key2future:
                        continue
                    future = c.submit(query, kwargs=dict(name=feature, netuid=netuid, block=block, max_age=max_age))
                    key2future[feature] = future
                futures = list(key2future.values())
                future2key = {v:k for k,v in key2future.items()}
                for f in c.as_completed(futures, timeout=timeout):
                    feature = future2key[f]
                    key2future.pop(feature)
                    result = f.result()
                    if c.is_error(result):
                        c.print('Failed: ', feature,  color='red')
                        continue
                    progress.update(1)
                    state[feature] = f.result()
                    break

            uid2key = state['key']
            uids = list(uid2key.keys())
            modules = []
            for uid in uids:
                module = {}
                for feature in features:
                    if uid in state[feature] or isinstance(state[feature], list):
                        module[feature] = state[feature][uid]
                    else:
                        uid_key = uid2key[uid]
                        module[feature] = state[feature].get(uid_key, name2default.get(uid_key, None))
                modules.append(module)
            self.put(path, modules)

            
        if len(modules) > 0:
            for i in range(len(modules)):
                modules[i] = self.format_module(modules[i], fmt=fmt)

        if search != None:
            modules = [m for m in modules if search in m['name']]

        return modules

    


    def min_stake(self, netuid: int = 0, network: str = 'main', fmt:str='j', **kwargs) -> int:
        min_stake = self.query('MinStake', netuid=netuid, network=network, **kwargs)
        return self.format_amount(min_stake, fmt=fmt)

    def registrations_per_block(self, network: str = network, fmt:str='j', **kwargs) -> int:
        return self.query('RegistrationsPerBlock', params=[], network=network, **kwargs)
    regsperblock = registrations_per_block
    
    def max_registrations_per_block(self, network: str = network, fmt:str='j', **kwargs) -> int:
        return self.query('MaxRegistrationsPerBlock', params=[], network=network, **kwargs)
 
    def uids(self, netuid = 0, **kwargs):
        return list(self.uid2key(netuid=netuid, **kwargs).keys())
   
    def keys(self,
             netuid = 0,
              update=False, 
              max_age=1000,
             network : str = 'main', 
             **kwargs) -> List[str]:
        keys =  self.query_map('Keys', netuid=netuid, update=update, network=network, max_age=max_age, **kwargs)
        if netuid == 'all':
            keys = [list(k.values()) for k in keys.values()]
        else:
            keys = list(keys.values())
        return keys

    def uid2key(self, uid=None, 
             netuid = 0,
              update=False, 
             network=network, 
             max_age= 1000,
             **kwargs):
        netuid = self.resolve_netuid(netuid)
        uid2key =  self.query_map('Keys',  netuid=netuid, update=update, network=network, max_age=max_age, **kwargs)
        # sort by uid
        if uid != None:
            return uid2key[uid]
        return uid2key
    

    def key2uid(self, key = None, network:str=  'main' ,netuid: int = 0, update=False, **kwargs):
        uid2key =  self.uid2key(network=network, netuid=netuid, update=update, **kwargs)
        key2uid = {v:k for k,v in uid2key.items()}
        if key == 'all':
            return key2uid
        if key != None:
            key_ss58 = self.resolve_key_ss58(key)
            return key2uid[key_ss58]
        return key2uid
        

    def uid2name(self, netuid: int = 0, update=False,  **kwargs) -> List[str]:
        netuid = self.resolve_netuid(netuid)
        names = {k: v for k,v in enumerate(self.query_map('Name', update=update,**kwargs)[netuid])}
        names = {k: names[k] for k in sorted(names)}
        return names
    
    def names(self, 
              netuid: int = 0, 
              update=False,
                **kwargs) -> List[str]:
        uid2name = self.query_map('Name', update=update, netuid=netuid,**kwargs)
        if isinstance(netuid, int):
            names = list(uid2name.values())
        else:
            for netuid, uid2name in uid2name.items():
                names[netuid] = list(netuid.values())
        return names

    def addresses(self, netuid: int = 0, update=False, **kwargs) -> List[str]:
        addresses = self.query_map('Address',netuid=netuid, update=update, **kwargs)
        
        if isinstance(netuid, int):
            addresses = list(addresses.values())
        else:
            for k,v in addresses.items():
                addresses[k] = list(v.values())
        return addresses

    def namespace(self, search=None, netuid: int = 0, update:bool = False, timeout=30, local=False, max_age=1000, **kwargs) -> Dict[str, str]:
        namespace = {}  
        results = {
            'names': None,
            'addresses': None
        }
        netuid = self.resolve_netuid(netuid)
        while any([v == None for v in results.values()]):
            future2key = {}
            for k,v in results.items():
                if v == None:
                    f =  c.submit(getattr(self, k), kwargs=dict(netuid=netuid, update=update, max_age=max_age, **kwargs))
                    future2key[f] = k
            for future in c.as_completed(list(future2key.keys()), timeout=timeout):
                key = future2key.pop(future)
                r = future.result()
                if not c.is_error(r) and r != None:
                    results[key] = r
        namespace = {k:v for k,v in zip(results['names'], results['addresses'])}

        if search != None:
            namespace = {k:v for k,v in namespace.items() if search in k}

        if local:
            ip = c.ip()
            namespace = {k:v for k,v in namespace.items() if ip in str(v)}

        return namespace

    
    def weights(self,  netuid = 0,  network = 'main', update=False, **kwargs) -> list:
        weights =  self.query_map('Weights',netuid=netuid, network = network, update=update, **kwargs)

        return weights

    def proposals(self, netuid = netuid, block=None,   network="main", nonzero:bool=False, update:bool = False,  **kwargs):
        proposals = [v for v in self.query_map('Proposals', network = 'main', block=block, update=update, **kwargs)]
        return proposals

    def save_weights(self, nonzero:bool = False, network = "main",**kwargs) -> list:
        self.query_map('Weights',network = 'main', update=True, **kwargs)
        return {'success': True, 'msg': 'Saved weights'}

    def pending_deregistrations(self, netuid = 0, update=False, **kwargs):
        pending_deregistrations = self.query_map('PendingDeregisterUids',update=update,**kwargs)[netuid]
        return pending_deregistrations
    
    def num_pending_deregistrations(self, netuid = 0, **kwargs):
        pending_deregistrations = self.pending_deregistrations(netuid=netuid, **kwargs)
        return len(pending_deregistrations)
        
    def emissions(self, netuid = 0, network = "main", block=None, update=False, **kwargs):

        return self.query_vector('Emission', network=network, netuid=netuid, block=block, update=update, **kwargs)
    
    emission = emissions
    
    def incentives(self, 
                  netuid = 0, 
                  block=None,  
                  network = "main", 
                  update:bool = False, 
                  **kwargs):
        return self.query_vector('Incentive', netuid=netuid, network=network, block=block, update=update, **kwargs)
    incentive = incentives

    def trust(self, 
                  netuid = 0, 
                  block=None,  
                  network = "main", 
                  update:bool = False, 
                  **kwargs):
        return self.query_vector('Trust', netuid=netuid, network=network, block=block, update=update, **kwargs)
    
    incentive = incentives
    
    def query_vector(self, name='Trust', netuid = 0, network="main", update=False, **kwargs):
        if isinstance(netuid, int):
            query_vector = self.query(name,  netuid=netuid, network=network, update=update, **kwargs)
        else:
            query_vector = self.query_map(name, netuid=netuid, network=network, update=update, **kwargs)
            if len(query_vector) == 0:
                query_vector = {_: [] for _ in range(len(self.netuids()))}
        return query_vector
    
    def last_update(self, netuid = 0, network='main', update=False, **kwargs):
        return self.query_vector('LastUpdate', netuid=netuid,  network=network, update=update, **kwargs)

    def dividends(self, netuid = 0, network = 'main',  update=False, **kwargs):
        return  self.query_vector('Dividends', netuid=netuid, network=network,  update=update,  **kwargs)
            

    dividend = dividends

    def registration_block(self, netuid: int = 0, update=False, **kwargs):
        registration_blocks = self.query_map('RegistrationBlock', netuid=netuid, update=update, **kwargs)
        return registration_blocks

    regblocks = registration_blocks = registration_block

    def stake_from(self, netuid = 0,
                    block=None, 
                    update=False,
                    network=network,
                    total = False,
                    fmt='nano', **kwargs) -> List[Dict[str, Union[str, int]]]:
        
        stake_from = self.query_map('StakeFrom', netuid=netuid, block=block, update=update, network=network, **kwargs)
        format_tuples = lambda x: [[_k, self.format_amount(_v, fmt=fmt)] for _k,_v in x]
        if netuid == 'all':
            stake_from = {netuid: {k: format_tuples(v) for k,v in stake_from[netuid].items()} for netuid in stake_from}
            # if total:
            #     stake = {}
            #     for netuid, subnet_stake_from in stake_from.items():
            #         for k, v in subnet_stake_from.items():
            #             stake[k] = stake.get(k, 0) + v
            #     return stake
        else:
            stake_from = {k: format_tuples(v) for k,v in stake_from.items()}

    
        return stake_from
    

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
        


    def get_nonce(self, key:str=None, network=None, **kwargs):
        key_ss58 = self.resolve_key_ss58(key)
        self.resolve_network(network)   
        return self.substrate.get_account_nonce(key_ss58)

    history_path = f'history'

    chain_path = c.libpath + '/subspace'
    spec_path = f"{chain_path}/specs"
    snapshot_path = f"{chain_path}/snapshots"

    @classmethod
    def check(cls, netuid=0):
        self = cls()
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
              features : list = ['name', 'emission','incentive', 'dividends', 'stake', 'vote_staleness', 'serving', 'address'],
              sort_features = ['emission', 'stake'],
              fmt : str = 'j',
              modules = None,
              servers = None,
              **kwargs
              ):

            
        if isinstance(netuid, str):
            netuid = self.subnet2netuid(netuid)

        if search == 'all':
            netuid = search
            search = None

        
        if netuid == 'all':
            all_modules = self.my_modules(netuid=netuid, update=update, network=network, fmt=fmt, search=search)
            servers = c.servers(network='local')
            stats = {}
            netuid2subnet = self.netuid2subnet(update=update)
            for netuid, modules in all_modules.items():
                subnet_name = netuid2subnet[netuid]
                stats[netuid] = self.stats(modules=modules, netuid=netuid, servers=servers)

                color = c.random_color()
                c.print(f'\n {subnet_name.upper()} :: (netuid:{netuid})\n', color=color)
                c.print(stats[netuid], color=color)
            

        modules = modules or self.my_modules(netuid=netuid, update=update, network=network, fmt=fmt, search=search)

        stats = []

        local_key_addresses = list(c.key2address().values())
        servers = servers or c.servers(network='local')
        for i, m in enumerate(modules):
            if m['key'] not in local_key_addresses :
                continue
            # sum the stake_from
            # we want to round these values to make them look nice
            for k in ['emission', 'dividends', 'incentive']:
                m[k] = c.round(m[k], sig=4)

            m['serving'] = bool(m['name'] in servers)
            m['stake'] = int(m['stake'])
            stats.append(m)
        df_stats =  c.df(stats)
        if len(stats) > 0:
            df_stats = df_stats[features]
            if 'emission' in features:
                epochs_per_day = self.epochs_per_day(netuid=netuid, network=network)
                df_stats['emission'] = df_stats['emission'] * epochs_per_day
            sort_features = [c for c in sort_features if c in df_stats.columns]  
            df_stats.sort_values(by=sort_features, ascending=False, inplace=True)
            if search is not None:
                df_stats = df_stats[df_stats['name'].str.contains(search, case=True)]

        if not df:
            return df_stats.to_dict('records')
        else:
            return df_stats


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
                   netuid = 'all',
                   update=False, 
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

        block = block or self.block

        path = f'state_dict/{network}.block-{block}-time-{int(c.time())}'

        feature2params = {}

        feature2params['balances'] = [self.get_feature, dict(feature='balances', update=update, block=block, timeout=timeout)]
        feature2params['subnets'] = [self.get_feature, dict(feature='subnet_params', update=update, block=block, netuid=netuid, timeout=timeout)]
        feature2params['global'] = [self.get_feature, dict(feature='global_params', update=update, block=block, timeout=timeout)]
        feature2params['modules'] = [self.get_feature, dict(feature='modules', update=update, block=block, timeout=timeout)]
    
        feature2result = {}
        state_dict = {'block': block,'block_hash': self.block_hash(block)}
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
                if c.is_error(result):
                    c.print('ERROR IN FEATURE', feature, result)
                    continue
                state_dict[feature] = result

                feature2params.pop(feature, None)
                result2feature.pop(future, None)

                # verbose 
                msg = {
                    'features_left': list(feature2params.keys()),

                }
                c.print(msg)
            
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
        
        self.get_balances(update=1)
        

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
    

    
    """
    
    WALLET VIBES
    
    """

    """
    #########################################
                    CHAIN LAND
    #########################################
    
    """
    ##################
    #### Register ####
    ##################
    def min_register_stake(self, netuid: int = 0, network: str = network, fmt='j', **kwargs) -> float:
        min_burn = self.min_burn( network=network, fmt=fmt)
        min_stake = self.min_stake(netuid=netuid, network=network, fmt=fmt)
        return min_stake + min_burn
    def register(
        self,
        name: str , # defaults to module.tage
        address : str = 'NA',
        stake : float = None,
        subnet: str = 'commune',
        netuid = 0,
        key : str  = None,
        module_key : str = None,
        network: str = network,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        module : str = None,
        nonce=None,
        tag = None,
        fmt = 'nano',
        max_age = 1000,
    **kwargs
    ) -> bool:

        if name == None:
            name = module
        if tag != None:
            name = f'{module}::{tag}'
        # resolve module name and tag if they are in the server_name
        if c.server_exists(module) and not refresh :
            address = c.get_address(module)
        else:
            serve_info =  c.serve(module, name=name, **kwargs)
            address = serve_info['address']

        network =self.resolve_network(network)
        address = address or c.namespace(network='local').get(name, '0.0.0.0:8888')
        module_key = module_key or c.get_key(name).ss58_address
        netuid2subnet = self.netuid2subnet(max_age=max_age)
        subnet2netuid = {v:k for k,v in netuid2subnet.items()}
        netuid = subnet or netuid

        if netuid in netuid2subnet:
            subnet = netuid2subnet[netuid]
        if subnet in subnet2netuid:
            netuid = subnet2netuid[subnet]
        else:
            subnet2netuid = self.subnet2netuid(max_age=0)
            if subnet in subnet2netuid:
                netuid = subnet2netuid[subnet]
            else:
                netuid = 0
                response = input(f"Do you want to create a new subnet ({subnet}) (yes or y or dope): ")
                if response.lower() not in ["yes", 'y', 'dope']:
                    return {'success': False, 'msg': 'Subnet not found and not created'}
                
            # require prompt to create new subnet        


        stake = stake or 0
        min_register_stake = self.min_register_stake(netuid=netuid, network=network)
        if stake < min_register_stake:
            stake = min_register_stake
        
        if c.key_exists(name):
            mkey = c.get_key(name)
            mkey_balance = self.get_balance(key=mkey.ss58_address, network=network)
            if mkey_balance > stake:
                c.print(f'Using {name} key to register {name} with {stake} stake')
                key = mkey
            
        stake = stake * 1e9

    

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
        **kwargs
        
    ) -> bool:
        # this is a bit of a hack to allow for the amount to be a string for c send 500 0x1234 instead of c send 0x1234 500
        if type(dest) in [int, float]:
            assert isinstance(amount, str), f"Amount must be a string"
            new_amount = int(dest)
            dest = amount
            amount = new_amount
        key = self.resolve_key(key)
        network = self.resolve_network(network)
        dest = self.resolve_key_ss58(dest)
        amount = self.to_nanos(amount) # convert to nano (10^9 nanos = 1 token)

        response = self.compose_call(
            module='Balances',
            fn='transfer',
            params={
                'dest': dest, 
                'value': amount
            },
            key=key,
            nonce = nonce,
            **kwargs
        )
        
        return response


    send = transfer

    ##################
    #### profit share ####
    ##################

    def profit_shares(self, key=None, network: str = 'main', **kwargs) -> List[Dict[str, Union[str, int]]]:
        key = self.resolve_module_key(key)

        return self.query_map('ProfitShares', network=network, **kwargs)

    def add_profit_shares(
        self,
        keys: List[str], # the keys to add profit shares to
        shares: List[float] = None , # the shares to add to the keys
        key: str = None,
        netuid : int = 0,
        network : str = 'main',
    ) -> bool:
        
        key = self.resolve_key(key)
        network = self.resolve_network(network)
        assert len(keys) > 0, f"Must provide at least one key"
        key2address = c.key2address()   
        keys = [key2address.get(k, k) for k in keys]             
        assert all([c.valid_ss58_address(k) for k in keys]), f"All keys must be valid ss58 addresses"
        shares = shares or [1 for _ in keys]

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


    def run_loop(self):
        while True:
            self.update_modules()
            self.subnet_params(netuid='all')
            self.stake_from(netuid='all')
            self.keys(netuid='all')



    def update_modules(self, search=None, 
                        timeout=60,
                         **kwargs) -> List[str]:
        namespace = c.namespace(search=search)
        my_modules = self.my_modules(search=search, **kwargs)

        self.keys()
        futures = []
        for m in my_modules:
            if m['name'] not in namespace:
                c.print(f"Module {m['name']} not found in local namespace, please deploy it ")
                continue
            name = m['name']
            address = namespace[m['name']]
            if m['address'] == address and m['name'] == name:
                c.print(f"Module {m['name']} already up to date")
                continue
            f = c.submit(c.update_module, kwargs={'module': name,
                                                    'name': name,
                                                    'address': address,
                                                  **kwargs}, timeout=timeout)
            futures+= [f]


        results = []

        for future in c.as_completed(futures, timeout=timeout):
            results += [future.result()]
            c.print(results[-1])
        return results


    def update_module(
        self,
        module: str, # the module you want to change
        address: str = None, # the address of the new module
        name: str = None, # the name of the new module
        delegation_fee: float = None, # the delegation fee of the new module
        fee : float = None, # the fee of the new module
        netuid: int = None, # the netuid of the new module
        network : str = "main", # the network of the new module
        nonce = None, # the nonce of the new module
        tip: int = 0, # the tip of the new module
    ) -> bool:
        self.resolve_network(network)
        key = self.resolve_key(module)
        netuid = self.resolve_netuid(netuid)  
        module_info = self.module_info(module)
        ip = c.ip(update=1)
        if module_info['key'] == None:
            return {'success': False, 'msg': 'not registered'}
        name = name or module_info['name']
        delegation_fee = fee or delegation_fee or module_info['delegation_fee']
        assert delegation_fee >= 0 and delegation_fee <= 100, f"Delegate fee must be between 0 and 100"


        if name != module_info['name']:
            c.print(f'Changing name from {module_info["name"]} to {name}, we need to serve the new module and swap the keys')
            c.print(c.mv_key(module_info['name'], name))
            address = c.serve(name)['address']
            
        address = address or module_info['address']
        if ip not in address:
            address = ip + ':'+ address.split(':')[-1]

        params = {
            'netuid': netuid, # defaults to module.netuid
             # PARAMS #
            'name': name, # defaults to module.tage
            'address': address, # defaults to module.tage
            'delegation_fee': delegation_fee, # defaults to module.delegate_fee
        }

        reponse  = self.compose_call('update_module',params=params, key=key, nonce=nonce, tip=tip)

        # IF SUCCESSFUL, MOVE THE KEYS, AS THIS IS A NON-REVERSIBLE OPERATION


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
        for k in ['max_stake', 'min_stake']:
            if k in params:
                params[k] = params[k] * 1e9
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
        return self.compose_call(fn='update_subnet',
                                     params=params, 
                                     key=key, 
                                     nonce=nonce)


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

    def resolve_module_key(self, x, netuid=0, max_age=10):
        if not c.valid_ss58_address(x):
            name2key = self.name2key(netuid=netuid, max_age=max_age)
            x = name2key.get(x)
        assert c.valid_ss58_address(x), f"Module key {x} is not a valid ss58 address"
        return x
                    
    
    def transfer_stake(
            self,
            module_key: str ,
            new_module_key: str ,
            amount: Union[int, float] = None, 
            key: str = None,
            netuid:int = 0,
            max_age=10,
            network:str = None,
        ) -> bool:
        # STILL UNDER DEVELOPMENT, DO NOT USE
        network = self.resolve_network(network)
        netuid = self.resolve_netuid(netuid)
        key = c.get_key(key)

        c.print(f':satellite: Staking to: [bold white]SubNetwork {netuid}[/bold white] {amount} ...')
        # Flag to indicate if we are using the wallet's own hotkey.

        module_key = self.resolve_module_key(module_key)
        new_module_key = self.resolve_module_key(new_module_key)
        assert module_key != new_module_key, f"Module key {module_key} is the same as new_module_key {new_module_key}"

        if amount == None:
            stake_to = self.get_stake_to( key=key , fmt='nanos', netuid=netuid, max_age=0)
            amount = stake_to.get(module_key, 0)
        else:
            amount = amount * 10**9

        assert amount > 0, f"Amount must be greater than 0"
                
        # Get current stake
        params={
                    'netuid': netuid,
                    'amount': int(amount),
                    'module_key': module_key,
                    'new_module_key': new_module_key

                    }

        return self.compose_call('transfer_stake',params=params, key=key)




    def stake(
            self,
            module: Optional[str] = None, # defaults to key if not provided
            amount: Union['Balance', float] = None, 
            key: str = None,  # defaults to first key
            netuid:int = None,
            network:str = None,
            existential_deposit: float = 0,
            **kwargs
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

        if c.valid_ss58_address(module):
            module_key = module
        else:
            module_key = self.name2key(netuid=netuid).get(module)

        # Flag to indicate if we are using the wallet's own hotkey.
        
        if amount == None:
            amount = self.get_balance( key.ss58_address , fmt='nano') - existential_deposit*10**9
        else:
            amount = int(self.to_nanos(amount - existential_deposit))
        assert amount > 0, f"Amount must be greater than 0 and greater than existential deposit {existential_deposit}"
        
        # Get current stake
        params={
                    'netuid': netuid,
                    'amount': amount,
                    'module_key': module_key
                    }

        return self.compose_call('add_stake',params=params, key=key)



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
    
        
        network = self.resolve_network(network)
        key = c.get_key(key)
        netuid = self.resolve_netuid(netuid)
        # get most stake from the module


        if isinstance(module, int):
            module = amount
            amount = module

        assert module != None or amount != None, f"Must provide a module or an amount"



        if c.valid_ss58_address(module):
            module_key = module
        elif isinstance(module, str):
            module_key = self.name2key(netuid=netuid).get(module)
        else: 
            raise Exception('Invalid input')

        if amount == None:
            stake_to = self.get_stake_to(netuid=netuid, names = False, fmt='nano', key=module_key)
            amount = stake_to[module_key] - 100000
        else:
            amount = int(self.to_nanos(amount))
        # convert to nanos
        params={
            'amount': amount ,
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
                        n:str = 10,
                        network: str = None) -> Optional['Balance']:
        key2address = c.key2address()
        network = self.resolve_network( network )
        key = self.resolve_key( key )
        balance = self.get_balance(key=key, fmt='j')
        for i, destination in enumerate(destinations):
            if not c.valid_ss58_address(destination):
                if destination in key2address:
                    destinations[i] = key2address[destination]
                else:
                    raise Exception(f"Invalid destination address {destination}")
        if type(amounts) in [float, int]: 
            amounts = [amounts] * len(destinations)
        assert len(set(destinations)) == len(destinations), f"Duplicate destinations found"
        assert len(destinations) == len(amounts), f"Length of modules and amounts must be the same. Got {len(modules)} and {len(amounts)}."
        assert all([c.valid_ss58_address(d) for d in destinations]), f"Invalid destination address {destinations}"
        total_amount = sum(amounts)
        assert total_amount < balance, f'The total amount is {total_amount} > {balance}'

        # convert the amounts to their interger amount (1e9)
        amounts = [self.to_nanos(a) for a in amounts]

        params = {
            "destinations": destinations,
            "amounts": amounts
        }

        return self.compose_call('transfer_multiple', params=params, key=key)

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
            stake_to = self.get_stake_to(key=key, netuid=netuid, names=False, update=True, fmt='nanos') # name to amount
            module_keys = [k for k in stake_to.keys()]
            # RESOLVE AMOUNTS
            if amounts == None:
                amounts = [stake_to[m] for m in module_keys]

        else:
            name2key = {}

            module_keys = []
            for i, module in enumerate(modules):
                if c.valid_ss58_address(module):
                    module_keys += [module]
                else:
                    if name2key == {}:
                        name2key = self.name2key(netuid=netuid, update=True)
                    assert module in name2key, f"Invalid module {module} not found in SubNetwork {netuid}"
                    module_keys += [name2key[module]]
                
            # RESOLVE AMOUNTS
            if amounts == None:
                stake_to = self.get_staketo(key=key, netuid=netuid, names=False, update=True, fmt='nanos') # name to amounts
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
                    
    def unstake2key( self,
                    modules = 'all',
                    netuid = 0,
                    network = network,
                    to = None):
        if modules == 'all':
            modules = self.my_modules()
        else:
            assert isinstance(modules, list), f"Modules must be a list of module names"
            for m in modules:
                assert m in self.my_modules_names(), f"Module {m} not found in your modules"
            modules = [m for m in self.my_modules() if m['name'] in modules or m['key'] in modules]
        c.print(f'Unstaking {len(modules)} modules')

    def unstake_all( self, 
                        key: str = 'model.openai', 
                        netuid = 0,
                        network = network,
                        to = None,
                        existential_deposit = 1) -> Optional['Balance']:
        
        network = self.resolve_network( network )
        key = self.resolve_key( key )
        key_stake_to = self.get_stake_to(key=key, netuid=netuid, names=False, update=True, fmt='nanos') # name to amount
        
        params = {
            "netuid": netuid,
            "module_keys": list(key_stake_to.keys()),
            "amounts": list(key_stake_to.values())
        }

        response = {}

        if len(key_stake_to) > 0:
            c.print(f'Unstaking all of {len(key_stake_to)} modules')
            response['stake'] = self.compose_call('remove_stake_multiple', params=params, key=key)
            total_stake = (sum(key_stake_to.values())) / 1e9
        else: 
            c.print(f'No modules found to unstake')
            total_stake = self.get_balance(key)
        total_stake = total_stake - existential_deposit
        
        return response


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
        if netuid == 'all':
            for netuid, netuid_keys in key2uid.items():
                key2uid[netuid] = {k: v for k,v in netuid_keys.items() if k in key_addresses}

        my_key2uid = { k: v for k,v in key2uid.items() if k in key_addresses}
        return my_key2uid
    
    def staked(self, 
                       search = None,
                        key = None, 
                        netuid = 0, 
                        network = 'main',
                        df = True,
                        keys = None,
                        max_age = 1000,
                        features = ['name', 'key', 'stake', 'stake_from', 'dividends', 'delegation_fee', 'vote_staleness'],
                        sort_by = 'stake_from',
                        **kwargs):
        
        key = self.resolve_key(key)
        netuid = self.resolve_netuid(netuid)

        if keys == None:
            staked_modules = self.get_stake_to(key=key, 
                                               netuid=netuid, 
                                               names=False, 
                                               network=network,
                                               max_age=max_age,
                                                 **kwargs)

            if netuid == 'all':
                staked = {}
                for netuid, netuid_staked_modules in staked_modules.items():
                    keys = list(netuid_staked_modules.keys())
                    if len(keys) == 0:
                        continue
                    c.print(f'Getting staked modules for SubNetwork {netuid} with {len(keys)} modules')
                    staked_netuid = self.staked(search=search, 
                                                key=key, 
                                                netuid=netuid, 
                                                network=network, 
                                                df=df, 
                                                keys=keys)
                    if len(staked_netuid) > 0:
                        staked[netuid] = staked_netuid
                
                return staked
            else: 
                keys = list(staked_modules.keys())
                
        block = self.block
        modules = self.get_modules(keys, block=block)

        for m in modules:          
            if isinstance(m['stake_from'], dict): 
                m['stake_from'] =  int(m['stake_from'].get(key.ss58_address, 0))
            m['stake'] = int(m['stake'])
        if search != None:
            modules = [m for m in modules if search in m['name']]


        if df:
            
            modules = [{k: v for k,v in m.items()  if k in features} for m in modules]

            if len(modules) == 0: 
                return modules
            modules = c.df(modules)

            modules = modules.sort_values(sort_by, ascending=False)
            del modules['key']
        return modules

    staked_modules = staked

    
    
    
    def my_keys(self, search=None, netuid=0, max_age=1000, update=False, **kwargs):
        netuid = self.resolve_netuid(netuid)
        keys = self.keys(netuid=netuid, max_age=max_age, update=update, **kwargs)
        key2address = c.key2address(search=search, max_age=max_age, update=update)
        if search != None:
            key2address = {k: v for k,v in key2address.items() if search in k}
        addresses = list(key2address.values())
        if netuid == 'all':
            my_keys = []
            for netuid, netuid_keys in enumerate(keys):
                my_keys += [[k for k in netuid_keys if k in addresses]]
        else:
            my_keys = [k for k in keys if k in addresses]
        return my_keys

    def set_weights(
        self,
        modules: Union['torch.LongTensor', list] = None,
        weights: Union['torch.FloatTensor', list] = None,
        uids = None,
        netuid: int = 0,
        key: 'c.key' = None,
        network = None,
        update=False,
        min_value = 0,
        max_value = 1,
        max_age = 100,
        **kwargs
    ) -> bool:
        import torch

        network = self.resolve_network(network)
        netuid = self.resolve_netuid(netuid)
        key = self.resolve_key(key)
        global_params = self.global_params( network=network)
        subnet_params = self.subnet_params( netuid = netuid )
        module_info = self.module_info(key.ss58_address, netuid=netuid)
        min_stake = global_params['min_weight_stake'] * subnet_params['min_allowed_weights']
        assert module_info['stake'] > min_stake
        max_num_votes = module_info['stake'] // global_params['min_weight_stake']
        n = int(min(max_num_votes, subnet_params['max_allowed_weights']))
        
        modules = uids or modules
        if modules == None:
            modules = c.shuffle(self.uids(netuid=netuid, update=update))
        # checking if the "uids" are passed as names -> strings
        for i, module in enumerate(modules):
            if isinstance(module, str):
                if module in key2name:
                    modules[i] = key2name[module]
                elif module in name2uid:
                    modules[i] = name2uid[module]
                    
        uids = modules
        
        if weights is None:
            weights = [1 for _ in uids]
        max_weight = max(weights)
        if len(uids) < subnet_params['min_allowed_weights']:
            n = self.n(netuid=netuid)
            while len(uids) < subnet_params['min_allowed_weights']:
                uid = c.choice(list(range(n)))
                if uid not in uids:
                    uids.append(uid)
                    weights.append(min_value)

        uid2weight = dict(sorted(zip(uids, weights), key=lambda item: item[1], reverse=True))
        uids = list(uid2weight.keys())
        weights = list(uid2weight.values())
        assert len(uids) == len(weights), f"Length of uids {len(uids)} must be equal to length of weights {len(weights)}"
        uids = torch.tensor(uids)[:n]
        weights = torch.tensor(weights)[:n]
        weights = weights / weights.sum() # normalize the weights between 0 and 1

        # STEP 2: CLAMP THE WEIGHTS BETWEEN 0 AND 1 WITH MIN AND MAX VALUES
        assert min_value >= 0 and max_value <= 1, f"min_value and max_value must be between 0 and 1"
        weights = torch.clamp(weights, min_value, max_value) # min_value and max_value are between 0 and 1

        weights = weights * (2**16 - 1)
        weights = list(map(lambda x : int(min(x, U16_MAX)), weights.tolist()))
        uids = list(map(int, uids.tolist()))

        params = {'uids': uids,
                  'weights': weights, 
                  'netuid': netuid}

        response = self.compose_call('set_weights',params = params , key=key, **kwargs)
            
        if response['success']:
            return {'success': True, 
                    'message': 'Voted', 
                    'num_uids': len(uids)}
        
        else:
            return response



    vote = set_weights



    def register_servers(self, search=None, netuid = 0, network = 'main',  timeout=42, key=None,  transfer_multiple=0, extra_amount=0,**kwargs):
        netuid = self.resolve_netuid(netuid)
        network = self.resolve_network(network)
        register_servers = self.unregistered_servers(search=search, netuid=netuid, network=network, update=True,  **kwargs)
        c.print(f'Registered servers: {register_servers}')
        min_stake = self.min_register_stake(netuid=netuid) + extra_amount
        balances = self.get_balances(keys=register_servers)
        key2balance = dict(zip(register_servers, list(balances.values())))
        destinations = []
        amounts = []
        for k,v in key2balance.items():
            if v < min_stake: 
                destinations += [k]
                amounts += [min_stake - v]
    
        if len(destinations) > 0:
            c.transfer_multiple(destinations=destinations, amounts=amounts, key=key)

        futures = []
        for s in register_servers:
            c.print(f'Registering {s}')
            f = c.submit(c.register, kwargs={'name':s, 'key':s, 'stake': min_stake + 1}, timeout=timeout)
            futures += [f]

        results = []
        for f in c.as_completed(futures,timeout=timeout):
            result = f.result()
            results += [result]
            if c.is_error(result):
                c.print(result, color='red')
            else:
                c.print(result, color='green')
        return results
        return registered_keys

    def unregistered_servers(self, search=None, netuid = 0, network = network,  timeout=42, key=None, max_age=None, update=False, transfer_multiple=True,**kwargs):
        netuid = self.resolve_netuid(netuid)
        network = self.resolve_network(network)
        servers = c.servers(search=search)
        key2address = c.key2address(update=1)
        keys = self.keys(netuid=netuid, max_age=max_age, update=update)
        uniregistered_keys = []
        unregister_servers = []
        for s in servers:
            if  key2address[s] not in keys:
                unregister_servers += [s]
        return unregister_servers

    def get_balances(self, 
                    keys=None,
                    search=None, 
                    workers = 1,
                    network = 'main',  
                    timeout=100,
                    batch_size = 128,
                    fmt = 'j',
                    n = 10000,
                    max_trials = 3,
                    names = False,
                    **kwargs):

        key2balance  = {}

        self.resolve_network(network)
        key2address = c.key2address(search=search)
        if keys == None:
            keys = list(key2address.keys())
        if len(keys) > n:
            c.print(f'Getting balances for {len(keys)} keys > {n} keys, using batch_size {batch_size}')
            balances = self.balances(network=network, **kwargs)
            key2balance = {}
            for k,a in key2address.items():
                if a in balances:
                    key2balance[k] = balances[a]
        else:
            keys = keys[:n]
            batch_size = min(batch_size, len(keys))
            batched_keys = c.chunk(keys, batch_size)
            num_batches = len(batched_keys)
            progress = c.progress(num_batches)
            futures = []
            c.print(f'Getting balances for {len(keys)} keys')

            def batch_fn(batch_keys):
                substrate = self.get_substrate(network=network)
                batch_keys = [key2address.get(k, k) for k in batch_keys]
                c.print(f'Getting balances for {len(batch_keys)} keys')
                results = substrate.query_multi([ substrate.create_storage_key("System", "Account", [k]) for k in batch_keys])
                return  {k.params[0]: v['data']['free'].value for k, v in results}
            key2balance = {}
            progress = c.progress(num_batches)


            for batch_keys in batched_keys:
                fails = 0
                while fails < max_trials:
                    if fails > max_trials:
                        raise Exception(f'Error getting balances {fails}/{max_trials}')
                    try:
                        result = batch_fn(batch_keys)
                        progress.update(1)
                        break # if successful, break
                    except Exception as e:
                        fails += 1
                        c.print(f'Error getting balances {fails}/{max_trials} {e}')
                if c.is_error(result):
                    c.print(result, color='red')
                else:
                    progress.update(1)
                    key2balance.update(result)
        for k,v in key2balance.items():
            key2balance[k] = self.format_amount(v, fmt=fmt)
        if names:
            address2key = c.address2key()
            key2balance = {address2key[k]: v for k,v in key2balance.items()}
        return key2balance
        
    def registered_servers(self, netuid = 0, network = 'main',  **kwargs):
        netuid = self.resolve_netuid(netuid)
        network = self.resolve_network(network)
        servers = c.servers(network='local')
        keys = self.keys(netuid=netuid)
        registered_keys = []
        key2address = c.key2address()
        for s in servers:
            key_address = key2address[s]
            if key_address in keys:
                registered_keys += [s]
        return registered_keys

    reged  = registered_servers

    unreged  = unregistered_servers
               
    def key2balance(self, search=None, 
                    batch_size = 64,
                    timeout = 10,
                    max_age = 1000,
                    fmt = 'j',
                    update=False,
                    names = False,
                    min_value=0.0001,
                      **kwargs):

        input_hash = c.hash(c.locals2kwargs(locals()))
        path = f'key2balance/{input_hash}'
        key2balance = self.get(path, max_age=max_age, update=update)

        if key2balance == None:
            key2balance = self.get_balances(search=search, 
                                    batch_size=batch_size, 
                                timeout=timeout, 
                                fmt = 'nanos',
                                min_value=min_value, 
                                **kwargs)
            self.put(path, key2balance)
        for k,v in key2balance.items():
            key2balance[k] = self.format_amount(v, fmt=fmt)
        key2balance = sorted(key2balance.items(), key=lambda x: x[1], reverse=True)
        key2balance = {k: v for k,v in key2balance if v > min_value}
        if names:
            address2key = c.address2key()
            key2balance = {address2key[k]: v for k,v in key2balance.items()}
        return key2balance
    
    def my_value(
                 self, *args, **kwargs
                 ):
        return sum(list(self.key2value( *args, **kwargs).values()))
    
    my_supply   = my_value

    def my_total_stake(self, netuid='all', network = 'main', fmt='j', update=False):
        my_stake_to = self.my_stake_to(netuid=netuid, network=network, fmt=fmt, update=update)
        return sum([sum(list(v.values())) for k,v in my_stake_to.items()])

    def check_valis(self, **kwargs):
        return self.check_servers(search='vali', **kwargs)
    
    def check_servers(self, search='vali',update:bool=False, netuid=0, min_lag=100, remote=False, **kwargs):
        if remote:
            kwargs = c.locals2kwargs(locals())
            return self.remote_fn('check_servers', kwargs=kwargs)
        module_stats = self.stats(search=search, netuid=netuid, df=False, update=update)
        module2stats = {m['name']:m for m in module_stats}
        block = self.block
        response_batch = {}
        c.print(f"Checking {len(module2stats)} {search} servers")
        for module, stats in module2stats.items():
            # check if the module is serving
            lag = stats['vote_staleness']
            port = int(stats['address'].split(':')[-1])
            if not c.server_exists(module) or lag > min_lag:
                c.print(f"Server {module} is not serving or has a lag of {lag} > {min_lag}")
                response_batch[module]  = c.submit(c.serve, kwargs=dict(module=module, network=f'subspace.{netuid}', port=port))

        futures = list(response_batch.values())
        future2key = {f: k for k,f in response_batch.items()}
        for f in c.as_completed(futures):
            key = future2key[f]
            c.print(f.result())
            response_batch[key] = f.result()
        return response_batch


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
                    sudo:bool  = False,
                    nonce: int = None,
                    remote_module: str = None,
                    unchecked_weight: bool = False,
                    network = network,
                    mode='ws',
                    max_tip = 10000,
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
        params = {k: int(v) if type(v) in [float]  else v for k,v in params.items()}
        compose_kwargs = dict(
                call_module=module,
                call_function=fn,
                call_params=params,
        )
        c.print(f'Sending 📡 using 🔑(ss58={key.ss58_address}, name={key.path})🔑', compose_kwargs,color=color)
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
        if tip < max_tip:
            tip = tip * 1e9
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
            address2key = c.address2key()
            key2address = {v:k for k,v in address2key.items()}
            if key in address2key:
                key = address2key[key]
            assert key in key2address, f"Key {key} not found in your keys, please make sure you have it"
            if key == None:
                raise ValueError(f"Key {key} not found in your keys, please make sure you have it")
            key = c.get_key(key)

        assert hasattr(key, 'ss58_address'), f"Invalid Key {key} as it should have ss58_address attribute."
        return key
    
    
    def unstake2key(self, key=None):
        key2stake = self.key2stake()
        c.print(key2stake)


    def test_subnet_storage(self):

        all_subnet_params = self.subnet_params(netuid='all')
        assert isinstance(all_subnet_params, list)
        for subnet_params in all_subnet_params: 
            assert isinstance(subnet_params, dict)
        subnet_params = self.subnet_params(netuid=10)
        assert isinstance(subnet_params, dict)
        return {'success': True, 'msg': 'All subnet params are dictionaries', 'n': len(all_subnet_params)}
    
    def test_global_storage(self):
        global_params = self.global_params(fmt='j')
        assert isinstance(global_params, dict)
        return global_params
    
    def test_module_storage(self):
        modules = self.get_modules(netuid=0)
        return modules 

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
        assert isinstance(stats, list) 


    


Subspace.run(__name__)
