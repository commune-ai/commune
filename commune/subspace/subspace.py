
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


    block_time = 8 # (seconds)
    default_config = c.get_config('subspace', to_munch=False)
    token_decimals = 9
    network = default_config['network']
    libpath = chain_path = c.libpath + '/subspace'
    netuid = 0
    local = default_config['local']

    def __init__( 
        self, 
        **kwargs,
    ):
        self.set_config(kwargs=kwargs)


    def resolve_url(self, url = None, mode='http', **kwargs):
        
        network = self.resolve_network()
        def is_match(x):
            url_search_terms = [x.strip() for x in self.config.url_search.split(',')]
            return any([url in x for url in url_search_terms])
        mode =  mode or self.config.network_mode
        assert mode in ['http', 'ws', 'wss', 'https']
        if url == None:
            urls_map = getattr(self.config.urls,  network)
            urls = []
            for provider, mode2url in urls_map.items():
                if is_match(provider):
                    urls += list(mode2url[mode])
            url = c.choice(urls)
        return url
    

    _substrate = None
    @property
    def substrate(self):
        if self._substrate == None:
            self.set_network()
        return self._substrate
    
    @substrate.setter
    def substrate(self, value):
        self._substrate = value
    
    url2substrate = {}
    def get_substrate(self, 
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
                mode = 'http'):

        
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
        while trials > 0:
            try:
                url = self.resolve_url(url, mode=mode)
                if cache:
                    if url in self.url2substrate:
                        substrate = self.url2substrate[url]
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
                c.print('ERROR IN CONNECTION: ', c.detailed_error(e))
                trials = trials - 1
                if trials > 0:
                    raise e
                
        self.url = url
        if cache:
            self.url2substrate[url] = substrate

        return substrate


    def set_network(self, 
                network:str = None,
                mode = 'http',
                trials = 10,
                url : str = None, 
                **kwargs):
        self.network = self.resolve_network(network)
        self.substrate = self.get_substrate( url=url, mode=mode, trials=trials , **kwargs)
        response =  {'network': self.network, 'url': self.url}

        return response
    @property
    def network(self):
        return self.resolve_network(self.config.network)

    @network.setter
    def network(self, value):
        self.config.network = value

    def __repr__(self) -> str:
        return f'<Subspace: network={self.config.network}>'
    def __str__(self) -> str:

        return f'<Subspace: network={self.config.network}>'


    

    def my_staked_module_keys(self, netuid = 0, **kwargs):
        my_stake_to = self.my_stake_to(netuid=netuid, **kwargs)
        module_keys = {} if netuid == 'all' else []
        for subnet_netuid, stake_to_key in my_stake_to.items():
            if netuid == 'all':
                for _netuid, stake_to_subnet in stake_to_key.items():
                    module_keys[_netuid] = list(stake_to_subnet.keys()) + module_keys.get(_netuid, [])
            else:
                module_keys += list(stake_to_key.keys())
        return module_keys

    def my_stake_to(self, netuid = 0, **kwargs):
        stake_to = self.stake_to(netuid=netuid, **kwargs)
        key2address = c.key2address()
        my_stake_to = {}

        for key, address in key2address.items():
            if netuid == 'all':
                my_stake_to[address] = my_stake_to.get(address, {})
                for _netuid, stake_to_subnet in stake_to.items():
                    if address in stake_to_subnet:
                        my_stake_to[address][_netuid] = {k:v  for k,v in stake_to_subnet.get(address, [])}
                        if my_stake_to[address][_netuid] == 0:
                            del my_stake_to[address][_netuid]
            else:
                my_stake_to[address] = {k:v  for k,v in stake_to.get(address, [])}

        stake_to_keys = list(my_stake_to.keys())
        for key in stake_to_keys:
            if len(my_stake_to[key]) == 0:
                del my_stake_to[key]

        return my_stake_to
    

    def staked_module_keys(self, netuid = 0, **kwargs):
        stake_to = self.stake_to(netuid=netuid, **kwargs)
        module_keys = []
        for key, stake_to_key in stake_to.items():
            module_keys += list(stake_to_key.keys())
        return module_keys

    def my_stake_from(self, netuid = 0, block=None, update=False,  fmt='j', max_age=1000 , **kwargs):
        stake_from_tuples = self.stake_from(netuid=netuid,
                                             block=block,
                                               update=update, 
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

    
    def delegation_fee(self, netuid = 0, block=None, update=False, fmt='j'):
        delegation_fee = self.query_map('DelegationFee', netuid=netuid, block=block ,update=update)
        return delegation_fee

    def stake_to(self, netuid = 0,block=None,  max_age=1000, update=False, fmt='nano',**kwargs):
        stake_to = self.query_map('StakeTo', netuid=netuid, block=block, max_age=max_age, update=update,  **kwargs)
        format_tuples = lambda x: [[_k, self.format_amount(_v, fmt=fmt)] for _k,_v in x]
        if netuid == 'all':
            stake_to = {netuid: {k: format_tuples(v) for k,v in stake_to[netuid].items()} for netuid in stake_to}
        else:
            stake_to = {k: format_tuples(v) for k,v in stake_to.items()}
    
        return stake_to
    
    
    def key2stake(self, netuid = 0,
                     block=None, 
                    update=False, 
                    names = True,
                    max_age = 1000,fmt='j'):
        stake_to = self.stake_to(netuid=netuid, 
                                block=block, 
                                max_age=max_age,
                                update=update, 
                                 
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
        if names:
            stake_to_total = {address2key.get(k, k): v for k,v in stake_to_total.items()}
        return stake_to_total


    def empty_keys(self,  block=None, update=False, max_age=1000, fmt='j'):
        key2address = c.key2address()
        key2value = self.key2value( block=block, update=update, max_age=max_age, fmt=fmt)
        empty_keys = []
        for key,key_address in key2address.items():
            key_value = key2value.get(key_address, 0)
            if key_value == 0:
                empty_keys.append(key)
               
        return empty_keys

    def key2value(self, netuid = 'all', block=None, update=False, max_age=1000, fmt='j', min_value=0, **kwargs):
        key2balance = self.key2balance(block=block, update=update,  max_age=max_age, fmt=fmt)
        key2stake = self.key2stake(netuid=netuid, block=block, update=update,  max_age=max_age, fmt=fmt)
        key2value = {}
        keys = set(list(key2balance.keys()) + list(key2stake.keys()))
        for key in keys:
            key2value[key] = key2balance.get(key, 0) + key2stake.get(key, 0)
        key2value = {k:v for k,v in key2value.items()}
        key2value = dict(sorted(key2value.items(), key=lambda x: x[1], reverse=True))
        return key2value
    
    def min_burn(self,  block=None, update=False, fmt='j'):
        min_burn = self.query('MinBurn', block=block, update=update)
        return self.format_amount(min_burn, fmt=fmt)



    def query(self, 
              name:str,  
              params = None, 
              module:str='SubspaceModule',
              block=None,  
              netuid = None,
              save= True,
              max_age=1000,
              trials = 4,
              mode = 'ws',
              feature = None,
            update=False):
        
        """
        query a subspace storage function with params and block.
        """
        name = feature or name # feature is an alias for name

        path = f'query/{self.config.network}/{module}.{name}'
    
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
        
        while trials > 0:
            try:
                substrate = self.get_substrate( mode=mode)
                response =  substrate.query(
                    module=module,
                    storage_function = name,
                    block_hash = None if block == None else substrate.get_block_hash(block), 
                    params = params
                )
                value =  response.value
                break
            except Exception as e:
                trials = trials - 1
                if trials == 0:
                    raise e
        
        # if the value is a tuple then we want to convert it to a list
        if save:
            self.put(path, value)

        return value

    def query_constant( self, 
                        constant_name: str, 
                       module_name: str = 'SubspaceModule', 
                       block: Optional[int] = None ) -> Optional[object]:
        """ 
        Gets a constant from subspace with
        module_name, constant_name, and block. 
        """

        substrate = self.get_substrate()

        value =  substrate.query(
            module=module_name,
            storage_function=constant_name,
            block_hash = None if block == None else substrate.get_block_hash(block)
        )
            
        return value
    
    

    def resolve_storage_name(self, name):
        if name[0].islower():
            _splits = name.split('_')
            name = _splits[0].capitalize() + ''.join([s[0].capitalize() + s[1:] for s in _splits[1:]])
        return name

    def query_map(self, name: str = 'StakeFrom', 
                  params: list = None,
                  block: Optional[int] = None, 
                  network:str = 'main',
                  netuid = None,
                  page_size=1000,
                  max_results=100000,
                  module='SubspaceModule',
                  update: bool = False,
                  max_age : str = 1000, # max age in seconds
                  mode = 'ws',
                  trials = 4,
                  **kwargs
                  ) -> Optional[object]:
        """ Queries subspace map storage with params and block. """
        # if all lowercase then we want to capitalize the first letter

        if name  == 'Account':
            module = 'System'
        path = f'query/{self.config.network}/{module}.{name}'
        # resolving the params
        params = params or []
        is_single_subnet = bool(netuid != 'all' and netuid != None)
        if is_single_subnet:
            params = [netuid] + params
        if not isinstance(params, list):
            params = [params]
        if len(params) > 0 :
            path = path + f'::params::' + '-'.join([str(p) for p in params])

        value = self.get(path, None , max_age=max_age, update=update)

        if value == None:
            # block = block or self.block
            path = path + f'{block}'
            # if the value is a tuple then we want to convert it to a list
    
            while trials > 0:
                try:
                    substrate = self.get_substrate( mode=mode)
                    qmap =  substrate.query_map(
                        module=module,
                        storage_function = name,
                        params = params,
                        page_size = page_size,
                        max_results = max_results,
                        block_hash =substrate.get_block_hash(block)
                    )
                    break
                except Exception as e:
                    trials = trials - 1
                    if trials == 0:
                        raise e
                    
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
        c.print(self.substrate.runtime_config.__dict__)
        runtime_version = self.query_constant(module_name='System', constant_name='SpVersionRuntimeVersion')
        return runtime_version
        
        
    #####################################
    #### Hyper parameter calls. ####
    #####################################

    """ Returns network SubnetN hyper parameter """
    def n(self,  netuid: int = 0,block: Optional[int] = None, max_age=100, update=False, **kwargs ) -> int:
        if netuid == 'all':
            return sum(self.query_map('N', block=block , update=update, max_age=max_age,  **kwargs).values())
        else:
            return self.query( 'N', params=[netuid], block=block , update=update,  **kwargs)

    ##########################
    #### Account functions ###
    
    """ Returns network Tempo hyper parameter """
    def stakes(self, netuid: int = 0, fmt:str='nano', max_age = 100, update=False, **kwargs) -> int:
        stakes =  self.query_map('Stake', netuid=netuid, update=update, max_age=max_age, **kwargs)
        if netuid == 'all':
            subnet2stakes = c.copy(stakes)
            stakes = {}
            for netuid, subnet_stakes in subnet2stakes.items():
                for k,v in subnet_stakes.items():
                    stakes[k] = stakes.get(k, 0) + v
        
        return {k: self.format_amount(v, fmt=fmt) for k,v in stakes.items()}

    """ Returns the stake under a coldkey - hotkey pairing """
    
    def resolve_key_ss58(self, key:str,netuid:int=0, resolve_name=True, **kwargs):
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
                    name2key = self.name2key( netuid=netuid)

                    if key in name2key:
                        key_address = name2key[key]
                    else:
                        key_address = key 
        # if the key has an attribute then its a key
        elif hasattr(key, 'key'):
            key_address = key.ss58_address
        
        return key_address

    def subnet2modules(self, **kwargs):
        subnet2modules = {}

        for netuid in self.netuids():
            c.print(f'Getting modules for SubNetwork {netuid}')
            subnet2modules[netuid] = self.my_modules(netuid=netuid, **kwargs)

        return subnet2modules
    
    def module2netuids(self, network:str='main', **kwargs):
        subnet2modules = self.subnet2modules( **kwargs)
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


    def all_key_info(self, netuid='all', timeout=10, update=False, **kwargs):
        my_keys = c.my_keys()


    def key_info(self, key:str = None, netuid='all', detail=0, timeout=10, update=False, **kwargs):
        key_info = {
            'balance': c.get_balance(key=key, **kwargs),
            'stake_to': c.get_stake_to(key=key, netuid=netuid, **kwargs),
        }
        if detail: 
            pass
        else: 
            for netuid, stake_to in key_info['stake_to'].items():
                key_info['stake_to'][netuid] = sum(stake_to.values())


        return key_info

    def my_total_stake_to( self, 
                     key: str = None, 
                     module_key=None,
                       block: Optional[int] = None, 
                       timeout=20,
                       names = False,
                        fmt='j' ,
                          update=False,
                        max_age = 1000,
                         **kwargs) -> Optional['Balance']:
        kwargs['netuid'] = 'all'
        return sum(list(self.my_netuid2stake(key=key, module_key=module_key,
                                              block=block, timeout=timeout, names=names, fmt=fmt, 
                                  update=update, 
                                 max_age=max_age, **kwargs).values()))
        




    def staking_rewards( self, 
                     key: str = None, 
                     module_key=None,
                       block: Optional[int] = None, 
                       timeout=20,
                       period = 100, 
                       names = False,
                        fmt='j' , update=False,
                        max_age = 1000,
                         **kwargs) -> Optional['Balance']:

        block = int(block or self.block)
        block_yesterday = int(block - period)
        day_before_stake = self.my_total_stake_to(key=key, module_key=module_key, block=block_yesterday, timeout=timeout, names=names, fmt=fmt,  update=update, max_age=max_age, **kwargs)
        day_after_stake = self.my_total_stake_to(key=key, module_key=module_key, block=block, timeout=timeout, names=names, fmt=fmt,  update=update, max_age=max_age, **kwargs) 
        return (day_after_stake - day_before_stake)
    

    def clear_query_history(self):
        return self.rm('query')


    def my_netuid2stake( self, 
                     key: str = None, 
                     module_key=None,
                       block: Optional[int] = None, 
                       timeout=20,
                       names = False,
                        fmt='j' , update=False,
                        max_age = 1000,
                         **kwargs) -> Optional['Balance']:
        kwargs['netuid'] = 'all'
        return self.get_stake_to(key=key, module_key=module_key,  block=block, timeout=timeout, names=names, fmt=fmt, 
                                  update=update, 
                                 max_age=max_age, **kwargs)
        


    def get_stake_to( self, 
                     key: str = None, 
                     module_key=None,
                     netuid:int = 0 ,
                       block: Optional[int] = None, 
                       names = False,
                        fmt='j' , update=False,
                        max_age = 60,
                        timeout = 10,
                         **kwargs) -> Optional['Balance']:
        

        if netuid == 'all':
            future2netuid = {}
            key2stake_to = {}
            for netuid in self.netuids():
                future = c.submit(self.get_stake_to, kwargs=dict(key=key, module_key=module_key, netuid=netuid, block=block, names=names, fmt=fmt,  update=update, max_age=max_age, **kwargs), timeout=timeout)
                future2netuid[future] = netuid
            try:
                for f in c.as_completed(future2netuid, timeout=timeout):
                    netuid = future2netuid[f]
                    result = f.result()
                    if len(result) > 0:
                        key2stake_to[netuid] = result
            except Exception as e:
                c.print(e)
                c.print('Error getting stake to')
            sorted_key2stake_to = {k: key2stake_to[k] for k in sorted(key2stake_to.keys())}
            return sorted_key2stake_to
        
        key_address = self.resolve_key_ss58( key )

        netuid = self.resolve_netuid( netuid )
        stake_to = self.query( 'StakeTo', params=[netuid, key_address], block=block, update=update,  max_age=max_age)
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
                        fmt='j' , update=True,
                         **kwargs) -> Optional['Balance']:
        stake_to = self.get_stake_to(key=key, module_key=module_key, netuid=netuid, block=block, timeout=timeout, names=names, fmt=fmt,  update=update, **kwargs)
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
    def block(self) -> int:
        return self.get_block()


    


    def subnet_exists(self, subnet:str) -> bool:
        subnets = self.subnets()
        return bool(subnet in subnets)

    def subnet_emission(self, netuid:str = 0, block=None, update=False, **kwargs):
        emissions = self.emission(block=block, update=update,  netuid=netuid, **kwargs)
        if isinstance(emissions[0], list):
            emissions = [sum(e) for e in emissions]
        return sum(emissions)
    
    
    def unit_emission(self, block=None, **kwargs):
        return self.query_constant( "UnitEmission", block=block)

    def subnet_state(self,  netuid='all', block=None, update=False, fmt='j', **kwargs):

        subnet_state = {
            'params': self.subnet_params(netuid=netuid,  block=block, update=update, fmt=fmt, **kwargs),
            'modules': self.modules(netuid=netuid,  block=block, update=update, fmt=fmt, **kwargs),
        }
        return subnet_state


    def subnet2stakes(self,  block=None, update=False, fmt='j', **kwargs):
        subnet2stakes = {}
        for netuid in self.netuids( update=update):
            subnet2stakes[netuid] = self.stakes(netuid=netuid,  block=block, update=update, fmt=fmt, **kwargs)
        return subnet2stakes


    def total_stake(self,  block: Optional[int] = None, netuid:int='all', fmt='j', update=False) -> 'Balance':
        return sum([sum([sum(list(map(lambda x:x[1], v))) for v in vv.values()]) for vv in self.stake_to( block=block,update=update, netuid='all')])

    def subnet2stake(self, fmt='j'):
        netuid2subnet = self.netuid2subnet()
        netuid2stake = self.netuid2stake(fmt=fmt)
        subnet2stake = {}
        for netuid, subnet in netuid2subnet.items():
            subnet2stake[subnet] = netuid2stake[netuid]
        return subnet2stake
        

    def netuid2stake(self, fmt='j',  **kwargs):
        netuid2stake = self.query_map('TotalStake',  **kwargs)
        for netuid, stake in netuid2stake.items():
            netuid2stake[netuid] = self.format_amount(stake, fmt=fmt)

        return netuid2stake

    def netuid2n(self, fmt='j',  **kwargs):
        netuid2n = self.query_map('N',  **kwargs)
        return netuid2n
    
    def subnet2n(self, fmt='j',  **kwargs):
        netuid2n = self.netuid2n(fmt=fmt, **kwargs)
        netuid2subnet = self.netuid2subnet()
        subnet2n = {}
        for netuid, subnet in netuid2subnet.items():
            subnet2n[subnet] = netuid2n[netuid]
        return subnet2n
    

    def netuid2emission(self, fmt='j',  **kwargs):
        netuid2emission = self.query_map('SubnetEmission',  **kwargs)
        for netuid, emission in netuid2emission.items():
            netuid2emission[netuid] = self.format_amount(emission, fmt=fmt)
        netuid2emission = dict(sorted(netuid2emission.items(), key=lambda x: x[1], reverse=True))

        return netuid2emission
    
    def subnet2emission(self, fmt='j',  **kwargs):
        netuid2emission = self.netuid2emission(fmt=fmt, **kwargs)
        netuid2subnet = self.netuid2subnet()
        subnet2emission = {}
        for netuid, subnet in netuid2subnet.items():
            subnet2emission[subnet] = netuid2emission[netuid]
        # sort by emission
        subnet2emission = dict(sorted(subnet2emission.items(), key=lambda x: x[1], reverse=True))
       

        return subnet2emission


    def subnet2state(self, fmt='j',  **kwargs):
        netuid2n = self.netuid2n(fmt=fmt, **kwargs)
        netuid2stake = self.netuid2stake(fmt=fmt, **kwargs)
        netuid2emission = self.netuid2emission(fmt=fmt, **kwargs)
        for netuid, state in netuid2state.items():
            netuid2state[netuid] = self.format_amount(state, fmt=fmt)
        return netuid2state




    def total_balance(self,  block: Optional[int] = None, fmt='j', update=False) -> 'Balance':
        return sum(list(self.balances( block=block, fmt=fmt).values()), update=update)

    def mcap(self,  block: Optional[int] = None, fmt='j', update=False) -> 'Balance':
        total_balance = self.total_balance( block=block, update=update)
        total_stake = self.total_stake( block=block, update=update)
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
        subnet_params = self.subnet_params(block=block, update=update, netuid='all', **kwargs)
        subnet2netuid = {}
        for netuid, subnet_params in subnet_params.items():
            if subnet_params['founder'] in address2key:
                subnet2netuid[subnet_params['name']] = netuid
        return subnet2netuid
    
    def my_subnets(self, key=None, update=True, **kwargs):
        return list(self.my_subnet2netuid(key=key,  update=update, **kwargs).keys())


    @classmethod
    def feature2name(cls, feature='MinStake'):
        chunks = []
        for i, ch in enumerate(feature):
            if ch.isupper():
                if i == 0:
                    chunks += [ch.lower()]
                else:
                    chunks += [f'_{ch.lower()}']
            else:
                chunks += [ch]

        name =  ''.join(chunks)
        if name == 'vote_mode_subnet':
            name =  'vote_mode'
        elif name == 'subnet_names':
            name  = 'name'
            
        return name
        

        

    @classmethod
    def name2feature(cls, name='min_stake_fam'):
        chunks = name.split('_')
        return ''.join([c.capitalize() for c in chunks])


    def query_multi(self, params_batch , 
                    substrate=None, 
                    module='SubspaceModule', 
                    feature='SubnetNames', 
                    trials = 6):
        # check if the params_batch is a list of lists
        for i,p in enumerate(params_batch):
            if isinstance(p, dict):
                p = [p.get('module', module), p.get('feature', feature), p.get('netuid', 0)]
            if len(p) == 1:
                p = [module, feature, p]
            assert len(p) == 3, f"[module, feature, netuid] should be of length 4. Got {p}"
            params_batch[i] = p
            
        assert isinstance(params_batch, list), f"params_batch should be a list of lists"
        while True:
            substrate = substrate or self.get_substrate()
            try:
                multi_query = [substrate.create_storage_key(*p) for p in params_batch]
                results = substrate.query_multi(multi_query)
                break
            except Exception as e:
                trials -= 1 
                if trials == 0: 
                    raise e
        return results

    def blocks_until_vote(self, netuid=0, **kwargs):
        netuid = self.resolve_netuid(netuid)
        tempo = self.subnet_params(netuid=netuid, **kwargs)['tempo']
        block = self.block
        return tempo - ((block + netuid) % tempo)
    
    def subnet_params(self, 
                    netuid=0,
                    update = False,
                    max_age = 1000,
                    timeout=40,
                    fmt:str='j', 
                    features  = None,
                    value_features = [],
                    **kwargs
                    ) -> list:  
        
        features = features or self.config.subnet_features
        netuid = self.resolve_netuid(netuid)
        path = f'query/{self.network}/SubspaceModule.SubnetParams.{netuid}'          
        subnet_params = self.get(path, None, max_age=max_age, update=update)
        names = [self.feature2name(f) for f in features]
        future2name = {}
        name2feature = dict(zip(names, features))
        for name, feature in name2feature.items():
            if netuid == 'all':
                query_kwargs = dict(name=feature, block=None, max_age=max_age, update=update)
                fn = c.query_map
            else:
                query_kwargs = dict(name=feature, 
                                    netuid=netuid,
                                     block=None, 
                                     max_age=max_age, 
                                     update=update)
                fn = c.query
            f = c.submit(fn, kwargs=query_kwargs, timeout=timeout)
            future2name[f] = name
        
        subnet_params = {}

        for f in c.as_completed(future2name, timeout=timeout):
            result = f.result()
            c.print(result)
            subnet_params[future2name.pop(f)] = result
        for k in value_features:
            subnet_params[k] = self.format_amount(subnet_params[k], fmt=fmt)

        if netuid == 'all':
            subnet_params_keys = list(subnet_params.keys())
            for k in subnet_params_keys:
                netuid2value = subnet_params.pop(k)
                for netuid, value in netuid2value.items():
                    if netuid not in subnet_params:
                        subnet_params[netuid] = {}
                    subnet_params[netuid][k] = value
        return subnet_params




    
    def global_params(self, 
                    update = False,
                    max_age = 1000,
                    timeout=30,
                    fmt:str='j', 
                    features  = None,
                    value_features = [],
                    path = f'global_params',
                    **kwargs
                    ) -> list:  
        
        features = features or self.config.global_features
        subnet_params = self.get(path, None, max_age=max_age, update=update)
        names = [self.feature2name(f) for f in features]
        future2name = {}
        name2feature = dict(zip(names, features))
        for name, feature in name2feature.items():
            c.print(f'Getting {name} for {feature}')
            query_kwargs = dict(name=feature, params=[], block=None, max_age=max_age, update=update)
            f = c.submit(self.query, kwargs=query_kwargs, timeout=timeout)
            future2name[f] = name
        
        subnet_params = {}

        for f in c.as_completed(future2name):
            result = f.result()
            subnet_params[future2name.pop(f)] = result
        for k in value_features:
            subnet_params[k] = self.format_amount(subnet_params[k], fmt=fmt)
        return subnet_params



    subnet = subnet_params





    def subnet2params( self,  block: Optional[int] = None ) -> Optional[float]:
        netuids = self.netuids()
        subnet2params = {}
        netuid2subnet = self.netuid2subnet()
        for netuid in netuids:
            subnet = netuid2subnet[netuid]
            subnet2params[subnet] = self.subnet_params(netuid=netuid, block=block)
        return subnet2params
    
    

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


    def regblock(self, netuid: int = 0, block: Optional[int] = None,  update=False ) -> Optional[float]:
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
    
    

    def balance(self,
                 key: str = None ,
                 block: int = None,
                 fmt='j',
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

        result = self.query(
                module='System',
                name='Account',
                params=[key_ss58],
                block = block,
                
                update=update,
                max_age=max_age
            )

        return  self.format_amount(result['data']['free'] , fmt=fmt)
        
    get_balance = balance 

    def get_account(self, key = None,  update=True):
        key = self.resolve_key_ss58(key)
        account = self.substrate.query(
            module='System',
            storage_function='Account',
            params=[key],
        )
        return account
    
    def accounts(self, key = None, update=True, block=None, max_age=100000, **kwargs):
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
        accounts = self.accounts( update=update, block=block)
        balances =  {k:v['data']['free'] for k,v in accounts.items()}
        balances = {k: self.format_amount(v, fmt=fmt) for k,v in balances.items()}
        return balances
    
    
    def resolve_network(self, 
                        network: Optional[int] = None,
                        spliters: List[str] = [ '::', ':'], 
                        **kwargs) -> int:
        """
        Resolve the network to use for the current session.
        
        """
        network = network or self.config.network

        for spliter in spliters:
            if spliter in str(network):
                network = network.split(spliter)[-1]
                break
        if network == 'subspace':
            network = 'main'
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
    

    def subnet_names(self , search=None, update=False, block=None, max_age=60, **kwargs) -> Dict[str, str]:
        records = self.query_map('SubnetNames', update=update,  block=block, max_age=max_age, **kwargs)
        subnet_names = sorted(list(map(lambda x: str(x), records.values())))
        if search != None:
            subnet_names = [s for s in subnet_names if search in s]
        return subnet_names
    
    
    def netuid2subnet(self, netuid=None,  update=False, block=None, **kwargs ) -> Dict[str, str]:
        netuid2subnet = self.query_map('SubnetNames', update=update,  block=block, **kwargs)
        if netuid != None:
            return netuid2subnet[netuid]
        return netuid2subnet


    def subnet2netuid(self, subnet=None,  update=False,  **kwargs ) -> Dict[str, str]:
        subnet2netuid =  {v:k for k,v in self.netuid2subnet( update=update, **kwargs).items()}
        # sort by subnet 
        subnet2netuid = {k:v for k,v in sorted(subnet2netuid.items(), key=lambda x: x[0].lower())}
        if subnet != None:
            return subnet2netuid[subnet] if subnet in subnet2netuid else len(subnet2netuid)
        return subnet2netuid


    def netuids(self,  update=False, block=None) -> Dict[int, str]:
        return list(self.netuid2subnet( update=update, block=block).keys())


    subnet_namespace = subnet2netuid

    def resolve_netuid(self, netuid: int = None) -> int:
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
            if netuid not in subnet2netuid: # if still not found, try lower case
                subnet2netuid =self.subnet2netuid(update=True)
            if netuid not in subnet2netuid: # if still not found, try lower case
                subnet2netuid = {k.lower():v for k,v in subnet2netuid.items()}
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
    @staticmethod
    def search_dict(x, search=None):
        if search != None:
            x = {k:v for k,v in x.items() if search in k}
        return x
              
    def name2uid(self, name = None, netuid: int = 0, search=None, network: str = 'main') -> int:
        netuid = self.resolve_netuid(netuid)
        uid2name = self.uid2name(netuid=netuid)

        if netuid == 'all':
            netuid2name2uid = {}
            for netuid, netuid_uid2name in uid2name.items():
                name2uid = self.search_dict(netuid_uid2name)
                if name != None:
                    name2uid = name2uid[name] 
                netuid2name2uid[netuid] = name2uid
            return netuid2name2uid
            
        else:
            name2uid =  self. search_dict({v:k for k,v in uid2name.items()}, search=search)
            if name != None:
                return name2uid[name] 
            
        return name2uid


    def name2key(self, name:str=None, 
                 max_age=1000, 
                 timeout=30, 
                 netuid: int = 0, 
                 update=False, 
                 trials=3,
                 **kwargs ) -> Dict[str, str]:
        # netuid = self.resolve_netuid(netuid)
        netuid = self.resolve_netuid(netuid)

        names = c.submit(self.get_feature, kwargs={'feature': 'names', 'netuid':netuid, 'update':update, 'max_age':max_age})
        keys = c.submit(self.get_feature, kwargs={'feature': 'keys', 'netuid':netuid, 'update':update, 'max_age':max_age})
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
                    return self.name2key(name=name,
                                        timeout=timeout, netuid=netuid, update=True, 
                                        trials=trials, **kwargs)
                
        return name2key





    def key2name(self, key=None, netuid: int = None, update=False) -> Dict[str, str]:
        
        key2name =  {v:k for k,v in self.name2key(netuid=netuid,  update=update).items()}
        if key != None:
            return key2name[key]
        return key2name
        

    def epoch_time(self, netuid=0, update=False, **kwargs):
        return self.subnet_params(netuid=netuid, update=update, **kwargs)['tempo']*self.block_time

    def blocks_per_day(self):
        return 24*60*60/self.block_time

    def epochs_per_day(self, netuid=None):
        return 24*60*60/self.epoch_time(netuid=netuid)
    
    def emission_per_epoch(self, netuid=None):
        return self.subnet(netuid=netuid)['emission']*self.epoch_time(netuid=netuid)

    def get_block(self,  block_hash=None, max_age=8): 
        path = f'cache/{self.network}.block'
        block = self.get(path, None, max_age=max_age)
        if block == None:
            block_header = self.substrate.get_block( block_hash=block_hash)['header']
            block = block_header['number']
            block_hash = block_header['hash']
            self.put(path, block)
        return block

    def block_hash(self, block = None,): 
        if block == None:
            block = self.block
        substrate = self.get_substrate()
        return substrate.get_block_hash(block)
    

    def seconds_per_epoch(self, netuid=None):
        netuid =self.resolve_netuid(netuid)
        return self.block_time * self.subnet(netuid=netuid)['tempo']

    
    def get_module(self, 
                    module='vali',
                    netuid=0,
                    trials = 4,
                    fmt='j',
                    mode = 'http',
                    block = None,
                    max_age = None,
                    lite = True, 
                    **kwargs ) -> 'ModuleInfo':

        url = self.resolve_url( mode=mode)
        module_key = module
        if not c.valid_ss58_address(module):
            module_key = self.name2key(name=module,  netuid=netuid, **kwargs)
        netuid = self.resolve_netuid(netuid)
        json={'id':1, 'jsonrpc':'2.0',  'method': 'subspace_getModuleInfo', 'params': [module_key, netuid]}
        module = None
        for i in range(trials):
            try:
                module = requests.post(url,  json=json).json()
                break
            except Exception as e:
                c.print(e)
                continue
        assert module != None, f"Failed to get module {module_key} after {trials} trials"
        module = {**module['result']['stats'], **module['result']['params']}
        # convert list of u8 into a string Vector<u8> to a string
        module['name'] = self.vec82str(module['name'])
        module['address'] = self.vec82str(module['address'])
        module['dividends'] = module['dividends'] / (U16_MAX)
        module['incentive'] = module['incentive'] / (U16_MAX)
        module['stake_from'] = {k:self.format_amount(v, fmt=fmt) for k,v in module['stake_from'].items()}
        module['stake'] = sum([v for k,v in module['stake_from'].items() ])
        module['emission'] = self.format_amount(module['emission'], fmt=fmt)
        module['key'] = module.pop('controller', None)
        module['metadata'] = module.pop('metadata', {})

        module['vote_staleness'] = (block or self.block) - module['last_update']
        if lite :
            features = self.config.module_features + ['stake', 'vote_staleness']
            module = {f: module[f] for f in features}
        assert module['key'] == module_key, f"Key mismatch {module['key']} != {module_key}"
        return module


    minfo = module_info = get_module
    
    @staticmethod
    def vec82str(l:list):
        return ''.join([chr(x) for x in l]).strip()

    def get_modules(self, keys:list = None,
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
            all_keys = self.keys(update=update, netuid=netuid)
            modules = {}
            for netuid in self.netuids():
                module = self.get_modules(keys=all_keys[netuid], netuid=netuid,   **kwargs)
                modules[netuid] = module.get(netuid, []) + [module]
            return modules
        if keys == None:
            keys = self.keys(update=update, netuid=netuid)
        c.print(f'Querying {len(keys)} keys for modules')
        if len(keys) >= batch_size:
            key_batches = c.chunk(keys, chunk_size=batch_size)
            futures = []
            for key_batch in key_batches:
                c.print(key_batch)
                f = c.submit(self.get_modules, kwargs=dict(keys=key_batch,
                                                        block=block, 
                                                         
                                                        netuid=netuid, 
                                                        batch_size=len(keys) + 1,
                                                        timeout=timeout))
                futures += [f]
            module_batches = c.wait(futures, timeout=timeout)
            c.print(module_batches)
            name2module = {}
            for module_batch in module_batches:
                if isinstance(module_batch, list):
                    for m in module_batch:
                        if isinstance(m, dict) and 'name' in m:
                            name2module[m['name']] = m
                    
            modules = list(name2module.values())
            return modules
        elif len(keys) == 0:
            c.print('No keys found')
            return []

        progress_bar = c.tqdm(total=len(keys), desc=f'Querying {len(keys)} keys for modules')
        modules = []
        for key in keys:
            module = self.module_info(module=key, block=block, netuid=netuid,  fmt=fmt, **kwargs)
            if isinstance(module, dict) and 'name' in module:
                modules.append(module)
                progress_bar.update(1)
        
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
        modules =  self.get_modules(keys=keys, netuid=netuid, **kwargs)
        return modules
    

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

    def df(self,
            netuid=0, 
            features=['name', 'address', 'incentive', 'dividends', 'emission', 'last_update', 'delegation_fee', 'stake'], 
            **kwargs) -> 'pd.DataFrame':
        df =  c.df(self.modules(netuid=netuid, **kwargs))
        if len(df) > 0:
            df = df[features]
        return df


    def modules(self,
                search:str= None,
                netuid: int = 0,
                block: Optional[int] = None,
                fmt='nano', 
                features : List[str] = None,
                timeout = 100,
                max_age=1000,
                subnet = None,
                df = False,
                vector_features =['dividends', 'incentive', 'trust', 'last_update', 'emission'],
                **kwargs
                ) -> Dict[str, 'ModuleInfo']:
        
        features = features or self.config.module_features
    

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
        state = {}
        path = f'query/{self.network}/SubspaceModule.Modules:{netuid}'
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
            for m in modules:
                m['stake'] =  sum([v
                                   for k,v in m['stake_from'].items()])

        if search != None:
            modules = [m for m in modules if search in m['name']]

        return modules

    


    def min_stake(self, netuid: int = 0, network: str = 'main', fmt:str='j', **kwargs) -> int:
        min_stake = self.query('MinStake', netuid=netuid,  **kwargs)
        return self.format_amount(min_stake, fmt=fmt)

    def registrations_per_block(self, network: str = network, fmt:str='j', **kwargs) -> int:
        return self.query('RegistrationsPerBlock', params=[],  **kwargs)
    regsperblock = registrations_per_block
    
    def max_registrations_per_block(self, network: str = network, fmt:str='j', **kwargs) -> int:
        return self.query('MaxRegistrationsPerBlock', params=[],  **kwargs)
 
   
    def uids(self,
             netuid = 0,
              update=False, 
              max_age=1000,
             network : str = 'main', 
             **kwargs) -> List[str]:
        netuid = self.resolve_netuid(netuid)
        keys =  self.query_map('Keys', netuid=netuid, update=update,  max_age=max_age, **kwargs)
        if netuid == 'all':
            for netuid, netuid_keys in keys.items():
                keys[netuid] = list(netuid_keys.keys ())
        else:
            keys = list(keys.keys())
        return keys

    def keys(self,
             netuid = 0,
              update=False, 
              max_age=1000,
             network : str = 'main', 
             **kwargs) -> List[str]:
        keys =  self.query_map('Keys', netuid=netuid, update=update,  max_age=max_age, **kwargs)
        if netuid == 'all':
            for netuid, netuid_keys in keys.items():
                keys[netuid] = list(netuid_keys.values())
        else:
            keys = list(keys.values())
        return keys

    def uid2key(self, uid=None, 
             netuid = 0,
              update=False, 
              
             max_age= 1000,
             **kwargs):
        netuid = self.resolve_netuid(netuid)
        uid2key =  self.query_map('Keys',  netuid=netuid, update=update,  max_age=max_age, **kwargs)
        # sort by uid
        if uid != None:
            return uid2key[uid]
        return uid2key
    

    def key2uid(self, key = None, network:str=  'main' ,netuid: int = 0, update=False, netuids=None , **kwargs):
        uid2key =  self.uid2key( netuid=netuid, update=update, **kwargs)
        reverse_map = lambda x: {v: k for k,v in x.items()}
        if netuid == 'all':
            key2uid =  {netuid: reverse_map(_key2uid) for netuid, _key2uid in uid2key.items()  if   netuids == None or netuid in netuids  }
        else:
            key2uid = reverse_map(uid2key)
        if key != None:
            key_ss58 = self.resolve_key_ss58(key)
            return key2uid[key_ss58]
        return key2uid
        

    def uid2name(self, netuid: int = 0, update=False,  **kwargs) -> List[str]:
        netuid = self.resolve_netuid(netuid)
        names = self.query_map('Name', netuid=netuid, update=update,**kwargs)
        return names
    
    def names(self, 
              netuid: int = 0, 
              update=False,
                **kwargs) -> List[str]:
        netuid = self.resolve_netuid(netuid)
        names = self.query_map('Name', update=update, netuid=netuid,**kwargs)
        if netuid == 'all':
            for netuid, netuid_names in names.items():
                names[netuid] = list(netuid_names.values())
        else:
            names = list(names.values())
        return names

    def addresses(self, netuid: int = 0, update=False, **kwargs) -> List[str]:
        netuid = self.resolve_netuid(netuid)
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

        
        if netuid == 'all':
            netuid2subnet = self.netuid2subnet()
            namespace = {}
            for netuid, netuid_addresses in results['addresses'].items():
                for uid,address in enumerate(netuid_addresses):
                    name = results['names'][netuid][uid]
                    subnet = netuid2subnet[netuid]
                    namespace[f'{subnet}/{name}'] = address

        else:
            namespace = {k:v for k,v in zip(results['names'], results['addresses'])}

        if search != None:
            namespace = {k:v for k,v in namespace.items() if search in str(k)}

        if local:
            ip = c.ip()
            namespace = {k:v for k,v in namespace.items() if ip in str(v)}

        return namespace

    
    def weights(self,  netuid = 0,  update=False, **kwargs) -> list:
        weights =  self.query_map('Weights',netuid=netuid, update=update, **kwargs)

        return weights

    def proposals(self,  block=None,  nonzero:bool=False, update:bool = False,  **kwargs):
        proposals = [v for v in self.query_map('Proposals', block=block, update=update, **kwargs)]
        return proposals

    def save_weights(self, **kwargs) -> list:
        self.query_map('Weights', update=True, **kwargs)
        return {'success': True, 'msg': 'Saved weights'}

    def pending_deregistrations(self, netuid = 0, update=False, **kwargs):
        pending_deregistrations = self.query_map('PendingDeregisterUids',update=update,**kwargs)[netuid]
        return pending_deregistrations
    
    def num_pending_deregistrations(self, netuid = 0, **kwargs):
        pending_deregistrations = self.pending_deregistrations(netuid=netuid, **kwargs)
        return len(pending_deregistrations)
        
    def emissions(self, netuid = 0, block=None, update=False, **kwargs):

        return self.query_vector('Emission',  netuid=netuid, block=block, update=update, **kwargs)
    
    emission = emissions
    
    def incentives(self, 
                  netuid = 0, 
                  block=None,  
                  update:bool = False, 
                  **kwargs):
        return self.query_vector('Incentive', netuid=netuid,  block=block, update=update, **kwargs)
    incentive = incentives

    def trust(self, 
                  netuid = 0, 
                  block=None,  
                  update:bool = False, 
                  **kwargs):
        return self.query_vector('Trust', netuid=netuid,  block=block, update=update, **kwargs)
    
    incentive = incentives
    
    def query_vector(self, name='Trust', netuid = 0, update=False, **kwargs):
        if isinstance(netuid, int):
            query_vector = self.query(name,  netuid=netuid,  update=update, **kwargs)
        else:
            query_vector = self.query_map(name, netuid=netuid,  update=update, **kwargs)
            if len(query_vector) == 0:
                query_vector = {_: [] for _ in range(len(self.netuids()))}
        return query_vector
    
    def last_update(self, netuid = 0, update=False, **kwargs):
        return self.query_vector('LastUpdate', netuid=netuid,   update=update, **kwargs)

    def dividends(self, netuid = 0, update=False, **kwargs):
        return  self.query_vector('Dividends', netuid=netuid,   update=update,  **kwargs)
            

    dividend = dividends

    def registration_block(self, netuid: int = 0, update=False, **kwargs):
        registration_blocks = self.query_map('RegistrationBlock', netuid=netuid, update=update, **kwargs)
        return registration_blocks

    regblocks = registration_blocks = registration_block

    def stake_from(self, netuid = 0,
                    block=None, 
                    update=False,
                    max_age=10000,
                    fmt='nano', 
                    **kwargs) -> List[Dict[str, Union[str, int]]]:
        
        stake_from = self.query_map('StakeFrom', netuid=netuid, block=block, update=update, max_age=max_age,  **kwargs)
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

        datetime2archive =  self.datetime2archive( **kwargs) 
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

        datetime2archive =  self.datetime2archive( **kwargs) 
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
                      
                     netuid= 0 , 
                     update=True,  
                     **kwargs):
        
        path = f'history/{self.network}.{netuid}.json'

        archive_history = []
        if not update:
            archive_history = cls.get(path, [])
        if len(archive_history) == 0:
            archive_history =  cls.search_archives(*args, netuid=netuid, **kwargs)
            cls.put(path, archive_history)
            
        return archive_history
        


    def get_nonce(self, key:str=None,  **kwargs):
        key_ss58 = self.resolve_key_ss58(key)
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
            all_modules = self.my_modules(netuid=netuid, update=update,  fmt=fmt, search=search)
            servers = c.servers()
            stats = {}
            netuid2subnet = self.netuid2subnet(update=update)
            for netuid, modules in all_modules.items():
                subnet_name = netuid2subnet[netuid]
                stats[netuid] = self.stats(modules=modules, netuid=netuid, servers=servers)

                color = c.random_color()
                c.print(f'\n {subnet_name.upper()} :: (netuid:{netuid})\n', color=color)
                c.print(stats[netuid], color=color)
            

        modules = modules or self.my_modules(netuid=netuid, update=update,  fmt=fmt, search=search)

        stats = []

        local_key_addresses = list(c.key2address().values())
        servers = servers or c.servers()
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
                epochs_per_day = self.epochs_per_day(netuid=netuid)
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


    def storage_functions(self,  block_hash = None):
        return self.substrate.get_metadata_storage_functions( block_hash=block_hash)
    
    storage_fns = storage_functions
        

    def storage_names(self,  search=None,  block_hash = None):
        storage_names =  [f['storage_name'] for f in self.substrate.get_metadata_storage_functions( block_hash=block_hash)]
        if search != None:
            storage_names = [s for s in storage_names if search in s.lower()]
        return storage_names

    def state_dict(self , 
                   timeout=1000, 
                   netuid = 'all',
                   update=False, 
                   mode='http', 
                   save = False,
                   block=None):
        
        
        start_time = c.time()

        if save:
            update = True
        if not update:
            state_path = self.latest_archive_path() # get the latest archive path
            state_dict = c.get(state_path, None)
            if state_path != None:
                return state_dict

        block = block or self.block

        path = f'state_dict/{self.network}.block-{block}-time-{int(c.time())}'

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
    

    def sync(self, netuid='all', max_age=30, timeout=60, **kwargs):
        
        batch = {
            'stake_from': {'netuid': netuid, 'max_age': max_age},
            'subnet_params': {'netuid': netuid,  'max_age': max_age},
            'global_params': {'max_age': max_age},
        }
        futures = []
        for fn, params in batch.items():
            c.print(dict(fn=fn, params=params))
            fn = getattr(self, fn)
            f = c.submit(fn, params=params, timeout=timeout)
            futures.append(f)

        results = []
        for f in c.as_completed(futures, timeout=timeout):
            result = f.result()
            c.print(result)
            results.append(result)
            
        return results

    def check_storage(self, block_hash = None):
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
        netuid = self.resolve_netuid(netuid)
        min_burn = self.min_burn(  fmt=fmt)
        min_stake = self.min_stake(netuid=netuid,  fmt=fmt)
        return min_stake + min_burn
    
    def register(
        self,
        name: str , # defaults to module.tage
        address : str = None,
        stake : float = None,
        netuid = None,
        subnet: str = 'commune',
        key : str  = None,
        module_key : str = None,
        network: str = network,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        module : str = None,
        metadata = None,
        nonce=None,
        tag = None,
        ensure_server = True,
    **kwargs
    ) -> bool:

        if name == None:
            name = module
        if tag != None:
            name = f'{module}::{tag}'
        # resolve module name and tag if they are in the server_name
        if not c.server_exists(name):
            address = c.serve(name)['address'] 
        else:
            address = c.namespace().get(name,address)

        module_key = module_key or c.get_key(name).ss58_address
        subnet2netuid = self.subnet2netuid(update=False)
        netuid2subnet = self.netuid2subnet(update=False)    

        if isinstance(netuid, str):
            subnet = netuid
        if isinstance(netuid, int):
            subnet = netuid2subnet[netuid]

        assert isinstance(subnet, str), f"Subnet must be a string"

        if not subnet in subnet2netuid:
            subnet2netuid = self.subnet2netuid(update=True)
            if subnet not in subnet2netuid:
                subnet2netuid[subnet] = len(subnet2netuid)
                response = input(f"Do you want to create a new subnet ({subnet}) (yes or y or dope): ")
                if response.lower() not in ["yes", 'y', 'dope']:
                    return {'success': False, 'msg': 'Subnet not found and not created'}
                
        # require prompt to create new subnet        
        stake = (stake or 0) * 1e9

        if '0.0.0.0' in address:
            address = address.replace('0.0.0.0', c.ip())

        if len(address) > 32:
            address = address[-32:]

        params = { 
                    'network': subnet.encode('utf-8'),
                    'address': address.encode('utf-8'),
                    'name': name.encode('utf-8'),
                    'stake': stake,
                    'module_key': module_key,
                    'metadata': json.dumps(metadata or {}).encode('utf-8'),
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
        dest = self.resolve_key_ss58(dest)
        amount = self.to_nanos(amount) # convert to nano (10^9 nanos = 1 token)

        response = self.compose_call(
            module='Balances',
            fn='transfer_keep_alive',
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

        return self.query_map('ProfitShares',  **kwargs)

    def add_profit_shares(
        self,
        keys: List[str], # the keys to add profit shares to
        shares: List[float] = None , # the shares to add to the keys
        key: str = None,
        netuid : int = 0,
    ) -> bool:
        
        key = self.resolve_key(key)
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
                'shares': shares,
                'netuid': netuid
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
                        netuid=0,
                         **kwargs) -> List[str]:
        
        netuid = self.resolve_netuid(netuid)
        my_modules = self.my_modules(search=search, netuid=netuid, **kwargs)

        self.keys()
        futures = []
        namespace = c.namespace()
        for m in my_modules:

            name = m['name']
            if name in namespace:
                address = namespace[name]
            else:
                address = c.serve(name)['address']

            if m['address'] == address and m['name'] == name:
                c.print(f"Module {m['name']} already up to date")

            f = c.submit(c.update_module, kwargs={'module': name,
                                                    'name': name,
                                                    'netuid': netuid,
                                                    'address': address,
                                                  **kwargs}, timeout=timeout)
            futures+= [f]


        results = []

        for future in c.as_completed(futures, timeout=timeout):
            results += [future.result()]
            c.print(future.result())
        return results


    def update_module(
        self,
        module: str, # the module you want to change
        address: str = None, # the address of the new module
        name: str = None, # the name of the new module
        delegation_fee: float = None, # the delegation fee of the new module
        metadata = None, # the metadata of the new module
        fee : float = None, # the fee of the new module
        netuid: int = 0, # the netuid of the new module
        nonce = None, # the nonce of the new module
        tip: int = 0, # the tip of the new module
    ) -> bool:
        key = self.resolve_key(module)
        netuid = self.resolve_netuid(netuid)  
        module_info = self.module_info(key.ss58_address, netuid=netuid)
        assert module_info['name'] == module
        assert module_info['key'] == key.ss58_address
            
        params = {
            'name': name , # defaults to module.tage
            'address': address , # defaults to module.tage
            'delegation_fee': fee or delegation_fee, # defaults to module.delegate_fee
            'metadata': c.serialize(metadata or {}), # defaults to module.metadata
        }


        should_update_module = False

        for k,v in params.items(): 
            if params[k] == None:
                params[k] = module_info[k]
            if k in module_info and params[k] != module_info[k]:
                should_update_module = True

        if not should_update_module: 
            return {'success': False, 'message': f"Module {module} is already up to date"}
               
        c.print('Updating with', params, color='cyan')
        params['netuid'] = netuid
        reponse  = self.compose_call('update_module', params=params, key=key, nonce=nonce, tip=tip)

        # IF SUCCESSFUL, MOVE THE KEYS, AS THIS IS A NON-REVERSIBLE OPERATION


        return reponse

    update_server = update_module

    #################
    #### UPDATE SUBNET ####
    #################
    def update_subnet(
        self,
        params: dict,
        netuid: int,
        key: str = None,
        nonce = None,
        update= True,
    ) -> bool:
            
        netuid = self.resolve_netuid(netuid)
        subnet_params = self.subnet_params( netuid=netuid , update=update, fmt='nanos')
        # infer the key if you have it
        for k in ['min_stake']:
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
        for k in ['name']:
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
        nonce = None,
        **params,
    ) -> bool:

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
                    
    
    def stake_transfer(
            self,
            module_key: str ,
            new_module_key: str ,
            amount: Union[int, float] = None, 
            key: str = None,
            netuid:int = 0,
        ) -> bool:
        # STILL UNDER DEVELOPMENT, DO NOT USE
        netuid = self.resolve_netuid(netuid)
        key = c.get_key(key)

        c.print(f':satellite: Staking to: [bold white]SubNetwork {netuid}[/bold white] {amount} ...')
        # Flag to indicate if we are using the wallet's own hotkey.

        module_key = self.resolve_module_key(module_key, netuid=netuid)
        new_module_key = self.resolve_module_key(new_module_key, netuid=netuid)
        c.print(f':satellite: Staking to: [bold white]SubNetwork {netuid}[/bold white] {amount} ...')
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
                        n:str = 100) -> Optional['Balance']:
        
        netuid = self.resolve_netuid( netuid )
        key = self.resolve_key( key )

        if modules == None:
            my_modules = self.my_modules(netuid=netuid,  update=False)
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
                        n:str = 10) -> Optional['Balance']:
        key2address = c.key2address()
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
        assert len(destinations) == len(amounts), f"Length of modules and amounts must be the same. Got {len(destinations)} and {len(amounts)}."
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
                        netuid:int = 0) -> Optional['Balance']:
        
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
    
    def unstake_all( self, 
                        key: str = None, 
                        netuid = 0,
                        existential_deposit = 1) -> Optional['Balance']:
        
        key = self.resolve_key( key )
        netuid = self.resolve_netuid( netuid )
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

    def my_key2uid(self, *args, netuid=0, update=False, **kwargs):
        key2uid = self.key2uid(*args,  netuid=netuid, **kwargs)

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
                        update = False,
                        n = None,
                        netuid = 0, 
                        df = True,
                        keys = None,
                        max_age = 1000,
                        min_stake = 100,
                        features = ['name','stake_from', 'dividends', 'delegation_fee',  'key'],
                        sort_by = 'stake_from',
                        **kwargs):
        
        key = self.resolve_key(key)
        netuid = self.resolve_netuid(netuid)

        if keys == None:
            staked_modules = self.my_staked_module_keys(netuid=netuid,  max_age=max_age, update=update)
            if netuid == 'all':
                staked = {}
                for netuid, keys in staked_modules.items():
                    if len(keys) == 0:
                        continue
                    staked_netuid = self.staked(search=search, 
                                                key=key, 
                                                netuid=netuid, 
                                                 
                                                df=df, 
                                                keys=keys)
                    if len(staked_netuid) > 0:
                        staked[netuid] = staked_netuid
                
                return staked
            else: 
                keys = staked_modules
                

        c.print(f'Getting staked modules for SubNetwork {netuid} with {len(keys)} modules')
        if search != None:
            key2name = self.my_key2name(search=search, netuid=netuid, max_age=max_age)
            keys = [k for k in keys if search in key2name.get(k, k)]
        block = self.block
        if n != None:
            keys = keys
        modules = self.get_modules(keys, block=block, netuid=netuid)
        for m in modules:          
            if isinstance(m['stake_from'], dict): 
                m['stake_from'] =  int(m['stake_from'].get(key.ss58_address, 0))
            m['stake'] = int(m['stake'])
        if search != None:
            modules = [m for m in modules if search in m['name']]


        if len(modules) == 0: 
            return modules
        modules = c.df(modules)[features]
        modules = modules.sort_values(sort_by, ascending=False)
        # filter out everything where stake_from > min_stake
        modules = modules[modules['stake_from'] > min_stake]
        if not df:
            modules = modules.to_dict(orient='records')
            modules = [{k: v for k,v in m.items()  if k in features} for m in modules]


        if n != None:
            modules = modules[:n]
        return modules

    staked_modules = staked

    
    
    
    def my_keys(self, search=None, netuid=0, max_age=None, update=False, **kwargs):
        netuid = self.resolve_netuid(netuid)
        keys = self.keys(netuid=netuid, max_age=max_age, update=update, **kwargs)
        key2address = c.key2address(search=search, max_age=max_age, update=update)
        if search != None:
            key2address = {k: v for k,v in key2address.items() if search in k}
        addresses = list(key2address.values())
        if netuid == 'all':
            my_keys = {}
            c.print(keys)
            for netuid, netuid_keys in enumerate(keys):
                if len(netuid_keys) > 0:
                    my_keys[netuid] = [k for k in netuid_keys if k in addresses]

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
        update=False,
        min_value = 0,
        max_value = 1,
        max_age = 100,
        **kwargs
    ) -> bool:
        import torch

        netuid = self.resolve_netuid(netuid)
        key = self.resolve_key(key)
        global_params = self.global_params()
        subnet_params = self.subnet_params( netuid = netuid , max_age=None, update=False)
        module_info = self.module_info(key.ss58_address, netuid=netuid)
        min_stake = global_params['min_weight_stake'] * subnet_params['min_allowed_weights']
        assert module_info['stake'] > min_stake
        max_num_votes = module_info['stake'] // global_params['min_weight_stake']
        n = int(min(max_num_votes, subnet_params['max_allowed_weights']))
        modules = uids or modules
        if modules == None:
            modules = c.shuffle(self.uids(netuid=netuid, update=update))
        # checking if the "uids" are passed as names -> strings
        key2name, name2uid = None, None
        for i, module in enumerate(modules):
            if isinstance(module, str):
                if key2name == None or name2uid == None:
                    key2name = self.key2name(netuid=netuid, update=update)
                    name2uid = self.name2uid(netuid=netuid, update=update)
                if module in key2name:
                    modules[i] = key2name[module]
                elif module in name2uid:
                    modules[i] = name2uid[module]
                    
        uids = modules
        
        if weights is None:
            weights = [1 for _ in uids]
        if len(uids) < subnet_params['min_allowed_weights']:
            n = self.n(netuid=netuid)
            while len(uids) < subnet_params['min_allowed_weights']:
                uid = c.choice(list(range(n)))
                if uid not in uids:
                    uids.append(uid)
                    weights.append(min_value)

        uid2weight = dict(sorted(zip(uids, weights), key=lambda item: item[1], reverse=True))
        
        self_uid = self.key2uid(netuid=netuid).get(key.ss58_address, None)
        uid2weight.pop(self_uid, None)

        uids = list(uid2weight.keys())
        weights = list(uid2weight.values())


        if len(uids) > subnet_params['max_allowed_weights']:
            uids = uids[:subnet_params['max_allowed_weights']]
            weights = weights[:subnet_params['max_allowed_weights']]


        c.print(f'Voting for {len(uids)} modules')
        assert len(uids) == len(weights), f"Length of uids {len(uids)} must be equal to length of weights {len(weights)}"
        uids = torch.tensor(uids)
        weights = torch.tensor(weights)
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

    def register_servers(self,  
                         search=None, 
                         infos=None,  
                         netuid = 0, 
                         timeout=60, 
                         max_age=None, 
                         key=None, update=False, 
                         parallel = True,
                         **kwargs):
        '''
        key2address : dict
            A dictionary of module names to their keys
        timeout : int 
            The timeout for each registration
        netuid : int
            The netuid of the modules
        
        '''
        keys = c.submit(self.keys, dict(netuid=netuid, update=update, max_age=max_age))
        names = c.submit(self.names, dict(netuid=netuid, update=update, max_age=max_age))
        keys, names = c.wait([keys, names], timeout=timeout)

        if infos==None:
            infos = c.infos(search=search, **kwargs)
            should_register_fn = lambda x: x['key'] not in keys and x['name'] not in names
            infos = [i for i in infos if should_register_fn(i)]
            c.print(f'Found {infos} modules to register')
        if parallel:
            launcher2balance = c.key2balance()
            min_stake = self.min_register_stake(netuid=netuid)
            launcher2balance = {k: v for k,v in launcher2balance.items() if v > min_stake}
            launcher_keys = list(launcher2balance.keys())
            futures = []
            for i, info in enumerate(infos):
                if info['key'] in keys:
                    continue
                    
                launcher_key = launcher_keys[i % len(launcher_keys)]
                c.print(f"Registering {info['name']} with module_key {info['key']} using launcher {launcher_key}")
                f = c.submit(c.register, kwargs=dict(name=info['name'], 
                                                    address= info['address'],
                                                    netuid = netuid,
                                                    module_key=info['key'], 
                                                    key=launcher_key), timeout=timeout)
                futures+= [f]

                if len(futures) == len(launcher_keys):
                    for future in c.as_completed(futures, timeout=timeout):
                        r = future.result()
                        c.print(r, color='green')
                        futures.remove(future)
                        break

            for future in c.as_completed(futures, timeout=timeout):
                r = future.result()
                c.print(r, color='green')
                futures.remove(future)

            return infos
                
        else:

            for info in infos:
                r = c.register(name=info['name'], 
                            address= info['address'],
                            module_key=info['key'], 
                            key=key)
                c.print(r, color='green')
  
        return {'success': True, 'message': 'All modules registered'}


    def unregistered_servers(self, search=None, netuid = 0, key=None, max_age=None, update=False, transfer_multiple=True,**kwargs):
        netuid = self.resolve_netuid(netuid)
        servers = c.servers(search=search)
        key2address = c.key2address(update=update)
        keys = self.keys(netuid=netuid, max_age=max_age, update=update)
        unregister_servers = []
        for s in servers:
            if  key2address[s] not in keys:
                unregister_servers += [s]
        return unregister_servers

    def get_balances(self, 
                    keys=None,
                    search=None, 
                    batch_size = 128,
                    fmt = 'j',
                    n = 100,
                    max_trials = 3,
                    names = False,
                    **kwargs):

        key2balance  = {}

        key2address = c.key2address(search=search)
        if keys == None:
            keys = list(key2address.keys())
        if len(keys) > n:
            c.print(f'Getting balances for {len(keys)} keys > {n} keys, using batch_size {batch_size}')
            balances = self.balances(**kwargs)
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
            c.print(f'Getting balances for {len(keys)} keys')

            def batch_fn(batch_keys):
                substrate = self.get_substrate()
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
        
    def registered_servers(self, netuid = 0, **kwargs):
        netuid = self.resolve_netuid(netuid)
        servers = c.servers()
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
                                update=1,
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
    
    def my_value( self, *args, **kwargs ):
        return sum(list(self.key2value( *args, **kwargs).values()))
    

    def my_total_stake(self, netuid='all', fmt='j', update=False):
        my_stake_to = self.my_stake_to(netuid=netuid,  fmt=fmt, update=update)
        return sum([sum(list(v.values())) for k,v in my_stake_to.items()])

    def check_valis(self, **kwargs):
        return self.check_servers(search='vali', **kwargs)
    
    def check_servers(self, search='vali',update:bool=False, netuid=0, max_staleness=100, timeout=30, remote=False, **kwargs):
        if remote:
            kwargs = c.locals2kwargs(locals())
            return self.remote_fn('check_servers', kwargs=kwargs)
        module_stats = self.stats(search=search, netuid=netuid, df=False, update=update)
        module2stats = {m['name']:m for m in module_stats}
        response_batch = {}
        c.print(f"Checking {len(module2stats)} {search} servers")
        for module, stats in module2stats.items():
            # check if the module is serving
            should_serve = not c.server_exists(module) or stats['vote_staleness'] > max_staleness
            if should_serve:

                c.print(f"Serving {module}")
                port = int(stats['address'].split(':')[-1])
                response_batch[module]  = c.submit(c.serve, 
                                                    kwargs=dict(module=module, 
                                                                network=f'subspace.{netuid}', 
                                                                port=port),
                                                    timeout=timeout)

        futures = list(response_batch.values())
        future2key = {f: k for k,f in response_batch.items()}
        for f in c.as_completed(futures, timeout=timeout):
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
                    mode='ws',
                    trials = 4,
                    max_tip = 10000,
                     **kwargs):

        """
        Composes a call to a Substrate chain.

        """
        key = self.resolve_key(key)

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
        paths = {m: f'history/{self.config.network}/{ss58_address}/{m}/{start_time}.json' for m in ['complete', 'pending']}
        params = {k: int(v) if type(v) in [float]  else v for k,v in params.items()}
        compose_kwargs = dict(
                call_module=module,
                call_function=fn,
                call_params=params,
        )
        c.print(f'Sending 📡 using 🔑(ss58={key.ss58_address}, name={key.path})🔑', compose_kwargs,color=color)
        tx_state = dict(status = 'pending',start_time=start_time, end_time=None)

        self.put_json(paths['pending'], tx_state)

        for t in range(trials):
            try:
                substrate = self.get_substrate( mode='ws')
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
                        response =  {'success': True, 'tx_hash': response.extrinsic_hash, 'msg': f'Called {module}.{fn} on {self.config.network} with key {key.ss58_address}'}
                    else:
                        response =  {'success': False, 'error': response.error_message, 'msg': f'Failed to call {module}.{fn} on {self.config.network} with key {key.ss58_address}'}
                else:
                    response =  {'success': True, 'tx_hash': response.extrinsic_hash, 'msg': f'Called {module}.{fn} on {self.config.network} with key {key.ss58_address}'}
                break
            except Exception as e:
                if t == trials - 1:
                    raise e
                

        tx_state['end_time'] = c.datetime()
        tx_state['status'] = 'completed'
        tx_state['response'] = response
        # remo 
        self.rm(paths['pending'])
        self.put_json(paths['complete'], tx_state)
        return response
            
    def tx_history(self, key:str=None, mode='complete', **kwargs):
        key_ss58 = self.resolve_key_ss58(key)
        assert mode in ['pending', 'complete']
        pending_path = f'history/{self.network}/{key_ss58}/{mode}'
        return self.glob(pending_path)
    
    def pending_txs(self, key:str=None, **kwargs):
        return self.tx_history(key=key, mode='pending', **kwargs)

    def complete_txs(self, key:str=None, **kwargs):
        return self.tx_history(key=key, mode='complete', **kwargs)

    def clean_tx_history(self):
        return self.ls(f'tx_history')
        
    def resolve_tx_dirpath(self, key:str=None, mode:'str([pending,complete])'='pending',  **kwargs):
        key_ss58 = self.resolve_key_ss58(key)
        assert mode in ['pending', 'complete']
        pending_path = f'history/{self.network}/{key_ss58}/{mode}'
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

        assert hasattr(key, 'key'), f"Invalid Key {key} as it should have ss58_address attribute."
        return key



    def launcher_keys(self, netuid=0, min_stake=500, **kwargs):
        keys = c.keys()
        key2balance =  c.key2balance(netuid=netuid,**kwargs)
        key2balance = {k: v for k,v in key2balance.items() if v > min_stake}
        return [k for k in keys]
    
    def load_launcher_keys(self, amount=600, **kwargs):
        launcher_keys = self.launcher_keys()
        key2address = c.key2address()
        destinations = []
        amounts = []
        launcher2balance = c.get_balances(launcher_keys)
        for k in launcher_keys:
            k_address = key2address[k]
            amount_needed = amount - launcher2balance.get(k_address, 0)
            if amount_needed > 0:
                destinations.append(k_address)
                amounts.append(amount_needed)
            else:
                c.print(f'{k} has enough balance --> {launcher2balance.get(k, 0)}')

        return c.transfer_many(amounts=amounts, destinations=destinations, **kwargs)
    
            
    @classmethod
    def get_feature(cls, feature='names', network='main', netuid=0, update=False, max_age=1000, **kwargs):
        return getattr(cls(network=network), feature)(netuid=netuid, update=update, max_age=max_age, **kwargs)
        
    



Subspace.run(__name__)
