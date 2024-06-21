
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


    def filter_url(self, url):
        """
        Filter urls based on the url_search parameter
        """
        if self.config.url_search == None:
            return True
        url_search_terms = [url.strip() for x in self.config.url_search.split(',')]
        return any([x in url for x in url_search_terms])
    
    def resolve_url(self, url = None, mode=None, **kwargs):
        mode =  mode or self.config.network_mode
        url = url or self.config.url
        assert mode in ['http', 'ws']
        if url != None:
            return url
        
        network = self.resolve_network()
        if url == None:
            urls_map = getattr(self.config.urls,  network)
            urls = urls_map.get(mode, [])
            assert len(urls) > 0, f'No urls found for network {network} and mode {mode}'
            if len(urls) > 1:
                urls_map = list(filter(self.filter_url, urls))
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
                update : bool = False,
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

                if not update:
                    if url in self.url2substrate:
                        substrate = self.url2substrate[url]
                        break

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
                if trials == 0:
                    raise e
                
        self.url = url
        self.url2substrate[url] = substrate
                
  

        return substrate


    def set_network(self, 
                network:str = None,
                mode = 'http',
                trials = 10,
                url : str = None, 
                save = False,
                **kwargs):
        self.network = self.resolve_network(network)
        self.substrate = self.get_substrate( url=url, mode=mode, trials=trials , **kwargs)
        if save:
            self.save_config(self.config)
        return  {'network': self.network, 'url': self.url, 'save': save}
    
    
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


    

    def staked_module_keys(self, netuid = 0, **kwargs):
        stake_to = self.stake_to(netuid=netuid, **kwargs)
        module_keys = []
        for key, stake_to_key in stake_to.items():
            module_keys += list(stake_to_key.keys())
        return module_keys

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
            progress_bar = c.progress(qmap, desc=f'Querying {name}(network={self.network})')
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

    
    def module2netuids(self, **kwargs):
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

        names = c.submit(self.get_feature, kwargs={'feature': 'names', 'netuid':netuid, 'update':update, 'max_age':max_age, 'network': self.network})
        keys = c.submit(self.get_feature, kwargs={'feature': 'keys', 'netuid':netuid, 'update':update, 'max_age':max_age, 'network': self.network})
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
        module['stake_from'] = {k:self.format_amount(v, fmt=fmt) for k,v in module['stake_from']}
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

    


    def min_stake(self, netuid: int = 0, fmt:str='j', **kwargs) -> int:
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
    

    def key2uid(self, key = None, netuid: int = 0, update=False, netuids=None , **kwargs):
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
        
    def emissions(self, netuid = 0, block=None, update=False, fmt = 'nanos', **kwargs):

        emissions = self.query_vector('Emission',  netuid=netuid, block=block, update=update, **kwargs)
        if netuid == 'all':
            for netuid, netuid_emissions in emissions.items():
                emissions[netuid] = [self.format_amount(e, fmt=fmt) for e in netuid_emissions]
        else:
            emissions = [self.format_amount(e, fmt=fmt) for e in emissions]
        
        return emissions
    
    def total_emissions(self, netuid = 0, block=None, update=False, fmt = 'nanos', **kwargs):

        emissions = self.query_vector('Emission',  netuid=netuid, block=block, update=update, **kwargs)
        if netuid == 'all':
            for netuid, netuid_emissions in emissions.items():
                emissions[netuid] = [self.format_amount(e, fmt=fmt) for e in netuid_emissions]
        else:
            emissions = [self.format_amount(e, fmt=fmt) for e in emissions]
        
        return sum(emissions)
    
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

    def check_storage(self, block_hash = None):
        return self.substrate.get_metadata_storage_functions( block_hash=block_hash)
 
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
    

    ##################
    #### profit share ####
    ##################

    def profit_shares(self, key=None, **kwargs) -> List[Dict[str, Union[str, int]]]:
        key = self.resolve_module_key(key)

        return self.query_map('ProfitShares',  **kwargs)

    def run_loop(self):
        while True:
            self.update_modules()
            self.subnet_params(netuid='all')
            self.stake_from(netuid='all')
            self.keys(netuid='all')


    def resolve_module_key(self, x, netuid=0, max_age=10):
        if not c.valid_ss58_address(x):
            name2key = self.name2key(netuid=netuid, max_age=max_age)
            x = name2key.get(x)
        assert c.valid_ss58_address(x), f"Module key {x} is not a valid ss58 address"
        return x

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
    

    @classmethod
    def get_feature(cls, feature='names', network='main', netuid=0, update=False, max_age=1000, **kwargs):
        return getattr(cls(network=network), feature)(netuid=netuid, update=update, max_age=max_age, **kwargs)
        
    



Subspace.run(__name__)
