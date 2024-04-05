
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
    whitelist = ['query', 
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
        """
        Initializes the object with the provided keyword arguments.

        :param kwargs: keyword arguments to configure the object
        :return: None
        """
        self.set_config(kwargs=kwargs)

    connection_mode = 'ws'

    def resolve_url(self, url:str = None, network:str = network, mode=None , **kwargs):
        """
        Resolve the URL based on the provided parameters.

        Parameters:
            url (str): The URL to be resolved.
            network (str): The network type.
            mode: The connection mode.
            **kwargs: Additional keyword arguments.
        
        Returns:
            str: The resolved URL.
        """
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
        """
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
                
        """
        
        network = network or self.config.network
        
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
        """
        A method to set up the network configuration.

        Parameters:
            network (str): The network to connect to. Default is 'main'.
            mode: The mode of connection. Default is 'http'.
            trials: Number of connection trials. Default is 10.
            url (str): The URL to connect to.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing network and URL information.
        """
               
        self.substrate = self.get_substrate(network=network, url=url, mode=mode, trials=trials , **kwargs)
        response =  {'network': self.network, 'url': self.url}
        c.print(response)
        
        return response

    def __repr__(self) -> str:
        """
        Return a string representation of the Subspace object.
        This function returns a formatted string showing the network attribute.
        Returns:
            str: A string containing the network attribute of the Subspace object.
        """
        return f'<Subspace: network={self.network}>'
    def __str__(self) -> str:
        """
        Returns a string representation of the Subspace object with network information.
        """

        return f'<Subspace: network={self.network}>'



    def wasm_file_path(self):
        """
        A method that generates the path to the WebAssembly (Wasm) file.
        Returns:
            str: The path to the Wasm file.
        """
        wasm_file_path = self.libpath + '/target/release/wbuild/node-subspace-runtime/node_subspace_runtime.compact.compressed.wasm'
        return wasm_file_path
    

    def my_stake_from(self, netuid = 0, block=None, update=False, network=network, fmt='j', max_age=1000 , **kwargs):
        """
        A function that calculates the total stake from a given netuid, considering various parameters.
        
        Parameters:
            netuid (int): The unique identifier for the net.
            block (str): The block to consider.
            update (bool): Whether to update the stake information.
            network (str): The network to operate on.
            fmt (str): The format of the stake.
            max_age (int): The maximum age to consider.
            **kwargs: Additional keyword arguments.
        
        Returns:
            dict: A dictionary containing the total stake information.
        """
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
        """
        Retrieve the delegation fee for a specific netuid, block, and network. 

        Args:
            netuid (int): The unique identifier for the network.
            block (object): The block for which the delegation fee is being retrieved.
            network (object): The network for which the delegation fee is being retrieved.
            update (bool): Whether to update the delegation fee information.
            fmt (str): The format of the returned delegation fee.

        Returns:
            delegation_fee: The delegation fee for the specified parameters.
        """
        delegation_fee = self.query_map('DelegationFee', netuid=netuid, block=block ,update=update, network=network)
        return delegation_fee

    def stake_to(self, netuid = 0, network=network, block=None, update=False, fmt='nano',**kwargs):
        """
        A function that queries 'StakeTo' information based on the provided parameters.
        
        Parameters:
            netuid (int): The netuid to query the information for.
            network (str): The network to query the information from.
            block (None or int): The block to query the information for.
            update (bool): Whether to update the information.
            fmt (str): The format for the amount.
            **kwargs: Additional keyword arguments for the query.
        
        Returns:
            dict: A dictionary containing the queried 'StakeTo' information with formatted amounts.
        """
        stake_to = self.query_map('StakeTo', netuid=netuid, block=block, update=update, network=network, **kwargs)
        format_tuples = lambda x: [[_k, self.format_amount(_v, fmt=fmt)] for _k,_v in x]
        if netuid == 'all':
            stake_to = {netuid: {k: format_tuples(v) for k,v in stake_to[netuid].items()} for netuid in stake_to}
        else:
            stake_to = {k: format_tuples(v) for k,v in stake_to.items()}
    
        return stake_to
    
    def my_stake_to(self, netuid = 0,
                     block=None, 
                    update=False, 
                    names = False,
                    network='main', fmt='j'):
        """
        A function that calculates the total stake for each staker_address based on the given parameters.

        Parameters:
            netuid (int): The netuid for which the stake is being calculated.
            block (str): The block to consider for the stake calculation.
            update (bool): Flag indicating whether to update the stake information.
            names (bool): Flag indicating whether to include names in the calculation.
            network (str): The network to consider for the stake calculation.
            fmt (str): The format of the stake information.

        Returns:
            dict: A dictionary containing the total stake for each staker_address.
        """
        stake_to = self.stake_to(netuid=netuid, block=block, update=update, network=network, fmt=fmt)
        address2key = c.address2key()
        stake_to_total = {}
        if netuid == 'all':
            stake_to_dict = stake_to
           
            netuid2subnet = self.netuid2subnet()
            for staker_address in address2key.keys():
                stake_to_total[staker_address] = {}
                for netuid, stake_to in stake_to_dict.items(): 
                    if names:
                        netuid = netuid2subnet[netuid]              
                    if staker_address in stake_to:
                        stake_to_total[staker_address][netuid] = sum([v[1] for v in stake_to[staker_address]])
                if len(stake_to_total[staker_address]) == 0:
                    del stake_to_total[staker_address]
        else:
            for staker_address in address2key.keys():
                if staker_address in stake_to:
                    stake_to_total[staker_address] = stake_to_total.get(staker_address, 0) + sum([v[1] for v in stake_to[staker_address]])
        return stake_to_total
    
    def min_burn(self,  network='main', block=None, update=False, fmt='j'):
        """
        A function to calculate the minimum burn with optional parameters for network, block, update, and format.
        Returns the formatted minimum burn value.
        """
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
        A method to query data from a specified network and module.
        
        Parameters:
            name (str): The name of the data to query.
            params (list): Additional parameters for the query (default None).
            module (str): The module to query from (default 'SubspaceModule').
            block: The block to query (default None).
            netuid: The unique identifier for the network (default None).
            network (str): The network to query from (default network).
            save (bool): Flag to save the query result (default True).
            max_age (int): Maximum age of the cached data (default 1000).
            mode (str): The mode of query (default 'http').
            update (bool): Flag to update the query result (default False).
        
        Returns:
            The queried value from the specified network and module.
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

        value = self.get(path, None, max_age=max_age)
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
        A function to query a constant value from a Substrate module.
        
        Args:
            constant_name (str): The name of the constant value to query.
            module_name (str, optional): The name of the module where the constant value resides. Defaults to 'SubspaceModule'.
            block (Optional[int], optional): The block number to query the constant value at. Defaults to None.
            network (str): The network to query the constant value from.
        
        Returns:
            Optional[object]: The queried constant value.
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
                  mode = 'ws',
                  **kwargs
                  ) -> Optional[object]:
        """
        A function to query a map with various parameters and options, and return the resulting map.
        
        Parameters:
            name: str - the name of the map to query
            params: list - additional parameters for the query (default: None)
            block: Optional[int] - block number for the query (default: None)
            network: str - the network to query on (default: 'main')
            netuid - unique identifier for the network (default: None)
            page_size: int - the size of the page for the query (default: 1000)
            max_results: int - the maximum number of results for the query (default: 100000)
            module: str - the module to query (default: 'SubspaceModule')
            update: bool - flag to indicate if the query should be updated (default: True)
            max_age: str - maximum age in seconds for the query (default: 1000)
            mode: str - the mode for the query (default: 'ws')
            **kwargs - additional keyword arguments for the query
        
        Returns:
            Optional[object] - the resulting map from the query
        """
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
            value = self.get(sorted(paths, reverse=True)[-1],None , max_age=max_age)
        else:
            value = None

        if value == None:
            block = block or self.block
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
            progress_bar = c.progress(qmap, desc=f'Querying {name} map with params {params}')
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
            for p in paths:
                c.rm(p)
            self.put(path, new_qmap)
        
        else: 
            new_qmap = value

        def convert_dict_k_digit_to_int(d):
            """
            Convert dictionary keys from strings to integers and sort the dictionary by key.
            Params:
                d (dict): The input dictionary
            Returns:
                dict: The dictionary with keys converted to integers and sorted
            """
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
        """
        Get the runtime version of the substrate with the given network.

        :param network: The network to get the runtime version from (default is 'main').
        :return: The runtime version of the substrate.
        """
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
        """
        A function that returns the sum of a query if netuid is 'all', otherwise returns a query result.
        
        Parameters:
            netuid (int): The unique identifier for the network.
            network (str): The network to query.
            block (Optional[int]): The block to query.
            update (bool): A flag to indicate whether to update the query.
            **kwargs: Additional keyword arguments for the query.
        
        Returns:
            int: The result of the query operation.
        """
        if netuid == 'all':
            return sum(self.query_map('N', block=block , update=update, network=network, **kwargs))
        else:
            return self.query( 'N', params=[netuid], block=block , update=update, network=network, **kwargs)

    ##########################
    #### Account functions ###
    
    """ Returns network Tempo hyper parameter """
    def stakes(self, netuid: int = 0, block: Optional[int] = None, fmt:str='nano', max_age = 100,network=None, update=False, **kwargs) -> int:
        """
        Generate the stakes for a given netuid. 
        
        Parameters:
            netuid (int): The unique identifier for the stakes. Default is 0.
            block (Optional[int]): The block number. Default is None.
            fmt (str): The format for the stakes. Default is 'nano'.
            max_age: The maximum age for the stakes. Default is 100.
            network: The network for the stakes. Default is None.
            update (bool): Whether to update the stakes. Default is False.
            **kwargs: Additional keyword arguments.
        
        Returns:
            int: The formatted stakes.
        """
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
        """
        Resolve the SS58 address for the given key.

        Args:
            key (str): The key for which the SS58 address needs to be resolved.
            network (str, optional): The network for which the address needs to be resolved. Defaults to 'main'.
            netuid (int, optional): The unique identifier for the network. Defaults to 0.
            resolve_name (bool, optional): Flag to indicate whether to resolve the name. Defaults to True.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The resolved SS58 address of the key.
        """
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
        """
        Generate a dictionary mapping each SubNetwork unique ID to a list of modules associated with it.

        Parameters:
            network (str): The network to operate on.
            **kwargs: Additional keyword arguments to pass to my_modules.

        Returns:
            dict: A dictionary mapping SubNetwork unique IDs to lists of modules.
        """
        subnet2modules = {}
        self.resolve_network(network)

        for netuid in self.netuids():
            c.print(f'Getting modules for SubNetwork {netuid}')
            subnet2modules[netuid] = self.my_modules(netuid=netuid, **kwargs)

        return subnet2modules
    
    def module2netuids(self, network:str='main', **kwargs):
        """
        Generate the mapping of module names to a list of network identifiers based on the provided network. 

        :param network: The network for which the module to network identifiers mapping is generated (default is 'main').
        :param kwargs: Additional keyword arguments.
        :return: A dictionary mapping module names to a list of network identifiers.
        """
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
        """
        A class method to convert a value from Nano to the token unit.
        """
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
        """
        A class method to format an amount based on the specified format. Supports various formats and rounding options.
        Parameters:
            x: the amount to be formatted
            fmt: the format to use for formatting (default is 'nano')
            decimals: the number of decimals to round to (default is None)
            format: alias for fmt
            features: additional features (default is None)
            **kwargs: additional keyword arguments
        Returns:
            the formatted amount
        """
        fmt = format or fmt # format is an alias for fmt

        if fmt in ['token', 'unit', 'j', 'J']:
            x = x / 10**9
        
        if decimals != None:
            x = c.round_decimals(x, decimals=decimals)
  

        return x
    
    def get_stake( self, key_ss58: str, block: Optional[int] = None, netuid:int = None , fmt='j', update=True ) -> Optional['Balance']:
        """
        Get the stake amount for a given key in SS58 format.

        Parameters:
            key_ss58 (str): The SS58 key for the stake.
            block (Optional[int]): The block number to query the stake amount at. Defaults to None.
            netuid (int): The network UID to query the stake amount from.
            fmt (str): The format of the amount to return. Defaults to 'j'.
            update (bool): Whether to update the stake amount before returning. Defaults to True.

        Returns:
            Optional['Balance']: The stake amount in the specified format.
        """
        
        key_ss58 = self.resolve_key_ss58( key_ss58)
        netuid = self.resolve_netuid( netuid )
        stake = self.query( 'Stake',params=[netuid, key_ss58], block=block , update=update)
        return self.format_amount(stake, fmt=fmt)

    

    def all_key_info(self, netuid='all', timeout=10, update=False, **kwargs):
        """
        A function to retrieve all key information.
        
        Parameters:
            netuid (str): The network UID to retrieve key information for. Default is 'all'.
            timeout (int): The timeout value for the request. Default is 10.
            update (bool): Flag to indicate if the information should be updated. Default is False.
            **kwargs: Additional keyword arguments that can be passed to the function.
        
        Returns:
            None
        """
        my_keys = c.my_keys()


    def key_info(self, key:str = None, netuid='all', timeout=10, update=False, **kwargs):
        """
        A function that retrieves key information.

        Parameters:
            key (str): The key to retrieve information for. Default is None.
            netuid (str): The network ID. Default is 'all'.
            timeout (int): Timeout value. Default is 10.
            update (bool): Whether to update the information. Default is False.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing key information.
        """
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
        """
        Calculate the total stake for a given key and module_key, at a specific block, 
        with an optional timeout and network. 
        Optionally update the stake and set a maximum age for the data. 
        Return the total stake as an optional Balance object.
        
        Args:
            key (str): The key to calculate the total stake for.
            module_key: The module key for the stake calculation.
            block (Optional[int]): The specific block for the stake calculation.
            timeout (int): The timeout for the stake calculation, default is 20.
            names (bool): Whether to include names in the stake calculation, default is False.
            fmt (str): The format for the stake calculation, default is 'j'.
            network: The network for the stake calculation.
            update (bool): Whether to update the stake, default is False.
            max_age (int): The maximum age for the stake data, default is 1000.
            **kwargs: Additional keyword arguments for the stake calculation.
            
        Returns:
            Optional['Balance']: The total stake as an optional Balance object.
        """
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
        """
        Calculate staking rewards based on the difference in total stake before and after a certain block.
        
        Parameters:
            key (str): The key for the staking rewards.
            module_key: The module key for the staking rewards.
            block (int): The block number to calculate rewards at.
            timeout (int): The timeout for the calculation.
            period (int): The period over which to calculate the rewards.
            names (bool): Flag to include names in the calculation.
            fmt (str): The format for the calculation.
            network: The network to calculate rewards on.
            update (bool): Flag to update the calculation.
            max_age (int): The maximum age for the calculation.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Balance: The staking rewards as a Balance object.
        """

        block = int(block or self.block)
        block_yesterday = int(block - period)
        day_before_stake = self.my_total_stake_to(key=key, module_key=module_key, block=block_yesterday, timeout=timeout, names=names, fmt=fmt, network=network, update=update, max_age=max_age, **kwargs)
        day_after_stake = self.my_total_stake_to(key=key, module_key=module_key, block=block, timeout=timeout, names=names, fmt=fmt, network=network, update=update, max_age=max_age, **kwargs) 
        return (day_after_stake - day_before_stake)
    

    def clear_query_history(self):
        """
        Clears the query history by removing the 'query' key.
        """
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
        """
        Generate the function comment for the given function body in a markdown code block with the correct language syntax.
        
        :param key: str = None
        :param module_key: Any
        :param block: Optional[int] = None
        :param timeout: int = 20
        :param names: bool = False
        :param fmt: str = 'j'
        :param network: Any
        :param update: bool = False
        :param max_age: int = 1000
        :param **kwargs: Any
        
        :return: Optional['Balance']
        """
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
                        max_age = 1000,
                         **kwargs) -> Optional['Balance']:
        """
        A function that retrieves stake information for a given key on a specific network.
        
        Parameters:
            key (str): The key for which stake information is requested.
            module_key: The module key.
            netuid (int): The unique identifier for the network.
            block (int, optional): The block number.
            names (bool): Flag to indicate whether to return names.
            fmt (str): The format of the data.
            network: The network to query.
            update (bool): Flag to indicate whether to update the information.
            max_age (int): The maximum age of the data to consider.
            **kwargs: Additional keyword arguments.
        
        Returns:
            Optional['Balance']: The stake information for the given key.
        """
        

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
        stake_to = self.query( 'StakeTo', params=[netuid, key_address], block=block, update=update, network=network)
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
        """
        A function to calculate the total stake amount based on the provided parameters.

        Parameters:
            key (str): The key parameter for stake calculation.
            module_key: The key parameter for the specific module.
            netuid (int): The unique identifier for the network. Default is 'all'.
            block (int): The block number for stake calculation. Default is None.
            timeout: The timeout value for the function. Default is 20.
            names: A boolean flag to indicate whether to include names in the calculation. Default is False.
            fmt (str): The format for the stake calculation. Default is 'j'.
            network: The network parameter for the stake calculation.
            update: A boolean flag to indicate whether to update the stake amount. Default is True.
            **kwargs: Additional keyword arguments.

        Returns:
            Optional['Balance']: The total stake amount calculated based on the provided parameters.
        """
        stake_to = self.get_stake_to(key=key, module_key=module_key, netuid=netuid, block=block, timeout=timeout, names=names, fmt=fmt, network=network, update=update, **kwargs)
        if netuid == 'all':
            return sum([sum(list(x.values())) for x in stake_to])
        else:
            return sum(stake_to.values())
    
        return stake_to
    
    get_staketo = get_stake_to
    
    def get_value(self, key=None):
        """
        A function that calculates the total value associated with a given key.
        
        Parameters:
            key (str): The key to look up the value for.
            
        Returns:
            int: The total value associated with the key.
        """
        key = self.resolve_key_ss58(key)
        value = self.get_balance(key)
        netuids = self.netuids()
        for netuid in netuids:
            stake_to = self.get_stake_to(key, netuid=netuid)
            value += sum(stake_to.values())
        return value    



    def get_stake_from( self, key: str, from_key=None, block: Optional[int] = None, netuid:int = None, fmt='j', update=True  ) -> Optional['Balance']:
        """
        A method to get the stake from a specified key.
        
        :param key: The key for which stake is to be retrieved.
        :param from_key: The key to retrieve stake from (default is None).
        :param block: Optional parameter to specify a block number (default is None).
        :param netuid: The network ID (default is None).
        :param fmt: The format to return the stake amount in (default is 'j').
        :param update: Whether to update the stake information (default is True).
        
        :return: Optional 'Balance' object representing the stake from the specified key.
        """
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
        """
        This is a property function that returns the block for a given network.
        
        :param network: A string indicating the network for which the block is requested.
        :type network: str
        :return: An integer representing the block for the specified network.
        :rtype: int
        """
        return self.get_block(network=network)


    
   
    @classmethod
    def archived_blocks(cls, network:str=network, reverse:bool = True) -> List[int]:
        """
        returns a list of archived blocks
        """
        # returns a list of archived blocks 
        
        blocks =  [f.split('.B')[-1].split('.json')[0] for f in cls.glob(f'archive/{network}/state.B*')]
        blocks = [int(b) for b in blocks]
        sorted_blocks = sorted(blocks, reverse=reverse)
        return sorted_blocks

    @classmethod
    def oldest_archive_path(cls, network:str=network) -> str:
        """
        Return the path of the oldest archive block for the specified network.

        Args:
            cls: The class object.
            network (str): The network for which the oldest archive path is needed.

        Returns:
            str: The path of the oldest archive block.
        """
        oldest_archive_block = cls.oldest_archive_block(network=network)
        assert oldest_archive_block != None, f"No archives found for network {network}"
        return cls.resolve_path(f'state_dict/{network}/state.B{oldest_archive_block}.json')
    @classmethod
    def newest_archive_block(cls, network:str=network) -> str:
        """
        Return the newest archive block for a given network.

        Parameters:
            network (str): The network for which to retrieve the newest archive block.

        Returns:
            str: The newest archive block.
        """
        blocks = cls.archived_blocks(network=network, reverse=True)
        return blocks[0]
    @classmethod
    def newest_archive_path(cls, network:str=network) -> str:
        """
        Returns the path to the newest archive file for the specified network.

        Args:
            network (str): The name of the network.

        Returns:
            str: The path to the newest archive file.
        """
        oldest_archive_block = cls.newest_archive_block(network=network)
        return cls.resolve_path(f'archive/{network}/state.B{oldest_archive_block}.json')
    @classmethod
    def oldest_archive_block(cls, network:str=network) -> str:
        """
        A method to retrieve the oldest archive block for a given network.

        Parameters:
            cls: the class object
            network (str): the network for which to retrieve the oldest archive block

        Returns:
            str: the oldest archive block for the specified network
        """
        blocks = cls.archived_blocks(network=network, reverse=True)
        if len(blocks) == 0:
            return None
        return blocks[-1]

    @classmethod
    def ls_archives(cls, network=network):
        """
        A class method to list all archives for a given network. Defaults to the class network if none provided. Returns a list of files.
        """
        if network == None:
            network = cls.network 
        return [f for f in cls.ls(f'state_dict') if os.path.basename(f).startswith(network)]

    
    @classmethod
    def block2archive(cls, network=network):
        """
        Generate a mapping of block number to archive path for a specific network.

        :param network: The network for which to generate the mapping (default is the network of the class).
        :return: A dictionary mapping block numbers to archive paths.
        """
        paths = cls.ls_archives(network=network)

        block2archive = {int(p.split('-')[-1].split('-time')[0]):p for p in paths if p.endswith('.json') and f'{network}.block-' in p}
        return block2archive

    def latest_archive_block(self, network=network) -> int:
        """
        Generate the latest archive block number for a given network.

        :param network: The network for which to retrieve the latest archive block (default is set to the current network).
        :return: The latest archive block number as an integer.
        """
        latest_archive_path = self.latest_archive_path(network=network)
        block = int(latest_archive_path.split(f'.block-')[-1].split('-time')[0])
        return block


        

    @classmethod
    def time2archive(cls, network=network):
        """
        Generate a dictionary mapping block times to their corresponding archive paths.

        :param network: The network to retrieve archives from.
        :return: A dictionary where keys are block times and values are paths to archive files.
        """
        paths = cls.ls_archives(network=network)

        block2archive = {int(p.split('time-')[-1].split('.json')[0]):p for p in paths if p.endswith('.json') and f'time-' in p}
        return block2archive

    @classmethod
    def datetime2archive(cls,search=None, network=network):
        """
        A class method to convert time data to archive data and sort it by datetime.
        
        Parameters:
            search (str): A string to search for in the keys of the resulting dictionary.
            network (str): The network to use for the conversion.
        
        Returns:
            dict: A dictionary mapping datetime strings to archive data, sorted by datetime.
        """
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
        """
        A class method to retrieve the latest archive path for a given network.

        Args:
            cls: The class itself.
            network: The network for which the latest archive path is to be retrieved.

        Returns:
            The latest archive path for the specified network, or None if the latest archive time is not available.
        """
        latest_archive_time = cls.latest_archive_time(network=network)
    
        if latest_archive_time == None:
            return None
        time2archive = cls.time2archive(network=network)
        return time2archive[latest_archive_time]

    @classmethod
    def latest_archive_time(cls, network=network):
        """
        A method to retrieve the latest archive time for a given network.

        :param cls: The class reference.
        :param network: The network for which the latest archive time is needed.
        :return: The latest archive time if available, otherwise None.
        """
        time2archive = cls.time2archive(network=network)
        if len(time2archive) == 0:
            return None
        latest_time = max(time2archive.keys())
        return latest_time

    @classmethod
    def lag(cls, network:str = network):
        """
        A class method to calculate the time lag between the current timestamp and the latest archive time for a given network.
        
        Parameters:
            network (str): The network for which to calculate the time lag. Defaults to the value of the 'network' parameter.
        
        Returns:
            int: The time difference in seconds between the current timestamp and the latest archive time for the specified network.
        """
        return c.timestamp() - cls.latest_archive_time(network=network) 

    @classmethod
    def latest_archive(cls, network=network):
        """
        A class method to retrieve the latest archive for a given network.
        
        :param network: the network for which to retrieve the latest archive
        :return: a dictionary containing the contents of the latest archive
        """
        path = cls.latest_archive_path(network=network)
        if path == None:
            return {}
        return cls.get(path, {})
    
 


    def light_sync(self, network=None, remote:bool=True, netuids=None, local:bool=True, save:bool=True, timeout=20, **kwargs):
        """
        A method to synchronize the light nodes, fetching stake, namespace, and weight information.
        
        Args:
            network (str): The network to synchronize with.
            remote (bool): Flag to indicate if the synchronization should be done remotely.
            netuids (list): List of netuids to synchronize.
            local (bool): Flag to indicate if the synchronization should be done locally.
            save (bool): Flag to indicate if the synchronization results should be saved.
            timeout (int): The maximum time to wait for the synchronization to complete.
            **kwargs: Additional keyword arguments.
        
        Returns:
            dict: A dictionary with 'success' flag and 'block' information.
        """
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
        """
        A function to loop through data with specified intervals.

        Parameters:
            intervals (dict): A dictionary with keys 'light' and 'full' indicating the intervals for different operations.
            network (None): Optional, a network parameter.
            remote (bool): Optional, a boolean flag to specify if the function should be run remotely.

        Returns:
            None
        """
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
        """
        A function that checks if a given subnet exists within a list of subnets.
        
        Parameters:
            subnet (str): The subnet to check for existence.
            network (optional): The network to search subnets in. Defaults to None.
        
        Returns:
            bool: True if the subnet exists in the list of subnets, False otherwise.
        """
        subnets = self.subnets(network=network)
        return bool(subnet in subnets)

    def subnet_emission(self, netuid:str = 0, network=None, block=None, update=False, **kwargs):
        """
        A function to calculate the emission for a subnet based on the given parameters.
        
        Parameters:
            netuid (str): The unique identifier for the network.
            network: The network information.
            block: The block information.
            update (bool): A flag to indicate if the emissions need to be updated.
            **kwargs: Additional keyword arguments.
        
        Returns:
            int: The total sum of emissions for the subnet.
        """
        emissions = self.emission(block=block, update=update, network=network, netuid=netuid, **kwargs)
        if isinstance(emissions[0], list):
            emissions = [sum(e) for e in emissions]
        return sum(emissions)
    
    
    def unit_emission(self, network=None, block=None, update=False, **kwargs):
        """
        Function to query the unit emission constant from the specified network and block.
        
        :param network: the network to query the constant from
        :param block: the block to query the constant from
        :param update: flag indicating whether to update the constant
        :param kwargs: additional keyword arguments
        :return: the queried unit emission constant
        """
        return self.query_constant( "UnitEmission", block=block,network=network)

    def subnet_state(self,  netuid='all',  network='main', block=None, update=False, fmt='j', **kwargs):

        subnet_state = {
        """
        Retrieves the state of a subnet with the given parameters.

        :param netuid: str, optional, the unique identifier of the subnet (default is 'all')
        :param network: str, optional, the network to which the subnet belongs (default is 'main')
        :param block: None or str, optional, the block within the subnet to retrieve (default is None)
        :param update: bool, optional, whether to update the state information (default is False)
        :param fmt: str, optional, the format of the returned state information (default is 'j')
        :param kwargs: dict, additional keyword arguments
        :return: dict, a dictionary containing the parameters and modules of the subnet
        """
            'params': self.subnet_params(netuid=netuid, network=network, block=block, update=update, fmt=fmt, **kwargs),
            'modules': self.modules(netuid=netuid, network=network, block=block, update=update, fmt=fmt, **kwargs),
        }
        return subnet_state


    def total_stake(self, network=network, block: Optional[int] = None, netuid:int='all', fmt='j', update=False) -> 'Balance':
        """
        A function to calculate the total stake based on network, block, netuid, format, and update settings.
        Parameters:
            network: The network for which to calculate the stake.
            block: The block number to consider for stake calculation.
            netuid: The unique identifier for the network stake.
            fmt: The format of the stake calculation.
            update: A boolean flag indicating whether to update the stake.
        Returns:
            Balance: The total stake calculated based on the provided parameters.
        """
        return sum([sum([sum(list(map(lambda x:x[1], v))) for v in vv.values()]) for vv in self.stake_to(network=network, block=block,update=update, netuid='all')])

    def total_balance(self, network=network, block: Optional[int] = None, fmt='j', update=False) -> 'Balance':
        return sum(list(self.balances(network=network, block=block, fmt=fmt).values()), update=update)

        """
        Calculate the total balance for a given network and block.

        Args:
            network: The network for which the balance is to be calculated.
            block: The block at which the balance is to be calculated.
            fmt: The format of the balance calculation.
            update: Whether to update the balance.

        Returns:
            Balance: The total balance calculated.
        """
    def mcap(self, network=network, block: Optional[int] = None, fmt='j', update=False) -> 'Balance':
        """
        A function that calculates the total balance and total stake, then formats and returns the sum of the two.
        
        :param network: The network for which the balances and stakes are calculated.
        :param block: The block number at which the balances and stakes are calculated.
        :param fmt: The format type for the returned sum.
        :param update: A flag to indicate if the balances and stakes should be updated.
        :return: A Balance object representing the formatted sum of total stake and total balance.
        """
        total_balance = self.total_balance(network=network, block=block, update=update)
        total_stake = self.total_stake(network=network, block=block, update=update)
        return self.format_amount(total_stake + total_balance, fmt=fmt)
    
    market_cap = total_supply = mcap  
            
        
    @classmethod
    def feature2storage(cls, feature:str):
        """
        Convert a given feature string to storage format by capitalizing the first letter of each word and removing underscores. 

        Args:
            feature (str): The input feature string.

        Returns:
            str: The converted storage format string.
        """
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
    
    def my_subnets(self, key=None, block=None, update=False, **kwargs):
        """
        Generate the mapping of subnet names to network unique ids based on subnet parameters.

        :param key: Key parameter for the function (default is None)
        :param block: Block parameter for the function (default is None)
        :param update: Update parameter for the function (default is False)
        :param kwargs: Additional keyword arguments
        :return: Dictionary mapping subnet names to network unique ids
        """
        address2key = c.address2key()
        subnet_params_list = self.subnet_params(block=block, update=update, netuid='all', **kwargs)
        subnet2netuid = {}
        for netuid, subnet_params in enumerate(subnet_params_list):
            if subnet_params['founder'] in address2key:
                subnet2netuid[subnet_params['name']] = netuid
        return subnet2netuid
                
    

    
    def subnet_params(self, 
                    netuid=0,
                    network = 'main',
                    block : Optional[int] = None,
                    update = False,
                    timeout = 30,
                    max_age = 1000,
                    fmt:str='j', 
                    rows:bool = True,
                    value_features = ['min_stake', 'max_stake']
                    ) -> list:        
        """
        A function to retrieve subnet parameters based on the provided network and netuid. 
        The function allows for optional parameters such as the block, update, timeout, max_age, fmt, rows, and value_features. 
        It returns a list containing the subnet parameters after resolving the network and netuid and making the necessary queries. 
        """
        network = self.resolve_network(network)  
        netuid = self.resolve_netuid(netuid)
        path = f'query/{network}/SubspaceModule.SubnetParams'          
        subnet_params = self.get(path, None, max_age=max_age)


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
                'max_weight_age': 'MaxWeightAge',
                'name': 'SubnetNames',
                'max_stake': 'MaxStake',
            }

        if subnet_params == None:
            features = list(name2feature.keys())
            block = block or self.block
            subnet_params = {}
            n = len(features)
            progress = c.tqdm(total=n, desc=f'Querying {n} features')
            futures = []
            while len(features) > 0 :
                c.print(f'Querying {len(features)} features')
                feature2future = {}
                for f in features:
                    if f in subnet_params:
                        continue
                    feature2future[f] = c.submit(self.query_map, dict(name=name2feature[f], 
                                                                    update=update, 
                                                                    block=block))
                future2feature = {v:k for k,v in feature2future.items()}
                futures = list(feature2future.values())
                for f in c.as_completed(futures, timeout=timeout):
                    feature = future2feature[f]
                    result = f.result()
                    if c.is_error(result):
                        continue
                    subnet_params[feature] = result
                    features.remove(feature)
                    progress.update(1)

            self.put(path, subnet_params)
        subnet_params = {f: {int(k):v for k,v in subnet_params[f].items()} for f in subnet_params}
        if netuid == 'all':
            num_subnets = len(subnet_params['tempo'])
            subnets_param_rows = []
            for netuid in range(num_subnets):
                subnets_param_row = {}
                for k in subnet_params.keys():
                    subnets_param_row[k] = subnet_params[k][netuid]
                    if k in value_features:
                        subnets_param_row[k] = self.format_amount(subnets_param_row[k], fmt=fmt)
                subnets_param_rows.append(subnets_param_row)
            subnet_params = subnets_param_rows    
        else: 
            for k,v in subnet_params.items():
                subnet_params[k] = v.get(netuid, None)
                if k in value_features:
                    subnet_params[k] = self.format_amount(subnet_params[k], fmt=fmt)
                            
        return subnet_params
    
    subnet = subnet_params


    def subnet2params( self, network: int = None, block: Optional[int] = None ) -> Optional[float]:
        """
        Generate subnet parameters for a given network and block.

        Parameters:
            network (int): The network ID.
            block (Optional[int]): The block ID.

        Returns:
            Optional[float]: A dictionary mapping subnets to their corresponding parameters.
        """
        netuids = self.netuids(network=network)
        subnet2params = {}
        netuid2subnet = self.netuid2subnet()
        for netuid in netuids:
            subnet = netuid2subnet[netuid]
            subnet2params[subnet] = self.subnet_params(netuid=netuid, block=block)
        return subnet2params
    
    def subnet2emission( self, network: int = None, block: Optional[int] = None ) -> Optional[float]:
        """
        Function to calculate emission from subnet, with optional network and block parameters.
        Returns optional float value.
        """
        subnet2emission = self.subnet2params(network=network, block=block)
        return subnet2emission

    

    def subnet2state( self, network: int = None, block: Optional[int] = None ) -> Optional[float]:
        """
        A function that converts subnet information to state information.

        Parameters:
            network (int): The network information.
            block (Optional[int]): The block information.

        Returns:
            Optional[float]: The state information corresponding to the subnet information.
        """
        subnet2state = self.subnet2params(network=network, block=block)

        return subnet2state
            

    def is_registered( self, key: str, netuid: int = None, block: Optional[int] = None) -> bool:
        """
        Check if a given key is registered, resolving the netuid if necessary.

        Parameters:
            key (str): The key to check for registration.
            netuid (int, optional): The network uid. Defaults to None.
            block (int, optional): The block to check. Defaults to None.

        Returns:
            bool: True if the key is registered, False otherwise.
        """
        netuid = self.resolve_netuid( netuid )
        if not c.valid_ss58_address(key):
            name2key = self.name2key(netuid=netuid)
            if key in name2key:
                key = name2key[key]
        assert c.valid_ss58_address(key), f"Invalid key {key}"
        is_reged =  bool(self.query('Uids', block=block, params=[ netuid, key ]))
        return is_reged
    is_reg = is_registered

    def get_uid( self, key: str, netuid: int = 0, block: Optional[int] = None, update=False, **kwargs) -> int:
        """
        Get the unique identifier for a given key and network identifier.
        
        Args:
            key (str): The key for which the unique identifier is requested.
            netuid (int, optional): The network identifier. Defaults to 0.
            block (int, optional): The block number. Defaults to None.
            update (bool): Whether to update the unique identifier. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            int: The unique identifier.
        """
        return self.query( 'Uids', block=block, params=[ netuid, key ] , update=update, **kwargs)  



    def register_subnets( self, *subnets, module='vali', **kwargs ) -> Optional['Balance']:
        """
        Register subnets and return a list of responses.

        Args:
            *subnets: Variable length argument list of subnets to register.
            module (str): The module to use for registration. Defaults to 'vali'.
            **kwargs: Additional keyword arguments for registration.

        Returns:
            Optional['Balance']: A list of responses for each registered subnet.
        """
        if len(subnets) == 1:
            subnets = subnets[0]
        subnets = list(subnets)
        assert isinstance(subnets, list), f"Subnets must be a list. Got {subnets}"
        
        responses = []
        for subnet in subnets:
            response = c.register(module=module, tag=subnet, subnet=subnet , **kwargs)
            responses.append(response)

        return responses
        

    def total_emission( self, netuid: int = 0, block: Optional[int] = None, fmt:str = 'j', **kwargs ) -> Optional[float]:
        """
        Calculate the total emission based on the given parameters.

        Parameters:
            netuid (int): The unique identifier for the net.
            block (Optional[int]): The block number.
            fmt (str): The format for the output.
            **kwargs: Additional keyword arguments.

        Returns:
            Optional[float]: The total emission amount.
        """
        total_emission =  sum(self.emission(netuid=netuid, block=block, **kwargs))
        return self.format_amount(total_emission, fmt=fmt)


    def regblock(self, netuid: int = 0, block: Optional[int] = None, network=network, update=False ) -> Optional[float]:
        """
        A function that retrieves a registration block based on the provided parameters.

        Parameters:
            netuid (int): The unique identifier of the network.
            block (Optional[int]): The block number to retrieve.
            network: The network to retrieve the block from.
            update (bool): A flag indicating whether to update the block.

        Returns:
            Optional[float]: The registration block corresponding to the given parameters.
        """
        regblock =  self.query_map('RegistrationBlock',block=block, update=update )
        if isinstance(netuid, int):
            regblock = regblock[netuid]
        return regblock

    def age(self, netuid: int = None) -> Optional[float]:
        """
        A function that calculates the age based on the difference between block and regblock values.

        Parameters:
            netuid (int): The user ID to resolve netuid. Defaults to None.

        Returns:
            Optional[float]: A dictionary containing the calculated age values.
        """
        netuid = self.resolve_netuid( netuid )
        regblock = self.regblock(netuid=netuid)
        block = self.block
        age = {}
        for k,v in regblock.items():
            age[k] = block - v
        return age
    
    
     
    def global_params(self, 
                      network: str = 'main',
                         timeout = 2,
                         update = False,
                         block : Optional[int] = None,
                         fmt = 'nanos',
                         max_age = 10000,
                          ) -> Optional[float]:
        """
        A function to retrieve global parameters for a given network.

        Parameters:
            network (str): The network to retrieve the parameters for.
            timeout (int): The timeout for the function call.
            update (bool): Whether to update the global parameters.
            block (Optional[int]): The block number to query.
            fmt (str): The format for the returned global parameters.
            max_age (int): The maximum age for cached global parameters.

        Returns:
            Optional[float]: The global parameters for the specified network.
        """
        
        path = f'cache/{network}.global_params.json'
        global_params = None if update else self.get(path, None, max_age=max_age)
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
        for k in ['min_stake', 'min_burn', 'unit_emission', 'min_weight_stake']:
            global_params[k] = self.format_amount(global_params[k], fmt=fmt)
        return global_params



    def balance(self,
                 key: str = None ,
                 block: int = None,
                 fmt='j',
                 network=None,
                 update=True) -> Optional['Balance']:
        """
        This function balances the specified key and returns the formatted amount. 

        Args:
            key (str): The key to be balanced. Defaults to None.
            block (int): The block number. Defaults to None.
            fmt (str): The format of the amount. Defaults to 'j'.
            network: The network to be used. Defaults to None.
            update (bool): Whether to update the balance. Defaults to True.

        Returns:
            Optional['Balance']: The formatted balance amount, or None if the key is not found.
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
        """
        Get account information from the substrate based on the provided key and network.
        
        :param key: The key used to identify the account.
        :param network: The network from which to fetch the account information.
        :param update: A boolean flag indicating whether to update the account information.
        :return: The account information retrieved from the substrate.
        """
        self.resolve_network(network)
        key = self.resolve_key_ss58(key)
        account = self.substrate.query(
            module='System',
            storage_function='Account',
            params=[key],
        )
        return account
    
    def accounts(self, key = None, network=None, update=True, block=None):
        """
        Retrieves accounts from the specified network, with optional key, network, update, and block parameters.
        Returns the retrieved accounts.
        """
        self.resolve_network(network)
        key = self.resolve_key_ss58(key)
        accounts = self.query_map(
            module='System',
            name='Account',
            update=update,
            block = block,
        )
        return accounts
    
    def balances(self,fmt:str = 'n', network:str = network, block: int = None, n = None, update=False , **kwargs) -> Dict[str, 'Balance']:
        """
        Retrieve balances for all accounts in the given network.

        Args:
            fmt (str): The format to display the balances in. Defaults to 'n'.
            network (str): The network to retrieve balances for.
            block (int): The block to retrieve balances at. Defaults to None.
            n: Additional parameter.
            update (bool): Whether to update the balances. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, 'Balance']: A dictionary containing account IDs as keys and their balances as values.
        """
        accounts = self.accounts(network=network, update=update, block=block)
        balances =  {k:v['data']['free'] for k,v in accounts.items()}
        balances = {k: self.format_amount(v, fmt=fmt) for k,v in balances.items()}
        return balances
    
    
    def resolve_network(self, network: Optional[int] = None, new_connection =False, mode='ws', **kwargs) -> int:
        """
        Resolve the network connection and return the network value.

        Parameters:
            self: the instance of the class
            network (Optional[int]): the network value, defaults to None
            new_connection (bool): flag to indicate if a new connection should be established, defaults to False
            mode (str): the mode of connection, defaults to 'ws'
            **kwargs: additional keyword arguments

        Returns:
            int: the resolved network value
        """
        if  not hasattr(self, 'substrate') or new_connection:
            self.set_network(network, **kwargs)

        if network == None:
            network = self.network
        
        return network
    
    def resolve_subnet(self, subnet: Optional[int] = None) -> int:
        """
        A function that resolves the subnet based on the given input. 

        Parameters:
            subnet (Optional[int]): The subnet to be resolved. Defaults to None.

        Returns:
            int: The resolved subnet.
        """
        if isinstance(subnet, int):
            assert subnet in self.netuids()
            subnet = self.netuid2subnet(netuid=subnet)
        subnets = self.subnets()
        assert subnet in subnets, f"Subnet {subnet} not found in {subnets}"
        return subnet


    def subnets(self, **kwargs) -> Dict[int, str]:
        """
        This function returns a dictionary of subnet names based on the provided keyword arguments.
        """
        return self.subnet_names(**kwargs)
    
    def num_subnets(self, **kwargs) -> int:
        """
        Calculate the number of subnets based on the given keyword arguments and return the count as an integer.
        """
        return len(self.subnets(**kwargs))
    
    def netuids(self, network=network, update=False, block=None) -> Dict[int, str]:
        return list(self.netuid2subnet(network=network, update=update, block=block).keys())

    def subnet_names(self, network=network , update=False, block=None, **kwargs) -> Dict[str, str]:
        """
        Retrieves the names of subnets for the given network.

        Args:
            network (optional): The network for which to retrieve subnet names. Defaults to the 'network' attribute.
            update (bool, optional): Whether to update the data before retrieving the subnet names. Defaults to False.
            block (optional): Optional block parameter.
            **kwargs: Additional keyword arguments to pass to the query.

        Returns:
            dict: A dictionary containing the names of subnets.
        """
        records = self.query_map('SubnetNames', update=update, network=network, block=block, **kwargs)
        return list(records.values())
    
    netuid2subnet = subnet_names

    def subnet2netuid(self, subnet=None, network=network, update=False,  **kwargs ) -> Dict[str, str]:
        """
        A function that converts subnet to network unique ID and returns a dictionary with the mappings. 

        :param subnet: The subnet to convert to network unique ID. Defaults to None.
        :param network: The network to use for conversion. Defaults to the class attribute 'network'.
        :param update: A boolean indicating whether to update the network unique ID mappings. Defaults to False.
        :param kwargs: Additional keyword arguments.
        :return: A dictionary containing the mappings of subnets to network unique IDs. If a specific subnet is provided, 
                 returns the corresponding network unique ID, or the length of mappings if not found.
        :rtype: Dict[str, str]
        """
        subnet2netuid =  {v:k for k,v in self.netuid2subnet(network=network, update=update, **kwargs).items()}
        if subnet != None:
            return subnet2netuid[subnet] if subnet in subnet2netuid else len(subnet2netuid)
        return subnet2netuid
    
    def netuid2subnet(self, netuid=None, network=network, update=False, block=None, **kwargs ) -> Dict[str, str]:
        """
        This function takes a netuid and returns the corresponding subnet. 
        It also has optional parameters for network, update, and block. 
        It returns a dictionary with string keys and string values.
        """
        netuid2subnet = self.query_map('SubnetNames', update=update, network=network, block=block, **kwargs)
        if netuid != None:
            return netuid2subnet[netuid]
        return netuid2subnet


    subnet_namespace = subnet2netuid

    def resolve_netuid(self, netuid: int = None, network=network, update=False) -> int:
        """
        A function to resolve the netuid based on the provided parameters.
        
        Parameters:
            netuid (int): The netuid to resolve. Defaults to None.
            network: The network to use for resolution.
            update (bool): Whether to update the netuid. Defaults to False.
            
        Returns:
            int: The resolved netuid.
        """

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
        """
        A function that maps keys to names based on the given key. 

        Parameters:
            key (str): The key to look up the corresponding name for.
            netuid (int): The netuid parameter for the function.

        Returns:
            str: The name corresponding to the given key.
        """
        modules = self.keys()
        key2name =  { m['key']: m['name']for m in modules}
        if key != None:
            return key2name[key]
        
    def name2uid(self,name = None, search:str=None, netuid: int = None, network: str = None) -> int:
        """
        A function to convert a name to a unique identifier (uid).
        
        Args:
            name (str): The name to be converted to uid.
            search (str): A string to search for in the name.
            netuid (int): The network uid.
            network (str): The network name.
        
        Returns:
            int: The unique identifier (uid) corresponding to the input name.
        """
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
        """
        A method to get a specific feature, with optional parameters for key, network, netuid, update, max_age, and additional kwargs.
        Returns the result of calling the specified feature with the given parameters.
        """
        s = cls(network=network)
        return getattr(s, key)(netuid=netuid, update=update, max_age=max_age, **kwargs)
        
    def name2key(self, name:str=None, 
                 network=network, 
                 max_age=1000, 
                 timeout=30, 
                 netuid: int = 0, 
                 update=False, 
                 **kwargs ) -> Dict[str, str]:
        # netuid = self.resolve_netuid(netuid)
        self.resolve_network(network)
        """
        A function to map names to keys using specified parameters and return the corresponding key. 

        Parameters:
            name (str): The name to be mapped to a key.
            network: The specified network to use.
            max_age (int): The maximum age parameter.
            timeout (int): The timeout parameter.
            netuid (int): The netuid parameter.
            update (bool): Flag to update the mapping.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, str]: A dictionary mapping names to keys or the key corresponding to the input name.
        """
        names = c.submit(self.get_feature, args=['names'])
        keys = c.submit(self.get_feature, args=['keys'])
        names, keys = c.wait([names, keys], timeout=timeout)
        name2key = dict(zip(names, keys))
        if name != None:
            return name2key[name]
        return name2key





    def key2name(self, key=None, netuid: int = None, network=network, update=False) -> Dict[str, str]:
        """
        A function that maps keys to names using the provided parameters and returns a dictionary.
        
        Parameters:
            key: Optional, the key to be mapped to a name.
            netuid: Optional, an integer representing network UID.
            network: Default parameter, represents the network.
            update: Optional, a boolean indicating if the mapping should be updated.
            
        Returns:
            A dictionary mapping keys to names.
        """
        key2name =  {v:k for k,v in self.name2key(netuid=netuid, network=network, update=update).items()}
        if key != None:
            return key2name[key]
        return key2name
        
    def is_unique_name(self, name: str, netuid=None):
        """
        Check if the given name is unique within the namespace.
        
        Args:
            name (str): The name to be checked for uniqueness.
            netuid: The namespace identifier. Defaults to None.

        Returns:
            bool: True if the name is unique, False otherwise.
        """
        return bool(name not in self.get_namespace(netuid=netuid))
    
    def epoch_time(self, netuid=0, network='main', update=False, **kwargs):
        """
        Generate epoch time based on subnet parameters.

        :param netuid: integer, unique identifier for the network
        :param network: string, the network to use
        :param update: boolean, whether to update the network settings
        :param kwargs: additional keyword arguments
        :return: float, the epoch time calculated based on subnet parameters
        """
        return self.subnet_params(netuid=netuid, network=network)['tempo']*self.block_time

    def blocks_per_day(self, netuid=None, network=None):
        """
        Calculate the number of blocks per day based on the block time.
        
        Args:
            netuid: (optional) The unique identifier for the network.
            network: (optional) The name of the network.
        
        Returns:
            float: The number of blocks per day.
        """
        return 24*60*60/self.block_time
    

    def epochs_per_day(self, netuid=None, network=None):
        return 24*60*60/self.epoch_time(netuid=netuid, network=network)
    
        """
        Calculate the number of epochs per day based on the epoch time.
        
        :param netuid: Optional parameter for the network unique identifier.
        :param network: Optional parameter for the network.
        :return: The number of epochs per day.
        """
    def emission_per_epoch(self, netuid=None, network=None):
        """
        Calculates the emission per epoch based on the given netuid and network.

        :param netuid: The unique identifier of the network.
        :param network: The network to calculate the emission for.
        :return: The emission per epoch.
        """
        return self.subnet(netuid=netuid, network=network)['emission']*self.epoch_time(netuid=netuid, network=network)


    def get_block(self, network='main', block_hash=None, max_age=8): 
        """
        Retrieve a block from the specified network cache, or fetch it from the network if not found.
        
        :param network: str, the network to retrieve the block from (default is 'main')
        :param block_hash: str, the hash of the block to retrieve
        :param max_age: int, the maximum age in seconds for a cached block to be considered valid (default is 8)
        :return: str, the block number
        """
        network = network or 'main'
        block = self.get(path, block_hash, max_age=max_age)
        if block == None:
            self.resolve_network(network)
            block_header = self.substrate.get_block( block_hash=block_hash)['header']
            block = block_header['number']
            block_hash = block_header['hash']
            self.put(path, block)
        return block

    def block_hash(self, block = None, network='main'): 
        """
        Calculate the hash of the given block.

        Args:
            block: The block for which the hash will be calculated. If not provided, the hash of the self.block will be calculated.
            network: The network for which the hash will be calculated. Default is 'main'.

        Returns:
            The hash of the specified block.
        """
        if block == None:
            block = self.block

        substrate = self.get_substrate(network=network)
        
        return substrate.get_block_hash(block)
    

    def seconds_per_epoch(self, netuid=None, network=None):
        """
        Calculate the number of seconds per epoch based on the block time and subnet tempo.

        Parameters:
            netuid (Optional): A unique identifier for the network.
            network (Optional): The network to resolve.

        Returns:
            The number of seconds per epoch calculated based on the block time and subnet tempo.
        """
        self.resolve_network(network)
        netuid =self.resolve_netuid(netuid)
        return self.block_time * self.subnet(netuid=netuid)['tempo']

    
    def get_module(self, module='vali',
                    netuid=0,
                    network='main',
                    fmt='j',
                    method='subspace_getModuleInfo',
                    mode = 'http',
                    lite = True, **kwargs ) -> 'ModuleInfo':
        """
        Get module information from the specified network using the given parameters and return the module information.
        
        Args:
            module (str): The module name or key. Defaults to 'vali'.
            netuid (int): The network UID. Defaults to 0.
            network (str): The network name. Defaults to 'main'.
            fmt (str): The format of the output. Defaults to 'j'.
            method (str): The method for getting module information. Defaults to 'subspace_getModuleInfo'.
            mode (str): The mode of communication. Defaults to 'http'.
            lite (bool): Flag to indicate if lite mode is enabled. Defaults to True.
            **kwargs: Additional keyword arguments.
            
        Returns:
            ModuleInfo: The module information.
        """
        url = self.resolve_url(network=network, mode=mode)
        if isinstance(module, int):
            module = self.uid2key(uid=module, netuid=netuid)
        if isinstance(module, str):
            module_key = self.resolve_key_ss58(module)
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
        module['vote_staleness'] = self.block - module['last_update']
        if lite :
            features = self.module_features + ['stake', 'vote_staleness']
            module = {f: module[f] for f in features}

        assert module['key'] == module_key, f"Key mismatch {module['key']} != {module_key}"

        return module
    

    @staticmethod
    def vec82str(l:list):
        """
        Convert a list of integers to a string by joining the characters represented by the integers and stripping any leading or trailing whitespace.

        :param l: A list of integers to be converted to a string.
        :return: A string representing the characters of the input integers with leading and trailing whitespace removed.
        """
        return ''.join([chr(x) for x in l]).strip()

    def get_modules(self, keys:list = None,
                     network='main',
                          timeout=20,
                         netuid=0, 
                         fmt='j',
                         update = False,
                         batch_size = 16,
                           **kwargs) -> List['ModuleInfo']:
        """
        A function to retrieve modules based on given keys and network information.

        Parameters:
            keys (list): A list of keys to retrieve modules for.
            network (str): The network to retrieve modules from (default is 'main').
            timeout (int): The timeout value for the request (default is 20).
            netuid (int): The unique identifier for the network (default is 0).
            fmt (str): The format of the response (default is 'j').
            update (bool): A flag indicating whether to update the modules (default is False).
            batch_size (int): The batch size for retrieving modules (default is 16).
            **kwargs: Additional keyword arguments for customization.

        Returns:
            List['ModuleInfo']: A list of ModuleInfo objects retrieved based on the provided keys and network.
        """
        netuid = self.resolve_netuid(netuid)
        
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
        progress_bar = c.tqdm(total=len(keys), desc=f'Querying {len(keys)} keys for modules')
        modules = []
        for key in keys:
            try:
                module = self.get_module(module=key, netuid=netuid, network=network, fmt=fmt, **kwargs)
            except Exception as e:
                c.print(e)
                continue
            if isinstance(module, dict) and 'name' in module:
                modules.append(module)
                progress_bar.update(1)
            else:
                c.print(f'Error querying module for key {key} {module}')
    
        return modules
        
    def my_modules(self, netuid=0, generator=False,  **kwargs):
        """
        A function that generates modules based on the given netuid. 
        :param netuid: The netuid to generate modules for (default is 0).
        :param generator: Flag indicating if generator mode is enabled (default is False).
        :param kwargs: Additional keyword arguments.
        :return: Dictionary containing modules generated based on the netuid.
        """
        keys = self.my_keys(netuid=netuid)
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
        """
        Formats the module information by applying specific formatting to certain fields.

        Parameters:
            module (ModuleInfo): The module information to be formatted.
            fmt (str): The format type to apply. Defaults to 'j'.

        Returns:
            ModuleInfo: The formatted module information.
        """
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
        """
        A function to query modules with various features from a specified network.
        
        Parameters:
            search (str): A string to search within module names (default is None).
            network (str): The network to query modules from (default is 'main').
            netuid (int): The unique identifier for the network (default is 0).
            block (Optional[int]): The block number to query modules from (default is None).
            fmt (str): The format in which to return the modules (default is 'nano').
            features (List[str]): A list of features to query (default is module_features).
            timeout: The timeout for each query (default is 100).
            max_age: The maximum age for cached data (default is 1000).
            subnet: The subnet to query modules from (default is None).
            vector_features: A list of vector features (default is ['dividends', 'incentive', 'trust', 'last_update', 'emission']).
            **kwargs: Additional keyword arguments for future expansion.
        
        Returns:
            Dict[str, 'ModuleInfo']: A dictionary containing information about modules with specified features.
        """
    

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



        netuid = self.resolve_netuid(netuid or subnet)
        network = self.resolve_network(network)
        state = {}
        path = f'query/{network}/SubspaceModule.Modules:{netuid}'
        modules = self.get(path, None, max_age=max_age)
        if modules == None:

            progress = c.tqdm(total=len(features), desc=f'Querying {len(features)} features')
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
                c.print(f'Querying {len(features)} features')
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
                        assert uid_key in state[feature], f"Key {uid_key} not found in {feature}"
                        module[feature] = state[feature][uid_key]
                modules.append(module)
            self.put(path, modules)

            
        if len(modules) > 0:
            for i in range(len(modules)):
                modules[i] = self.format_module(modules[i], fmt=fmt)

        if search != None:
            modules = [m for m in modules if search in m['name']]

        return modules

    


    def min_stake(self, netuid: int = 0, network: str = 'main', fmt:str='j', **kwargs) -> int:
        """
        A function that calculates the minimum stake required for a network.
        
        Parameters:
            netuid (int): The unique identifier of the network.
            network (str): The name of the network ('main' by default).
            fmt (str): The format of the stake amount ('j' by default).
            **kwargs: Additional keyword arguments.
        
        Returns:
            int: The formatted minimum stake amount.
        """
        min_stake = self.query('MinStake', netuid=netuid, network=network, **kwargs)
        return self.format_amount(min_stake, fmt=fmt)

    def registrations_per_block(self, network: str = network, fmt:str='j', **kwargs) -> int:
        """
        A function to calculate the number of registrations per block in the network.
        
        Parameters:
            network (str): The network to query.
            fmt (str): The format of the query.
            **kwargs: Additional keyword arguments for the query.
        
        Returns:
            int: The number of registrations per block.
        """
        return self.query('RegistrationsPerBlock', params=[], network=network, **kwargs)
    regsperblock = registrations_per_block
    
    def max_registrations_per_block(self, network: str = network, fmt:str='j', **kwargs) -> int:
        """
        A function that calculates the maximum number of registrations per block based on the network and format provided.

        Parameters:
            network (str): The network to query for.
            fmt (str): The format of the query.
            **kwargs: Additional keyword arguments to pass to the query.

        Returns:
            int: The maximum number of registrations per block.
        """
        return self.query('MaxRegistrationsPerBlock', params=[], network=network, **kwargs)
 
    def uids(self, netuid = 0, **kwargs):
        """
        Return a list of unique identifiers based on the given netuid and additional keyword arguments.
        """
        return list(self.uid2key(netuid=netuid, **kwargs).keys())
   
    def keys(self,
             netuid = 0,
              update=False, 
              max_age=1000,
             network : str = 'main', 
             **kwargs) -> List[str]:
        """
        A function to retrieve keys based on the specified parameters.

            netuid: int, optional, default 0
                The network UID.
            update: bool, optional, default False
                Whether to update the keys.
            max_age: int, optional, default 1000
                The maximum age of the keys.
            network: str, optional, default 'main'
                The network to retrieve keys from.
            **kwargs: dict
                Additional keyword arguments.
            
            Returns:
                List[str]: A list of keys based on the specified parameters.
        """
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
    	"""
    	uid2key function comment with parameters and return types.
    	"""
        netuid = self.resolve_netuid(netuid)
        uid2key =  self.query_map('Keys',  netuid=netuid, update=update, network=network, max_age=max_age, **kwargs)
        # sort by uid
        if uid != None:
            return uid2key[uid]
        return uid2key
    

    def key2uid(self, key = None, network:str=  'main' ,netuid: int = 0, update=False, **kwargs):
        """
        A function to map keys to user IDs and vice versa in a given network.
        
        Args:
            key: Optional, the key to be mapped to a user ID. Default is None.
            network: Optional, the network in which to perform the mapping. Default is 'main'.
            netuid: Optional, the network UID. Default is 0.
            update: Optional, a boolean indicating whether to update the mapping. Default is False.
            **kwargs: Additional keyword arguments to be passed to other functions.

        Returns:
            dict: A dictionary mapping keys to user IDs.
        """
        uid2key =  self.uid2key(network=network, netuid=netuid, update=update, **kwargs)
        key2uid = {v:k for k,v in uid2key.items()}
        if key == 'all':
            return key2uid
        if key != None:
            key_ss58 = self.resolve_key_ss58(key)
            return key2uid[key_ss58]
        return key2uid
        

    def uid2name(self, netuid: int = 0, update=False,  **kwargs) -> List[str]:
        """
        Resolve netuid and retrieve names from query_map, then return a sorted list of names.
        
        Args:
            netuid (int): The network user ID. Defaults to 0.
            update (bool): Whether to update the query map. Defaults to False.
            **kwargs: Additional keyword arguments for query_map.
            
        Returns:
            List[str]: A sorted list of names.
        """
        netuid = self.resolve_netuid(netuid)
        names = {k: v for k,v in enumerate(self.query_map('Name', update=update,**kwargs)[netuid])}
        names = {k: names[k] for k in sorted(names)}
        return names
    
    def names(self, 
              netuid: int = 0, 
              update=False,
                **kwargs) -> List[str]:
        """
        Generate a list of names based on the provided netuid, with an option to update the query map.
        
        :param netuid: An integer representing the unique ID.
        :param update: A boolean indicating whether to update the query map.
        :param **kwargs: Additional keyword arguments.
        :return: A list of strings representing names.
        """
        uid2name = self.query_map('Name', update=update, netuid=netuid,**kwargs)
        if isinstance(netuid, int):
            names = list(uid2name.values())
        else:
            for netuid, uid2name in uid2name.items():
                names[netuid] = list(netuid.values())
        return names

    def addresses(self, netuid: int = 0, update=False, **kwargs) -> List[str]:
        """
        A description of the entire function, its parameters, and its return types.
        """
        addresses = self.query_map('Address',netuid=netuid, update=update, **kwargs)
        
        if isinstance(netuid, int):
            addresses = list(addresses.values())
        else:
            for k,v in addresses.items():
                addresses[k] = list(v.values())
        return addresses

    def namespace(self, search=None, netuid: int = 0, update:bool = False, timeout=30, local=False, max_age=1000, **kwargs) -> Dict[str, str]:
        """
        A function that retrieves namespace information based on the given parameters.

        Args:
            search (optional): A string to filter the namespace by. Defaults to None.
            netuid (int): The unique identifier for the network. Defaults to 0.
            update (bool): Flag to indicate whether to update the namespace. Defaults to False.
            timeout (int): The maximum time to wait for a response. Defaults to 30.
            local (bool): Flag to indicate whether to restrict the namespace to the local machine. Defaults to False.
            max_age (int): The maximum age of the namespace information. Defaults to 1000.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, str]: A dictionary containing the namespace information, with names as keys and addresses as values.
        """
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
        """
        A function to retrieve weights based on the given parameters.

        Parameters:
            netuid (int): The unique identifier of the network.
            network (str): The network name.
            update (bool): Flag indicating whether to update weights.
            **kwargs: Additional keyword arguments.

        Returns:
            list: A list of weights retrieved based on the parameters.
        """
        weights =  self.query_map('Weights',netuid=netuid, network = network, update=update, **kwargs)

        return weights

    def proposals(self, netuid = netuid, block=None,   network="main", nonzero:bool=False, update:bool = False,  **kwargs):
        """
        Function to retrieve proposals from the specified network and block.

        Args:
            netuid: The unique identifier for the network.
            block: The block from which to retrieve the proposals.
            network: The network from which to retrieve the proposals.
            nonzero: A boolean flag to indicate whether to include only nonzero proposals.
            update: A boolean flag to indicate whether to update the proposals.
            **kwargs: Additional keyword arguments.

        Returns:
            List: A list of proposals.
        """
        proposals = [v for v in self.query_map('Proposals', network = 'main', block=block, update=update, **kwargs)]
        return proposals

    def save_weights(self, nonzero:bool = False, network = "main",**kwargs) -> list:
        """
        Save the weights with the option to only save non-zero weights.

        :param self: The object itself.
        :param nonzero: Whether to save only non-zero weights. Default is False.
        :param network: The network to save the weights for. Default is "main".
        :param **kwargs: Additional keyword arguments.
        :return: A dictionary with a success flag and a message indicating the weights were saved.
        """
        self.query_map('Weights',network = 'main', update=True, **kwargs)
        return {'success': True, 'msg': 'Saved weights'}

    def pending_deregistrations(self, netuid = 0, update=False, **kwargs):
        """
        A function to retrieve pending deregistrations for a specific network UID.
        
        :param netuid: The network UID for which pending deregistrations are to be retrieved (default is 0).
        :param update: A boolean flag to indicate whether to update the pending deregistrations (default is False).
        :param **kwargs: Additional keyword arguments to be passed to the query_map function.
        
        :return: A dictionary containing pending deregistrations for the specified network UID.
        """
        pending_deregistrations = self.query_map('PendingDeregisterUids',update=update,**kwargs)[netuid]
        return pending_deregistrations
    
    def num_pending_deregistrations(self, netuid = 0, **kwargs):
        """
        Calculate the number of pending deregistrations for a given netuid.

        :param netuid: int, optional, the netuid for which pending deregistrations are being calculated (default is 0)
        :param kwargs: additional keyword arguments
        :return: int, the number of pending deregistrations
        """
        pending_deregistrations = self.pending_deregistrations(netuid=netuid, **kwargs)
        return len(pending_deregistrations)
        
    def emissions(self, netuid = 0, network = "main", block=None, update=False, **kwargs):
        """
        A function to retrieve emissions data based on the given parameters.
        
        :param netuid: int, default 0, the unique identifier of the network
        :param network: str, default "main", the name of the network
        :param block: None or int, the block number
        :param update: bool, whether to update the data
        :param **kwargs: additional keyword arguments
        
        :return: the result of querying the vector for 'Emission'
        """

        return self.query_vector('Emission', network=network, netuid=netuid, block=block, update=update, **kwargs)
    
    emission = emissions
    
    def incentives(self, 
                  netuid = 0, 
                  block=None,  
                  network = "main", 
                  update:bool = False, 
                  **kwargs):
        """
        A function to retrieve incentives with optional parameters netuid, block, network, update, and additional keyword arguments.
        Returns the result of querying the 'Incentive' vector.
        """
        return self.query_vector('Incentive', netuid=netuid, network=network, block=block, update=update, **kwargs)
    incentive = incentives

    def trust(self, 
                  netuid = 0, 
                  block=None,  
                  network = "main", 
                  update:bool = False, 
                  **kwargs):
        """
        A description of the entire function, its parameters, and its return types.
        """
        return self.query_vector('Trust', netuid=netuid, network=network, block=block, update=update, **kwargs)
    
    incentive = incentives
    
    def query_vector(self, name='Trust', netuid = 0, network="main", update=False, **kwargs):
        """
        A function to query a vector based on the provided parameters.
        
        Parameters:
            name (str): The name of the vector (default is 'Trust').
            netuid (int): The unique identifier of the network.
            network (str): The network to query (default is "main").
            update (bool): A flag indicating whether to update the vector.
            **kwargs: Additional keyword arguments.
        
        Returns:
            dict: The queried vector based on the parameters provided.
        """
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
        """
        A function to retrieve registration blocks based on the provided parameters.
        
        :param netuid: An integer representing the unique identifier for the registration block.
        :param update: A boolean indicating whether to update the registration block.
        :param kwargs: Additional keyword arguments for querying registration blocks.
        
        :return: A list of registration blocks retrieved based on the input parameters.
        """
        registration_blocks = self.query_map('RegistrationBlock', netuid=netuid, update=update, **kwargs)
        return registration_blocks

    regblocks = registration_blocks = registration_block

    def stake_from(self, netuid = 0,
                    block=None, 
                    update=False,
                    network=network,
                    total = False,
                    fmt='nano', **kwargs) -> List[Dict[str, Union[str, int]]]:
        """
        A function to retrieve stake information based on the provided parameters.

        Parameters:
            netuid (int): The unique identifier of the network stake to retrieve.
            block (str): The block information for which stake data is requested.
            update (bool): A flag to indicate if the stake information needs to be updated.
            network (str): The network for which stake information is required.
            total (bool): A flag to indicate if the total stake information is requested.
            fmt (str): The format in which the stake information should be formatted.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Dict[str, Union[str, int]]]: A list of dictionaries containing formatted stake information.
        """
        
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
        """
        A function to get the blockchain archives for a given network and datetime, returning a list of blockchain IDs and archive paths.
        
        Parameters:
            netuid: The unique identifier for the network.
            network: The network for which the archives are being retrieved.
            **kwargs: Additional keyword arguments.
        
        Returns:
            List[str]: A list of dictionaries containing blockchain IDs, archive paths, and block numbers.
        """

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
        """
        Retrieves blockchain information from archives based on the provided network and optional keyword arguments.

        Args:
            netuid: The unique identifier for the network.
            network: The network to retrieve blockchain information from.
            **kwargs: Additional keyword arguments.

        Returns:
            List[str]: A list of blockchain information containing blockchain ID, archive path, block number, and earliest block.
        """

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
        """
        A class method to retrieve the most recent archives by searching the archives.
        No input parameters.
        Returns the list of archives.
        """
        archives = cls.search_archives()
        return archives
    
    @classmethod
    def num_archives(cls, *args, **kwargs):
        return len(cls.datetime2archive(*args, **kwargs))

    def keep_archives(self, loockback_hours=24, end_time='now'):
        """
        Keep archives for a specified period and remove the rest.

        Parameters:
            loockback_hours (int): The number of hours to look back for keeping archives (default is 24).
            end_time (str): The end time for considering the archives (default is 'now').

        Returns:
            list: The paths of the kept archives.
        """
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
        """
        A function to search archives within a specified time range.

        Parameters:
            lookback_hours (int): The number of hours to look back in the archives (default is 24 hours).
            end_time (str): The end time for the search (default is 'now').
            start_time (Union[int, str]): The start time for the search, can be an integer timestamp or a string (default is None).
            netuid: The network ID for the search.
            n: The number of archives to return (default is 1000).
            **kwargs: Additional keyword arguments.

        Returns:
            list: A list of dictionaries containing information about archives within the specified time range.
        """
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
        
    def key_usage_path(self, key:str):
        """
        This function generates a usage path based on the provided key.

        Parameters:
            key (str): The key for which the usage path is generated.

        Returns:
            str: The usage path for the provided key.
        """
        key_ss58 = self.resolve_key_ss58(key)
        return f'key_usage/{key_ss58}'

    def key_used(self, key:str):
        """
        Check if the specified key is used and return the result.
        
        :param key: A string representing the key to be checked.
        :return: True if the key is used, False otherwise.
        """
        return self.exists(self.key_usage_path(key))
    
    def use_key(self, key:str):
        """
        Use the provided key to update the key usage path with the current time.
        
        :param key: A string representing the key to be used.
        :return: The result of putting the key usage path with the current time.
        """
        return self.put(self.key_usage_path(key), c.time())
    
    def unuse_key(self, key:str):
        """
        Removes the usage of the specified key and returns the result.
        
        Args:
            key (str): The key to be removed.

        Returns:
            The result of removing the key usage.
        """
        return self.rm(self.key_usage_path(key))
    
    def test_key_usage(self):
        """
        Test the usage of a key by adding, using, unusing, and removing it,
        then checking its existence and usage status. Returns a dictionary 
        indicating the success of the test and a message.
        """
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
        """
        A function to get the nonce for a given key on a specific network.

        :param key: Optional key in string format.
        :param network: Optional network information.
        :param kwargs: Additional keyword arguments.
        :return: The nonce of the account associated with the key.
        """
        key_ss58 = self.resolve_key_ss58(key)
        self.resolve_network(network)   
        return self.substrate.get_account_nonce(key_ss58)

    history_path = f'history'

    chain_path = c.libpath + '/subspace'
    spec_path = f"{chain_path}/specs"
    snapshot_path = f"{chain_path}/snapshots"

    @classmethod
    def check(cls, netuid=0):
        """
        Check function to perform various queries and print the length of the results.
        
        Parameters:
            cls: The class itself.
            netuid (int): The netuid to use in the queries. Defaults to 0.
        """
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
              features : list = ['name', 'emission','incentive', 'dividends', 'stake', 'last_update', 'serving'],
              sort_features = ['emission', 'stake'],
              fmt : str = 'j',
              modules = None,
              servers = None,
              **kwargs
              ):
        """
        Function to calculate statistics based on the given parameters and return the results in a pandas DataFrame format.
        
        Parameters:
            search: str, optional
                Search term to filter the results.
            netuid: int, optional
                Unique identifier for the network.
            network: object
                Network object to be used for the statistics.
            df: bool, optional
                Boolean flag to indicate whether the results should be returned as a DataFrame.
            update: bool, optional
                Boolean flag to indicate whether the data should be updated before calculating statistics.
            features: list, optional
                List of features to include in the statistics calculation.
            sort_features: list, optional
                List of features to use for sorting the statistics results.
            fmt: str, optional
                Format of the statistics results.
            modules: object, optional
                Modules object to be used for the statistics calculation.
            servers: object, optional
                Servers object to be used for the statistics calculation.
            **kwargs: 
                Additional keyword arguments for future expansion.
        
        Returns:
            pandas.DataFrame or list
                Depending on the value of the 'df' parameter, it returns the statistics results either as a DataFrame or a list of records.
        """

            
        if isinstance(netuid, str):
            netuid = self.subnet2netuid(netuid)

        if search == 'all':
            netuid = search
            search = None

        
        if netuid == 'all':
            all_modules = self.my_modules(netuid=netuid, update=update, network=network, fmt=fmt)
            servers = c.servers(network='local')
            stats = {}
            netuid2subnet = self.netuid2subnet(update=update)
            for netuid, modules in all_modules.items():
                subnet_name = netuid2subnet[netuid]
                stats[netuid] = self.stats(modules=modules, netuid=netuid, servers=servers)

                color = c.random_color()
                c.print(f'\n {subnet_name.upper()} :: (netuid:{netuid})\n', color=color)
                c.print(stats[netuid], color=color)
            

        modules = modules or self.my_modules(netuid=netuid, update=update, network=network, fmt=fmt)

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
        """
        Class method to get the status with the specified current working directory and return the status.
        """
        return c.status(cwd=cls.libpath)


    def storage_functions(self, network=network, block_hash = None):
        """
        A description of the entire function, its parameters, and its return types.
        """
        self.resolve_network(network)
        return self.substrate.get_metadata_storage_functions( block_hash=block_hash)
    storage_fns = storage_functions
        

    def storage_names(self,  search=None, network=network, block_hash = None):
        """
        Retrieve storage names based on the provided search criteria, network, and block hash.

        Args:
            search (str): The search criteria for filtering storage names.
            network (str, optional): The network to resolve.
            block_hash (str, optional): The block hash for retrieving storage names.

        Returns:
            list: A list of storage names filtered based on the search criteria.
        """
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
        """
        A method to retrieve the state dictionary with various features for a given network and block. 

        Parameters:
            timeout (int): The timeout for the request in milliseconds. Default is 1000.
            network (str): The network to retrieve the state dictionary for. Default is 'main'.
            netuid (str): The netuid parameter for the request. Default is 'all'.
            update (bool): Whether to update the state dictionary. Default is False.
            mode (str): The mode for the request. Default is 'http'.
            save (bool): Whether to save the state dictionary. Default is False.
            block (str): The block to retrieve the state dictionary for. 

        Returns:
            dict: A dictionary containing the state dictionary and metadata about the request.
        """
        
        
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
        """
        A method to synchronize the state using the given arguments and keyword arguments, and return the updated state dictionary.
        """
        return  self.state_dict(*args, save=True, update=True, **kwargs)

    def check_storage(self, block_hash = None, network=network):
        """
        A function to check storage, taking in parameters block_hash and network, and returning metadata storage functions.
        """
        self.resolve_network(network)
        return self.substrate.get_metadata_storage_functions( block_hash=block_hash)

    @classmethod
    def sand(cls): 
        """
        Class method to perform sand operation. Retrieves node keys, spec, and address, then iterates through datetime archive to print balances.
        """
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
        """
        A function to test balance with specified network, number of transfers, balance amount, and timeout period.
        Parameters:
            network (str): The network to test balance on.
            n (int): Number of transfers to perform.
            timeout (int): Time limit for the test.
            verbose (bool): Verbosity flag.
            min_amount: Minimum amount for balance.
            key: Optional key parameter.
        """
        key = c.get_key(key)

        balance = self.get_balance(network=network)
        assert balance > 0, f'balance must be greater than 0, not {balance}'
        balance = int(balance * 0.5)
        c.print(f'testing network {network} with {n} transfers of {balance} each')


    def test_commands(self, network:str = network, n:int = 10, timeout:int = 10, verbose:bool = False, min_amount = 10, key=None):
        """
        A function to test various commands with specified parameters.

        Parameters:
            network (str): The network to test on.
            n (int): Number of transfers to perform.
            timeout (int): Timeout for each transfer.
            verbose (bool): Verbosity flag.
            min_amount: Minimum amount for transfer.
            key: Optional key parameter.

        Returns:
            None
        """
        key = c.get_key(key)

        key2 = c.get_key('test2')
        
        balance = self.get_balance(network=network)
        assert balance > 0, f'balance must be greater than 0, not {balance}'
        c.transfer(dest=key, amount=balance, timeout=timeout, verbose=verbose)
        balance = int(balance * 0.5)
        c.print(f'testing network {network} with {n} transfers of {balance} each')


    @classmethod
    def fix(cls):
        """
        This class method fixes something. It does not take any parameters and does not return any value.
        """
        avoid_ports = []
        free_ports = c.free_ports(n=3, avoid_ports=avoid_ports)
        avoid_ports += free_ports

    def num_holders(self, **kwargs):
        balances = self.balances(**kwargs)
        return len(balances)
        """
        This function calculates the number of holders based on the balances retrieved using the given keyword arguments and returns the count.
        """

    def total_balance(self, **kwargs):
        """
        Calculate the total balance based on the balances retrieved using the provided keyword arguments.

        :param kwargs: additional keyword arguments to pass to the balances method
        :return: total balance calculated as the sum of all balances
        """
        balances = self.balances(**kwargs)
        return sum(balances.values())
    

    def sand(self, **kwargs):
        """
        Calculate the total sum of balances from a dictionary of balances.

        Parameters:
            **kwargs (dict): Additional keyword arguments to pass to the my_balances method.

        Returns:
            int: The total sum of balances.
        """
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
        """
        A function that chains the input arguments using the 'subspace.chain' module.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        
        Returns:
            The result of calling the 'subspace.chain' module with the input arguments.
        """
        return c.module('subspace.chain')(*args, **kwargs)
    
    def chain_config(self, *args, **kwargs):
        """
        A method that chains the configuration settings with the provided arguments and returns the configuration.
        
        Parameters:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        
        Returns:
            The configuration after chaining with the provided arguments.
        """
        return self.chain(*args, **kwargs).config
    
    def chains(self, *args, **kwargs):
        """
        A function that generates chains based on the arguments passed and returns the chains.
        """
        return self.chain(*args, **kwargs).chains()

    """
    #########################################
                    CHAIN LAND
    #########################################
    
    """
    ##################
    #### Register ####
    ##################
    def min_register_stake(self, netuid: int = 0, network: str = network, fmt='j', **kwargs) -> float:
        """
        Calculate the minimum stake considering the netuid, network, fmt, and any additional keyword arguments.

        :param netuid: An integer representing the netuid. Default is 0.
        :param network: A string representing the network. Default is the network parameter of the class.
        :param fmt: A string specifying the format. Default is 'j'.
        :param kwargs: Any additional keyword arguments.
        :return: A float representing the calculated minimum stake.
        """
        min_burn = self.min_burn( network=network, fmt=fmt)
        min_stake = self.min_stake(netuid=netuid, network=network, fmt=fmt)
        return min_stake + min_burn
    def register(
        self,
        name: str , # defaults to module.tage
        address : str = 'NA',
        stake : float = 0,
        subnet: str = 'commune',
        netuid = None,
        key : str  = None,
        module_key : str = None,
        network: str = network,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        nonce=None,
        fmt = 'nano',
        max_age = 1000,
    **kwargs
    ) -> bool:
        """
        A function to register a user with optional parameters like name, address, stake, subnet, etc.
        
        Parameters:
            name (str): The name of the user. Defaults to module.tage.
            address (str): The address of the user. Defaults to 'NA'.
            stake (float): The stake amount. Defaults to 0.
            subnet (str): The subnet of the user. Defaults to 'commune'.
            netuid: Optional network UID.
            key (str): The key for registration.
            module_key (str): The module key.
            network (str): The network name.
            wait_for_inclusion (bool): Whether to wait for inclusion.
            wait_for_finalization (bool): Whether to wait for finalization.
            nonce: Optional nonce value.
            fmt: The format of the registration.
            max_age: The maximum age for registration.
            **kwargs: Additional keyword arguments.
        
        Returns:
            bool: True if successful, False otherwise.
        """

        network =self.resolve_network(network)
        key = self.resolve_key(key)
        address = address or c.namespace(network='local').get(name, '0.0.0.0:8888')
        module_key = module_key or c.get_key(name).ss58_address
        netuid2subnet = self.netuid2subnet(max_age=max_age)
        subnet2netuid = {v:k for k,v in netuid2subnet.items()}
        
        if stake == None :
            if isinstance(subnet, str):
                if subnet in subnet2netuid:
                    netuid = subnet2netuid[subnet]
                else:
                    netuid = len(subnet2netuid)
            min_stake = self.min_register_stake(netuid=netuid, network=network)
            stake = min_stake + 1
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
        """
        A description of the entire function, its parameters, and its return types.
        
            Parameters:
                dest (str): Destination address for the transfer.
                amount (float): Amount to transfer.
                key (str): Optional key for the transfer.
                network (str): Optional network for the transfer.
                nonce: Optional nonce value.
                **kwargs: Additional keyword arguments.
            
            Returns:
                bool: True if the transfer was successful, False otherwise.
        """
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
    def add_profit_shares(
        self,
        keys: List[str], # the keys to add profit shares to
        shares: List[float] = None , # the shares to add to the keys
        key: str = None,
        network : str = None,
    ) -> bool:
        """
        Add profit shares to the specified keys.

        Args:
            keys (List[str]): the keys to add profit shares to
            shares (List[float], optional): the shares to add to the keys. Defaults to None.
            key (str, optional): the key to resolve. Defaults to None.
            network (str, optional): the network to resolve. Defaults to None.

        Returns:
            bool: True if the operation is successful
        """
        
        key = self.resolve_key(key)
        network = self.resolve_network(network)
        assert len(keys) > 0, f"Must provide at least one key"
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



    def update_module(
        self,
        module: str, # the module you want to change
        address: str = None, # the address of the new module
        name: str = None, # the name of the new module
        delegation_fee: float = None, # the delegation fee of the new module
        netuid: int = None, # the netuid of the new module
        network : str = "main", # the network of the new module
        nonce = None, # the nonce of the new module
        tip: int = 0, # the tip of the new module
    ) -> bool:
        """
        A function to update a module with new information. The function takes in the module name, address, name, delegation fee, netuid, network, nonce, and tip as parameters. It ensures the network is resolved, gets the key for the module, resolves the netuid, and fetches information about the module. It then checks if the module is registered, sets default values if necessary, validates the delegate fee, and composes a call to update the module. Returns a boolean indicating the success of the update.
        """
        self.resolve_network(network)
        key = self.resolve_key(module)
        netuid = self.resolve_netuid(netuid)  
        module_info = self.get_module(module)

        if module_info['key'] == None:
            return {'success': False, 'msg': 'not registered'}
        
        if name == None:
            name = module_info['name']
        if address == None:
            address = module_info['address'][:32]
        # Validate that the module is already registered with the same address
        # ENSURE DELEGATE FEE IS BETWEEN 0 AND 100
        if delegation_fee == None:
            delegation_fee = module_info['delegation_fee']
        assert delegation_fee >= 0 and delegation_fee <= 100, f"Delegate fee must be between 0 and 100"


        params = {
            'netuid': netuid, # defaults to module.netuid
             # PARAMS #
            'name': name, # defaults to module.tage
            'address': address, # defaults to module.tage
            'delegation_fee': delegation_fee, # defaults to module.delegate_fee
        }

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
        """
        A method to update a subnet with the provided netuid, key, network, nonce, and parameters. 
        Returns a boolean indicating the success of the update operation.
        """
            
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
        """
        Propose a subnet update.

        Args:
            netuid (int): The unique identifier of the subnet.
            key (str): The key parameter.
            network (str): The network to resolve.
            nonce: The nonce parameter.
            **params: Additional parameters.

        Returns:
            bool: True if successful.
        """

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
        netuid = 0,
        **params,

    ) -> bool:
        """
        A function to vote on a proposal.
        
        Parameters:
            proposal_id (int): The ID of the proposal to vote on.
            key (str): The key for the vote.
            network (str): The network to use for voting.
            nonce: A unique identifier for the vote.
            netuid (int): The unique ID of the network.
            **params: Additional keyword arguments.
        
        Returns:
            bool: True if the vote was successful.
        """

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
        network : str = 'main',
        sudo:  bool = True,
        **params,
    ) -> bool:
        """
        A method to update a global setting with the given key and network. 
        It resolves the key and network, updates the global parameters, and encodes string parameters to utf-8. 
        Finally, it composes a call to update the global setting and returns a boolean indicating the success of the call. 
        Parameters:
            key: str, optional - The key of the setting to be updated.
            network: str, default 'main' - The network on which the setting should be updated.
            sudo: bool, default True - Indicates whether the call should be made with sudo privileges.
            **params - Additional parameters to update the global setting.
        Returns:
            bool - Indicates the success of the call to update the global setting.
        """

        key = self.resolve_key(key)
        network = self.resolve_network(network)
        global_params = self.global_params( )
        global_params.update(params)
        params = global_params
        for k,v in params.items():
            if isinstance(v, str):
                params[k] = v.encode('utf-8')
        # this is a sudo call
        return self.compose_call(fn='update_global',
                                     params=params, 
                                     key=key, 
                                     sudo=sudo)





    #################
    #### set_code ####
    #################
    def set_code(
        self,
        wasm_file_path = None,
        key: str = None,
        network = network,
    ) -> bool:
        """
        A function to set the code for the given wasm file path, key, and network. 
        It checks if the wasm file path is provided, resolves the network, resolves the key, 
        reads the wasm file, constructs the extrinsic, and returns the response.
        """

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
            module_key: str ,
            new_module_key: str ,
            amount: Union[int, float] = None, 
            key: str = None,
            netuid:int = 0,
            max_age=10,
            network:str = None,
        ) -> bool:
        """
        Transfer stake from one module to another module on a specified network. 
        :param module_key: The key of the module to transfer stake from.
        :param new_module_key: The key of the module to transfer stake to.
        :param amount: The amount of stake to transfer. Defaults to None.
        :param key: The key to use for the transfer. Defaults to None.
        :param netuid: The network UID. Defaults to 0.
        :param max_age: The maximum age of the stake. Defaults to 10.
        :param network: The network to use for the transfer. Defaults to None.
        :return: A boolean indicating the success of the stake transfer.
        """
        # STILL UNDER DEVELOPMENT, DO NOT USE
        network = self.resolve_network(network)
        netuid = self.resolve_netuid(netuid)
        key = c.get_key(key)

        c.print(f':satellite: Staking to: [bold white]SubNetwork {netuid}[/bold white] {amount} ...')
        # Flag to indicate if we are using the wallet's own hotkey.
        name2key = self.name2key(netuid=netuid, max_age=max_age)

        def resolve_module_key(x, netuid=0):
            """
            Resolve a module key by checking if it is a valid SS58 address. 
            If the key is not valid, attempt to retrieve it from a dictionary.
            
            Parameters:
                x: str, the module key to be resolved
                netuid: int, optional, the network UID to use (default is 0)
                
            Returns:
                str, the resolved module key
            """
            if not c.valid_ss58_address(x):
                x = name2key.get(x)
            assert c.valid_ss58_address(x), f"Module key {x} is not a valid ss58 address"
            return x
        
        c.print(module_key, new_module_key)
            
        module_key = resolve_module_key(module_key, netuid=netuid)
        new_module_key = resolve_module_key(new_module_key, netuid=netuid)
        assert module_key != new_module_key, f"Module key {module_key} is the same as new_module_key {new_module_key}"

        if amount == None:
            stake_to = self.get_stake_to( key=key , fmt='nanos', netuid=netuid)
            amount = stake_to.get(new_module_key, 0)
        else:
            amount = amount * 10**9
                
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
        if isinstance(module, int):
            amount = module
            module = None
        network = self.resolve_network(network)
        key = c.get_key(key)
        netuid = self.resolve_netuid(netuid)
        # get most stake from the module

        stake_to = self.get_stake_to(netuid=netuid, names = False, fmt='nano', key=key)

        if c.valid_ss58_address(module):
            module_key = module
        elif module == None and amount != None:
            # find the largest staked module
            for k,v in stake_to.items():
                if v > amount:
                    module_key = k      
                    break
        elif module != None and amount == None:
            module_key = self.name2key(netuid=netuid).get(module)
            amount = int(self.to_nanos(amount)) if amount else stake_to[module_key]
        else: 
            raise Exception('Invalid input')

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
        """
        A function to stake amounts to multiple modules. 

        Args:
            modules (List[str]): List of module names to stake amounts to.
            amounts (Union[List[str], float, int]): Amounts to stake. If a single value is provided, it will be divided evenly among the modules.
            key (str): Key for the stake operation.
            netuid (int): User ID for the stake operation.
            min_balance (int): Minimum balance required for the stake operation.
            n (str): Number of modules to stake to.
            network (str): Network to stake on.

        Returns:
            Optional['Balance']: The response from the stake operation.
        """
        
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
                        network: str = None) -> Optional['Balance']:
        """
        Transfer multiple amounts to multiple destinations.
        
        Args:
            destinations (List[str]): A list of destination addresses.
            amounts (Union[List[str], float, int]): A list of amounts to transfer, or a single amount to be distributed among the destinations.
            key (str, optional): The key to use for the transfer. Defaults to None.
            netuid (int): The network UID. Defaults to 0.
            n (str): The maximum number of destinations to transfer to. Defaults to 10.
            network (str, optional): The network to use for the transfer. Defaults to None.
        
        Returns:
            Optional['Balance']: The balance after the transfer, or None if the transfer failed.
        """
        network = self.resolve_network( network )
        key = self.resolve_key( key )
        balance = self.get_balance(key=key, fmt='j')
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
        """
        A function to unstake multiple amounts from a given network.
        
        :param modules: A list of module names or 'all' to specify which modules to unstake from.
        :param amounts: A list of amounts to unstake corresponding to the modules.
        :param key: A key for authorization.
        :param netuid: An integer representing the network ID.
        :param network: A string representing the network to interact with.
        
        :return: An optional 'Balance' object representing the response from removing stakes.
        """
        
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
        """
        unstake2key method to unstake modules in the network.

        :param modules: The modules to unstake. Defaults to 'all'.
        :param netuid: The netuid to use. Defaults to 0.
        :param network: The network to unstake from.
        :param to: The recipient of the unstaked modules.

        :return: None
        """
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
        """
        A function to unstake all modules associated with a given key on a specified network.
        
        Parameters:
            key (str): The key associated with the modules to be unstaked. Default is 'model.openai'.
            netuid (int): The network UID where the modules are located. Default is 0.
            network: The network where the modules are located.
            to: The destination for the unstaked funds.
            existential_deposit (int): The minimum deposit required for the unstaking operation.
        
        Returns:
            Optional['Balance']: A dictionary containing the response of the unstaking and transfer operations.
        """
        
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
        to = c.get_key(to)
        c.print(f'Transfering {total_stake} to ')
        response['transfer'] = self.transfer(dest=to, amount=total_stake, key=key)
        return response


    def my_servers(self, search=None,  **kwargs):
        """
        A function that returns a list of server names based on the modules retrieved. 

        :param search: A string to filter server names by.
        :param kwargs: Additional keyword arguments to pass to the 'my_modules' function.
        :return: A list of server names filtered by the search string if provided.
        """
        servers = [m['name'] for m in self.my_modules(**kwargs)]
        if search != None:
            servers = [s for s in servers if search in s]
        return servers
    
    def my_modules_names(self, *args, **kwargs):
        """
        Generate a list of module names from the result of calling self.my_modules with the given arguments.

        Parameters:
            *args: variable length argument list
            **kwargs: arbitrary keyword arguments

        Returns:
            list: a list of module names extracted from the 'name' key of each dictionary in my_modules
        """
        my_modules = self.my_modules(*args, **kwargs)
        return [m['name'] for m in my_modules]

    def my_module_keys(self, *args,  **kwargs):
        """
        A function that retrieves keys from modules returned by my_modules.

        Parameters:
            *args: positional arguments to be passed to my_modules.
            **kwargs: keyword arguments to be passed to my_modules.

        Returns:
            A list of keys extracted from the modules.
        """
        modules = self.my_modules(*args, **kwargs)
        return [m['key'] for m in modules]

    def my_key2uid(self, *args, network=None, netuid=0, update=False, **kwargs):
        """
        This function `my_key2uid` takes in `*args`, `network`, `netuid`, `update`, and `**kwargs` as parameters,
        and returns a dictionary `my_key2uid` after performing certain operations on the input parameters.
        """
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
                        **kwargs):
        """
        A function to calculate staked amounts based on various parameters like key, network, and features.
        
        Parameters:
            search (str): A string to search for in the module names.
            key (str): The key to be resolved.
            netuid (int): The unique identifier for the network.
            network (str): The network to be considered (default is 'main').
            df (bool): A flag to indicate whether to return the result as a DataFrame (default is True).
            keys (list): A list of keys to be used.
            max_age (int): The maximum age for the data.
            features (list): A list of features to include in the result.
            **kwargs: Additional keyword arguments.

        Returns:
            dict or pandas.DataFrame: A dictionary or DataFrame containing the staked amounts based on the provided parameters.
        """
        
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
        modules = self.get_modules(keys)
        if search != None:
            modules = [m for m in modules if search in m['name']]
        if df:
            
            for m in modules:
                m['stake_from'] =  int(m['stake_from'].get(key.ss58_address, 0))
                m['stake'] = int(m['stake'])
                m['vote_staleness'] =  max(block - m['last_update'], 0)
            
            modules = [{k: v for k,v in m.items()  if k in features} for m in modules]

            if len(modules) == 0: 
                return modules
            modules = c.df(modules)

            modules = modules.sort_values('stake_from', ascending=False)
            del modules['key']
        return modules

    staked_modules = staked

    
    
    
    def my_keys(self, *args, netuid=0, **kwargs):
        """
        A function that filters keys based on addresses and netuid and returns the filtered keys.
        
        Parameters:
            *args: variable length argument list
            netuid (int): the netuid to resolve
            **kwargs: variable keyword arguments
        
        Returns:
            list: a list of filtered keys based on addresses and netuid
        """
        netuid = self.resolve_netuid(netuid)
        keys = self.keys(*args, netuid=netuid, **kwargs)
        key2address = c.key2address()
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
        uids: Union['torch.LongTensor', list] = None,
        weights: Union['torch.FloatTensor', list] = None,
        netuid: int = 0,
        key: 'c.key' = None,
        network = None,
        update=False,
        max_age = 100,
        **kwargs
    ) -> bool:
        """
        Set weights for the given uids and network, and return the success status and relevant information.

        Args:
            uids (Union['torch.LongTensor', list], optional): The uids to set weights for. Defaults to None.
            weights (Union['torch.FloatTensor', list], optional): The weights corresponding to the uids. Defaults to None.
            netuid (int, optional): The network ID. Defaults to 0.
            key ('c.key', optional): The key to use. Defaults to None.
            network: The network to use. Defaults to None.
            update (bool, optional): Whether to update the weights. Defaults to False.
            max_age: The maximum age of the weights. Defaults to 100.
            **kwargs: Additional keyword arguments.

        Returns:
            bool: The success status of setting the weights.
        """

        network = self.resolve_network(network)
        netuid = self.resolve_netuid(netuid)
        key = self.resolve_key(key)
        module_info = self.get_module(key.ss58_address, netuid=netuid)
        global_params = self.global_params( network=network, update=update, fmt='j')
        subnet_params = self.subnet_params( netuid = netuid )
        
        stake = module_info['stake']
        min_stake = global_params['min_weight_stake'] * subnet_params['min_allowed_weights']
        assert stake > min_stake
        max_num_votes = stake // global_params['min_weight_stake']
        n = int(min(max_num_votes, subnet_params['max_allowed_weights']))
        # checking if the "uids" are passed as names -> strings
        if uids != None and all(isinstance(item, str) for item in uids):
            name2uid = self.name2uid(netuid=netuid, network=network, update=update, max_age=max_age)
            for i, uid in enumerate(uids):
                if uid in name2uid:
                    uids[i] = name2uid[uid]
                else:
                    raise Exception(f'Could not find {uid} in network {netuid}')

        if uids == None:
            # we want to vote for the nonzero dividedn
            uids = self.uids(netuid=netuid, network=network, update=update)
            assert len(uids) > 0, f"No nonzero dividends found in network {netuid}"
            # shuffle the uids
            uids = c.shuffle(uids)
            
        if weights is None:
            weights = [1 for _ in uids]
        
        if len(uids) < subnet_params['min_allowed_weights']:
            n = self.n(netuid=netuid)
            while len(uids) < subnet_params['min_allowed_weights']:
                uid = c.choice(list(range(n)))
                if uid not in uids:
                    uids.append(uid)
                    weights.append(1)

        uid2weight = {uid: weight for uid, weight in zip(uids, weights)}
        # sort the uids and weights
        uid2weight = {k: v for k, v in dict(sorted(uid2weight.items(), key=lambda item: item[1], reverse=True)).items()}
        
        uids = list(uid2weight.keys())[:n]
        weights = list(uid2weight.values())[:n]
        
        if len(uids) == 0:
            return {'success': False, 'message': f'No uids found in network {netuid}'}
        
        assert len(uids) == len(weights), f"Length of uids {len(uids)} must be equal to length of weights {len(weights)}"


        # get uniqe uids
        uid2weight = {uid: weight for uid, weight in zip(uids, weights)}
        uids = list(uid2weight.keys())
        weights = list(uid2weight.values())

        uids = uids[:subnet_params['max_allowed_weights']]
        weights = weights[:subnet_params['max_allowed_weights']]

        import torch
        # sort the uids and weights
        uids = torch.tensor(uids)
        weights = torch.tensor(weights)
        indices = torch.argsort(weights, descending=True)
        uids = uids[indices]
        weights = weights[indices]
        weight_sum = weights.sum()
        weights = weights / (weight_sum)
        weights = weights * (2**16 - 1)
        weights = list(map(lambda x : int(min(x, U16_MAX)), weights.tolist()))

        uids = list(map(int, uids.tolist()))

        params = {'uids': uids,
                  'weights': weights, 
                  'netuid': netuid}
        
        vote_stats = {
            'num_votes': len(uids),
            'num_weights': len(weights),
            'min_allowed_weights': subnet_params['min_allowed_weights'],
            'max_allowed_weights': subnet_params['max_allowed_weights'],
            'min_stake': min_stake,
            'netuid': netuid,
            'max_num_votes': max_num_votes
        }

        c.print('VOTE STATS -->', vote_stats)

        response = self.compose_call('set_weights',params = params , key=key, **kwargs)
            
        if response['success']:
            return {'success': True,  'num_weigts': len(uids), 'message': 'Set weights', 'key': key.ss58_address, 'netuid': netuid, 'network': network}
        

        return {'success': True, 
                'message': 'Voted', 
                'num_uids': len(response['uids']),
                'avg_weight': c.mean(response['weights']),
                'stdev_weight': c.stdev(response['weights'])}
    


    vote = set_weights





    def registered_servers(self, netuid = 0, network = network,  **kwargs):
        """
        Retrieves a list of registered servers for a given network and netuid.

        Parameters:
            netuid (int): The unique identifier of the network.
            network (object): The network object.
            **kwargs: Additional keyword arguments.

        Returns:
            list: A list of registered server keys.
        """
        netuid = self.resolve_netuid(netuid)
        network = self.resolve_network(network)
        servers = c.servers(network='local')
        registered_keys = []
        for s in servers:
            if self.is_registered(s, netuid=netuid):
                registered_keys += [s]
        return registered_keys
    reged = reged_servers = registered_servers

    def unregistered_servers(self, netuid = 0, network = network,  **kwargs):
        """
        A function that returns a list of unregistered servers for a given network and netuid.
        
        Parameters:
            netuid (int): The unique identifier for the network.
            network (str): The network name.
            **kwargs: Additional keyword arguments.
            
        Returns:
            list: A list of unregistered server keys.
        """
        netuid = self.resolve_netuid(netuid)
        network = self.resolve_network(network)
        network = self.resolve_network(network)
        servers = c.servers(network='local')
        unregistered_keys = []
        for s in servers:
            if not self.is_registered(s, netuid=netuid):
                unregistered_keys += [s]
        return unregistered_keys

    
    def check_reged(self, netuid = 0, network = network,  **kwargs):
        """
        Function to check the registration of a network user and retrieve information asynchronously.
        
        Args:
            netuid (int): The network user ID. Defaults to 0.
            network (str): The network to check registration for.
            **kwargs: Additional keyword arguments for the registration check.
            
        Returns:
            dict: A dictionary containing the results of the registration check for each module.
        """
        reged = self.reged(netuid=netuid, network=network, **kwargs)
        jobs = []
        for module in reged:
            job = c.async_call(module=module, fn='info',  network='subspace', netuid=netuid)
            jobs += [job]

        results = dict(zip(reged, c.gather(jobs)))

        return results 

    unreged = unreged_servers = unregistered_servers
               
    def my_balances(self, search=None, 
                    update=False, 
                    fmt='j', 
                    batch_size = 32,
                    timeout = 10,
                    full_scan = 0,
                    min_value=10, **kwargs):
        """
        Generate the balances for a given search, with optional updates and formatting options.

        Parameters:
            search (str): The search parameter to filter balances.
            update (bool): Update flag to refresh balances.
            fmt (str): The format of the balances.
            batch_size (int): The batch size for processing balances.
            timeout (int): The timeout for balance processing.
            full_scan (int): Flag to perform a full scan of balances.
            min_value (int): The minimum value threshold for balances.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the balances for the given search.
        """
        address2key = c.address2key(search)
        future2address = {}
        my_balance = {}
        
        addresses = list(address2key.keys())
        if full_scan:
            balances = self.balances(**kwargs)
        for a in addresses:
            if full_scan:
                if a in balances:
                    my_balance[a] = balances[a]
            else:
                futures = list(future2address.keys())
                if len(future2address) < batch_size:
                    f = c.submit(self.get_balance, args=[a], timeout=timeout)
                    future2address[f] = a
                else:
                    for f in c.as_completed(futures):
                        result = f.result()
                        result_address = future2address.pop(f)
                        if c.is_error(result):
                            c.print(result, color='red')
                        else:
                            balance = f.result()  
                            if balance > 0:
                                c.print(result_address, balance, color='green')
                                my_balance[result_address] = balance
                        break

        return my_balance

        


        # my_balances = {key:balances[address] for address,key in address2key.items() if address in balances}
        # if min_value > 0:
        #     my_balances = {k:v for k,v in my_balances.items() if v > min_value}
        # return my_balances
    

    # def my_balances(self, search=None, update=False, network="main", min_value=10, **kwargs):
    #     address2key = c.address2key(search)
    #     key2balance = {}
    #     for address, key in address2key.items():
    #         c.print(f'Getting balance for {key}')
    #         key2balance[address] = self.get_balance(key)
    #     return key2balance

    


    def my_balance(self, search:str=None, update=False, network:str = 'main', fmt=fmt,  block=None, min_value:int = 0):
        """
        Generate the user's balance by filtering balances and sorting them.  

        :param search: Optional. A string to search for in the key of the balance dictionary.
        :param update: Optional. A boolean indicating whether to update balances.
        :param network: Optional. A string specifying the network to use.
        :param fmt: Required. The format of the balance.
        :param block: Optional. A block to retrieve balances for.
        :param min_value: Optional. An integer specifying the minimum balance value to include.
        
        :return: A dictionary containing the user's filtered and sorted balance.
        """

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

    def my_value(
                 self, 
                 network = 'main',
                 update=False,
                 fmt='j'
                 ):
        """
        Calculate the value by adding the total stake and total balance for the specified network.

        Args:
            self: The instance of the class.
            network (str): The network for which the value is calculated (default is 'main').
            update (bool): Whether to update the data before calculating the value (default is False).
            fmt (str): The format of the value (default is 'j').

        Returns:
            The calculated value as the sum of the total stake and total balance.
        """
        return self.my_total_stake(network=network, update=update, fmt=fmt,) + \
                    self.my_total_balance(network=network, update=update, fmt=fmt)
    
    my_supply   = my_value

    def subnet2stake(self, network=None, update=False) -> dict:
        """
        A function to calculate stake for each subnet in the network and return a dictionary mapping subnet names to stake values.
        Parameters:
            network: Optional parameter to specify the network to calculate stake for.
            update: Optional boolean parameter to indicate whether to update stake values.
        Returns:
            A dictionary containing subnet names as keys and their respective stake values as values.
        """
        subnet2stake = {}
        for subnet_name in self.subnet_names(network=network):
            c.print(f'Getting stake for subnet {subnet_name}')
            subnet2stake[subnet_name] = self.my_total_stake(network=network, netuid=subnet_name , update=update)
        return subnet2stake

    def my_total_stake(self, netuid='all', network = 'main', fmt=fmt, update=False):
        """
        A function that calculates the total stake based on the user's stake, network, and format.

        Parameters:
            netuid (str): The unique identifier of the network stake (default is 'all').
            network (str): The network to calculate the stake on (default is 'main').
            fmt: The format of the stake (assumed to be defined elsewhere in the code).
            update (bool): Flag to indicate if stake information should be updated (default is False).

        Returns:
            int: The total stake calculated based on the parameters provided.
        """
        my_stake_to = self.my_stake_to(netuid=netuid, network=network, fmt=fmt, update=update)
        return sum([sum(list(v.values())) for k,v in my_stake_to.items()])
    




    def staker2stake(self,  update=False, network='main', fmt='j', local=False):
        """
        A function to convert staker information to stake values.
        
        Parameters:
            self: The object instance.
            update (bool): Flag indicating whether to update the information.
            network (str): The network to consider (default is 'main').
            fmt (str): The format of the information (default is 'j').
            local (bool): Flag indicating whether to use local information (default is False).
        
        Returns:
            dict: A dictionary mapping stakers to their corresponding stake values.
        """
        staker2netuid2stake = self.staker2netuid2stake(update=update, network=network, fmt=fmt, local=local)
        staker2stake = {}
        for staker, netuid2stake in staker2netuid2stake.items():
            if staker not in staker2stake:
                staker2stake[staker] = 0
            
        return staker2stake
    

    def staker2netuid2stake(self,  update=False, network='main', fmt='j', local=False, **kwargs):
        """
        A function that converts stake data from one format to another.
        
        Parameters:
            update (bool): Flag indicating whether to update the data.
            network (str): Specifies the network to use (default is 'main').
            fmt (str): Format of the data (default is 'j').
            local (bool): Flag indicating whether the data is local.
            **kwargs: Additional keyword arguments.
        
        Returns:
            dict: A dictionary containing the converted stake data.
        """
        stake_to = self.query_map("StakeTo", update=update, network=network, **kwargs)
        staker2netuid2stake = {}
        for netuid , stake_to_subnet in stake_to.items():
            for staker, stake_tuples in stake_to_subnet.items():
                staker2netuid2stake[staker] = staker2netuid2stake.get(staker, {})
                staker2netuid2stake[staker][netuid] = staker2netuid2stake[staker].get(netuid, [])
                staker2netuid2stake[staker][netuid] = sum(list(map(lambda x: x[-1], stake_tuples )))
                staker2netuid2stake[staker][netuid] +=  self.format_amount(staker2netuid2stake[staker][netuid],fmt=fmt)
        
        if local:
            address2key = c.address2key()
            staker2netuid2stake = {address:staker2netuid2stake.get(address,{}) for address in address2key.keys()}

        return staker2netuid2stake
    

 
    def my_total_balance(self, network = None, fmt=fmt, update=False):
        """
        A function that calculates the total balance based on the individual balances in the network.

        :param network: The network for which the balance needs to be calculated.
        :param fmt: The format of the balance.
        :param update: A boolean indicating whether to update the balance.
        
        :return: The total balance calculated from the individual balances in the network.
        """
        return sum(self.my_balance(network=network, fmt=fmt, update=update ).values())


    def check_valis(self, **kwargs):
        """
        A function that checks for 'vali' servers using the check_servers method.
        
        Parameters:
            **kwargs: additional keyword arguments to pass to check_servers method.
        
        Returns:
            The result of calling check_servers method with 'search' set to 'vali' and additional keyword arguments if any.
        """
        return self.check_servers(search='vali', **kwargs)
    
    
    def check_servers(self, search='vali',update:bool=False, netuid=0, min_lag=100, remote=False, **kwargs):
        """
        A function to check servers with various parameters like search keyword, update flag, netuid, min_lag, and remote flag.
        Returns a dictionary with server responses for each module.
        """
        if remote:
            kwargs = c.locals2kwargs(locals())
            return self.remote_fn('check_servers', kwargs=kwargs)
        features = ['name', 'serving', 'address', 'last_update', 'stake', 'dividends']
        module_stats = self.stats(search=search, netuid=netuid, features=features, df=False, update=update)
        module2stats = {m['name']:m for m in module_stats}
        block = self.block
        response_batch = {}
        c.print(f"Checking {len(module2stats)} {search} servers")
        for module, stats in module2stats.items():
            # check if the module is serving
            lag = block - stats['last_update']
            if not c.server_exists(module) or lag > min_lag:
                response  = c.serve(module)
            else:
                response = f"{module} is already serving or has a lag of {lag} blocks but less than {min_lag} blocks"
            response_batch[module] = response

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
        A function to compose a call with various parameters and options, including network, module, key, and more. 
        It resolves the key and network, then either connects to a remote module or composes a transaction locally. 
        It handles various options like verbosity, nonce, and unchecked weight for calls. 
        Finally, it submits the extrinsic, waits for its inclusion and finalization, and processes events if needed.
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

        c.print(f'Sending Transaction: 📡', compose_kwargs, color=color)
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
        """
        A method to retrieve transaction history based on the provided key, mode, and network.
        
        Parameters:
            key (str): The key to retrieve transaction history for. Defaults to None.
            mode (str): The mode of history to retrieve, must be either 'pending' or 'complete'.
            network: The network to retrieve transaction history from.
            **kwargs: Additional keyword arguments to pass to the function.
        
        Returns:
            The transaction history based on the provided key, mode, and network.
        """
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
        """
        Resolve the given key by checking if it exists, getting it if it does, or setting a default value.
        
        Parameters:
            key (str): The key to be resolved. If None, it defaults to the key from the configuration or 'module'.
        
        Returns:
            str: The resolved key with ss58_address attribute.
        """
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
    
    
    def unstake2key(self, key=None):
        """
        Method for unstaking 2key tokens, with an optional key parameter.
        """
        key2stake = self.key2stake()
        c.print(key2stake)


    def test_subnet_storage(self):
        """
        Test the subnet storage function by checking the types of subnet parameters and returning a success message with the count of all subnet parameters.
        """

        all_subnet_params = self.subnet_params(netuid='all')
        assert isinstance(all_subnet_params, list)
        for subnet_params in all_subnet_params: 
            assert isinstance(subnet_params, dict)
        subnet_params = self.subnet_params(netuid=10)
        assert isinstance(subnet_params, dict)
        return {'success': True, 'msg': 'All subnet params are dictionaries', 'n': len(all_subnet_params)}
    
    def test_global_storage(self):
        """
        This function retrieves global parameters and returns them as a dictionary.
        """
        global_params = self.global_params()
        assert isinstance(global_params, dict)
        return global_params
    
    def test_module_storage(self):
        """
        A function to test the module storage by retrieving modules with a specific netuid.
        
        Parameters:
        self (obj): The object instance
       
        Returns:
        modules (list): A list of modules retrieved based on the netuid provided
        """
        modules = self.get_modules(netuid=0)
        return modules 

    @classmethod
    def test(cls):
        """
        A class method to test the functionality of the 'subspace' module.
        Checks various attributes and their types for validity.
        """
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
