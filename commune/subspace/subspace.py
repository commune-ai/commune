
from retry import retry
from typing import *
import json
import os
import commune as c
import requests 
from substrateinterface import SubstrateInterface

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
                print('ERROR IN CONNECTION: ', c.detailed_error(e), self.config)
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
                    self.dict_put(new_qmap, k, v)

            self.put(path, new_qmap)
        
        else: 
            new_qmap = value

        def process_qmap(d):
            is_int_bool = False
            for k,v in c.copy(d).items():
                if c.is_int(k):
                    is_int_bool = True
                    d[int(k)] = d.pop(k)
                    if isinstance(v, dict):
                        d[int(k)] = process_qmap(v)
            if is_int_bool:
                # sort the dictionary by key
                d = dict(sorted(d.items()))
            return d
        
        new_map = process_qmap(new_qmap)

        return new_map
    
    def runtime_spec_version(self, network:str = 'main'):
        # Get the runtime version
        c.print(self.substrate.runtime_config.__dict__)
        runtime_version = self.query_constant(module_name='System', constant_name='SpVersionRuntimeVersion')
        return runtime_version
        
        
    #####################################
    #### Hyper parameter calls. ####
    #####################################


    """ Returns the stake under a coldkey - hotkey pairing """

    def from_nano(self,x):
        return x / (10**self.config.token_decimals)
    to_token = from_nano


    def to_nanos(self,x):
        """
        Converts a token amount to nanos
        """
        return x * (10**self.config.token_decimals)
    from_token = to_nanos

    def format_amount(self, x, fmt='nano', decimals = None, format=None, features=None, **kwargs):
        fmt = format or fmt # format is an alias for fmt

        if fmt in ['token', 'unit', 'j', 'J']:
            x = x / 10**9
        
        if decimals != None:
            x = c.round_decimals(x, decimals=decimals)

        return x
    



    ###########################
    #### Global Parameters ####
    ###########################

    @property
    def block(self) -> int:
        return self.substrate.get_block_number(block_hash=None)

    
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

    def feature2storage(self, feature:str):
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

    def feature2name(self, feature='MinStake'):
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

    def name2feature(self, name='min_stake_fam'):
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

    def get_account(self, key = None,  update=True):
        key = self.resolve_key_ss58(key)
        account = self.substrate.query(
            module='System',
            storage_function='Account',
            params=[key],
        )
        return account
    
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

    def query_vector(self, name='Trust', netuid = 0, update=False, **kwargs):
        if isinstance(netuid, int):
            query_vector = self.query(name,  netuid=netuid,  update=update, **kwargs)
        else:
            query_vector = self.query_map(name, netuid=netuid,  update=update, **kwargs)
            if len(query_vector) == 0:
                query_vector = {_: [] for _ in range(len(self.netuids()))}
        return query_vector

    def get_nonce(self, key:str=None,  **kwargs):
        key_ss58 = self.resolve_key_ss58(key)
        return self.substrate.get_account_nonce(key_ss58)

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
    

    def get_feature(self, feature='names', network='main', netuid=0, update=False, max_age=1000, **kwargs):
        return getattr(self(network=network), feature)(netuid=netuid, update=update, max_age=max_age, **kwargs)
        




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
    
    def pending_txs(self, key:str=None, **kwargs):
        return self.tx_history(key=key, mode='pending', **kwargs)

    def complete_txs(self, key:str=None, **kwargs):
        return self.tx_history(key=key, mode='complete', **kwargs)

    def clean_tx_history(self):
        return self.ls(f'tx_history')
        
    def resolve_tx_dirpath(self, key:str=None, mode:str ='pending',  **kwargs):
        key_ss58 = self.resolve_key_ss58(key)
        assert mode in ['pending', 'complete']
        pending_path = f'history/{self.network}/{key_ss58}/{mode}'
        return pending_path
     
    def tx_history(self, key:str=None, mode='complete', **kwargs):
        key_ss58 = self.resolve_key_ss58(key)
        assert mode in ['pending', 'complete']
        pending_path = f'history/{self.network}/{key_ss58}/{mode}'
        return self.glob(pending_path)
    

    def emissions(self, netuid = 0, network = "main", block=None, update=False, **kwargs):
        return self.query_vector('Emission', network=network, netuid=netuid, block=block, update=update, **kwargs)

    def subnet2stake(self, network=None, update=False) -> dict:
        subnet2stake = {}
        for subnet_name in self.subnet_names(network=network):
            c.print(f'Getting stake for subnet {subnet_name}')
            subnet2stake[subnet_name] = self.my_total_stake(network=network, netuid=subnet_name , update=update)
        return subnet2stake


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
    


    def subnet2netuid(self, subnet=None,  update=False,  **kwargs ) -> Dict[str, str]:
        subnet2netuid =  {v:k for k,v in self.netuid2subnet( update=update, **kwargs).items()}
        # sort by subnet 
        subnet2netuid = {k:v for k,v in sorted(subnet2netuid.items(), key=lambda x: x[0].lower())}
        if subnet != None:
            return subnet2netuid[subnet] if subnet in subnet2netuid else len(subnet2netuid)
        return subnet2netuid



    def netuids(self,  update=False, block=None) -> Dict[int, str]:
        return list(self.netuid2subnet( update=update, block=block).keys())


    def netuid2subnet(self, netuid=None,  update=False, block=None, **kwargs ) -> Dict[str, str]:
        netuid2subnet = self.query_map('SubnetNames', update=update,  block=block, **kwargs)
        if netuid != None:
            return netuid2subnet[netuid]
        return netuid2subnet


    def key2name(self, key: str = None, netuid: int = 0) -> str:
        modules = self.keys(netuid=netuid)
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
        



    def subnet_names(self , search=None, update=False, block=None, max_age=60, **kwargs) -> Dict[str, str]:
        records = self.query_map('SubnetNames', update=update,  block=block, max_age=max_age, **kwargs)
        subnet_names = sorted(list(map(lambda x: str(x), records.values())))
        if search != None:
            subnet_names = [s for s in subnet_names if search in s]
        return subnet_names
    

    def subnets(self, **kwargs) -> Dict[int, str]:
        return self.subnet_names(**kwargs)
    
    def num_subnets(self, **kwargs) -> int:
        return len(self.subnets(**kwargs))
    


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


    """ Returns network SubnetN hyper parameter """
    def n(self,  netuid: int = 0,block: Optional[int] = None, max_age=100, update=False, **kwargs ) -> int:
        if netuid == 'all':
            return sum(self.query_map('N', block=block , update=update, max_age=max_age,  **kwargs).values())
        else:
            return self.query( 'N', params=[netuid], block=block , update=update,  **kwargs)

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
    


    def min_register_stake(self, netuid: int = 0, fmt='j', **kwargs) -> float:
        netuid = self.resolve_netuid(netuid)
        min_burn = self.min_burn(  fmt=fmt)
        min_stake = self.min_stake(netuid=netuid,  fmt=fmt)
        return min_stake + min_burn
    



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

    


    def format_module(self, module: 'ModuleInfo', fmt:str='j') -> 'ModuleInfo':
        for k in ['emission']:
            module[k] = self.format_amount(module[k], fmt=fmt)
        for k in ['incentive', 'dividends']:
            module[k] = module[k] / (U16_MAX)
        
        module['stake_from'] = {k: self.format_amount(v, fmt=fmt)  for k, v in module['stake_from']}
        return module
    


    def min_stake(self, netuid: int = 0, fmt:str='j', **kwargs) -> int:
        min_stake = self.query('MinStake', netuid=netuid,  **kwargs)
        return self.format_amount(min_stake, fmt=fmt)




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
    
    

    def pending_deregistrations(self, netuid = 0, update=False, **kwargs):
        pending_deregistrations = self.query_map('PendingDeregisterUids',update=update,**kwargs)[netuid]
        return pending_deregistrations
    
    def num_pending_deregistrations(self, netuid = 0, **kwargs):
        pending_deregistrations = self.pending_deregistrations(netuid=netuid, **kwargs)
        return len(pending_deregistrations)
        



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
    

    def get_stake_from( self, key: str, from_key=None, block: Optional[int] = None, netuid:int = None, fmt='j', update=True  ) -> Optional['Balance']:
        key = self.resolve_key_ss58( key )
        netuid = self.resolve_netuid( netuid )
        stake_from = self.query( 'StakeFrom', params=[netuid, key], block=block,  update=update )
        state_from =  [(k, self.format_amount(v, fmt=fmt)) for k, v in stake_from ]
 
        if from_key != None:
            from_key = self.resolve_key_ss58( from_key )
            state_from ={ k:v for k, v in state_from}.get(from_key, 0)

        return state_from
    


    def get_stake( self, key_ss58: str, block: Optional[int] = None, netuid:int = None , fmt='j', update=True ) -> Optional['Balance']:
        
        key_ss58 = self.resolve_key_ss58( key_ss58)
        netuid = self.resolve_netuid( netuid )
        stake = self.query( 'Stake',params=[netuid, key_ss58], block=block , update=update)
        return self.format_amount(stake, fmt=fmt)



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
    



    def weights(self,  netuid = 0,  update=False, **kwargs) -> list:
        weights =  self.query_map('Weights',netuid=netuid, update=update, **kwargs)

        return weights



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
    
    
    def get_uid( self, key: str, netuid: int = 0, block: Optional[int] = None, update=False, **kwargs) -> int:
        return self.query( 'Uids', block=block, params=[ netuid, key ] , update=update, **kwargs)  


    def total_emission( self, netuid: int = 0, block: Optional[int] = None, fmt:str = 'j', **kwargs ) -> Optional[float]:
        total_emission =  sum(self.emission(netuid=netuid, block=block, **kwargs))
        return self.format_amount(total_emission, fmt=fmt)


    def blocks_until_vote(self, netuid=0, **kwargs):
        netuid = self.resolve_netuid(netuid)
        tempo = self.subnet_params(netuid=netuid, **kwargs)['tempo']
        block = self.block
        return tempo - ((block + netuid) % tempo)



    def epoch_time(self, netuid=0, update=False, **kwargs):
        return self.subnet_params(netuid=netuid, update=update, **kwargs)['tempo']*self.block_time


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
                    module=None,
                    netuid=0,
                    trials = 4,
                    fmt='j',
                    mode = 'http',
                    block = None,
                    max_age = None,
                    lite = True, 
                    update = False,
                    **kwargs ) -> 'ModuleInfo':
        if module == None:
            module = self.keys(netuid=netuid, update=update, max_age=max_age)[0]
            c.print(f'No module specified, using {module}')

        url = self.resolve_url( mode=mode)
        module_key = module
        is_valid_key = c.valid_ss58_address(module)
        print(is_valid_key, module_key)
        if not is_valid_key:
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
        print(module)
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




    @staticmethod
    def vec82str(l:list):
        return ''.join([chr(x) for x in l]).strip()





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



       
    
Subspace.enable_routes()
Subspace.run(__name__)


