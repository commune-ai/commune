
from retry import retry
from typing import *
import json
import os
import commune as c
import requests 
from .subnet import SubspaceSubnet
from .wallet import SubspaceWallet
from substrateinterface import SubstrateInterface

class Subspace( SubspaceSubnet, SubspaceWallet, c.Module):
    """
    Handles interactions with the subspace chain.
    """
    block_time = 8
    token_decimals = 9

    whitelist = ['query', 
                 'score',
                 'query_map', 
                 'get_module', 
                 'get_balance', 
                 'get_stake_to', 
                 'get_stake_from']

    supported_modes = ['http', 'ws']

    def __init__(self, 
                network: str =  'main',
                network_mode: str =  'ws',
                subnet: str = 'commune',
                url: str = None,
                url_search: str  = 'commune',
                url_path: str = None,
                netuid : int =  0,
                max_age: int = 1000,
                sync_loop = False,
                **kwargs,
        ):


        
        self.config = self.set_config(locals())
        self.url_path = self.dirpath() +  '/urls.yaml'
        # merge the config with the subspace config
        self.config = c.dict2munch({**Subspace.config(), **self.config})
        self.set_network(network )
        self.set_netuid(netuid)
        if sync_loop:
            c.thread(self.sync_loop)
    
    init_subspace = __init__    

    ###########################
    #### Global Parameters ####
    ###########################

    def set_netuid(self, netuid:int):
        self.netuid = netuid
        return self.netuid

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

    def name2feature(self, name='min_stake_fam'):
        chunks = name.split('_')
        return ''.join([c.capitalize() for c in chunks])

    def get_account(self, key = None):
        key = self.resolve_key_ss58(key)
        account = self.substrate.query(
            module='System',
            storage_function='Account',
            params=[key],
        )
        return account


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
    

    @staticmethod
    def vec82str(l:list):
        return ''.join([chr(x) for x in l]).strip()


    def clean_keys(self, 
                   min_value=1,
                   update = False):
        """
        description:
            Removes keys with a value less than min_value
        params:
            network: str = 'main', # network to remove keys from
            min_value: int = 1, # min value of the key
            update: bool = True, # update the key2value cache
            max_age: int = 0 # max age of the key2value cache
        """
        key2value= self.key2value(netuid='all', update=update, fmt='j', min_value=0)
        address2key = c.address2key()
        keys_left = []
        rm_keys = []
        for key in address2key.values():
            key_value = key2value.get(key, 0)
            if key_value < min_value:
                c.print(f'Removing key {key} with value {key_value}')
                c.rm_key(key)
                rm_keys += [key]
            else:
                keys_left += [key]
        return {'success': True, 'msg': 'cleaned keys', 'keys_left': len(keys_left), 'rm_keys': len(rm_keys)}

    
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
       
    def launcher_keys(self, netuid=0, min_stake=500, **kwargs):
        keys = c.keys()
        key2balance =  c.key2balance(netuid=netuid,**kwargs)
        key2balance = {k: v for k,v in key2balance.items() if v > min_stake}
        return [k for k in keys]

    def resolve_key(self, key = None):

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


    def filter_url(self, url):
        """
        Filter urls based on the url_search parameter
        """
        if self.config.url_search == None:
            return True
        url_search_terms = [url.strip() for x in self.config.url_search.split(',')]
        return any([x in url for x in url_search_terms])
    

    def resolve_url(self, 
                    url = None, 
                    mode='ws', 
                    network=None, 
                    **kwargs):
        mode =  mode or self.config.network_mode
        url = url or self.config.url
        assert mode in self.supported_modes
        if url != None:
            return url
        network = self.resolve_network(network)
        if url == None:
            urls_map = getattr(self.urls(),  network)
            urls = urls_map.get(mode, [])
            assert len(urls) > 0, f'No urls found for network {network} and mode {mode}'
            if len(urls) > 1:
                urls_map = list(filter(self.filter_url, urls))
            url = c.choice(urls)
        return url
    

    @property
    def network(self):
        return self.config.network
    
    @network.setter
    def network(self, value):
        self.config.network = value
    

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
                network=None,
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
        

        for i in range(trials):
            try:
          
                url = self.resolve_url(url, mode=mode, network=network)

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
                    
                self.url = url
                self.url2substrate[url] = substrate
                return substrate
            except Exception as e:
                print('ERROR IN CONNECTION: ', c.detailed_error(e), self.config)
                if i == trials - 1:
                    raise e
                
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
        response =  {'network': self.network, 'url': self.url}
        c.print(response)
        return response

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

        module = self.resolve_query_module_from_name(name)

        path = f'query/{self.config.network}/{module}.{name}'
        params = params or []
        if not isinstance(params, list):
            params = [params]
        if netuid != None and netuid != 'all':
            params = [netuid] + params
        # we want to cache based on the params if there are any
        path = path + f'::params::' + '-'.join([str(p) for p in params]) if len(params) > 0 else path
        value = self.get(path, default=None, max_age=max_age, update=update)
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
    

    def events(self, block=None, max_age=1000, update=False):
        path = f'events/{self.network}'
        events = self.get(path, None, max_age=max_age, update=update)
        if events == None:
            events = self.substrate.get_events(block_hash=self.block_hash(block))
            events = list(map(lambda x: x.value, events))
            self.put(path, events)
        return events
    

    def resolve_query_module_from_name(self, name):
        if name  == 'Account':
            module = 'System'
        elif name in ['SubnetGovernanceConfig', 'GlobalGovernanceConfig'] :
            module = 'GovernanceModule'
        else:
            module = 'SubspaceModule'

        return module

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
                  trials = 1,
                  **kwargs
                  ) -> Optional[object]:
        """ Queries subspace map storage with params and block. """
        # if all lowercase then we want to capitalize the first letter

        module = self.resolve_query_module_from_name(name)
        path = f'query_map/{self.config.network}/{module}.{name}'
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
                    print(f'Querying {name} with params {params} and block {block}')
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
            keys = list(c.copy(d).keys())
            for k in keys:
                v = d[k]
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
    
    def runtime_spec_version(self):
        # Get the runtime version
        c.print(self.substrate.runtime_config.__dict__)
        runtime_version = self.query_constant(module_name='System', constant_name='SpVersionRuntimeVersion')
        return runtime_version
        
        


    def from_nano(self,x):
        return x / (10**self.token_decimals)
    to_token = from_nano


    def to_nanos(self,x):
        """
        Converts a token amount to nanos
        """
        return x * (10**self.token_decimals)
    from_token = to_nanos



    """ Returns the stake under a coldkey - hotkey pairing """

    def format_amount(self, x, fmt='nano', decimals = None, format=None, features=None, **kwargs):
        fmt = format or fmt # format is an alias for fmt

        if fmt in ['token', 'unit', 'j', 'J']:
            x = x / 10**9
        
        if decimals != None:
            x = c.round_decimals(x, decimals=decimals)

        return x
    

    @property
    def block(self) -> int:
        return self.substrate.get_block_number(block_hash=None)




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


    
    def resolve_network(self, 
                        network: Optional[int] = None,
                        **kwargs) -> int:
        """
        Resolve the network to use for the current session.
        
        """
        network = network or self.config.network
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
            storage_names = [s for s in storage_names if search.lower() in s.lower()]
        return storage_names

    def check_storage(self, block_hash = None):
        return self.substrate.get_metadata_storage_functions( block_hash=block_hash)
    
    def get_feature(self, feature='names', network=None, netuid=0, update=False, max_age=1000, **kwargs):
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
                    network = None,
                     **kwargs):

        """
        Composes a call to a Substrate chain.

        """
        network = self.resolve_network(network)
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
        paths = {m: f'history/{network}/{ss58_address}/{m}/{start_time}.json' for m in ['complete', 'pending']}
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
    

    # set the network if network is specified

    protected_attributes = [ 'set_network', 'protected_attributes']
    def __getattr__(self, key):
        if key in self.protected_attributes:
            return getattr(self, key)
        else:
            def wrapper(*args, network=None, **kwargs):
                if network is not None:
                    self.set_network(network)
                elif 'network' in kwargs:
                    self.set_network(kwargs['network'])
                return getattr(self, key)(*args, **kwargs)
            return wrapper


    def unit_emission(self, block=None, **kwargs):
        return self.query_constant( "UnitEmission", block=block)

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

    global_param_features = [
            'MaxNameLength',
            'MaxAllowedModules',
            'MaxAllowedSubnets',
            'MaxRegistrationsPerBlock',
            'MinWeightStake',
    ]

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
        
        features = features or self.global_param_features
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

    def urls(self):
        return c.dict2munch(c.load_yaml(self.url_path))
    

    def global_state(self, max_age=None, update=False):
        max_age = max_age or self.config.max_age
        global_state = self.get('global_state', None, max_age=max_age, update=update)
        if global_state == None :
            params = self.global_params(max_age=max_age)
            subnet2netuid = self.subnet2netuid(max_age=max_age)
            subnet_names = list(subnet2netuid.keys())
            netuids = list(subnet2netuid.values())
            subnet2emission = self.subnet2emission(max_age=max_age)
            global_state =  {
                'params': params,
                'subnet2netuid': subnet2netuid,
                'subnet_names': subnet_names,
                'netuids': netuids,
                'subnet2emission': subnet2emission
            }
        return global_state

    def subnet_state(self, netuid=0, max_age=None, timeout=60):

        max_age = max_age or self.config.max_age
        subnet_state = self.get(f'subnet_state/{netuid}', None, netuid=netuid, max_age=max_age)
        if subnet_state == None:
            subnet_params = self.subnet_params(netuid=netuid, max_age=max_age)
            subnet_modules = self.get_modules(netuid=netuid,  max_age=max_age)
            subnet_name = subnet_params['name']
            subnet_state = {
                'params': subnet_params,
                'netuid': netuid,
                'name': subnet_name,
                'modules': subnet_modules
            }
        return subnet_state
    
    def sync(self, max_age=None):
        try:
            self.state(max_age=max_age)
        except Exception as e:
            c.print(f'Error in syncing {e}')
    
    def state(self, max_age=None):
        max_age = max_age or self.config.max_age
        path = f'state/{self.network}'
        state_dict = self.get(path, None, max_age=max_age)
        if state_dict == None:
            global_state = self.global_state( max_age=max_age)
            progress_bar = c.tqdm(total=len(self.netuids))
            subnet_state = {}
            for netuid in self.netuids:
                c.print(f"Syncing {netuid}")
                subnet_state[netuid] = self.subnet_state(**{'netuid': netuid, 'max_age': max_age})     
                progress_bar.update(1)
            state = {'global': global_state, 'subnets': subnet_state}
            self.put(path, state)
        return {'msg': 'synced', 'netuids': self.netuids, 'subnet_names': self.subnet_names}
    
    def sync_loop(self,max_age=None):
        max_age = max_age or self.config.max_age

        c.print(f'Starting Sync Loop max_age={max_age}')
        futures = []
        while True:
            if len(futures) > 0:
                c.print('Waiting for futures to complete')
                
            futures += [c.submit(self.sync)]
            c.print(c.wait(futures, timeout=self.config.max_age))
            c.print('Synced all subnets, sleeping')
            c.sleep(self.config.max_age)

Subspace.run(__name__)

