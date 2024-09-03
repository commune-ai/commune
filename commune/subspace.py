
from retry import retry
from typing import *
import json
import os
import commune as c
import requests 
from substrateinterface import SubstrateInterface

__ss58_format__ = 42


class Subspace(c.Module):
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
            urls_map = self.urls()
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
    
    def urls(self):
        return c.get_yaml(self.dirpath() + '/urls.yaml').get(self.network)

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
        # resolving the params
        params = params or []
        params = [netuid] + params if bool(netuid != 'all' and netuid != None) else params
        params = params if isinstance(params, list) else [params]
        path = f'query_map/{self.network}/{module}.{name}'
        if len(params) > 0 :
            path = path + f'::params::' + '-'.join([str(p) for p in params])
        value = self.get(path, None , max_age=max_age, update=update)
        if value == None:
            # if the value is a tuple then we want to convert it to a list
            substrate = self.get_substrate( mode=mode)
            c.print(f'Querying {name} with params {params} and block {block}')
            qmap =  substrate.query_map(
                module=module,
                storage_function = name,
                params = params,
                page_size = page_size,
                max_results = max_results,
                block_hash =substrate.get_block_hash(block)
            )
            new_qmap = {} 
            for (k,v) in qmap:
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




 ##################
    #### Transfer ####
    ##################
    def transfer(
        self,
        dest: str, 
        amount: float , 
        key: str = None,
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
        amounts = [a*(10**9) for a in amounts]

        params = {
            "destinations": destinations,
            "amounts": amounts
        }

        return self.compose_call('transfer_multiple', params=params, key=key)

    transfer_many = transfer_multiple

    def my_modules(self,
                    modules : list = None,
                    netuid=0,
                    timeout=30,
                    **kwargs):
        if modules == None:
            modules = self.my_keys(netuid=netuid)
        futures = [c.submit(self.get_module, kwargs=dict(module=module, netuid=netuid, **kwargs)) for module in modules]
        for future in c.as_completed(futures, timeout=timeout):
            module = future.result()
            print(module)
            if not c.is_error(module):
                modules += [module]
        return modules

    def unstake_many( self, 
                        
                        modules:Union[List[str], str] = None,
                        amounts:Union[List[str], float, int] = None,
                        key: str = None, 
                        netuid=0, 
                        update=True,
                        
                        ) -> Optional['Balance']:
        
        key = self.resolve_key( key )
        name2key = {}

        module_keys = []
        for i, module in enumerate(modules):
            if c.valid_ss58_address(module):
                module_keys += [module]
            else:
                if name2key == {}:
                    name2key = self.name2key(netuid=netuid, update=update)
                assert module in name2key, f"Invalid module {module} not found in SubNetwork {netuid}"
                module_keys += [name2key[module]]
            
        # RESOLVE AMOUNTS
        if amounts == None:
            stake_to = self.get_stake_to(key=key, names=False, update=update, fmt='j') # name to amounts
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
        params = None
    ) -> bool:
        key = self.resolve_key(module)
        netuid = self.resolve_netuid(netuid)  
        assert self.is_registered(key.ss58_address, netuid=netuid), f"Module {module} is not registered in SubNetwork {netuid}"
        if params == None:
            params = {
                'name': name , # defaults to module.tage
                'address': address , # defaults to module.tage
                'delegation_fee': fee or delegation_fee, # defaults to module.delegate_fee
                'metadata': c.python2str(metadata or {}), # defaults to module.metadata
            }

        should_update_module = False
        module_info = self.get_module(key.ss58_address, netuid=netuid)

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

    update = update_server = update_module

    def stake_transfer(
            self,
            module_key: str ,
            new_module_key: str ,
            amount: Union[int, float] = None, 
            key: str = None,
        ) -> bool:
        # STILL UNDER DEVELOPMENT, DO NOT USE
        key = c.get_key(key)
        netuid = 0
        module_key = self.resolve_module_key(module_key, netuid=netuid)
        new_module_key = self.resolve_module_key(new_module_key, netuid=netuid)

        assert module_key != new_module_key, f"Module key {module_key} is the same as new_module_key {new_module_key}"

        if amount == None:
            amount = self.get_stake_to( key=key , fmt='j', max_age=0).get(module_key, 0)
        
        # Get current stake
        params={
                    'amount': int(amount * 10**9),
                    'module_key': module_key,
                    'new_module_key': new_module_key

                    }

        return self.compose_call('transfer_stake',params=params, key=key)

    def unstake(
            self,
            module : str = None, # defaults to most staked module
            amount: float =None, # defaults to all of the amount
            key : 'c.Key' = None,  # defaults to first key
            netuid=0,
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
        # get most stake from the module

        if isinstance(module, int):
            module = amount
            amount = module

        assert module != None or amount != None, f"Must provide a module or an amount"
        key2address = c.key2address()
        if module in key2address:
            module_key = key2address[module]
        else:
            name2key = self.name2key(netuid=netuid)
            if module in name2key:
                module_key = name2key[module]
            else:
                module_key = module
        assert self.is_registered(module_key, netuid=netuid), f"Module {module} is not registered in SubNetwork {netuid}"
        
        if amount == None:
            stake_to = self.get_stake_to(names = False, fmt='nano', key=module_key)
            amount = stake_to[module_key] - 100000
        else:
            amount = int(self.to_nanos(amount))
        # convert to nanos
        params={
            'amount': amount ,
            'module_key': module_key
            }
        response = self.compose_call(fn='remove_stake',params=params, key=key, **kwargs)

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

    def my_keys(self,  
                netuid=0, 
                search=None, 
                max_age=None, 
                names  = False,
                update=False, **kwargs):
        netuid = self.resolve_netuid(netuid)
        keys = self.keys(netuid=netuid, max_age=max_age, update=update, **kwargs)
        key2address = c.key2address(search=search, max_age=max_age, update=update)
        if search != None:
            key2address = {k: v for k,v in key2address.items() if search in k}
        address2key = {v:k for k,v in key2address.items()}
        addresses = list(key2address.values())
        convert_fn = lambda x : address2key.get(x, x) if names else x
        if netuid == 'all':
            my_keys = {}
            for netuid, netuid_keys in keys.items():
                
                my_netuid_keys = [convert_fn(k) for k in netuid_keys if k in addresses]
                
                if len(my_netuid_keys) > 0:
                    my_keys[netuid] = my_netuid_keys
        else:
            my_keys = [convert_fn(k) for k in keys if k in addresses]
        return my_keys


    def my_value( self, *args, **kwargs ):
        return sum(list(self.key2value( *args, **kwargs).values()))
    

    def my_total_stake(self, netuid='all', fmt='j', update=False):
        my_stake_to = self.my_stake_to(netuid=netuid,  fmt=fmt, update=update)
        return sum([sum(list(v.values())) for k,v in my_stake_to.items()])
    
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

    def my_stake_to(self, fmt='j', **kwargs):
        stake_to = self.stake_to(fmt=fmt, **kwargs)
        key2address = c.key2address()
        my_stake_to = {}
        for key, address in key2address.items():
            my_stake_to[address] = {k:v  for k,v in stake_to.get(address, {}).items()}
        stake_to_keys = list(my_stake_to.keys())
        for key in stake_to_keys:
            if len(my_stake_to[key]) == 0:
                del my_stake_to[key]

        return my_stake_to
    



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



    def my_total_stake_to( self, 
                     key: str = None, 
                       block: Optional[int] = None, 
                       timeout=20,
                       names = False,
                        fmt='j' ,
                        update=False,
                        max_age = 1000,
                         **kwargs) -> Optional['Balance']:
        return sum(list(self.get_stake_to(key=key,  block=block, timeout=timeout, names=names, fmt=fmt, 
                                  update=update, 
                                 max_age=max_age, **kwargs).values()))
        
    
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





    def stake(
            self,
            module: Optional[str] = None, # defaults to key if not provided
            amount: Union['Balance', float] = None, 
            key: str = None,  # defaults to first key
            existential_deposit: float = 0,
            netuid=0,
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
                    'amount': amount,
                    'module_key': module_key
                    }

        return self.compose_call('add_stake',params=params, key=key)




    def key_info(self, key:str = None, netuid='all', detail=0, timeout=10, update=False, **kwargs):
        key_info = {
            'balance': self.get_balance(key=key, **kwargs),
            'stake_to': self.get_stake_to(key=key, **kwargs),
        }
        if detail: 
            pass
        else: 
            for netuid, stake_to in key_info['stake_to'].items():
                key_info['stake_to'][netuid] = sum(stake_to.values())


        return key_info

    


    def subnet2modules(self, **kwargs):
        subnet2modules = {}

        for netuid in self.netuids():
            c.print(f'Getting modules for SubNetwork {netuid}')
            subnet2modules[netuid] = self.my_modules(netuid=netuid, **kwargs)

        return subnet2modules
    
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
    
    def empty_keys(self,  block=None, update=False, max_age=1000, fmt='j'):
        key2address = c.key2address()
        key2value = self.key2value( block=block, update=update, max_age=max_age, fmt=fmt)
        empty_keys = []
        for key,key_address in key2address.items():
            key_value = key2value.get(key_address, 0)
            if key_value == 0:
                empty_keys.append(key) 
        return empty_keys
    
    def profit_shares(self, key=None, **kwargs) -> List[Dict[str, Union[str, int]]]:
        key = self.resolve_module_key(key)
        return self.query_map('ProfitShares',  **kwargs)

    def key2stake(self, 
                     block=None, 
                    update=False, 
                    names = True,
                    max_age = 1000,fmt='j'):
        stake_to = self.stake_to(
                                block=block, 
                                max_age=max_age,
                                update=update, 
                                 
                                fmt=fmt)
        address2key = c.address2key()
        stake_to_total = {}
 
        for staker_address in address2key.keys():
            if staker_address in stake_to:
                stake_to_total[staker_address] = sum(stake_to.get(staker_address, {}).values())
        # sort the dictionary by value
        stake_to_total = dict(sorted(stake_to_total.items(), key=lambda x: x[1], reverse=True))
        if names:
            stake_to_total = {address2key.get(k, k): v for k,v in stake_to_total.items()}
        return stake_to_total

    def key2value(self,  block=None, update=False, max_age=1000, fmt='j', min_value=0, **kwargs):
        key2balance = self.key2balance(block=block, update=update,  max_age=max_age, fmt=fmt)
        key2stake = self.key2stake( block=block, update=update,  max_age=max_age, fmt=fmt)
        key2value = {}
        keys = set(list(key2balance.keys()) + list(key2stake.keys()))
        for key in keys:
            key2value[key] = key2balance.get(key, 0) + key2stake.get(key, 0)
        key2value = {k:v for k,v in key2value.items()}
        key2value = dict(sorted(key2value.items(), key=lambda x: x[1], reverse=True))
        return key2value
    
    def resolve_module_key(self, x, netuid=0, max_age=60):
        if not c.valid_ss58_address(x):
            name2key = self.name2key(netuid=netuid, max_age=max_age)
            x = name2key.get(x)
        assert c.valid_ss58_address(x), f"Module key {x} is not a valid ss58 address"
        return x
    
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

    def get_balance(self,
                 key: str = None ,
                 block: int = None,
                 fmt='j',
                 max_age=0,
                 update=False) -> Optional['Balance']:
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

    balance = get_balance

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
    
    def balances(self,fmt:str = 'n', block: int = None, n = None, update=False , **kwargs) -> Dict[str, 'Balance']:
        accounts = self.accounts( update=update, block=block)
        balances =  {k:v['data']['free'] for k,v in accounts.items()}
        balances = {k: self.format_amount(v, fmt=fmt) for k,v in balances.items()}
        return balances
    
    def blocks_per_day(self):
        return 24*60*60/self.block_time
    
    def min_burn(self,  block=None, update=False, fmt='j'):
        min_burn = self.query('MinBurn', block=block, update=update)
        return self.format_amount(min_burn, fmt=fmt)
    

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
            future2key = {}
            for key in keys:
                f = c.submit(self.get_balance, kwargs=dict(key=key, fmt=fmt, **kwargs))
                future2key[f] = key
            
            for f in c.as_completed(future2key):
                key = future2key.pop(f)
                key2balance[key] = f.result()
                
        for k,v in key2balance.items():
            key2balance[k] = self.format_amount(v, fmt=fmt)


        if names:
            address2key = c.address2key()
            key2balance = {address2key[k]: v for k,v in key2balance.items()}
            
        return key2balance

    def total_balance(self, **kwargs):
        balances = self.balances(**kwargs)
        return sum(balances.values())

    def num_holders(self, **kwargs):
        balances = self.balances(**kwargs)
        return len(balances)

    def proposals(self,  block=None,  nonzero:bool=False, update:bool = False,  **kwargs):
        proposals = [v for v in self.query_map('Proposals', block=block, update=update, **kwargs)]
        return proposals

    def registrations_per_block(self,**kwargs) -> int:
        return self.query('RegistrationsPerBlock', params=[],  **kwargs)
    regsperblock = registrations_per_block
    
    def max_registrations_per_block(self, **kwargs) -> int:
        return self.query('MaxRegistrationsPerBlock', params=[], **kwargs)

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

        key = self.resolve_key(key)
        network = self.resolve_network(network)
        global_params = self.global_params(fmt='nanos')
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
       





    def register(
        self,
        name: str , # defaults to module.tage
        address : str = None,
        stake : float = None,
        netuid = 0,
        network_name : str = None,
        key : str  = None,
        module_key : str = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        max_address_characters = 32,
        metadata = None,
        network = None,
        nonce=None,
    **kwargs
    ) -> bool:
        module_key =  c.get_key(module_key or name).ss58_address
        netuid2subnet = self.netuid2subnet(update=False)  
        subnet2netuid = {v:k for k,v in netuid2subnet.items()}

        if network_name == None and netuid != 0:
            network_name = netuid2subnet[netuid]
        else:
            assert isinstance(network_name, str), f"Subnet must be a string"
            if not network_name in subnet2netuid:
                subnet2netuid = self.subnet2netuid(update=True)
                if network_name not in subnet2netuid:
                    subnet2netuid[network_name] = len(subnet2netuid)
                    response = input(f"Do you want to create a new subnet ({network_name}) (yes or y or dope): ")
                    if response.lower() not in ["yes", 'y', 'dope']:
                        return {'success': False, 'msg': 'Subnet not found and not created'}
                
        # require prompt to create new subnet        
        stake = (stake or 0) * 1e9

        if c.server_exists(name):
            address = c.namespace().get(name)
        else:
            address = address or 'NA'

        params = { 
                    'network_name': network_name.encode('utf-8'),
                    'address': address[-max_address_characters:].replace('0.0.0.0', c.ip()).encode('utf-8'),
                    'name': name.encode('utf-8'),
                    'stake': stake,
                    'module_key': module_key,
                    'metadata': json.dumps(metadata or {}).encode('utf-8'),
                }
        
        # create extrinsic call
        response = self.compose_call('register', params=params, key=key, wait_for_inclusion=wait_for_inclusion, wait_for_finalization=wait_for_finalization, nonce=nonce)
        return response
    
    def resolve_uids(self, uids: Union['torch.LongTensor', list], netuid: int = 0, update=False) -> None:
        name2uid = None
        key2uid = None
        for i, uid in enumerate(uids):
            if isinstance(uid, str):
                if name2uid == None:
                    name2uid = self.name2uid(netuid=netuid, update=update)
                if uid in name2uid:
                    uids[i] = name2uid[uid]
                else:
                    if key2uid == None:
                        key2uid = self.key2uid(netuid=netuid, update=update)
                    if uid in key2uid:
                        uids[i] = key2uid[uid]
        return uids

    def set_weights(
        self,
        uids: Union['torch.LongTensor', list] ,
        weights: Union['torch.FloatTensor', list] ,
        netuid: int = 0,
        key: 'c.key' = None,
        update=False,
        modules = None,
        vector_length = 2**16 - 1,
        nonce=None,
        **kwargs
    ) -> bool:
        import torch
        netuid = self.resolve_netuid(netuid)
        key = self.resolve_key(key)

        # ENSURE THE UIDS ARE VALID UIDS, CONVERTING THE NAMES AND KEYS ASSOCIATED WITH THEM
        uids = self.resolve_uids(modules or uids, netuid=netuid, update=update)
        weights = weights or  [1] * len(uids)
        
        # CHECK WEIGHTS LENGHTS
        assert len(uids) == len(weights), f"Length of uids {len(uids)} must be equal to length of weights {len(weights)}"
        
        # NORMALIZE WEIGHTS
        weights = torch.tensor(weights)
        weights = (weights / weights.sum()) * vector_length # normalize the weights between 0 and 1
        weights = torch.clamp(weights, 0, vector_length)  # min_value and max_value are between 0 and 1
        
        params = {'uids': list(map(int, uids)),
                  'weights': list(map(int, weights.tolist())), 
                  'netuid': netuid}
        
        return self.compose_call('set_weights',params = params , key=key, nonce=nonce, **kwargs)
        
    vote = set_weights
    
    def unstake_all( self, 
                        key: str = None, 
                        existential_deposit = 1,
                        min_stake = 0.5,
                        ) -> Optional['Balance']:
        
        key = self.resolve_key( key )
        key_stake_to = self.get_stake_to(key=key,  names=False, update=True, fmt='nanos') # name to amount

        min_stake = min_stake * 1e9
        key_stake_to = {k:v for k,v in key_stake_to.items() if v > min_stake }

        params = {
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

    def registered_keys(self, netuid='all'):
        key2address = c.key2address()
        address2key = {v:k for k,v in key2address.items()}
        
        if netuid == 'all':
            registered_keys = {}
            netuid2keys =  self.keys(netuid=netuid)
            for netuid, keys in netuid2keys.items():
                registered_keys[netuid] = []
                for k in keys:
                    if k in address2key:
                        registered_keys[netuid].append(k)
        else:
            registered_keys = [k for k in self.keys(netuid=netuid) if k in address2key]

        return registered_keys


    subnet_param_features = [
                "ImmunityPeriod", 
                "MinAllowedWeights",
                "MaxAllowedWeights", 
                "Tempo",
                "MaxAllowedUids",
                "Founder",
                "FounderShare",
                "IncentiveRatio",
                "TrustRatio",
                "SubnetNames", 
                "MaxWeightAge",
                "BondsMovingAverage", 
                "MaximumSetWeightCallsPerEpoch", 
                "MinValidatorStake",
                "MaxAllowedValidators",
    "ModuleBurnConfig",
     "SubnetMetadata",
        'SubnetGovernanceConfig'
    ]



    def stake_to(self, block=None,  max_age=1000, update=False, fmt='nano', **kwargs):
        stake_to = self.query_map('StakeTo', block=block, max_age=max_age, update=update, **kwargs)
        format_value = lambda v:  {v_k: self.format_amount(v_v, fmt=fmt) for v_k, v_v in v.items()}
        stake_to = {k: format_value(v) for k,v in stake_to.items()}
        return stake_to
    


    def netuid2founder(self, fmt='j',  **kwargs):
        netuid2founder = self.query_map('Founder',  **kwargs)
        return netuid2founder
    

    def stake_from(self, 
                    block=None, 
                    update=False,
                    max_age=10000,
                    fmt='nano', 
                    **kwargs) -> List[Dict[str, Union[str, int]]]:
        
        stake_from = self.query_map('StakeFrom', block=block, update=update, max_age=max_age )
        format_value = lambda v:  {v_k: self.format_amount(v_v, fmt=fmt) for v_k, v_v in v.items()}
        stake_from = {k: format_value(v) for k,v in stake_from.items()}
        return stake_from

    """ Returns network Tempo hyper parameter """
    def stakes(self, fmt:str='j', max_age = 100, update=False, stake_from=None, **kwargs) -> int:
        stake_from = stake_from or self.stake_from( update=update, max_age=max_age, fmt=fmt)
        stakes = {k: sum(v.values()) for k,v in stake_from.items()}
        return stakes
    
    def leaderboard(self, netuid = 0, block=None, update=False, columns = ['emission', 'name', 'incentive', 'dividends'], **kwargs):
        modules = self.get_modules(netuid=netuid, block=block, update=update, **kwargs)
        return c.df(modules)[columns]

    
    def min_stake(self, netuid: int = 0, fmt:str='j', **kwargs) -> int:
        min_stake = self.query('MinStake', netuid=netuid,  **kwargs)
        return self.format_amount(min_stake, fmt=fmt)
    
    def regblock(self, netuid: int = 0, block: Optional[int] = None,  update=False ) -> Optional[float]:
        regblock =  self.query_map('RegistrationBlock',block=block, update=update )
        if isinstance(netuid, int):
            regblock = regblock[netuid]
        return regblock

    def emissions(self, netuid = None, block=None, update=False, fmt = 'nanos', **kwargs):
        netuid = self.resolve_netuid(netuid)
        emissions = self.query_vector('Emission',  netuid=netuid, block=block, update=update, **kwargs)
        if netuid == 'all':
            for netuid, netuid_emissions in emissions.items():
                emissions[netuid] = [self.format_amount(e, fmt=fmt) for e in netuid_emissions]
        else:
            emissions = [self.format_amount(e, fmt=fmt) for e in emissions]
        
        return emissions

    emission = emissions


    def total_emission( self, netuid: int = 0, block: Optional[int] = None, fmt:str = 'j', **kwargs ) -> Optional[float]:
        total_emission =  sum(self.emission(netuid=netuid, block=block, **kwargs))
        return self.format_amount(total_emission, fmt=fmt)

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

    def emissions(self, netuid = None, network = "main", block=None, update=False, **kwargs):
        netuid = self.resolve_netuid(netuid)
        return self.query_vector('Emission', network=network, netuid=netuid, block=block, update=update, **kwargs)

    def key2name(self, key: str = None, netuid: int = 0) -> str:
        modules = self.keys(netuid=netuid)
        key2name =  { m['key']: m['name']for m in modules}
        if key != None:
            return key2name[key]

    def uid2name(self, netuid: int = 0, update=False,  **kwargs) -> List[str]:
        netuid = self.resolve_netuid(netuid)
        names = self.query_map('Name', netuid=netuid, update=update,**kwargs)
        return names
    
    def is_registered(self, key: str, netuid: int = 0, update=False, **kwargs) -> bool:
        key_address = self.resolve_key_ss58(key)
        try:
            uid =  self.get_uid(key_address, netuid=netuid, update=update, **kwargs)
            if isinstance(uid, int):
                return True
        except Exception as e:
            return False

    def keys(self,
             netuid = None,
              update=False, 
              max_age=1000,
             **kwargs) -> List[str]:
        netuid = self.resolve_netuid(netuid)
        keys =  self.query_map('Keys', netuid=netuid, update=update,  max_age=max_age, **kwargs)
        if netuid == 'all':
            for netuid, netuid_keys in keys.items():
                keys[netuid] = list(netuid_keys.values())
        else:
            keys = list(keys.values())
        return keys

    def delegation_fee(self, netuid = None, block=None, update=False, fmt='j'):
        netuid = self.resolve_netuid(netuid)
        delegation_fee = self.query_map('DelegationFee', netuid=netuid, block=block ,update=update)
        return delegation_fee


    def feature2name(self, feature='MinStake'):
        translations = {
            'subnet_names': 'name'
        }
        name = ''
        for i, ch in enumerate(feature):
            if ch.isupper():
                if i == 0:
                    name += ch.lower()
                else:
                    name += f'_{ch.lower()}'
            else:
                name += ch
        name = translations.get(name, name)
        return name


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
        if netuid == 'all':
            return self.all_subnet_params(update=update, 
                                          max_age=max_age, 
                                          timeout=timeout, 
                                          fmt=fmt, 
                                          features=features, 
                                          value_features=value_features, 
                                          **kwargs)
        

        default_params = {
            'maximum_set_weight_calls_per_epoch': 30,
            'max_allowed_validators': 50
        }
        
        features = features or self.subnet_param_features
        netuid = self.resolve_netuid(netuid)
        path = f'query/{self.network}/SubnetParams.{netuid}'          
        subnet_params = self.get(path, max_age=max_age, update=update)
        if subnet_params == None:

            names = [self.feature2name(f) for f in features]
            future2name = {}
            for name, feature in dict(zip(names, features)).items():
                query_kwargs = dict(name=feature, netuid=netuid,block=None, max_age=max_age, update=update)
                if name in ['SubnetGovernanceConfig']:
                    fn = self.query_map
                else:
                    fn = self.query

                f = c.submit(fn, kwargs=query_kwargs, timeout=timeout)
                future2name[f] = name
            subnet_params = {}
            for f in c.as_completed(future2name, timeout=timeout):
                result = f.result()
                subnet_params[future2name.pop(f)] = result
            for k in subnet_params.keys():
                v = subnet_params[k]
                if v == None:
                    v = default_params.get(k, v)
                if k in value_features:
                    v = self.format_amount(v, fmt=fmt)
                subnet_params[k] = v
            
            self.put(path, subnet_params)
            
        subnet_params.update(subnet_params.pop('subnet_governance_config')) 
        translation = {
            'subnet_names': 'name', 
            'bonds_moving_average': 'bonds_ma'
        }
        for k,v in translation.items():
            if k in subnet_params:
                subnet_params[v] = subnet_params.pop(k)
        return subnet_params


    def all_subnet_params(self, 
                    update = False,
                    max_age = 1000,
                    features  = None,
                    **kwargs
                    ) -> list:  
        
        features = features or self.subnet_param_features
        netuid = self.resolve_netuid(netuid)
        path = f'query/{self.network}/SubnetParams.all'          
        all_subnet_params = self.get(path, max_age=max_age, update=update)
        if all_subnet_params == None:
            all_subnet_params = {}
            for netuid in self.netuids(update=update):
                all_subnet_params[netuid] = self.subnet_params(netuid=netuid, update=update, max_age=max_age, **kwargs)
        return all_subnet_params

    def pending_deregistrations(self, netuid = None, update=False, **kwargs):
        netuid = self.resolve_netuid(netuid)
        pending_deregistrations = self.query_map('PendingDeregisterUids',update=update,**kwargs)[netuid]
        return pending_deregistrations
    
    def num_pending_deregistrations(self, netuid = 0, **kwargs):
        pending_deregistrations = self.pending_deregistrations(netuid=netuid, **kwargs)
        return len(pending_deregistrations)
        
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

    def trust(self, 
                  netuid = 0, 
                  block=None,  
                  update:bool = False, 
                  **kwargs):
        return self.query_vector('Trust', netuid=netuid,  block=block, update=update, **kwargs)

    def incentives(self, 
                  netuid = 0, 
                  block=None,  
                  update:bool = False, 
                  **kwargs):
        return self.query_vector('Incentive', netuid=netuid,  block=block, update=update, **kwargs)
    incentive = incentives

    def last_update(self, netuid = 0, update=False, **kwargs):
        return self.query_vector('LastUpdate', netuid=netuid,   update=update, **kwargs)

    def dividends(self, netuid = 0, update=False, **kwargs):
        return  self.query_vector('Dividends', netuid=netuid,   update=update,  **kwargs)
            
    dividend = dividends
    
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
    
    def subnet2n(self, fmt='j',  **kwargs):
        netuid2n = self.netuid2n(fmt=fmt, **kwargs)
        netuid2subnet = self.netuid2subnet()
        subnet2n = {}
        for netuid, subnet in netuid2subnet.items():
            subnet2n[subnet] = netuid2n[netuid]
        return subnet2n
    
    def subnet2stakes(self,  block=None, update=False, fmt='j', **kwargs):
        subnet2stakes = {}
        for netuid in self.netuids( update=update):
            subnet2stakes[netuid] = self.stakes(netuid=netuid,  block=block, update=update, fmt=fmt, **kwargs)
        return subnet2stakes

    def subnet_state(self,  netuid='all', block=None, update=False, fmt='j', **kwargs):

        subnet_state = {
            'params': self.subnet_params(netuid=netuid,  block=block, update=update, fmt=fmt, **kwargs),
            'modules': self.modules(netuid=netuid,  block=block, update=update, fmt=fmt, **kwargs),
        }
        return subnet_state

    def subnet2emission(self, fmt='j',  **kwargs):
        subnet2params = self.subnet_params(netuid='all')
        netuid2emission = self.netuid2emission(fmt=fmt, **kwargs)
        netuid2subnet = self.netuid2subnet()
        subnet2emission = {}
        for netuid, subnet in netuid2subnet.items():
            subnet2emission[subnet] = netuid2emission[netuid]
        # sort by emission
        subnet2emission = dict(sorted(subnet2emission.items(), key=lambda x: x[1], reverse=True))

        return subnet2emission


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

    """ Returns network SubnetN hyper parameter """
    def n(self,  netuid: int = 0,block: Optional[int] = None, max_age=100, update=False, **kwargs ) -> int:
        if netuid == 'all':
            return sum(self.query_map('N', block=block , update=update, max_age=max_age,  **kwargs).values())
        else:
            return self.query( 'N', params=[netuid], block=block , update=update,  **kwargs)

    def subnet_exists(self, subnet:str) -> bool:
        subnets = self.subnets()
        return bool(subnet in subnets)

    def subnet_emission(self, netuid:str = 0, block=None, update=False, **kwargs):
        emissions = self.emission(block=block, update=update,  netuid=netuid, **kwargs)
        if isinstance(emissions[0], list):
            emissions = [sum(e) for e in emissions]
        return sum(emissions)
    
    def get_modules(self,
                    keys : list = None,
                    netuid=None,
                    timeout=30,
                    min_emission=0,
                    max_age = 1000,
                    update=False,
                    **kwargs):
        netuid = self.resolve_netuid(netuid)
        modules = None
        path = None 
        if keys == None :
            path = f'subnet/{self.network}/{netuid}/modules'
            modules = self.get(path, None, max_age=max_age, update=update)
            keys = self.keys(netuid=netuid)

        n = len(keys)
        if modules == None:
            modules = []
            print(f'Getting modules {n}')
            futures = [c.submit(self.get_module, kwargs=dict(module=k, netuid=netuid, **kwargs)) for k in keys]
            progress = c.tqdm(n)
            modules = []


            should_pass = lambda x: isinstance(x, dict) \
                            and 'name' in x \
                            and len(x['name']) > 0 \
                            and x['emission'] >= min_emission
            
            for future in c.as_completed(futures, timeout=timeout):
                module = future.result()
                if should_pass(module):
                    modules += [module]
                    progress.update(1)
            if path != None:
                self.put(path, modules)
            
                    
        return modules

    module_param_features = [
        'key',
        'name',
        'address',
        'emission',
        'incentive',
        'dividends',
        'last_update',
        'stake_from',
        'delegation_fee'
    ]
    
    def get_module(self, 
                    module=None,
                    netuid=None,
                    trials = 4,
                    fmt='j',
                    mode = 'http',
                    block = None,
                    max_age = None,
                    lite = False, 
                    update = False,
                    **kwargs ) -> 'ModuleInfo':
        U16_MAX = 2**16 - 1

        netuid = self.resolve_netuid(netuid)
        
        if module == None:
            module = self.keys(netuid=netuid, update=update, max_age=max_age)[0]
            c.print(f'No module specified, using {module}')
        
        module = c.key2address().get(module, module)
        url = self.resolve_url( mode=mode)
        module_key = module
        is_valid_key = c.valid_ss58_address(module)
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
        assert module != None, f"Failed to get module {module_key} after {trials} trials"
        if not 'result' in module:
            return module
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
            features = self.module_param_features + ['stake', 'vote_staleness']
            module = {f: module[f] for f in features}
        assert module['key'] == module_key, f"Key mismatch {module['key']} != {module_key}"
        return module


    def root_valis(self, search=None, netuid = 0, update=False, **kwargs):
        root_valis = []
        for module in self.get_modules(netuid=netuid, update=update, **kwargs):
            if search != None:
                if search not in module['name']:
                    continue
            module.pop('stake_from')
            root_valis += [module ]
        
        return c.df(root_valis)[['name', 'key', 'stake']]
    

    def root_keys(self, netuid = 0, update=False, **kwargs):
        return self.keys(netuid=netuid, update=update, **kwargs)




    def registration_block(self, netuid: int = 0, update=False, **kwargs):
        registration_blocks = self.query_map('RegistrationBlock', netuid=netuid, update=update, **kwargs)
        return registration_blocks

    regblocks = registration_blocks = registration_block





    

    def key2name(self, key=None, netuid: int = None, update=False) -> Dict[str, str]:
        
        key2name =  {v:k for k,v in self.name2key(netuid=netuid,  update=update).items()}
        if key != None:
            return key2name[key]
        return key2name
        



    def name2key(self, name:str=None, 
                 max_age=1000, 
                 timeout=30, 
                 netuid: int = 0, 
                 update=False, 
                 trials=3,
                 **kwargs ) -> Dict[str, str]:
        # netuid = self.resolve_netuid(netuid)
        netuid = self.resolve_netuid(netuid)

        names = c.submit(self.names, kwargs={'feature': 'names', 'netuid':netuid, 'update':update, 'max_age':max_age, 'network': self.network})
        keys = c.submit(self.keys, kwargs={'feature': 'keys', 'netuid':netuid, 'update':update, 'max_age':max_age, 'network': self.network})
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



              
    def name2uid(self, name = None, netuid: int = 0, search=None) -> int:
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
            name2uid = {v:k for k,v in uid2name.items()}
            if search != None:
                name2uid =  self.search_dict(name2uid, search=search)
            if name != None:
                return name2uid[name] 
            
        return name2uid

    def netuids(self,  update=False, block=None) -> Dict[int, str]:
        return list(self.netuid2subnet( update=update, block=block).keys())

    def netuid2subnet(self, netuid=None,  update=False, block=None, **kwargs ) -> Dict[str, str]:
        netuid2subnet = self.query_map('SubnetNames', update=update,  block=block, **kwargs)
        netuid2subnet = dict(sorted(netuid2subnet.items(), key=lambda x: x[0]))
        if netuid != None:
            return netuid2subnet[netuid]
        return netuid2subnet
    netuid2name = netuid2subnet

    def subnet2netuid(self, subnet=None,  update=False,  **kwargs ) -> Dict[str, str]:
        subnet2netuid =  {v:k for k,v in self.netuid2subnet( update=update, **kwargs).items()}
        # sort by subnet 
        if subnet != None:
            return subnet2netuid[subnet] if subnet in subnet2netuid else len(subnet2netuid)
        return subnet2netuid
    name2netuid = subnet2netuid


    def get_uid( self, key: str, netuid: int = 0, block: Optional[int] = None, update=False, **kwargs) -> int:
        return self.query( 'Uids', block=block, params=[ netuid, key ] , update=update, **kwargs)  


    def weights(self,  netuid = 0,  update=False, **kwargs) -> list:
        weights =  self.query_map('Weights',netuid=netuid, update=update, **kwargs)

        tuples2list = lambda x: [list(v) for v in x]
        if netuid == 'all':
            for netuid, netuid_weights in weights.items():
                weights[netuid] = {k: tuples2list(v) for k,v in netuid_weights.items()}
        else:
            weights = {k: tuples2list(v) for k,v in weights.items()}
            
        return weights
    
    def resolve_uid(self, uid=None, netuid=None, **kwargs) -> int:
        netuid = self.resolve_netuid(netuid)
        if isinstance(uid, int):
            return uid
        elif isinstance(uid, str):
            if c.key_exists(uid):
                # for key
                uid = self.resolve_key_ss58(uid)
                uid = self.key2uid(netuid=netuid,**kwargs)[uid]
            else:
                # resolve name
                uid = self.name2uid(name=uid, netuid=netuid, **kwargs)
                
        return uid
        


    def get_weights(self, key=None, netuid = 0,  update=False, **kwargs) -> list:
        uid = self.resolve_uid(key, netuid=netuid)
        weights =  self.query('Weights', params=[netuid, uid], update=update, **kwargs)
        return weights
    
    




    def total_emissions(self, netuid = 9, block=None, update=False, fmt = 'j', **kwargs):

        emissions = self.query_vector('Emission',  netuid=netuid, block=block, update=update, **kwargs)
        if netuid == 'all':
            for netuid, netuid_emissions in emissions.items():
                emissions[netuid] = [self.format_amount(e, fmt=fmt) for e in netuid_emissions]
        else:
            emissions = [self.format_amount(e, fmt=fmt) for e in emissions]
        
        return sum(emissions)
    


    # def state(self, block=None, netuid='all', update=False, max_age=10000, fmt='j', **kwargs):
    #     subnet_params = self.subnet_params(block=block, netuid=netuid, max_age=max_age, **kwargs)
    #     subnet2emissions = self.emissions(netuid=netuid, max_age=max_age, block=block, **kwargs)
    #     subnet2staketo = self.stake_to(netuid=netuid, block=block, update=update, fmt=fmt, **kwargs)
    #     subnet2incentives = self.incentives(netuid=netuid, block=block, update=update, fmt=fmt, **kwargs)
    #     subnet2trust = self.trust(netuid=netuid, block=block, update=update, fmt=fmt, **kwargs)
    #     subnet2keys = self.keys(netuid=netuid, block=block, update=update, **kwargs)
                        
    #     subnet2state = {}
    #     for netuid, params in subnet_params.items():
    #         subnet_state = {
    #             'params': params,
    #             'incentives': subnet2incentives[netuid],
    #             'emissions': subnet2emissions[netuid],
    #             ''
    #             'stake_to': subnet2staketo[netuid],
    #             'keys': subnet2keys[netuid],

    #         }
    #         subnet_state[netuid] = subnet_state
        
    #     return subnet2state


    def netuid2emission(self, fmt='j',  period='day', names=None, **kwargs):
        netuid2emission = {}
        netuid2tempo = None
        emissions = self.query_vector('Emission',  netuid='all', **kwargs)
        for netuid, netuid_emissions in emissions.items():
            if period == 'day':
                if netuid2tempo == None:
                    netuid2tempo = self.query_map('Tempo', netuid='all', **kwargs)
                tempo = netuid2tempo.get(netuid, 100)
                multiplier = self.blocks_per_day() / tempo
            else:
                multiplier = 1
            netuid2emission[netuid] = self.format_amount(sum(netuid_emissions), fmt=fmt) * multiplier
        netuid2emission = {k: v   for k,v in netuid2emission.items()}
        if names:
            netuid2emission = {self.netuid2name(netuid=k): v for k,v in netuid2emission.items()}
        return netuid2emission

    def subnet2emission(self, fmt='j',  period='day', **kwargs):
        return self.netuid2emission(fmt=fmt, period=period, names=1, **kwargs)
    

    def global_emissions(self,  **kwargs):
        return sum(list(self.subnet2emissions( **kwargs).values()))


    

    def subnet2params( self,  block: Optional[int] = None ) -> Optional[float]:
        netuids = self.netuids()
        subnet2params = {}
        netuid2subnet = self.netuid2subnet()
        for netuid in netuids:
            subnet = netuid2subnet[netuid]
            subnet2params[subnet] = self.subnet_params(netuid=netuid, block=block)
        return subnet2params
    



    #################
    #### UPDATE SUBNET ####
    #################
    def update_subnet(
        self,
        params: dict= None,
        netuid: int = 0,
        key: str = None,
        nonce = None,
        update= True,
        **extra_params,
    ) -> bool:
        
        params = {**(params or {}), **extra_params}
            
        netuid = self.resolve_netuid(netuid)
        subnet_params = self.subnet_params( netuid=netuid , update=update, fmt='nanos')
        # infer the key if you have it
        for k in ['min_immunity_stake']:
            if k in params:
                params[k] = params[k] * 1e9
        if key == None:
            key2address = c.address2key()
            if subnet_params['founder'] not in key2address:
                return {'success': False, 'message': f"Subnet {netuid} not found in local namespace, please deploy it "}
            key = c.get_key(key2address.get(subnet_params['founder']))
            c.print(f'Using key: {key}')

        # remove the params that are the same as the module info
        params = {**subnet_params, **params}
        for k in ['name']:
            params[k] = params[k].encode('utf-8')
        
        params['netuid'] = netuid
        return self.compose_call(fn='update_subnet', params=params,   key=key,  nonce=nonce)


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



    def get_stake( self, key_ss58: str, block: Optional[int] = None, netuid:int = None , fmt='j', update=True ) -> Optional['Balance']:
        
        key_ss58 = self.resolve_key_ss58( key_ss58)
        netuid = self.resolve_netuid( netuid )
        stake = self.query( 'Stake',params=[netuid, key_ss58], block=block , update=update)
        return self.format_amount(stake, fmt=fmt)



    def min_register_stake(self, netuid: int = 0, fmt='j', **kwargs) -> float:
        netuid = self.resolve_netuid(netuid)
        min_burn = self.min_burn(  fmt=fmt)
        min_stake = self.min_stake(netuid=netuid,  fmt=fmt)
        return min_stake + min_burn
    



    def resolve_netuid(self, netuid: int = None) -> int:
        '''
        Resolves a netuid to a subnet name.
        '''
        if netuid == 'all':
            return netuid
        if netuid == None :
            # If the netuid is not specified, use the default.
            return self.netuid
        if isinstance(netuid, str):
            subnet2netuid = self.subnet2netuid()
            if netuid not in subnet2netuid: # if still not found, try lower case
                subnet2netuid =self.subnet2netuid(update=True)
            assert netuid in subnet2netuid, f"Subnet {netuid} not found in {subnet2netuid}"
            return subnet2netuid[netuid]

        elif isinstance(netuid, int):
            if netuid == 0: 
                return netuid
            # If the netuid is an integer, ensure it is valid.
            
        assert isinstance(netuid, int), "netuid must be an integer"
        return netuid
    

    def blocks_until_vote(self, netuid=None, **kwargs):
        netuid = self.resolve_netuid(netuid)
        tempo = self.subnet_params(netuid=netuid, **kwargs)['tempo']
        block = self.block
        return tempo - ((block + netuid) % tempo)

    def emission_per_epoch(self, netuid=None):
        return self.subnet(netuid=netuid)['emission']*self.epoch_time(netuid=netuid)



    

    def get_stake_to( self, 
                     key: str = None, 
                     module_key=None,
                       block: Optional[int] = None, 
                        fmt='j' , update=False,
                        max_age = 60,
                        timeout = 10,
                         **kwargs) -> Optional['Balance']:
        
        key_address = self.resolve_key_ss58( key )
        stake_to = self.query_map( 'StakeTo', params=[ key_address], block=block, update=update,  max_age=max_age)
        stake_to =  {k: self.format_amount(v, fmt=fmt) for k, v in stake_to.items()}
        return stake_to
    
    def get_stake_from( self, key: str, block: Optional[int] = None,  fmt='j', update=True  ) -> Optional['Balance']:
        key = self.resolve_key_ss58( key )
        stake_from = self.query_map( 'StakeFrom', params=[key], block=block,  update=update )
        stake_from =  {k: self.format_amount(v, fmt=fmt) for k, v in stake_from.items()}
        return stake_from


    def epoch_time(self, netuid=None, update=False, **kwargs):
        netuid = self.resolve_netuid(netuid)
        return self.subnet_params(netuid=netuid, update=update, **kwargs)['tempo']*self.block_time


    def seconds_per_day(self ):
        return 24*60*60
    
    def epochs_per_day(self, netuid=None):
        netuid = self.resolve_netuid(netuid)
        return self.seconds_per_day()/self.epoch_time(netuid=netuid)

    def seconds_per_epoch(self, netuid=0):
        netuid =self.resolve_netuid(netuid)
        return self.block_time * self.subnet_params(netuid=netuid)['tempo']


    def format_module(self, module: 'ModuleInfo', fmt:str='j') -> 'ModuleInfo':
        U16_MAX = 2**16 - 1
        for k in ['emission']:
            module[k] = self.format_amount(module[k], fmt=fmt)
        for k in ['incentive', 'dividends']:
            module[k] = module[k] / (U16_MAX)
        
        module['stake_from'] = {k: self.format_amount(v, fmt=fmt)  for k, v in module['stake_from']}
        return module
    


    def netuid2module(self, update=False, fmt:str='j', **kwargs) -> 'ModuleInfo':
        netuids = self.netuids(update=update)
        future2netuid = {}
        for netuid in netuids:
            f  = c.submit(self.get_module, dict(netuid=netuid, update=update, fmt=fmt, **kwargs))
            future2netuid[f] = netuid
        netuid2module = {}
        progress = c.tqdm(len(netuids))

        for future in c.as_completed(future2netuid):
            netuid = future2netuid.pop(future)
            module = future.result()
            if not c.is_error(module):
                netuid2module[netuid] = module
            progress.update(1)  
            
        return netuid2module
                
    def netuid2uid(self, key=None, update=False, **kwargs) -> Dict[str, str]:
        key = self.resolve_key_ss58(key)
        netuids = self.netuids(update=update)
        netuid2uid = {}
        progress = c.tqdm(len(netuids))

        future2netuid = {}
        for netuid in netuids:
            f = c.submit(self.get_uid, kwargs=dict(key=key, netuid=netuid, **kwargs))
            future2netuid[f] = netuid

        for future in c.as_completed(future2netuid):
            netuid = future2netuid.pop(future)
            uid = future.result()
            if uid != None:
                netuid2uid[netuid] = uid
            progress.update(1)
        # sort by netuid key
        netuid2uid = dict(sorted(netuid2uid.items(), key=lambda x: x[0]))

        return netuid2uid
    

    def subnet_state(self, netuid=0, update=False,  **kwargs):

        modules = self.get_modules(netuid=netuid, update=update, **kwargs)

        return {
            'params': self.subnet_params(netuid=netuid, 
                                         update=update, 
                                         fmt='nanos'),
            'modules': modules
        }
    
    def register_subnet(self, key: 'Keypair', name: str, metadata: str | None = None) -> 'c':
        """
        Registers a new subnet in the network.

        Args:
            key (Keypair): The keypair used for registering the subnet.
            name (str): The name of the subnet to be registered.
            metadata (str | None, optional): Additional metadata for the subnet. Defaults to None.

        Returns:
            ExtrinsicReceipt: A receipt of the subnet registration transaction.

        Raises:
            ChainTransactionError: If the transaction fails.
        """

        params = {
            "name": name,
            "metadata": metadata,
        }

        response = self.compose_call("register_subnet", params=params, key=key)

        return response


Subspace.run(__name__)

