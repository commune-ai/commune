
from retry import retry
from typing import *
import json
import os
import commune as c
import requests 
from substrateinterface import SubstrateInterface

U32_MAX = 2**32 - 1
U16_MAX = 2**16 - 1

class Network(c.Module):
    """
    Handles interactions with the subspace chain.
    """

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


    def storage_functions(self, network=network, block_hash = None):
        self.resolve_network(network)
        return self.substrate.get_metadata_storage_functions( block_hash=block_hash)
    

    def storage_names(self,  search=None, network=network, block_hash = None):
        self.resolve_network(network)
        storage_names =  [f['storage_name'] for f in self.substrate.get_metadata_storage_functions( block_hash=block_hash)]
        if search != None:
            storage_names = [s for s in storage_names if search in s.lower()]
        return storage_names


    def check_storage(self, block_hash = None, network=network):
        self.resolve_network(network)
        return self.substrate.get_metadata_storage_functions( block_hash=block_hash)
