

import os
import sys
from copy import deepcopy
from typing import Dict, List, Optional, Union
import commune as c


import lru
import requests
from requests.adapters import HTTPAdapter
from typing import Dict, Optional, Union
from web3 import WebsocketProvider
from web3 import HTTPProvider, WebsocketProvider
class CustomHTTPProvider(HTTPProvider):
    """
    Override requests to control the connection pool to make it blocking.
    """
    def make_request(self, method, params):
        self.logger.debug(
            "Making request HTTP. URI: %s, Method: %s", self.endpoint_uri, method
        )
        request_data = self.encode_rpc_request(method, params)
        raw_response = self.make_post_request(
            self.endpoint_uri, request_data, **self.get_request_kwargs()
        )
        response = self.decode_rpc_response(raw_response)
        self.logger.debug(
            "Getting response HTTP. URI: %s, " "Method: %s, Response: %s",
            self.endpoint_uri,
            method,
            response,
        )
        return response

    def _remove_session(self, key, session):
        session.close()

    _session_cache = lru.LRU(8, callback=_remove_session)

    def _get_session(self, *args, **kwargs):
        from web3._utils.caching import generate_cache_key
        cache_key = generate_cache_key((args, kwargs))
        if cache_key not in self._session_cache:
            # This is the main change from original Web3 `_get_session`
            session = requests.sessions.Session()
            session.mount(
                "http://",
                HTTPAdapter(pool_connections=25, pool_maxsize=25, pool_block=True),
            )
            self._session_cache[cache_key] = session
        return self._session_cache[cache_key]


    def make_post_request(self, endpoint_uri, data, *args, timeout:int=10, **kwargs):
        kwargs.setdefault("timeout", timeout)
        session = self._get_session(endpoint_uri)
        response = session.post(endpoint_uri, data=data, *args, **kwargs)
        response.raise_for_status()

        return response.content


class EVMNetwork(c.Module):
    

    def __init__(self, network:str = 'local.main'):
        self.set_config(kwargs=locals())
        self.set_network(network)

    @property
    def network(self):
        network = self.config['network']
        if len(network.split('.')) == 3:
            network = '.'.join(network.split('.')[:-1])
        assert len(network.split('.')) == 2
        return network


    @network.setter
    def network(self, network):
        assert network in self.networks, f'{network} is not here fam'
        self.config['network'] = network

    def set_network(self, network:str='local.main.ganache') -> 'Web3':
        network = network if network != None else self.config['network']
        url = self.get_url(network)
        self.network = network
        self.url = url 
        self.web3 = self.get_web3(self.url)
    
    connect_network = set_network

    @property
    def networks_config(self):
        return c.load_yaml(self.dirpath() + '/networks.yaml')

    @property
    def networks(self):
        return list(self.networks_config.keys())

    @property
    def available_networks(self):
        return self.get_available_networks()



    def get_url_options(self, network:str ) -> List[str]:
        assert len(network.split('.')) == 2
        network, subnetwork = network.split('.')
        return list(self.networks_config[network][subnetwork]['url'].keys())

    def get_url(self, network:str='local.main.ganache' ) -> str:
        from commune.utils.dict import dict_get
        
        if len(network.split('.')) == 2:
            url_key = self.get_url_options(network)[0]
            network_key, subnetwork_key = network.split('.')
        elif len(network.split('.')) == 3:
            network_key, subnetwork_key, url_key = network.split('.')
        else:
            raise NotImplementedError(network)

        key_path = [network_key, subnetwork_key, 'url',url_key ]
        return dict_get(self.networks_config, key_path )
    

    def get_web3_connection_provider(self, network_url):
        if network_url.startswith("http"):
            provider = CustomHTTPProvider(network_url)
        elif network_url.startswith("ws"):
            provider = WebsocketProvider(network_url)
        else:
            raise NotImplementedError
        return provider

    def get_web3(self, network_url: str) -> 'Web3':
        from web3.main import Web3
        from web3.middleware import geth_poa_middleware

        provider = self.get_web3_connection_provider(network_url)
        web3 = Web3(provider)

        if web3.eth.chain_id == 4:
            web3.middleware_onion.inject(geth_poa_middleware, layer=0)
        return web3

    def get_web3_connection_provider(
        self,
        network_url: str,
    ) -> Union[CustomHTTPProvider, WebsocketProvider]:
        if network_url.startswith("http"):
            return CustomHTTPProvider(network_url)
        elif network_url.startswith("ws"):
            return WebsocketProvider(network_url)
        else:
            msg = (
                f"The given network_url *{network_url}* does not start with either"
                f"`http` or `wss`. A correct network url is required."
            )
            raise AssertionError(msg)
        
    @classmethod
    def test_url(cls, url:str):
        # Setup
        from web3 import Web3

        alchemy_url = "https://eth-mainnet.g.alchemy.com/v2/RrtpZjiUVoViiDEaYxhN9o6m1CSIZvlL"
        w3 = Web3(Web3.HTTPProvider(alchemy_url))

        # Print if web3 is successfully connected
        print(w3.isConnected())

        # Get the latest block number
        latest_block = w3.eth.block_number
        print(latest_block)

        # Get the balance of an account
        balance = w3.eth.get_balance('0x742d35Cc6634C0532925a3b844Bc454e4438f44e')
        print(balance)

        # Get the information of a transaction
        tx = w3.eth.get_transaction('0x5c504ed432cb51138bcf09aa5e8a410dd4a1e204ef84bfed1be16dfba1b22060')
        print(tx)
