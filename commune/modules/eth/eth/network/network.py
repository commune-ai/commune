import os
from copy import deepcopy
from typing import *
import commune as c

class Network(c.Module):
    DECIMALS = 10**18
    def __init__(self, network: str = 'local'):       
        self.set_network(network)

    def is_valid_address(self, address:str) -> bool:
        return self.client.is_address(address)

    def resolve_address(self, key):
        if self.is_valid_address(key):
            return key
        if hasattr(key, 'address'):
            address = key.address
        else:
            key = self.get_key(key)
            address = key.address
        return address

    def get_transaction_count(self, key=None):
        address = self.resolve_address(key)
        return self.client.eth.get_transaction_count(address)
    
    nonce = get_transaction_count

    def tx_metadata(self, key=None) -> Dict[str, Union[int, str, bytes]]:
        key = self.get_key(key)
        return {
            'from': key.address,
            'nonce': self.get_transaction_count(key),
            'gasPrice':self.gas_price(),
            }

    def gas_price(self):
        return self.client.eth.generate_gas_price() or 0

    def send_tx(self, tx, key = None) -> Dict:
        key = self.get_key(key)
        rawTransaction = self.sign_tx(tx=tx)    
        tx_hash = self.client.eth.send_raw_transaction(rawTransaction)
        tx_receipt = self.client.eth.wait_for_transaction_receipt(tx_hash)
        return tx_receipt.__dict__

    def sign_tx( self, tx: Dict, key=None ) -> 'HexBytes':
        key = self.get_key(key)
        tx['nonce'] = self.get_transaction_count(key)
        tx["gasPrice"] = self.gas_price()
        signed_tx = self.client.eth.sign_transaction(tx, key.private_key)
        return signed_tx.rawTransaction

    @property
    def networks(self):
        return list(self.network_state.keys())
    
    @property
    def client(self):
        return  self.get_client(self.network)

    @property
    def network_state(self):
        return c.load_yaml(self.dirpath() + '/networks.yaml')

    def resolve_network(self, network):
        if network == None:
            network = self.network
        return network

    def set_network(self, network:str='local') -> 'Web3':
        self.network = network

    def get_urls(self, network:str ) -> List[str]:
        urls = self.network_state[network]['url']
        if isinstance(urls, str):
            urls = [urls]
        return urls

    def get_url(self, network:str='local' ) -> str:
        return self.get_urls(network)[0]


    def get_client(self, network: str) -> 'Web3':
        network_url = self.get_url(network)
        from web3.main import Web3
        if network_url.startswith("http"):
            from web3.providers import HTTPProvider
            provider =  HTTPProvider(network_url)
        elif network_url.startswith("ws"):
            from web3.providers import WebsocketProvider
            provider =  WebsocketProvider(network_url)
        else:
            raise AssertionError(f"Invalid network url: {network_url}")
        conn =  Web3(provider)
        if conn.eth.chain_id == 4:
            conn.middleware_onion.inject(conn.middleware.geth_poa_middleware, layer=0)
        return conn
    

