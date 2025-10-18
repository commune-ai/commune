import commune as c
from typing import Dict, Any, Optional

class Network:
    """
    Network manager for Solana operations
    """
    def __init__(self, rpc_url: str = 'https://api.devnet.solana.com'):
        self.rpc_url = rpc_url
        self.networks = {
            'mainnet': 'https://api.mainnet-beta.solana.com',
            'devnet': 'https://api.devnet.solana.com',
            'testnet': 'https://api.testnet.solana.com'
        }
    
    def get_network(self, network: str = 'devnet') -> str:
        """Get RPC URL for specified network"""
        return self.networks.get(network, self.rpc_url)
    
    def switch_network(self, network: str) -> str:
        """Switch to a different network"""
        if network in self.networks:
            self.rpc_url = self.networks[network]
            return self.rpc_url
        raise ValueError(f"Unknown network: {network}")
    
    def get_current_network(self) -> str:
        """Get the current network name"""
        for name, url in self.networks.items():
            if url == self.rpc_url:
                return name
        return 'custom'
    
    def add_custom_network(self, name: str, rpc_url: str):
        """Add a custom network"""
        self.networks[name] = rpc_url
