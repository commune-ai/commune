import commune as c
import bittensor as bt
from typing import List, Dict, Any, Optional
from bittensor.utils.balance import Balance
# import btwallet

class Bt:
    """Interface module for Subtensor network operations and wallet management"""
    
    def __init__(self, network: str = "finney", archive=False):
        """Initialize the Subtensor module
        Args:
            network (str): Network to connect to (e.g. finney, test)
        """
        self.network = network
        self.subtensor = bt.subtensor(network=network)
        if archive:
            self.subtensor = bt.subtensor(network=network, archive=True)

    def mod2json(self, mod: Any) -> Dict:
        """Convert a neuron object to JSON dictionary
        Args:
            neuron (Any): Neuron object
        Returns:
            Dictionary representation of the neuron
        """
        mod =  mod.__dict__
        mod['axon_info'] = mod['axon_info'].__dict__ 
        mod['prometheus_info'] = mod['prometheus_info'].__dict__ 
        mod['url'] = mod['axon_info']['ip'] + ':' + str(mod['axon_info']['port'])
        return mod

    def neurons(self, netuid: int = 2) -> List[Dict]:
        """List all neurons in a subnet
        Args:
            netuid (int): Network UID
        Returns:
            List of neuron information
        """
        return [self.mod2json(n) for n in self.subtensor.neurons(netuid=netuid)]

    modules = mods = neurons
    
    def n(self, netuid: int = 1) -> int:
        """Get number of neurons in a subnet
        Args:
            netuid (int): Network UID
        Returns:
            Number of neurons
        """
        return len(self.neurons(netuid=netuid))

    
    def subnet(self, netuid: int = 2, block: Optional = None) -> Dict:
        """Get subnet information
        Args:
            netuid (int): Network UID
            block (Optional): Block number
        Returns:
            Subnet information dictionary
        """

        


        return self.subtensor.subnet(netuid=netuid, block=block)





    def get_all_subnets_info(self, block: Optional = None, df=False) -> List[Dict]:
        """List all subnets
        Args:
            block (Optional): Block number
        Returns:
            List of subnet information dictionaries
        """
        return self.subtensor.get_all_subnets_info(block=block)

    def subnets(self, block: Optional = None, neurons=False, max_age=6000, update=False) -> List[Dict]:
        """List all subnets
        Args:
            block (Optional): Block number
        Returns:
            List of subnet information dictionaries
        """
        path = '~/.bt/subnets.json'
        subnets = c.get(path,  None, update=update, max_age=max_age)
        if subnets is not None:
            return subnets
        subnets_info =  self.get_all_subnets_info(block=block)
        subnets = []
        for subnet_info in subnets_info:
            netuid = subnet_info.netuid
            subnet = self.subnet(netuid=netuid, block=block).__dict__
            subnet['subnet_identity'] = subnet['subnet_identity'].__dict__ if subnet.get('subnet_identity', None) != None else None
            if neurons:
                neurons = self.neurons(netuid=netuid)
                subnet['neurons'] = neurons
            subnets.append(subnet)
        
            
        return subnets
    
    def create_wallet(self, name: str, hotkey: Optional = None) -> Dict:
        """Create a new wallet
        Args:
            name (str): Name of the wallet
            hotkey (str): Optional hotkey name
        Returns:
            Wallet information dictionary
        """
        wallet = bt.wallet(name=name, hotkey=hotkey)
        return wallet

    
    def get_wallet(self, name: str, hotkey: Optional = None) -> Dict:
        """Get wallet information
        Args:
            name (str): Name of the wallet
            hotkey (str): Optional hotkey name
        Returns:
            Wallet information dictionary
        """
        wallet = bt.wallet(name=name, hotkey=hotkey)
        return dir(wallet)

    
    
    def balance(self, address: str) -> float:
        """Get balance for an address
        Args:
            address (str): Wallet address
        Returns:
            Balance in TAO
        """
        balance = self.subtensor.get_balance(address)
        return balance.tao
    
    def transfer(self, 
                wallet_name: str,
                dest_address: str,
                amount: float) -> bool:
        """Transfer TAO between wallets
        Args:
            wallet_name (str): Source wallet name
            dest_address (str): Destination address
            amount (float): Amount to transfer
        Returns:
            Success boolean
        """
        wallet = bt.wallet(wallet_name)
        amount_bal = Balance.from_tao(amount)
        return self.subtensor.transfer(
            wallet=wallet,
            dest=dest_address,
            amount=amount_bal
        )
    
    def get_subnets(self) -> List[int]:
        """Get list of all subnets
        Returns:
            List of subnet UIDs
        """
        return self.subtensor.get_subnets()
    
    def metagraph(self, netuid: int = 1) -> Any:
        """Get metagraph for a subnet
        Args:
            netuid (int): Network UID
        Returns:
            Metagraph object
        """
        return self.subtensor.metagraph(netuid)
    
    meta = metagraph