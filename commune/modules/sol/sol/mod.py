import commune as c
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.system_program import TransferParams, transfer
from solders.transaction import Transaction
from solana.rpc.api import Client
import base58

class Base:
    """
    A Solana-integrated base class that provides functionality for interacting with the Solana blockchain.
    """
    def __init__(self, **kwargs):
        """
        Initialize the base class with Solana configuration.
        Args:
            **kwargs: Arbitrary keyword arguments to configure the instance
        """
        # Initialize OpenRouter model for explanations
        self.model = c.module('openrouter')()
        
        # Solana RPC endpoint (mainnet-beta, testnet, or devnet)
        self.rpc_url = kwargs.get('rpc_url', 'https://api.devnet.solana.com')
        self.client = Client(self.rpc_url)
        
        # Initialize or load keypair
        self.keypair = None
        if 'private_key' in kwargs:
            # Load from base58 encoded private key
            self.keypair = Keypair.from_bytes(base58.b58decode(kwargs['private_key']))
        else:
            # Generate new keypair
            self.keypair = Keypair()
            
    def forward(self, module: str='explain', *args, stream=1, **kwargs):
        """
        Dynamically call a method of the class or explain module code.
        Args:
            module (str): Name of the module to explain or method to call
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method
        Returns:
            Result of the called method or explanation
        """
        # Check if it's a Solana method
        if hasattr(self, module) and module.startswith('sol_'):
            method = getattr(self, module)
            return method(*args, **kwargs)
        
        # Default behavior: explain module code
        return self.model.forward(f'what does this do? {c.code(module)}', stream=stream)
    
    def sol_get_balance(self, address: str = None):
        """
        Get SOL balance for an address.
        Args:
            address: Base58 encoded public key (uses own address if None)
        Returns:
            Balance in SOL
        """
        if address is None:
            pubkey = self.keypair.pubkey()
        else:
            pubkey = Pubkey.from_string(address)
            
        balance_lamports = self.client.get_balance(pubkey).value
        balance_sol = balance_lamports / 1e9  # Convert lamports to SOL
        return balance_sol
    
    def sol_transfer(self, to_address: str, amount_sol: float):
        """
        Transfer SOL to another address.
        Args:
            to_address: Recipient's base58 encoded public key
            amount_sol: Amount of SOL to transfer
        Returns:
            Transaction signature
        """
        # Convert SOL to lamports
        amount_lamports = int(amount_sol * 1e9)
        
        # Create transfer instruction
        transfer_params = TransferParams(
            from_pubkey=self.keypair.pubkey(),
            to_pubkey=Pubkey.from_string(to_address),
            lamports=amount_lamports
        )
        transfer_instruction = transfer(transfer_params)
        
        # Get recent blockhash
        recent_blockhash = self.client.get_latest_blockhash().value.blockhash
        
        # Create and sign transaction
        transaction = Transaction.new_signed_with_payer(
            [transfer_instruction],
            self.keypair.pubkey(),
            [self.keypair],
            recent_blockhash
        )
        
        # Send transaction
        response = self.client.send_transaction(transaction)
        return response.value
    
    def sol_get_address(self):
        """
        Get the public key address of the current keypair.
        Returns:
            Base58 encoded public key
        """
        return str(self.keypair.pubkey())
    
    def sol_get_private_key(self):
        """
        Get the private key of the current keypair.
        Returns:
            Base58 encoded private key
        """
        return base58.b58encode(bytes(self.keypair)).decode('utf-8')
    
    def sol_airdrop(self, amount_sol: float = 1.0):
        """
        Request an airdrop (only works on devnet/testnet).
        Args:
            amount_sol: Amount of SOL to request
        Returns:
            Transaction signature
        """
        amount_lamports = int(amount_sol * 1e9)
        signature = self.client.request_airdrop(self.keypair.pubkey(), amount_lamports)
        return signature.value
