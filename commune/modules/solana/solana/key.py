import commune as c
import base58
from solders.keypair import Keypair
from typing import Optional, Dict, Any, List
import json
import os

class Key:
    """
    Key manager for Solana operations
    """
    def __init__(self, name: str = 'default', path: str = None):
        self.name = name
        self.path = path or os.path.expanduser(f'~/.commune/sol/key/{name}.json')
        self.keypair = None
        self._load_or_create()
    
    def _load_or_create(self):
        """Load existing key or create new one"""
        if os.path.exists(self.path):
            self.load()
        else:
            self.create()
    
    def create(self) -> Dict[str, str]:
        """Create a new keypair"""
        self.keypair = Keypair()
        self.save()
        return self.info()
    
    def save(self):
        """Save keypair to file"""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        data = {
            'name': self.name,
            'private_key': base58.b58encode(bytes(self.keypair)).decode('utf-8'),
            'public_key': str(self.keypair.pubkey())
        }
        with open(self.path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self):
        """Load keypair from file"""
        with open(self.path, 'r') as f:
            data = json.load(f)
        self.keypair = Keypair.from_bytes(base58.b58decode(data['private_key']))
        self.name = data.get('name', self.name)
    
    def info(self) -> Dict[str, str]:
        """Get key information"""
        return {
            'name': self.name,
            'address': str(self.keypair.pubkey()),
            'private_key': base58.b58encode(bytes(self.keypair)).decode('utf-8'),
            'path': self.path
        }
    
    def export(self) -> str:
        """Export private key as base58 string"""
        return base58.b58encode(bytes(self.keypair)).decode('utf-8')
    
    def import_key(self, private_key: str):
        """Import keypair from private key"""
        self.keypair = Keypair.from_bytes(base58.b58decode(private_key))
        self.save()
    
    @classmethod
    def keys(cls, path: str = None) -> List[str]:
        """List all saved keys"""
        path = path or os.path.expanduser('~/.commune/sol/keys/')
        if not os.path.exists(path):
            return []
        return [f.replace('.json', '') for f in os.listdir(path) if f.endswith('.json')]
    
    @classmethod
    def get_key(cls, name: str = 'default', **kwargs) -> 'Key':
        """Get or create a key by name"""
        return cls(name=name, **kwargs)
    key = get_key  # Alias for convenience
    
    
    def sign(self, message: str) -> str:
        """Sign a message with the keypair"""
        if not self.keypair:
            raise ValueError("Keypair not loaded")
        if isinstance(message, str):
            message = message.encode('utf-8')
        signature = self.keypair.sign_message(message)
        # convert signature to base58 for easier handling

        return str(signature)


    
    def verify(self, message: str, signature: str, public_key: str = None) -> bool:
        """Verify a signature against a message"""
        try:
            if isinstance(message, str):
                message = message.encode('utf-8')
            
            # Use provided public key or self keypair's public key
            if public_key:
                from solders.pubkey import Pubkey
                pubkey = Pubkey.from_string(public_key)
            else:
                if not self.keypair:
                    raise ValueError("Keypair not loaded")
                pubkey = self.keypair.pubkey()
            
            # Decode the signature
            sig_bytes = base58.b58decode(signature)
            
            # Verify using Solana's ed25519 verification
            from solders.signature import Signature
            sig = Signature.from_bytes(sig_bytes)
            
            # Reconstruct the signed message format that Solana uses
            from nacl.signing import VerifyKey
            import nacl.encoding
            
            verify_key = VerifyKey(bytes(pubkey))
            verify_key.verify(message, sig_bytes[:64])
            return True
            
        except Exception as e:
            print(f"Verification failed: {e}")
            return False
    
    def sign_message(self, message: str) -> str:
        """Sign a message with the keypair (alias for sign)"""
        return self.sign(message)
    
    def verify_signature(self, message: str, signature: str) -> bool:
        """Verify a signature against a message (alias for verify)"""
        return self.verify(message, signature)
    
    def test_signature(self, message: str = 'hey') -> Dict[str, Any]:
        """Test signing and verification of a message"""
        signature = self.sign_message(message)
        is_valid = self.verify_signature(message, signature)
        return {
            'message': message,
            'signature': signature,
            'is_valid': is_valid
        }



    @property
    def private_key(self) -> str:
        """Get the private key as a base58 string"""
        if not self.keypair:
            raise ValueError("Keypair not loaded")
        return base58.b58encode(bytes(self.keypair)).decode('utf-8')
    def set_private_key(self, private_key: str):
        """Set the private key from a base58 string"""
        self.import_key(private_key)

    @property
    def address(self) -> str:
        """Get the public address of the keypair"""
        if not self.keypair:
            raise ValueError("Keypair not loaded")
        return str(self.keypair.pubkey())

    def __str__(self):
        return f"Key(name={self.name}, address={self.address})"

    def get_key(self, name: str = 'default', **kwargs) -> 'Key':
        """Get or create a key by name"""
        return self.__class__(name=name, **kwargs)
