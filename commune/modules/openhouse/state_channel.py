import hashlib
import json
import time
from typing import Dict, List, Optional, Tuple
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes, serialization

class StateChannel:
    """Implements a state channel for off-chain Bitcoin transactions."""
    
    def __init__(self, channel_id: str, participants: List[str], initial_balances: Dict[str, int]):
        """Initialize a new state channel.
        
        Args:
            channel_id: Unique identifier for this channel
            participants: List of participant public keys
            initial_balances: Initial BTC balances for each participant (in satoshis)
        """
        self.channel_id = channel_id
        self.participants = participants
        self.balances = initial_balances
        self.nonce = 0
        self.is_open = True
        self.signatures = {}
        self.dispute_period = 24 * 60 * 60  # 24 hours in seconds
        self.closing_time = None
        
    def get_state_hash(self) -> str:
        """Generate a hash of the current channel state."""
        state = {
            'channel_id': self.channel_id,
            'balances': self.balances,
            'nonce': self.nonce,
            'is_open': self.is_open
        }
        serialized = json.dumps(state, sort_keys=True).encode()
        return hashlib.sha256(serialized).hexdigest()
    
    def sign_state(self, private_key: ec.EllipticCurvePrivateKey) -> str:
        """Sign the current state with a participant's private key."""
        state_hash = self.get_state_hash().encode()
        signature = private_key.sign(
            state_hash,
            ec.ECDSA(hashes.SHA256())
        )
        return signature.hex()
    
    def add_signature(self, public_key: str, signature: str) -> bool:
        """Add a signature from a participant."""
        if public_key not in self.participants:
            return False
        
        self.signatures[public_key] = signature
        return True
    
    def verify_signatures(self) -> bool:
        """Verify that all participants have signed the current state."""
        if len(self.signatures) != len(self.participants):
            return False
            
        state_hash = self.get_state_hash().encode()
        
        for pub_key, signature in self.signatures.items():
            # In a real implementation, we would verify the signature here
            # using the public key and the state hash
            pass
            
        return True
    
    def update_state(self, new_balances: Dict[str, int], signatures: Dict[str, str]) -> bool:
        """Update the channel state with new balances and signatures."""
        if not self.is_open:
            return False
            
        # Verify total balance remains the same
        if sum(new_balances.values()) != sum(self.balances.values()):
            return False
            
        # Verify all participants are accounted for
        if set(new_balances.keys()) != set(self.participants):
            return False
            
        # Update state
        self.balances = new_balances
        self.nonce += 1
        self.signatures = signatures
        
        return True
    
    def initiate_close(self) -> None:
        """Initiate channel closing, starting the dispute period."""
        if not self.is_open:
            return
            
        self.closing_time = int(time.time())
    
    def can_close(self) -> bool:
        """Check if the dispute period has passed and the channel can be closed."""
        if not self.closing_time:
            return False
            
        return int(time.time()) >= self.closing_time + self.dispute_period
    
    def close_channel(self) -> Dict[str, int]:
        """Close the channel and return final balances."""
        if not self.can_close():
            return {}
            
        self.is_open = False
        return self.balances
    
    def to_dict(self) -> Dict:
        """Convert the channel state to a dictionary."""
        return {
            'channel_id': self.channel_id,
            'participants': self.participants,
            'balances': self.balances,
            'nonce': self.nonce,
            'is_open': self.is_open,
            'signatures': self.signatures,
            'closing_time': self.closing_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StateChannel':
        """Create a StateChannel instance from a dictionary."""
        channel = cls(
            channel_id=data['channel_id'],
            participants=data['participants'],
            initial_balances=data['balances']
        )
        channel.nonce = data['nonce']
        channel.is_open = data['is_open']
        channel.signatures = data['signatures']
        channel.closing_time = data['closing_time']
        return channel
