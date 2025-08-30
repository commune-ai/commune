import time
import json
import hashlib
from typing import Dict, List, Optional, Set
from state_channel import StateChannel

class RollupAggregator:
    """Aggregates state channel updates and submits them to the Bitcoin blockchain."""
    
    def __init__(self, batch_size: int = 50, batch_timeout: int = 600):
        """Initialize the rollup aggregator.
        
        Args:
            batch_size: Maximum number of state updates to include in a batch
            batch_timeout: Maximum time to wait before submitting a batch (in seconds)
        """
        self.channels: Dict[str, StateChannel] = {}
        self.pending_updates: List[Dict] = []
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.last_batch_time = int(time.time())
        self.state_roots: List[str] = []
        
    def register_channel(self, channel: StateChannel) -> bool:
        """Register a new state channel with the aggregator."""
        if channel.channel_id in self.channels:
            return False
            
        self.channels[channel.channel_id] = channel
        return True
    
    def submit_state_update(self, channel_id: str, state_data: Dict) -> bool:
        """Submit a state update for a channel."""
        if channel_id not in self.channels:
            return False
            
        channel = self.channels[channel_id]
        
        # Verify the update has a higher nonce than current state
        if state_data['nonce'] <= channel.nonce:
            return False
            
        # Add to pending updates
        self.pending_updates.append({
            'channel_id': channel_id,
            'state': state_data,
            'timestamp': int(time.time())
        })
        
        return True
    
    def compute_merkle_root(self, items: List[Dict]) -> str:
        """Compute a Merkle root from a list of state updates."""
        if not items:
            return hashlib.sha256(b'empty').hexdigest()
            
        # Hash each item
        leaves = [hashlib.sha256(json.dumps(item, sort_keys=True).encode()).digest() for item in items]
        
        # Simple implementation - in a real system, we'd build a proper Merkle tree
        combined = b''.join(leaves)
        return hashlib.sha256(combined).hexdigest()
    
    def should_create_batch(self) -> bool:
        """Determine if a new batch should be created based on size or timeout."""
        if len(self.pending_updates) >= self.batch_size:
            return True
            
        current_time = int(time.time())
        if current_time - self.last_batch_time >= self.batch_timeout and self.pending_updates:
            return True
            
        return False
    
    def create_batch(self) -> Optional[str]:
        """Create a new batch of state updates and return the Merkle root."""
        if not self.should_create_batch():
            return None
            
        # Process pending updates
        updates_to_process = self.pending_updates[:self.batch_size]
        self.pending_updates = self.pending_updates[self.batch_size:]
        
        # Apply updates to channels
        for update in updates_to_process:
            channel_id = update['channel_id']
            state_data = update['state']
            
            if channel_id in self.channels:
                channel = self.channels[channel_id]
                # In a real implementation, we would verify signatures here
                channel.update_state(state_data['balances'], state_data['signatures'])
        
        # Compute Merkle root of all channel states
        channel_states = [channel.to_dict() for channel in self.channels.values()]
        merkle_root = self.compute_merkle_root(channel_states)
        
        # Record the state root
        self.state_roots.append(merkle_root)
        self.last_batch_time = int(time.time())
        
        return merkle_root
    
    def submit_to_blockchain(self, merkle_root: str) -> Dict:
        """Submit a batch's Merkle root to the Bitcoin blockchain.
        
        In a real implementation, this would create a Bitcoin transaction
        with the Merkle root embedded in an OP_RETURN output.
        """
        # Simulate blockchain submission
        return {
            'txid': hashlib.sha256(merkle_root.encode()).hexdigest(),
            'merkle_root': merkle_root,
            'timestamp': int(time.time()),
            'block_height': 700000 + len(self.state_roots)  # Simulated block height
        }
    
    def process_batch_if_needed(self) -> Optional[Dict]:
        """Process a batch if needed and submit it to the blockchain."""
        merkle_root = self.create_batch()
        if not merkle_root:
            return None
            
        return self.submit_to_blockchain(merkle_root)
    
    def get_channel_state(self, channel_id: str) -> Optional[Dict]:
        """Get the current state of a channel."""
        if channel_id not in self.channels:
            return None
            
        return self.channels[channel_id].to_dict()
    
    def process_channel_dispute(self, channel_id: str, state_data: Dict) -> bool:
        """Process a dispute for a channel."""
        if channel_id not in self.channels:
            return False
            
        channel = self.channels[channel_id]
        
        # If the provided state has a higher nonce and valid signatures,
        # update the channel state
        if state_data['nonce'] > channel.nonce and self._verify_state_signatures(state_data):
            channel.update_state(state_data['balances'], state_data['signatures'])
            return True
            
        return False
    
    def _verify_state_signatures(self, state_data: Dict) -> bool:
        """Verify signatures on a state update."""
        # In a real implementation, we would verify all signatures here
        return True
