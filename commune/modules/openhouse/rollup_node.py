import argparse
import json
import time
import threading
from typing import Dict, List, Optional

from state_channel import StateChannel
from rollup_aggregator import RollupAggregator
from bitcoin_interface import BitcoinInterface
from fraud_proof import FraudProof

class RollupNode:
    """Node implementation for the L2 Bitcoin rollup using state channels."""
    
    def __init__(self, node_id: str, network: str = 'testnet'):
        """Initialize the rollup node.
        
        Args:
            node_id: Unique identifier for this node
            network: Bitcoin network to connect to
        """
        self.node_id = node_id
        self.aggregator = RollupAggregator(batch_size=10, batch_timeout=60)
        self.bitcoin = BitcoinInterface(network=network)
        self.channels: Dict[str, StateChannel] = {}
        self.running = False
        self.batch_thread = None
    
    def create_channel(self, channel_id: str, participants: List[str], 
                      initial_balances: Dict[str, int]) -> Dict:
        """Create a new state channel."""
        # Create the channel
        channel = StateChannel(channel_id, participants, initial_balances)
        
        # Register with aggregator
        self.aggregator.register_channel(channel)
        
        # Store locally
        self.channels[channel_id] = channel
        
        # Create multisig address
        multisig = self.bitcoin.create_multisig_address(
            public_keys=participants,
            m=len(participants)  # Require all signatures (can be adjusted)
        )
        
        return {
            'channel': channel.to_dict(),
            'multisig_address': multisig
        }
    
    def fund_channel(self, channel_id: str, funding_tx: Dict) -> Dict:
        """Fund a channel with an on-chain transaction."""
        if channel_id not in self.channels:
            return {'success': False, 'error': 'Channel not found'}
        
        # Submit funding transaction to Bitcoin
        tx = self.bitcoin.fund_channel(channel_id, funding_tx)
        
        return {
            'success': True,
            'transaction': tx
        }
    
    def update_channel(self, channel_id: str, new_balances: Dict[str, int], 
                      signatures: Dict[str, str]) -> Dict:
        """Update a channel's state off-chain."""
        if channel_id not in self.channels:
            return {'success': False, 'error': 'Channel not found'}
        
        channel = self.channels[channel_id]
        
        # Update the channel state
        success = channel.update_state(new_balances, signatures)
        
        if not success:
            return {'success': False, 'error': 'Failed to update state'}
        
        # Submit state update to aggregator
        state_data = {
            'balances': new_balances,
            'nonce': channel.nonce,
            'signatures': signatures
        }
        
        self.aggregator.submit_state_update(channel_id, state_data)
        
        return {
            'success': True,
            'channel_state': channel.to_dict()
        }
    
    def initiate_channel_close(self, channel_id: str) -> Dict:
        """Initiate the closing of a channel."""
        if channel_id not in self.channels:
            return {'success': False, 'error': 'Channel not found'}
        
        channel = self.channels[channel_id]
        channel.initiate_close()
        
        return {
            'success': True,
            'channel_state': channel.to_dict(),
            'closing_time': channel.closing_time,
            'dispute_period_ends': channel.closing_time + channel.dispute_period
        }
    
    def close_channel(self, channel_id: str) -> Dict:
        """Close a channel and settle on-chain after the dispute period."""
        if channel_id not in self.channels:
            return {'success': False, 'error': 'Channel not found'}
        
        channel = self.channels[channel_id]
        
        if not channel.can_close():
            return {
                'success': False, 
                'error': 'Dispute period not over',
                'remaining_time': (channel.closing_time + channel.dispute_period) - int(time.time())
            }
        
        # Close the channel
        final_balances = channel.close_channel()
        
        # Submit closing transaction to Bitcoin
        tx = self.bitcoin.close_channel(
            channel_id=channel_id,
            final_state=channel.to_dict(),
            signatures=channel.signatures
        )
        
        return {
            'success': True,
            'transaction': tx,
            'final_balances': final_balances
        }
    
    def submit_fraud_proof(self, channel_id: str, valid_state: Dict, 
                          invalid_state: Dict, signatures: Dict[str, str]) -> Dict:
        """Submit a fraud proof for a channel."""
        if channel_id not in self.channels:
            return {'success': False, 'error': 'Channel not found'}
        
        # Generate fraud proof
        proof = FraudProof.generate_proof(valid_state, invalid_state, signatures)
        
        # Verify the proof locally
        channel = self.channels[channel_id]
        is_valid = FraudProof.verify_proof(proof, channel)
        
        if not is_valid:
            return {'success': False, 'error': 'Invalid fraud proof'}
        
        # In a real implementation, we would submit this proof to the Bitcoin blockchain
        # For simulation, we'll just update the channel state
        channel.update_state(valid_state['balances'], signatures)
        
        return {
            'success': True,
            'proof': proof,
            'updated_channel_state': channel.to_dict()
        }
    
    def get_channel_state(self, channel_id: str) -> Dict:
        """Get the current state of a channel."""
        if channel_id not in self.channels:
            return {'success': False, 'error': 'Channel not found'}
        
        return {
            'success': True,
            'channel_state': self.channels[channel_id].to_dict()
        }
    
    def _batch_processing_loop(self):
        """Background thread for processing batches."""
        while self.running:
            # Process batch if needed
            result = self.aggregator.process_batch_if_needed()
            
            if result:
                # Submit state root to Bitcoin
                tx = self.bitcoin.submit_state_root(result['merkle_root'])
                print(f"Submitted batch with root {result['merkle_root']} in tx {tx['txid']}")
            
            # Sleep for a short time
            time.sleep(5)
    
    def start(self):
        """Start the rollup node."""
        if self.running:
            return
            
        self.running = True
        self.batch_thread = threading.Thread(target=self._batch_processing_loop)
        self.batch_thread.daemon = True
        self.batch_thread.start()
        
        print(f"Rollup node {self.node_id} started")
    
    def stop(self):
        """Stop the rollup node."""
        self.running = False
        if self.batch_thread:
            self.batch_thread.join(timeout=2.0)
        print(f"Rollup node {self.node_id} stopped")

def main():
    parser = argparse.ArgumentParser(description='Run a Bitcoin L2 rollup node')
    parser.add_argument('--node-id', type=str, default='node1', help='Node identifier')
    parser.add_argument('--network', type=str, default='testnet', help='Bitcoin network')
    args = parser.parse_args()
    
    node = RollupNode(node_id=args.node_id, network=args.network)
    node.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        node.stop()

if __name__ == '__main__':
    main()
