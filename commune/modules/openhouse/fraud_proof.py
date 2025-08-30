import hashlib
import json
from typing import Dict, List, Optional, Tuple
from state_channel import StateChannel

class FraudProof:
    """Handles fraud proofs for the L2 rollup state channel."""
    
    @staticmethod
    def generate_proof(old_state: Dict, new_state: Dict, signatures: Dict[str, str]) -> Dict:
        """Generate a fraud proof when an invalid state transition is detected.
        
        Args:
            old_state: The previous valid state
            new_state: The claimed new state that is invalid
            signatures: Signatures of the valid state
            
        Returns:
            A fraud proof that can be submitted on-chain
        """
        # Verify the old state is valid (has proper signatures)
        # In a real implementation, we would verify signatures cryptographically
        
        proof = {
            'type': 'invalid_state_transition',
            'valid_state': old_state,
            'valid_state_signatures': signatures,
            'invalid_state': new_state,
            'proof_hash': hashlib.sha256(json.dumps({
                'old': old_state,
                'new': new_state
            }, sort_keys=True).encode()).hexdigest()
        }
        
        return proof
    
    @staticmethod
    def verify_proof(proof: Dict, channel: StateChannel) -> bool:
        """Verify a fraud proof against a channel state.
        
        Args:
            proof: The fraud proof to verify
            channel: The channel state to check against
            
        Returns:
            True if the proof is valid, False otherwise
        """
        if proof['type'] == 'invalid_state_transition':
            valid_state = proof['valid_state']
            invalid_state = proof['invalid_state']
            
            # Check if the valid state has a higher nonce than the current channel state
            if valid_state['nonce'] > channel.nonce:
                # Verify signatures on the valid state
                # In a real implementation, we would verify cryptographically
                
                # Check if the invalid state violates state transition rules
                # For example, check if balances don't add up correctly
                valid_sum = sum(valid_state['balances'].values())
                invalid_sum = sum(invalid_state['balances'].values())
                
                if valid_sum != invalid_sum:
                    return True
                    
                # Check for other violations like incorrect nonce progression
                if invalid_state['nonce'] != valid_state['nonce'] + 1:
                    return True
        
        return False
    
    @staticmethod
    def generate_inclusion_proof(channel_id: str, state: Dict, merkle_root: str, merkle_path: List[str]) -> Dict:
        """Generate a proof that a channel state was included in a rollup batch.
        
        Args:
            channel_id: The channel identifier
            state: The channel state
            merkle_root: The Merkle root submitted on-chain
            merkle_path: The Merkle path proving inclusion
            
        Returns:
            An inclusion proof
        """
        proof = {
            'type': 'inclusion_proof',
            'channel_id': channel_id,
            'state': state,
            'merkle_root': merkle_root,
            'merkle_path': merkle_path,
            'proof_hash': hashlib.sha256(json.dumps({
                'channel_id': channel_id,
                'state': state,
                'merkle_root': merkle_root
            }, sort_keys=True).encode()).hexdigest()
        }
        
        return proof
    
    @staticmethod
    def verify_inclusion_proof(proof: Dict) -> bool:
        """Verify an inclusion proof.
        
        Args:
            proof: The inclusion proof to verify
            
        Returns:
            True if the proof is valid, False otherwise
        """
        if proof['type'] != 'inclusion_proof':
            return False
            
        # In a real implementation, we would verify the Merkle path
        # by hashing the state and following the path to the root
        
        # For simulation, we'll assume the proof is valid if it has the correct structure
        return all(k in proof for k in ['channel_id', 'state', 'merkle_root', 'merkle_path'])
    
    @staticmethod
    def generate_exclusion_proof(channel_id: str, state: Dict, merkle_root: str) -> Dict:
        """Generate a proof that a channel state was NOT included in a rollup batch.
        
        Args:
            channel_id: The channel identifier
            state: The channel state that should have been included
            merkle_root: The Merkle root submitted on-chain
            
        Returns:
            An exclusion proof
        """
        # In a real implementation, this would be a complex zero-knowledge proof
        # For simulation, we'll create a simplified version
        
        proof = {
            'type': 'exclusion_proof',
            'channel_id': channel_id,
            'state': state,
            'merkle_root': merkle_root,
            'proof_hash': hashlib.sha256(json.dumps({
                'channel_id': channel_id,
                'state': state,
                'merkle_root': merkle_root,
                'excluded': True
            }, sort_keys=True).encode()).hexdigest()
        }
        
        return proof
