"""
Bittensor-specific validator implementation for the Polaris validator system.

This module defines the BittensorValidator class that handles validation and
weight submission for the Bittensor network.
"""
import logging
import time
import math
from typing import Dict, Any, List, Tuple, Optional

import bittensor as bt

from validator.src.validators.base_validator import BaseValidator
from validator.src.config import ValidatorConfig

logger = logging.getLogger(__name__)

class BittensorValidator(BaseValidator):
    """
    Bittensor-specific validator implementation.
    
    This class extends the BaseValidator to add Bittensor-specific functionality
    for validating miners and submitting weights on the Bittensor network.
    """
    
    def __init__(self, config: ValidatorConfig):
        """
        Initialize the Bittensor validator.
        
        Args:
            config: Validator configuration
        """
        # Initialize base validator
        super().__init__(config, network_name="bittensor")
        
        # Get Bittensor-specific settings
        self.netuid = config.network.bittensor['netuid']
        self.network = config.network.bittensor['network']
        self.wallet_name = config.network.bittensor['wallet_name']
        self.hotkey = config.network.bittensor['hotkey']
        self.subtensor_endpoint = config.network.bittensor['subtensor_endpoint']
        
        # Initialize Bittensor components
        self.subtensor = None
        self.wallet = None
        self.metagraph = None
        
        # Connect to Bittensor network
        self._initialize_bittensor()
        
        # Weight settings
        self.max_weight = config.max_weight_value
        self.min_score_for_weight = config.min_score_for_weight
    
    def _initialize_bittensor(self):
        """Initialize Bittensor components (subtensor, wallet, metagraph)."""
        try:
            # Configure logging
            bt.logging(level=self.config.log_level)
            
            # Initialize wallet
            self.wallet = bt.wallet(name=self.wallet_name, hotkey=self.hotkey)
            
            if not self.wallet:
                raise ValueError(f"Failed to load wallet: {self.wallet_name}/{self.hotkey}")
            
            logger.info(f"Loaded wallet: {self.wallet_name}/{self.hotkey}")
            
            # Initialize subtensor connection
            if self.subtensor_endpoint:
                self.subtensor = bt.subtensor(network=self.network, chain_endpoint=self.subtensor_endpoint)
            else:
                self.subtensor = bt.subtensor(network=self.network)
            
            if not self.subtensor:
                raise ValueError(f"Failed to connect to {self.network} network")
            
            logger.info(f"Connected to {self.network} network")
            
            # Initialize metagraph
            self.metagraph = self.subtensor.metagraph(netuid=self.netuid)
            
            if not self.metagraph:
                raise ValueError(f"Failed to load metagraph for netuid {self.netuid}")
            
            logger.info(f"Loaded metagraph for subnet {self.netuid} with {self.metagraph.n} neurons")
            
            # Check if wallet is registered on this subnet
            if not self.is_wallet_registered():
                logger.warning(f"Validator wallet {self.wallet.hotkey.ss58_address} is not registered on subnet {self.netuid}")
            else:
                logger.info(f"Validator is registered on subnet {self.netuid}")
                
        except Exception as e:
            logger.error(f"Failed to initialize Bittensor components: {e}")
            raise
    
    def is_wallet_registered(self) -> bool:
        """
        Check if the validator's wallet is registered on the network.
        
        Returns:
            True if registered, False otherwise
        """
        if not self.subtensor or not self.wallet:
            return False
        
        try:
            return self.subtensor.is_hotkey_registered(
                hotkey_ss58=self.wallet.hotkey.ss58_address,
                netuid=self.netuid
            )
        except Exception as e:
            logger.error(f"Error checking if wallet is registered: {e}")
            return False
    
    def verify_network_registration(self, miner_id: str, miner_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Verify that a miner is registered on the Bittensor network.
        
        Args:
            miner_id: The ID of the miner
            miner_data: Miner registration data
        
        Returns:
            Tuple of (is_registered, reason)
        """
        # Get miner's Bittensor hotkey
        bt_info = miner_data.get('bittensor', {})
        hotkey = bt_info.get('hotkey')
        
        if not hotkey:
            return False, "Missing Bittensor hotkey in registration data"
        
        # Check if hotkey is valid
        try:
            # Check if hotkey is registered on this subnet
            is_registered = self.subtensor.is_hotkey_registered(
                hotkey_ss58=hotkey,
                netuid=self.netuid
            )
            
            if not is_registered:
                return False, f"Hotkey {hotkey} is not registered on subnet {self.netuid}"
            
            # Find UID in metagraph
            uid = None
            for i, hotkey_i in enumerate(self.metagraph.hotkeys):
                if hotkey_i == hotkey:
                    uid = i
                    break
            
            if uid is None:
                return False, f"Hotkey {hotkey} is registered but not found in metagraph"
            
            # Store UID in miner data for later use
            bt_info['uid'] = uid
            
            return True, f"Registered on subnet {self.netuid} with UID {uid}"
            
        except Exception as e:
            logger.error(f"Error verifying registration for miner {miner_id}: {e}")
            return False, f"Error during verification: {str(e)}"
    
    def normalize_scores(self, miner_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize miner scores to weights that sum to 1.0.
        
        Args:
            miner_scores: Dictionary mapping miner IDs to scores
        
        Returns:
            Dictionary mapping miner IDs to normalized weights
        """
        if not miner_scores:
            return {}
        
        # Filter out miners with scores below threshold
        valid_miners = {
            miner_id: score 
            for miner_id, score in miner_scores.items() 
            if score >= self.min_score_for_weight
        }
        
        if not valid_miners:
            logger.warning("No miners have scores above the minimum threshold")
            return {}
        
        # Calculate total score
        total_score = sum(valid_miners.values())
        
        if total_score <= 0:
            logger.warning("Total score is zero or negative, no weights assigned")
            return {}
        
        # Normalize scores to weights
        weights = {
            miner_id: (score / total_score) * self.max_weight
            for miner_id, score in valid_miners.items()
        }
        
        logger.info(f"Normalized {len(weights)} miner scores to weights")
        return weights
    
    def submit_weights(self, miners: Dict[str, Dict[str, Any]]) -> bool:
        """
        Submit calculated weights to the Bittensor network.
        
        Args:
            miners: Dictionary mapping miner IDs to validation results
        
        Returns:
            True if submission was successful, False otherwise
        """
        if not miners:
            logger.warning("No miners to submit weights for")
            return False
        
        # Extract scores from validated miners
        miner_scores = {}
        uid_to_miner_id = {}
        
        for miner_id, result in miners.items():
            if result.get('status') != 'verified':
                continue
            
            score = result.get('score', 0)
            bt_info = result.get('miner_data', {}).get('bittensor', {})
            uid = bt_info.get('uid')
            
            if uid is not None:
                miner_scores[miner_id] = score
                uid_to_miner_id[uid] = miner_id
        
        if not miner_scores:
            logger.warning("No verified miners with UIDs found")
            return False
        
        # Normalize scores to weights
        normalized_weights = self.normalize_scores(miner_scores)
        
        if not normalized_weights:
            logger.warning("No weights to submit after normalization")
            return False
        
        try:
            # Prepare weights for submission
            uids = []
            weights = []
            
            # Get all UIDs in the network
            all_uids = list(range(self.metagraph.n))
            
            # Set weights for each UID
            for uid in all_uids:
                miner_id = uid_to_miner_id.get(uid)
                
                if miner_id and miner_id in normalized_weights:
                    uids.append(uid)
                    weights.append(float(normalized_weights[miner_id]))
            
            if not uids:
                logger.warning("No UIDs to set weights for")
                return False
            
            logger.info(f"Submitting weights for {len(uids)} miners")
            
            # Submit weights to the network
            result = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.netuid,
                uids=uids,
                weights=weights
            )
            
            # Log submission
            submission_data = {
                'uids': uids,
                'weights': weights,
                'result': str(result),
                'success': result is not None
            }
            
            self.firebase_client.log_weight_submission('bittensor', submission_data)
            
            if result:
                logger.info(f"Successfully submitted weights for {len(uids)} miners")
                return True
            else:
                logger.error("Failed to submit weights")
                return False
                
        except Exception as e:
            logger.error(f"Error submitting weights: {e}")
            return False 