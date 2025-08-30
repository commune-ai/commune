"""
Bittensor Validator Implementation

This module provides the Bittensor-specific implementation of the validator,
handling all network-specific concerns like connecting to the Bittensor network,
scoring miners, and setting weights.
"""

import time
import traceback
from typing import Dict, List, Any, Optional, Tuple

from loguru import logger
from substrateinterface import Keypair

try:
    import bittensor as bt
    import torch
    BITTENSOR_AVAILABLE = True
except ImportError:
    logger.warning("Bittensor not available. Bittensor validation will be disabled.")
    BITTENSOR_AVAILABLE = False

from validator.src.validator_node._config import ValidatorSettings
from validator.src.validator_node.core_validator import CoreValidator
from validator.src.validator_node.pog import compare_compute_resources, fetch_compute_specs


class BittensorValidator(CoreValidator):
    """Bittensor-specific validator implementation."""
    
    def __init__(self, key: Keypair, settings: ValidatorSettings | None = None):
        """Initialize the Bittensor validator.
        
        Args:
            key: The keypair used for authentication and signing
            settings: Validator settings
        """
        super().__init__(key, settings)
        
        # Bittensor-specific attributes
        self._subtensor = None
        self._wallet = None
        self._metagraph = None
        self.netuid = getattr(settings, 'bittensor_netuid', 49)  # Default to 49 (Polaris)
        self.network = getattr(settings, 'bittensor_network', 'finney')  # Default to mainnet
        
        # Initialize immediately
        self._initialized = False
    
    @property
    def subtensor(self):
        """Get the Bittensor subtensor client."""
        if not BITTENSOR_AVAILABLE:
            raise ImportError("Bittensor package is not available")
            
        if self._subtensor is None:
            logger.debug(f"Creating new subtensor connection for network {self.network}")
            self._subtensor = bt.subtensor(network=self.network)
        return self._subtensor
    
    @property
    def wallet(self):
        """Get the Bittensor wallet."""
        if self._wallet is None:
            raise ValueError("Wallet not initialized. Call initialize_network() first.")
        return self._wallet
    
    @property
    def metagraph(self):
        """Get the subnet metagraph."""
        if self._metagraph is None or time.time() - self._last_metagraph_update > 600:  # Refresh every 10 minutes
            logger.debug(f"Updating metagraph for netuid {self.netuid}")
            self._metagraph = self.subtensor.metagraph(netuid=self.netuid)
            self._last_metagraph_update = time.time()
        return self._metagraph
    
    async def initialize_network(self) -> bool:
        """Initialize Bittensor network connection and wallet.
        
        Returns:
            bool: True if initialization was successful
        """
        if not BITTENSOR_AVAILABLE:
            logger.error("Bittensor package is not available. Cannot initialize.")
            return False
            
        if self._initialized:
            logger.debug("Bittensor network already initialized")
            return True
            
        try:
            logger.info(f"Initializing Bittensor network connection to {self.network}")
            
            # Initialize subtensor connection
            self._subtensor = bt.subtensor(network=self.network)
            logger.info(f"Connected to Bittensor network: {self._subtensor.network}")
            
            # Initialize wallet
            self._wallet = self._create_wallet_from_key()
            if not self._wallet:
                logger.error("Failed to create Bittensor wallet")
                return False
                
            # Check if wallet is registered on the subnet
            if not self.subtensor.is_hotkey_registered(
                netuid=self.netuid,
                hotkey_ss58=self._wallet.hotkey.ss58_address
            ):
                logger.warning(
                    f"Validator hotkey {self._wallet.hotkey.ss58_address} is not registered " 
                    f"on subnet {self.netuid}"
                )
                
            # Initialize metagraph
            logger.info(f"Loading metagraph for subnet {self.netuid}")
            self._metagraph = self.subtensor.metagraph(netuid=self.netuid)
            self._last_metagraph_update = time.time()
            
            logger.info(f"Found {len(self._metagraph.uids)} miners on subnet {self.netuid}")
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Bittensor network: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _create_wallet_from_key(self) -> Optional["bt.wallet"]:
        """Create a Bittensor wallet from the existing key.
        
        Returns:
            bt.wallet: Bittensor wallet or None if creation failed
        """
        try:
            # Create a wallet with the validator name
            wallet = bt.wallet(name="polaris_validator", hotkey="validator")
            
            # TODO: Implement proper key conversion between Commune keypair and Bittensor wallet
            # This is a placeholder - we'd need to either:
            # 1. Create a wallet with the same mnemonic/seed as the Commune key
            # 2. Or use the same SS58 address format if compatible
            
            # For now, we'll generate a warning
            logger.warning(
                "Using placeholder wallet. In production, implement proper key conversion "
                "between Commune keypair and Bittensor wallet"
            )
            
            # Test that the wallet is usable
            try:
                ss58_address = wallet.hotkey.ss58_address
                logger.info(f"Wallet initialized with hotkey address: {ss58_address}")
                return wallet
            except Exception as e:
                logger.error(f"Wallet initialized but not usable: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create Bittensor wallet: {e}")
            logger.error(traceback.format_exc())
            return None
    
    async def validate_step(self) -> None:
        """Perform a complete validation step for Bittensor miners."""
        if not BITTENSOR_AVAILABLE:
            logger.error("Bittensor package not available. Cannot validate.")
            return
            
        if not self._initialized:
            success = await self.initialize_network()
            if not success:
                logger.error("Failed to initialize network. Skipping validation step.")
                return
        
        try:
            logger.info("Starting Bittensor validation cycle")
            
            # 1. Get miners to validate
            miners = await self.get_miners()
            if not miners:
                logger.info("No miners found to validate")
                return
                
            logger.info(f"Found {len(miners)} miners to validate")
            
            # 2. Score miners
            scores = await self.score_miners(miners)
            if not scores:
                logger.info("No valid scores, skipping weight update")
                return
                
            logger.info(f"Scored {len(scores)} miners")
            
            # 3. Normalize scores
            normalized_scores = self.normalize_scores(scores)
            
            # 4. Convert to weights
            weights_dict = {
                uid: self.assign_weight(score)
                for uid, score in normalized_scores.items()
            }
            
            # 5. Trim to maximum allowed weights
            weights_dict = self.cut_to_max_allowed_weights(weights_dict)
            
            # 6. Set weights on network
            success = await self.set_weights(weights_dict)
            if success:
                logger.info(f"Successfully set weights for {len(weights_dict)} miners")
            else:
                logger.error("Failed to set weights")
                
        except Exception as e:
            logger.error(f"Error during Bittensor validation: {e}")
            logger.error(traceback.format_exc())
    
    async def get_miners(self) -> Dict[str, Dict[str, Any]]:
        """Get miners to validate from the Bittensor subnet.
        
        Returns:
            Dict: Dictionary of miners by UID
        """
        try:
            # Refresh metagraph if needed
            metagraph = self.metagraph
            
            miners = {}
            for uid in metagraph.uids.tolist():
                # Skip our own validator UID if it's in the metagraph
                if metagraph.hotkeys[uid] == self.wallet.hotkey.ss58_address:
                    logger.debug(f"Skipping our own validator UID: {uid}")
                    continue
                    
                # Skip miners with no axon info
                if not metagraph.axons[uid]:
                    logger.debug(f"Skipping miner {uid}: No axon info")
                    continue
                
                # Extract miner information
                miner_info = {
                    'uid': uid,
                    'hotkey': metagraph.hotkeys[uid],
                    'axon_info': metagraph.axons[uid],
                    'stake': float(metagraph.stake[uid]),
                    'trust': float(metagraph.trust[uid]),
                    'consensus': float(metagraph.consensus[uid]),
                    'active': bool(metagraph.active[uid]),
                }
                
                miners[str(uid)] = miner_info
                
            return miners
            
        except Exception as e:
            logger.error(f"Error getting miners from Bittensor: {e}")
            return {}
    
    async def score_miners(self, miners: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Score miners based on their resources and performance.
        
        Args:
            miners: Dictionary of miners to score
            
        Returns:
            Dict: Dictionary mapping miner UIDs to their scores
        """
        scores = {}
        
        for uid, miner in miners.items():
            try:
                # Skip inactive miners
                if not miner.get('active', False):
                    logger.debug(f"Skipping inactive miner {uid}")
                    continue
                
                # Get axon info
                axon_info = miner.get('axon_info')
                if not axon_info:
                    logger.debug(f"Skipping miner {uid}: No axon info")
                    continue
                
                # Try to fetch compute specs via axon
                # This would be a custom request to the miner's axon
                # For now, we'll use a placeholder scoring method
                spec_score = await self._score_miner_specs(uid, miner)
                uptime_score = self._score_miner_uptime(miner)
                
                # Combine scores
                final_score = 0.7 * spec_score + 0.3 * uptime_score
                scores[uid] = final_score
                
                logger.debug(f"Scored miner {uid}: {final_score:.4f} (specs: {spec_score:.4f}, uptime: {uptime_score:.4f})")
                
            except Exception as e:
                logger.warning(f"Error scoring miner {uid}: {e}")
        
        return scores
    
    async def _score_miner_specs(self, uid: str, miner: Dict[str, Any]) -> float:
        """Score a miner based on their hardware specifications.
        
        This would normally interact with the miner's axon to get specs.
        For now, it uses a placeholder scoring method.
        
        Args:
            uid: Miner UID
            miner: Miner information dictionary
            
        Returns:
            float: Score from 0.0 to 1.0
        """
        # TODO: Implement real spec verification via axon
        # For now, we'll use stake as a proxy for specs
        # In a real implementation, you'd query the miner's axon
        
        stake = miner.get('stake', 0.0)
        max_stake = 100.0  # Assuming 100 TAO is a good amount
        
        # Normalize stake to a 0-1 range
        score = min(stake / max_stake, 1.0)
        
        # Add some randomness for testing
        import random
        random_factor = random.uniform(0.8, 1.2)
        score = min(score * random_factor, 1.0)
        
        return score
    
    def _score_miner_uptime(self, miner: Dict[str, Any]) -> float:
        """Score a miner based on their uptime/reliability.
        
        Args:
            miner: Miner information dictionary
            
        Returns:
            float: Score from 0.0 to 1.0
        """
        # Use the trust score from the metagraph as a proxy for uptime
        trust = miner.get('trust', 0.0)
        consensus = miner.get('consensus', 0.0)
        
        # Combine trust and consensus for an uptime score
        uptime_score = 0.5 * trust + 0.5 * consensus
        
        return min(uptime_score, 1.0)
    
    async def set_weights(self, weights: Dict[str, float]) -> bool:
        """Set weights for miners on the Bittensor network.
        
        Args:
            weights: Dictionary mapping miner UIDs to their weights
            
        Returns:
            bool: True if weights were successfully set
        """
        if not weights:
            logger.info("No weights to set")
            return False
            
        try:
            # Convert weights dict to format expected by Bittensor
            uids = [int(uid) for uid in weights.keys()]
            weight_values = [float(weights[str(uid)]) for uid in uids]
            
            # Record history before setting weights
            self.add_weights_history(
                uids=[str(uid) for uid in uids],
                weights=weight_values,
                network="bittensor"
            )
            
            # Format into a tensor for Bittensor
            # Create a zero tensor with the size of the full metagraph
            weight_tensor = torch.zeros(len(self.metagraph.uids))
            
            # Set the weights for the specific UIDs
            for i, uid in enumerate(uids):
                idx = (self.metagraph.uids == uid).nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    weight_tensor[idx[0]] = weight_values[i]
            
            logger.info(f"Setting weights for {len(uids)} miners on Bittensor network")
            
            # Submit weights to the network
            result = self.subtensor.set_weights(
                netuid=self.netuid,
                wallet=self.wallet,
                uids=self.metagraph.uids,
                weights=weight_tensor,
                version_key=1,
                wait_for_inclusion=False
            )
            
            if not result:
                logger.error("Failed to set weights on Bittensor network")
                return False
                
            logger.info(f"Successfully set weights for {len(uids)} miners on Bittensor network")
            return True
            
        except Exception as e:
            logger.error(f"Error setting weights on Bittensor network: {e}")
            logger.error(traceback.format_exc())
            return False 