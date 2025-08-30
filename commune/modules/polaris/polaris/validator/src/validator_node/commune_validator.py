"""
Commune Validator Implementation

This module provides the Commune-specific implementation of the validator,
handling all network-specific concerns like connecting to the Commune network,
scoring miners, and setting weights.
"""

import traceback
from typing import Dict, List, Any, Optional

from communex._common import get_node_url
from communex.client import CommuneClient
from loguru import logger
from substrateinterface import Keypair

from validator.src.validator_node._config import ValidatorSettings
from validator.src.validator_node.core_validator import CoreValidator
from validator.src.validator_node.base.utils import get_netuid
from validator.src.validator_node.pog import compute_resource_score, compare_compute_resources


class CommuneValidator(CoreValidator):
    """Commune-specific validator implementation."""
    
    def __init__(self, key: Keypair, settings: ValidatorSettings | None = None):
        """Initialize the Commune validator.
        
        Args:
            key: The keypair used for authentication and signing
            settings: Validator settings
        """
        super().__init__(key, settings)
        
        # Commune-specific attributes
        self._c_client = None
        self.netuid = None
        self.miner_data = {}
        self.container_start_times = {}
        
        # Initialize network connection
        self._initialized = False
    
    @property
    def c_client(self):
        """Get the Commune client."""
        if self._c_client is None:
            logger.warning("Client accessed but not initialized, attempting to create new client")
            node_url = get_node_url(use_testnet=self.settings.use_testnet)
            self._c_client = CommuneClient(node_url)
        return self._c_client
    
    async def initialize_network(self) -> bool:
        """Initialize Commune network connection.
        
        Returns:
            bool: True if initialization was successful
        """
        if self._initialized:
            logger.debug("Commune network already initialized")
            return True
            
        try:
            logger.info("Initializing Commune network connection")
            
            # Get node URL based on testnet setting
            node_url = get_node_url(use_testnet=self.settings.use_testnet)
            logger.debug(f"Using Commune node URL: {node_url}")
            
            # Initialize client
            self._c_client = CommuneClient(node_url)
            
            # Test connection
            try:
                subnets = self._c_client.query_map_subnet_names()
                logger.debug(f"Available subnets: {subnets}")
            except Exception as e:
                logger.error(f"Failed to query subnets: {e}")
                raise
            
            # Get netuid
            self.netuid = get_netuid(self._c_client)
            logger.info(f"Using Commune netuid: {self.netuid}")
            
            # Get validator ss58 address
            validator_ss58 = self.key.ss58_address
            
            # Check if validator is registered
            modules_keys = self._c_client.query_map_key(self.netuid)
            if validator_ss58 not in modules_keys.values():
                logger.warning(f"Validator key {validator_ss58} is not registered in subnet {self.netuid}")
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Commune network: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def validate_step(self) -> None:
        """Perform a complete validation step for Commune miners."""
        if not self._initialized:
            success = await self.initialize_network()
            if not success:
                logger.error("Failed to initialize network. Skipping validation step.")
                return
        
        try:
            logger.info("Starting Miner program validation cycle")
            
            # Track miner containers to update miner_data
            await self.track_miner_containers()
            
            # Get scores from miner_data
            scores = self.miner_data
            if not scores:
                logger.info("No valid scores in miner_data, skipping weight update")
                return
                
            logger.info(f"Found scores for {len(scores)} miners")
            
            # Normalize scores
            normalized_scores = self.normalize_scores(scores)
            
            # Convert to weights
            weights_dict = {
                uid: self.assign_weight(score)
                for uid, score in normalized_scores.items()
            }
            
            # Trim to maximum allowed weights
            weights_dict = self.cut_to_max_allowed_weights(weights_dict)
            
            # Set weights on network
            success = await self.set_weights(weights_dict)
            if success:
                logger.info(f"Successfully set weights for {len(weights_dict)} miners")
            else:
                logger.error("Failed to set weights")
                
        except Exception as e:
            logger.error(f"Error during Commune validation: {e}")
            logger.error(traceback.format_exc())
    
    async def get_miners(self) -> Dict[str, Any]:
        """Get miners to validate from the Commune subnet.
        
        Returns:
            Dict: Dictionary of miners by UID
        """
        try:
            miners = {}
            
            # Get modules from subnet
            modules = self.c_client.query_map_modules(self.netuid)
            
            # Extract module information
            for uid, module in modules.items():
                # Skip our own validator UID
                if module.get('key') == self.key.ss58_address:
                    logger.debug(f"Skipping our own validator UID: {uid}")
                    continue
                
                # Extract miner information
                miner_info = {
                    'uid': uid,
                    'key': module.get('key'),
                    'address': module.get('address'),
                    'stake': module.get('stake', 0),
                    'trust': module.get('trust', 0),
                    'consensus': module.get('consensus', 0),
                    'incentive': module.get('incentive', 0),
                    'dividends': module.get('dividends', 0),
                    'last_update': module.get('last_update', 0),
                }
                
                miners[str(uid)] = miner_info
            
            return miners
            
        except Exception as e:
            logger.error(f"Error getting miners from Commune: {e}")
            return {}
    
    async def score_miners(self, miners: Dict[str, Any]) -> Dict[str, float]:
        """Score miners based on their resources and performance.
        
        Args:
            miners: Dictionary of miners to score
            
        Returns:
            Dict: Dictionary mapping miner UIDs to their scores
        """
        # For Commune, we typically verify miners via SSH and store the scores in miner_data
        # This method would be used for direct scoring without container tracking
        
        scores = {}
        
        for uid, miner in miners.items():
            try:
                # For now, we'll just use any existing scores from miner_data
                if uid in self.miner_data:
                    scores[uid] = self.miner_data[uid]
                    continue
                
                # If we don't have a score in miner_data, we could calculate it here
                # based on miner information (stake, trust, etc.)
                stake = miner.get('stake', 0)
                trust = miner.get('trust', 0)
                consensus = miner.get('consensus', 0)
                
                # Simple placeholder scoring
                score = (stake * 0.4) + (trust * 0.3) + (consensus * 0.3)
                scores[uid] = score
                
            except Exception as e:
                logger.warning(f"Error scoring miner {uid}: {e}")
        
        return scores
    
    async def set_weights(self, weights: Dict[str, float]) -> bool:
        """Set weights for miners on the Commune network.
        
        Args:
            weights: Dictionary mapping miner UIDs to their weights
            
        Returns:
            bool: True if weights were successfully set
        """
        if not weights:
            logger.info("No weights to set")
            return False
            
        try:
            # Convert weights dict to format expected by Commune
            uids = list(int(uid) for uid in weights.keys())
            weight_values = list(float(weights[str(uid)]) for uid in uids)
            
            # Record history before setting weights
            self.add_weights_history(
                uids=[str(uid) for uid in uids],
                weights=weight_values,
                network="commune"
            )
            
            logger.info(f"Setting weights for {len(uids)} miners on Commune network")
            
            # Submit to Commune network
            self.c_client.vote(
                key=self.key,
                uids=uids,
                weights=weight_values,
                netuid=self.netuid,
            )
            
            logger.info(f"Successfully set weights for {len(uids)} miners on Commune network")
            return True
            
        except Exception as e:
            logger.error(f"Error setting weights on Commune network: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def track_miner_containers(self) -> None:
        """Track miner containers and update scores in miner_data.
        
        This method is specific to Commune validation and updates the miner_data
        dictionary with scores based on container tracking.
        """
        try:
            logger.info("Tracking miner containers")
            
            # 1. Get all miners
            miners = await self.get_miners()
            if not miners:
                logger.info("No miners found to track")
                return
                
            logger.debug(f"Found {len(miners)} miners")
            
            # 2. Verify miners
            verified_miners = await self.verify_miners(miners)
            if not verified_miners:
                logger.info("No verified miners")
                return
                
            logger.debug(f"Verified {len(verified_miners)} miners")
            
            # 3. For each verified miner, check containers and score
            for uid, miner in verified_miners.items():
                try:
                    # Get miner SSH info from address
                    # This would extract IP and credentials from the miner's address
                    # Placeholder implementation
                    ssh_info = self._extract_ssh_info(miner.get('address', ''))
                    if not ssh_info:
                        logger.debug(f"Could not extract SSH info for miner {uid}")
                        continue
                    
                    # Fetch compute specs via SSH
                    # This would connect to the miner via SSH and get hardware specs
                    # Placeholder implementation
                    compute_specs = self._fetch_compute_specs(ssh_info)
                    if not compute_specs:
                        logger.debug(f"Could not fetch compute specs for miner {uid}")
                        continue
                    
                    # Score based on compute specs
                    score = compute_resource_score(compute_specs)
                    
                    # Update miner_data with score
                    self.miner_data[uid] = score
                    logger.debug(f"Scored miner {uid}: {score}")
                    
                except Exception as e:
                    logger.warning(f"Error processing miner {uid}: {e}")
            
            logger.info(f"Updated scores for {len(self.miner_data)} miners")
            
        except Exception as e:
            logger.error(f"Error tracking miner containers: {e}")
            logger.error(traceback.format_exc())
    
    async def verify_miners(self, miners: Dict[str, Any]) -> Dict[str, Any]:
        """Verify miners before scoring them.
        
        Args:
            miners: Dictionary of miners to verify
            
        Returns:
            Dict: Dictionary of verified miners
        """
        verified = {}
        
        for uid, miner in miners.items():
            try:
                # Check last update time
                last_update = miner.get('last_update', 0)
                if self._is_recently_active(last_update):
                    verified[uid] = miner
                    
            except Exception as e:
                logger.warning(f"Error verifying miner {uid}: {e}")
        
        return verified
    
    def _is_recently_active(self, last_update: int) -> bool:
        """Check if a miner is recently active based on last update time.
        
        Args:
            last_update: Last update timestamp
            
        Returns:
            bool: True if miner is recently active
        """
        # Placeholder implementation
        # In a real implementation, you'd compare timestamps properly
        return last_update > 0
    
    def _extract_ssh_info(self, address: str) -> Optional[Dict[str, Any]]:
        """Extract SSH connection information from miner address.
        
        Args:
            address: Miner address string
            
        Returns:
            Dict: SSH connection info or None if extraction failed
        """
        # Placeholder implementation
        # In a real implementation, you'd parse the address to extract IP, port, etc.
        if not address:
            return None
            
        return {
            'host': '127.0.0.1',  # Placeholder
            'port': 22,
            'username': 'user',
            'password': None,
            'key_filename': None,
        }
    
    def _fetch_compute_specs(self, ssh_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fetch compute specifications from a miner via SSH.
        
        Args:
            ssh_info: SSH connection information
            
        Returns:
            Dict: Compute specifications or None if fetch failed
        """
        # Placeholder implementation
        # In a real implementation, you'd connect via SSH and run commands
        # to gather hardware information
        
        # Return example specs for testing
        return {
            'resource_type': 'CPU',
            'cpu_specs': {
                'total_cpus': 16,
                'threads_per_core': 2,
                'cpu_max_mhz': 3500,
            },
            'ram': '64GB',
            'storage': {
                'read_speed': '500MB/s',
            }
        } 