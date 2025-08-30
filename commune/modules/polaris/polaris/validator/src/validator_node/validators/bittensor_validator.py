import logging
from typing import Dict, List, Any
import time

import bittensor as bt
from bittensor import Keypair

from validator.src.validator_node.base.validator_base import BaseValidator
from validator.src.validator_node.settings import ValidatorNodeSettings
from validator.src.validator_node.utils.firebase_client import FirebaseClient
from validator.src.validator_node.utils.resource_validation import (
    calculate_cpu_score, calculate_gpu_score, calculate_memory_score,
    calculate_storage_score, calculate_network_score, calculate_container_usage,
    validate_miner_resources
)

logger = logging.getLogger(__name__)

class BittensorValidator(BaseValidator):
    """Validator implementation for Bittensor network."""
    
    def __init__(self, key: Keypair, settings: ValidatorNodeSettings) -> None:
        """Initialize the Bittensor validator."""
        super().__init__(key, settings)
        # Initialize Bittensor subtensor client
        self.subtensor = bt.subtensor(network=settings.network)
        self.netuid = settings.netuid
        self.wallet = bt.wallet(name=settings.wallet_name, hotkey=settings.hotkey)
        self.client = bt.subtensor(network=settings.network)
        self.firebase_client = FirebaseClient.get_instance()
        logger.info(f"Initialized Bittensor validator for netuid {self.netuid}")
    
    def get_miners(self) -> List[str]:
        """Fetch miners from the Bittensor network."""
        try:
            # First get registered miners from Firebase
            registered_miners = self.firebase_client.get_network_miners('bittensor')
            
            # Then get the hotkeys from the blockchain
            network_uids = self.client.query_map_key(self.netuid)
            miner_hotkeys = list(network_uids.keys())
            
            # Filter to only include miners that are registered on both Firebase and Bittensor
            valid_miners = []
            for miner_id, miner_data in registered_miners.items():
                hotkey = miner_data.get('hotkey')
                if hotkey in miner_hotkeys:
                    valid_miners.append(miner_id)
                    logger.debug(f"Miner {miner_id} with hotkey {hotkey} is registered on Bittensor")
                else:
                    logger.warning(f"Miner {miner_id} with hotkey {hotkey} is not registered on Bittensor")
            
            logger.info(f"Found {len(valid_miners)} valid miners on Bittensor network")
            return valid_miners
        except Exception as e:
            logger.error(f"Error fetching Bittensor miners: {e}")
            return []
    
    def get_blockchain_uid(self, hotkey: str) -> int:
        """Get the UID of a hotkey on the blockchain."""
        try:
            return self.subtensor.get_uid_for_hotkey(hotkey, self.netuid)
        except Exception as e:
            logger.error(f"Error getting UID for hotkey {hotkey}: {e}")
            return -1
    
    def process_miners(self, miners: List[str], miner_resources: Dict) -> List[Dict]:
        """Process and score miners based on their resources and container usage."""
        results = []
        
        for miner_id in miners:
            try:
                if miner_id not in miner_resources:
                    logger.warning(f"No resources found for miner {miner_id}, skipping")
                    continue
                
                resources = miner_resources[miner_id]
                
                # Get the hotkey for this miner
                hotkey = resources.get('hotkey')
                if not hotkey:
                    logger.warning(f"No hotkey found for miner {miner_id}, skipping")
                    continue
                
                # Get the UID on the blockchain
                uid = self.get_blockchain_uid(hotkey)
                if uid < 0:
                    logger.warning(f"Invalid UID for miner {miner_id} with hotkey {hotkey}, skipping")
                    continue
                
                # Get containers for this miner
                containers = self.get_containers_for_miner(miner_id)
                
                # Calculate hardware scores
                hw_specs = resources.get('hardware_specs', {})
                cpu_score = calculate_cpu_score(hw_specs)
                gpu_score = calculate_gpu_score(hw_specs)
                memory_score = calculate_memory_score(hw_specs)
                storage_score = calculate_storage_score(hw_specs)
                network_score = calculate_network_score(hw_specs)
                
                # Calculate total hardware score (max 40)
                hw_score = cpu_score + gpu_score + memory_score + storage_score + network_score
                
                # Calculate container usage scores
                container_scores = []
                for container_id in containers:
                    # Fetch container details from API
                    try:
                        # This would be an API call to get container details
                        # For now, we'll mock it with some sample data
                        container_data = {
                            'active_time': 60,  # 60 minutes
                            'cpu_utilization': 50,  # 50%
                            'memory_utilization': 30,  # 30%
                        }
                        container_score = calculate_container_usage(container_data)
                        container_scores.append(container_score)
                    except Exception as e:
                        logger.error(f"Error processing container {container_id}: {e}")
                
                # Calculate average container score
                avg_container_score = sum(container_scores) / max(1, len(container_scores)) if container_scores else 0
                
                # Calculate final score (hardware + container usage)
                # Weight hardware at 60% and container usage at 40%
                final_score = (hw_score * 0.6) + (avg_container_score * 0.4)
                
                # Add to results
                results.append({
                    'miner_uid': miner_id,
                    'blockchain_uid': uid,
                    'hotkey': hotkey,
                    'hardware_score': hw_score,
                    'container_score': avg_container_score,
                    'final_score': final_score,
                    'container_count': len(containers)
                })
                
                # Save validation results to Firebase
                self.firebase_client.save_validation_results(miner_id, {
                    'hardware_score': hw_score,
                    'container_score': avg_container_score,
                    'score': final_score,
                    'network': 'bittensor',
                    'blockchain_uid': uid
                })
                
            except Exception as e:
                logger.error(f"Error processing miner {miner_id}: {e}")
        
        return results
    
    def track_miner_containers(self) -> None:
        """Fetch and update active containers for Bittensor miners."""
        miners = self.get_miners()
        if not miners:
            logger.info("No Bittensor miners to work on")
            return
        
        # Get miner details and verify them
        miner_resources = self.get_miner_list_with_resources({miner_id: miner_id for miner_id in miners})
        verified_miners = self.verify_miners(miners)
        
        # Filter to only include verified miners
        verified_miner_ids = [miner_id for miner_id, verified in verified_miners.items() if verified]
        if not verified_miner_ids:
            logger.info("No verified Bittensor miners to process")
            return
        
        # Process verified miners
        logger.info(f"Processing {len(verified_miner_ids)} verified Bittensor miners...")
        results = self.process_miners(verified_miner_ids, miner_resources)
        
        # Update miner scores
        for result in results:
            self.miner_data[result['miner_uid']] = result['final_score']
        
        # Log completion
        logger.info(f"Processed {len(results)} Bittensor miners successfully")
        logger.debug(f"Updated miner_data: {self.miner_data}")
    
    def submit_weights(self, weights: Dict[str, float]) -> bool:
        """Submit weights to the Bittensor network."""
        try:
            if not weights:
                logger.warning("No weights to submit")
                return False
            
            # Convert miner IDs to UIDs and normalize
            weights_by_uid = {}
            for miner_id, score in weights.items():
                # Get resources for this miner
                miner_resources = self.get_miner_list_with_resources({miner_id: miner_id})
                if miner_id not in miner_resources:
                    logger.warning(f"No resources found for miner {miner_id}, skipping weight submission")
                    continue
                
                # Get hotkey and UID
                hotkey = miner_resources[miner_id].get('hotkey')
                if not hotkey:
                    logger.warning(f"No hotkey found for miner {miner_id}, skipping weight submission")
                    continue
                
                uid = self.get_blockchain_uid(hotkey)
                if uid < 0:
                    logger.warning(f"Invalid UID for miner {miner_id}, skipping weight submission")
                    continue
                
                weights_by_uid[uid] = score
            
            if not weights_by_uid:
                logger.warning("No valid UIDs found for weight submission")
                return False
            
            # Normalize and cap weights
            normalized_weights = self.normalize_scores(weights_by_uid)
            capped_weights = self.cut_to_max_allowed_weights(normalized_weights)
            
            # Prepare for submission
            uids = list(capped_weights.keys())
            weights_list = [capped_weights[uid] for uid in uids]
            
            # Submit weights
            result = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.netuid,
                uids=uids,
                weights=weights_list,
                wait_for_inclusion=True
            )
            
            # Record submission
            timestamp = time.time()
            self.submission_history.append({
                'timestamp': timestamp,
                'uids': uids,
                'weights': weights_list,
                'success': result is not None
            })
            
            logger.info(f"Weight submission {'successful' if result else 'failed'}")
            return result is not None
        
        except Exception as e:
            logger.error(f"Error submitting weights: {e}")
            return False 