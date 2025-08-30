import logging
from typing import Dict, List, Any
import time
import requests

from bittensor import Keypair
import commune as c

from validator.src.validator_node.base.validator_base import BaseValidator
from validator.src.validator_node.settings import ValidatorNodeSettings
from validator.src.validator_node.utils.firebase_client import FirebaseClient
from validator.src.validator_node.utils.resource_validation import (
    calculate_cpu_score, calculate_gpu_score, calculate_memory_score,
    calculate_storage_score, calculate_network_score, calculate_container_usage,
    validate_miner_resources
)

logger = logging.getLogger(__name__)

class CommuneValidator(BaseValidator):
    """Validator implementation for Commune network."""
    
    def __init__(self, key: Keypair, settings: ValidatorNodeSettings) -> None:
        """Initialize the Commune validator."""
        super().__init__(key, settings)
        # Initialize Commune client
        self.netuid = settings.commune_netuid
        self.key = key
        try:
            self.c_client = c.connect('http://commune.network')
            logger.info(f"Initialized Commune validator for netuid {self.netuid}")
        except Exception as e:
            logger.error(f"Failed to initialize Commune client: {e}")
            raise
        self.firebase_client = FirebaseClient.get_instance()
    
    def get_miners(self) -> List[str]:
        """Fetch miners from the Commune network."""
        try:
            # First get registered miners from Firebase
            registered_miners = self.firebase_client.get_network_miners('commune')
            
            # Then get the UIDs from the Commune network
            try:
                commune_miners = self.c_client.miners(netuid=self.netuid)
                commune_uids = [str(miner['uid']) for miner in commune_miners if 'uid' in miner]
            except Exception as e:
                logger.error(f"Error fetching miners from Commune network: {e}")
                commune_uids = []
            
            # Filter to only include miners that are registered on both Firebase and Commune
            valid_miners = []
            for miner_id, miner_data in registered_miners.items():
                commune_uid = miner_data.get('commune_uid')
                if commune_uid and str(commune_uid) in commune_uids:
                    valid_miners.append(miner_id)
                    logger.debug(f"Miner {miner_id} with Commune UID {commune_uid} is registered on Commune")
                else:
                    logger.warning(f"Miner {miner_id} with Commune UID {commune_uid} is not registered on Commune")
            
            logger.info(f"Found {len(valid_miners)} valid miners on Commune network")
            return valid_miners
        except Exception as e:
            logger.error(f"Error fetching Commune miners: {e}")
            return []
    
    def get_commune_uid(self, commune_key: str) -> int:
        """Get the UID of a key on the Commune network."""
        try:
            # This would be a call to the Commune API to get the UID
            # For now, we'll simply extract it from the miner resources
            response = requests.get(f"https://polaris-test-server.onrender.com/api/v1/miners/key/{commune_key}")
            if response.status_code == 200:
                data = response.json()
                return data.get('commune_uid', -1)
            return -1
        except Exception as e:
            logger.error(f"Error getting UID for Commune key {commune_key}: {e}")
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
                
                # Get the Commune UID for this miner
                commune_uid = resources.get('commune_uid')
                if not commune_uid:
                    logger.warning(f"No Commune UID found for miner {miner_id}, skipping")
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
                
                # Commune puts more emphasis on GPU resources, so we'll weight those higher
                hw_score = hw_score * 0.7 + gpu_score * 0.3  # Give extra weight to GPU
                
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
                # For Commune, we weight hardware at 70% and container usage at 30%
                final_score = (hw_score * 0.7) + (avg_container_score * 0.3)
                
                # Add to results
                results.append({
                    'miner_uid': miner_id,
                    'commune_uid': commune_uid,
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
                    'network': 'commune',
                    'commune_uid': commune_uid
                })
                
            except Exception as e:
                logger.error(f"Error processing miner {miner_id}: {e}")
        
        return results
    
    def track_miner_containers(self) -> None:
        """Fetch and update active containers for Commune miners."""
        miners = self.get_miners()
        if not miners:
            logger.info("No Commune miners to work on")
            return
        
        # Get miner details and verify them
        miner_resources = self.get_miner_list_with_resources({miner_id: miner_id for miner_id in miners})
        verified_miners = self.verify_miners(miners)
        
        # Filter to only include verified miners
        verified_miner_ids = [miner_id for miner_id, verified in verified_miners.items() if verified]
        if not verified_miner_ids:
            logger.info("No verified Commune miners to process")
            return
        
        # Process verified miners
        logger.info(f"Processing {len(verified_miner_ids)} verified Commune miners...")
        results = self.process_miners(verified_miner_ids, miner_resources)
        
        # Update miner scores
        for result in results:
            self.miner_data[result['miner_uid']] = result['final_score']
        
        # Log completion
        logger.info(f"Processed {len(results)} Commune miners successfully")
        logger.debug(f"Updated miner_data: {self.miner_data}")
    
    def submit_weights(self, weights: Dict[str, float]) -> bool:
        """Submit weights to the Commune network."""
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
                
                # Get Commune UID
                commune_uid = miner_resources[miner_id].get('commune_uid')
                if not commune_uid:
                    logger.warning(f"No Commune UID found for miner {miner_id}, skipping weight submission")
                    continue
                
                weights_by_uid[int(commune_uid)] = score
            
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
            result = self.c_client.vote(
                key=self.key,
                uids=uids,
                weights=weights_list,
                netuid=self.netuid,
            )
            
            # Record submission
            timestamp = time.time()
            self.submission_history.append({
                'timestamp': timestamp,
                'uids': uids,
                'weights': weights_list,
                'success': result
            })
            
            logger.info(f"Weight submission {'successful' if result else 'failed'}")
            return bool(result)
        
        except Exception as e:
            logger.error(f"Error submitting weights: {e}")
            return False 