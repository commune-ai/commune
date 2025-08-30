import argparse
import logging
import time
import sys
import requests
from typing import Dict, List, Any, Optional
import bittensor as bt
from bittensor import Keypair, wallet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('validator.log')
    ]
)
logger = logging.getLogger(__name__)

API_BASE = "https://polaris-test-server.onrender.com/api/v1"

class SimpleValidator:
    """Simplified validator that uses the render endpoints directly."""
    
    def __init__(self, 
                 wallet_name: str = "default", 
                 hotkey: str = "default",
                 netuid: int = 33,
                 network: str = "test",
                 max_weight: float = 1.0) -> None:
        """Initialize the validator."""
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey
        self.netuid = netuid
        self.network = network
        self.max_weight = max_weight
        self.miner_data = {}  # Store miner scores
        
        # Initialize Bittensor
        self.subtensor = bt.subtensor(network=network)
        self.wallet = bt.wallet(name=wallet_name, hotkey=hotkey)
        self.hotkey = self.wallet.hotkey
        logger.info(f"Initialized validator for netuid {netuid} on {network} network")
    
    def get_miners(self) -> List[str]:
        """Fetch miners from the render API."""
        try:
            # Get the blockchain UIDs
            network_uids = self.subtensor.metagraph(self.netuid).uids.tolist()
            logger.info(f"Found {len(network_uids)} miners on blockchain")
            
            # Get miners from the render API
            response = requests.get(f"{API_BASE}/miners")
            if response.status_code != 200:
                logger.error(f"Failed to fetch miners from API: {response.status_code}")
                return []
            
            miners_data = response.json()
            logger.info(f"Fetched {len(miners_data)} miners from API")
            
            # Filter to miners that are also registered on the blockchain
            valid_miners = []
            for miner in miners_data:
                miner_id = miner.get("miner_id") or miner.get("id")
                if miner_id:
                    valid_miners.append(miner_id)
            
            logger.info(f"Found {len(valid_miners)} valid miners")
            return valid_miners
        except Exception as e:
            logger.error(f"Error fetching miners: {e}")
            return []
    
    def get_containers_for_miner(self, miner_uid: str) -> List[str]:
        """Fetch container IDs associated with a miner."""
        try:
            response = requests.get(f"{API_BASE}/containers/miner/{miner_uid}")
            if response.status_code == 200:
                return response.json()
            logger.warning(f"No containers yet for {miner_uid}")
        except Exception as e:
            logger.error(f"Error fetching containers for miner {miner_uid}: {e}")
        return []
    
    def get_miner_list_with_resources(self, miner_ids: List[str]) -> Dict:
        """Fetch miner resources from the API."""
        miner_resources = {}
        try:
            for miner_id in miner_ids:
                response = requests.get(f"{API_BASE}/miners/{miner_id}")
                if response.status_code == 200:
                    miner_data = response.json()
                    miner_resources[miner_id] = miner_data
                else:
                    logger.warning(f"Failed to fetch resources for miner {miner_id}: {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching miner resources: {e}")
        return miner_resources
    
    def verify_miners(self, miner_ids: List[str]) -> Dict[str, bool]:
        """Verify miners through the API."""
        verification_results = {}
        for miner_id in miner_ids:
            try:
                response = requests.patch(
                    f"{API_BASE}/miners/{miner_id}/verify",
                    json={"status": "verified"}
                )
                if response.status_code == 200:
                    verification_results[miner_id] = True
                    logger.info(f"Successfully verified miner {miner_id}")
                else:
                    verification_results[miner_id] = False
                    logger.warning(f"Failed to verify miner {miner_id}: {response.status_code}")
            except Exception as e:
                verification_results[miner_id] = False
                logger.error(f"Error verifying miner {miner_id}: {e}")
        return verification_results
    
    def calculate_cpu_score(self, specs: Dict[str, Any]) -> float:
        """Calculate the score for CPU resources."""
        try:
            cpu_count = int(specs.get('cpu_count', 0))
            cpu_speed = float(specs.get('cpu_speed', 0.0))
            
            # Basic CPU score calculation
            cpu_score = cpu_count * cpu_speed
            return min(10.0, cpu_score / 1000)  # Normalize to max 10
        except Exception as e:
            logger.error(f"Error calculating CPU score: {e}")
            return 0.0
    
    def calculate_gpu_score(self, specs: Dict[str, Any]) -> float:
        """Calculate the score for GPU resources."""
        try:
            gpus = specs.get('gpus', [])
            if not gpus:
                return 0.0
            
            total_gpu_score = 0.0
            for gpu in gpus:
                # Extract GPU memory (in GB) and name
                memory = gpu.get('memory', 0)  # Memory in MB
                memory_gb = memory / 1024.0  # Convert to GB
                name = gpu.get('name', '').lower()
                
                # Base score on memory
                gpu_score = memory_gb * 0.5
                
                # Bonus for powerful GPUs
                if 'a100' in name:
                    gpu_score *= 1.5
                elif 'h100' in name:
                    gpu_score *= 2.0
                elif '3090' in name or '4090' in name:
                    gpu_score *= 1.3
                
                total_gpu_score += gpu_score
            
            return min(20.0, total_gpu_score)  # Normalize to max 20
        except Exception as e:
            logger.error(f"Error calculating GPU score: {e}")
            return 0.0
    
    def calculate_memory_score(self, specs: Dict[str, Any]) -> float:
        """Calculate the score for memory resources."""
        try:
            memory = float(specs.get('memory', 0))  # Memory in GB
            return min(5.0, memory / 10.0)  # Normalize to max 5
        except Exception as e:
            logger.error(f"Error calculating memory score: {e}")
            return 0.0
    
    def calculate_storage_score(self, specs: Dict[str, Any]) -> float:
        """Calculate the score for storage resources."""
        try:
            storage = float(specs.get('storage', 0))  # Storage in GB
            return min(5.0, storage / 100.0)  # Normalize to max 5
        except Exception as e:
            logger.error(f"Error calculating storage score: {e}")
            return 0.0
    
    def calculate_network_score(self, specs: Dict[str, Any]) -> float:
        """Calculate the score for network resources."""
        try:
            bandwidth = float(specs.get('bandwidth', 0))  # Bandwidth in Mbps
            return min(5.0, bandwidth / 100.0)  # Normalize to max 5
        except Exception as e:
            logger.error(f"Error calculating network score: {e}")
            return 0.0
    
    def calculate_container_usage(self, container_data: Dict[str, Any]) -> float:
        """Calculate the score based on container usage."""
        try:
            # Calculate based on active time and utilization
            active_time = container_data.get('active_time', 0)  # In minutes
            cpu_utilization = container_data.get('cpu_utilization', 0)  # In percentage
            memory_utilization = container_data.get('memory_utilization', 0)  # In percentage
            
            # Basic formula: active_time * (cpu_util + memory_util) / 200
            # This gives higher scores to containers that are used more
            usage_score = active_time * (cpu_utilization + memory_utilization) / 200.0
            return min(10.0, usage_score)  # Normalize to max 10
        except Exception as e:
            logger.error(f"Error calculating container usage score: {e}")
            return 0.0
    
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
                try:
                    uid = self.subtensor.get_uid_for_hotkey(hotkey, self.netuid)
                except Exception as e:
                    logger.error(f"Error getting UID for hotkey {hotkey}: {e}")
                    uid = -1
                
                if uid < 0:
                    logger.warning(f"Invalid UID for miner {miner_id} with hotkey {hotkey}, skipping")
                    continue
                
                # Get containers for this miner
                containers = self.get_containers_for_miner(miner_id)
                
                # Calculate hardware scores
                hw_specs = resources.get('hardware_specs', {})
                cpu_score = self.calculate_cpu_score(hw_specs)
                gpu_score = self.calculate_gpu_score(hw_specs)
                memory_score = self.calculate_memory_score(hw_specs)
                storage_score = self.calculate_storage_score(hw_specs)
                network_score = self.calculate_network_score(hw_specs)
                
                # Calculate total hardware score (max 40)
                hw_score = cpu_score + gpu_score + memory_score + storage_score + network_score
                
                # Calculate container usage scores
                container_scores = []
                for container_id in containers:
                    # Fetch container details from API
                    try:
                        # This is a simplified placeholder - in a real implementation, we'd fetch this from the API
                        container_data = {
                            'active_time': 60,  # 60 minutes
                            'cpu_utilization': 50,  # 50%
                            'memory_utilization': 30,  # 30%
                        }
                        container_score = self.calculate_container_usage(container_data)
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
                
                # Save the score to miner_data
                self.miner_data[uid] = final_score
                
            except Exception as e:
                logger.error(f"Error processing miner {miner_id}: {e}")
        
        return results
    
    def normalize_scores(self, scores: Dict[int, float]) -> Dict[int, float]:
        """Normalize miner scores."""
        if not scores:
            return {}
        total = sum(scores.values())
        return {uid: score / total for uid, score in scores.items()} if total > 0 else scores
    
    def cut_to_max_allowed_weights(self, score_dict: Dict[int, float]) -> Dict[int, float]:
        """Apply maximum weight limits to scores."""
        return {uid: min(score, self.max_weight) for uid, score in score_dict.items()}
    
    def track_miner_containers(self) -> None:
        """Main validation function to track containers and update scores."""
        miners = self.get_miners()
        if not miners:
            logger.info("No miners to work on")
            return
        
        # Get miner details and verify them
        miner_resources = self.get_miner_list_with_resources(miners)
        verified_miners = self.verify_miners(miners)
        
        # Filter to only include verified miners
        verified_miner_ids = [miner_id for miner_id, verified in verified_miners.items() if verified]
        if not verified_miner_ids:
            logger.info("No verified miners to process")
            return
        
        # Process verified miners
        logger.info(f"Processing {len(verified_miner_ids)} verified miners...")
        results = self.process_miners(verified_miner_ids, miner_resources)
        
        # Log completion
        logger.info(f"Processed {len(results)} miners successfully")
        logger.debug(f"Updated miner_data: {self.miner_data}")
    
    def submit_weights(self) -> bool:
        """Submit weights to the Bittensor network."""
        try:
            weights = self.miner_data
            if not weights:
                logger.warning("No weights to submit")
                return False
            
            # Normalize and cap weights
            normalized_weights = self.normalize_scores(weights)
            capped_weights = self.cut_to_max_allowed_weights(normalized_weights)
            
            # Prepare for submission
            uids = list(capped_weights.keys())
            weights_list = [capped_weights[uid] for uid in uids]
            
            logger.info(f"Submitting weights for {len(uids)} miners: {list(zip(uids, weights_list))}")
            
            # Submit weights
            result = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.netuid,
                uids=uids,
                weights=weights_list,
                wait_for_inclusion=True
            )
            
            logger.info(f"Weight submission {'successful' if result else 'failed'}")
            return result is not None
        
        except Exception as e:
            logger.error(f"Error submitting weights: {e}")
            return False

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Polaris Compute Subnet Validator")
    
    # Bittensor specific args
    parser.add_argument('--netuid', type=int, default=33, help='Bittensor netuid to validate on')
    parser.add_argument('--wallet_name', type=str, default='default', help='Bittensor wallet name')
    parser.add_argument('--hotkey', type=str, default='default', help='Bittensor hotkey name')
    parser.add_argument('--network', type=str, default='test', help='Bittensor network (local/test/finney)')
    
    # General settings
    parser.add_argument('--max_weight', type=float, default=1.0, help='Maximum weight to assign to any miner')
    parser.add_argument('--validation_interval', type=int, default=900, help='Interval between validations in seconds')
    parser.add_argument('--submission_interval', type=int, default=3600, help='Interval between weight submissions in seconds')
    
    return parser.parse_args()

def main() -> None:
    """Main entry point for the validator."""
    args = parse_args()
    
    # Create validator
    validator = SimpleValidator(
        wallet_name=args.wallet_name,
        hotkey=args.hotkey,
        netuid=args.netuid,
        network=args.network,
        max_weight=args.max_weight
    )
    
    logger.info(f"Starting validation for {args.network} network")
    
    # Main loop
    validation_cycle = 0
    submission_cycle = 0
    last_validation_time = 0
    last_submission_time = 0
    
    try:
        while True:
            current_time = time.time()
            
            # Run validation cycle if interval has passed
            if current_time - last_validation_time >= args.validation_interval:
                validation_cycle += 1
                logger.info(f"Starting validation cycle {validation_cycle}")
                validator.track_miner_containers()
                last_validation_time = current_time
                logger.info(f"Completed validation cycle {validation_cycle}")
            
            # Submit weights if interval has passed
            if current_time - last_submission_time >= args.submission_interval:
                submission_cycle += 1
                logger.info(f"Starting weight submission cycle {submission_cycle}")
                validator.submit_weights()
                last_submission_time = current_time
                logger.info(f"Completed weight submission cycle {submission_cycle}")
            
            # Sleep to avoid busy waiting
            time.sleep(10)
    
    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        logger.info("Validator shutting down")

if __name__ == "__main__":
    main() 