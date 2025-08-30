from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Any
import requests
from bittensor import Keypair
from validator.src.validator_node.settings import ValidatorNodeSettings

logger = logging.getLogger(__name__)

class BaseValidator(ABC):
    """Base class for network-specific validators."""
    
    def __init__(self, key: Keypair, settings: ValidatorNodeSettings) -> None:
        """Initialize base validator with common settings."""
        self.key = key
        self.settings = settings
        self.miner_data = {}  # Store miner scores
        self.submission_history = []  # Track weight submissions
    
    @abstractmethod
    def get_miners(self) -> List[str]:
        """Fetch miners from the network. To be implemented by subclasses."""
        pass
    
    @abstractmethod
    def submit_weights(self, weights: Dict[str, float]) -> bool:
        """Submit weights to the network. To be implemented by subclasses."""
        pass
    
    def get_containers_for_miner(self, miner_uid: str) -> List[str]:
        """Fetch container IDs associated with a miner."""
        try:
            response = requests.get(f"https://polaris-test-server.onrender.com/api/v1/containers/miner/{miner_uid}")
            if response.status_code == 200:
                return response.json()
            logger.warning(f"No containers yet for {miner_uid}")
        except Exception as e:
            logger.error(f"Error fetching containers for miner {miner_uid}: {e}")
        return []
    
    def get_miner_list_with_resources(self, miner_network_map: Dict[str, str]) -> Dict:
        """Fetch miner resources from the API."""
        miner_resources = {}
        try:
            for miner_id, network_id in miner_network_map.items():
                response = requests.get(f"https://polaris-test-server.onrender.com/api/v1/miners/{miner_id}")
                if response.status_code == 200:
                    miner_data = response.json()
                    # Add network ID to the miner data
                    miner_data['network_id'] = network_id
                    miner_resources[miner_id] = miner_data
                else:
                    logger.warning(f"Failed to fetch resources for miner {miner_id}: {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching miner resources: {e}")
        return miner_resources
    
    def verify_miners(self, miner_ids: List[str]) -> Dict[str, bool]:
        """Verify miners through SSH connection."""
        verification_results = {}
        for miner_id in miner_ids:
            try:
                response = requests.patch(
                    f"https://polaris-test-server.onrender.com/api/v1/miners/{miner_id}/verify",
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
    
    def extract_ssh_and_password(self, miner_resources: Dict) -> Dict:
        """Extract SSH credentials for verification."""
        ssh_credentials = {}
        for miner_id, resources in miner_resources.items():
            if 'ssh_endpoint' in resources and 'ssh_password' in resources:
                ssh_credentials[miner_id] = {
                    'endpoint': resources['ssh_endpoint'],
                    'password': resources['ssh_password']
                }
        return ssh_credentials
    
    def normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize miner scores."""
        if not scores:
            return {}
        total = sum(scores.values())
        return {uid: score / total for uid, score in scores.items()} if total > 0 else scores
    
    def cut_to_max_allowed_weights(self, score_dict: Dict[str, float]) -> Dict[str, float]:
        """Apply maximum weight limits to scores."""
        return {uid: min(score, self.settings.max_weight) for uid, score in score_dict.items()}
    
    @abstractmethod
    def process_miners(self, miners: List[str], miner_resources: Dict) -> List[Dict]:
        """Process and score miners. To be implemented by subclasses."""
        pass
    
    @abstractmethod
    def track_miner_containers(self) -> None:
        """Main validation function to track containers and update scores."""
        pass 