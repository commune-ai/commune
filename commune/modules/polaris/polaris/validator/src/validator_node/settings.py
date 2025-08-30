from dataclasses import dataclass
from typing import Optional

@dataclass
class ValidatorNodeSettings:
    """Settings for the validator node."""
    
    # Bittensor network settings
    netuid: int = 33  # Default to Polaris subnet UID
    wallet_name: str = "default"
    hotkey: str = "default"
    network: str = "local"  # Options: local, test, finney
    
    # Commune network settings
    commune_netuid: int = 0
    commune_key: str = "default"
    
    # General settings
    max_weight: float = 1.0  # Maximum weight to assign to a miner
    
    # API settings
    api_url: str = "https://polaris-test-server.onrender.com/api/v1"
    
    # Firebase settings
    firebase_creds_path: Optional[str] = None 