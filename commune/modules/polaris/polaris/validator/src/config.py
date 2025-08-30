"""
Configuration module for the Polaris validator system.

This module centralizes all configuration options, including API endpoints,
scoring parameters, and validation settings.
"""
import os
import logging
from typing import Dict, Any, List, Optional
import argparse

logger = logging.getLogger(__name__)

class ApiConfig:
    """API endpoint configuration."""
    
    def __init__(self):
        """Initialize API configuration with default values."""
        # Base API URL, can be overridden via environment variable
        self.base_url = os.environ.get(
            'POLARIS_API_BASE_URL', 
            'https://api.polaris.network/v1'
        )
        
        # API endpoints
        self.miners_endpoint = '/miners'
        self.containers_endpoint = '/containers'
        self.verification_endpoint = '/verification'
        
        # API timeout in seconds
        self.timeout = int(os.environ.get('API_TIMEOUT', '30'))
    
    def get_miners_url(self) -> str:
        """Get the full URL for the miners endpoint."""
        return f"{self.base_url}{self.miners_endpoint}"
    
    def get_containers_url(self) -> str:
        """Get the full URL for the containers endpoint."""
        return f"{self.base_url}{self.containers_endpoint}"
    
    def get_verification_url(self) -> str:
        """Get the full URL for the verification endpoint."""
        return f"{self.base_url}{self.verification_endpoint}"


class ScoringConfig:
    """Scoring algorithm configuration."""
    
    def __init__(self):
        """Initialize scoring configuration with default values."""
        # CPU scoring parameters
        self.cpu_max_score = 40.0
        self.cpu_normalization_factor = 100.0  # CPU cores * GHz
        
        # GPU scoring parameters
        self.gpu_max_score = 40.0
        self.gpu_base_factor = 1.0  # Base score per GB of GPU memory
        self.gpu_bonus_factors = {
            'rtx': 1.2,
            'tesla': 1.5,
            'a100': 2.0,
            'h100': 2.5,
            'v100': 1.8
        }
        
        # Memory scoring parameters
        self.memory_max_score = 10.0
        self.memory_normalization_factor = 100.0  # GB
        
        # Storage scoring parameters
        self.storage_max_score = 5.0
        self.storage_normalization_factor = 1000.0  # GB
        
        # Network scoring parameters
        self.network_max_score = 5.0
        self.network_normalization_factor = 1000.0  # Mbps
        
        # Container usage scoring parameters
        self.container_max_score = 100.0
        
        # Overall weighting factors
        self.hardware_weight = 0.7  # 70% weight for hardware
        self.container_weight = 0.3  # 30% weight for container usage


class NetworkConfig:
    """Network-specific configuration."""
    
    def __init__(self):
        """Initialize network configuration."""
        # Bittensor network settings
        self.bittensor = {
            'netuid': int(os.environ.get('BT_NETUID', '33')),
            'network': os.environ.get('BT_NETWORK', 'test'),
            'subtensor_endpoint': os.environ.get('BT_SUBTENSOR_ENDPOINT', ''),
            'wallet_name': os.environ.get('BT_WALLET_NAME', 'poli_validator'),
            'hotkey': os.environ.get('BT_HOTKEY', 'poli_validator_hot')
        }
        
        # Commune network settings
        self.commune = {
            'network': os.environ.get('COMMUNE_NETWORK', 'mainnet'),
            'wallet_path': os.environ.get('COMMUNE_WALLET_PATH', ''),
            'module_name': os.environ.get('COMMUNE_MODULE_NAME', 'polaris')
        }


class ValidatorConfig:
    """Main configuration for the validator."""
    
    def __init__(self):
        """Initialize validator configuration with default values."""
        # Set up nested configurations
        self.api = ApiConfig()
        self.scoring = ScoringConfig()
        self.network = NetworkConfig()
        
        # Validation settings
        self.validation_interval = int(os.environ.get('VALIDATION_INTERVAL', '3600'))  # In seconds
        self.submission_interval = int(os.environ.get('SUBMISSION_INTERVAL', '3600'))  # In seconds
        self.max_miners_per_validation = int(os.environ.get('MAX_MINERS_PER_VALIDATION', '100'))
        self.ssh_connection_timeout = int(os.environ.get('SSH_CONNECTION_TIMEOUT', '30'))  # In seconds
        self.ssh_command_timeout = int(os.environ.get('SSH_COMMAND_TIMEOUT', '60'))  # In seconds
        
        # Firebase settings
        self.firebase_credentials_path = os.environ.get(
            'FIREBASE_CREDENTIALS_PATH', 
            'firebase_credentials.json'
        )
        
        # Weights settings
        self.max_weight_value = float(os.environ.get('MAX_WEIGHT_VALUE', '1.0'))
        self.min_score_for_weight = float(os.environ.get('MIN_SCORE_FOR_WEIGHT', '5.0'))
        
        # Log settings
        self.log_level = os.environ.get('LOG_LEVEL', 'INFO')
        self.log_file = os.environ.get('LOG_FILE', 'validator.log')
        
        # Networks to validate
        self.networks_to_validate = os.environ.get('NETWORKS_TO_VALIDATE', 'bittensor').split(',')
    
    def load_from_env(self):
        """
        Load configuration from environment variables.
        This allows for dynamic configuration without code changes.
        """
        # This is a placeholder for now
        # The init method already loads from environment variables
        pass
    
    def load_from_args(self, args: argparse.Namespace):
        """
        Load configuration from command line arguments.
        
        Args:
            args: Command line arguments parsed by argparse
        """
        # Map argparse arguments to configuration attributes if they exist
        for arg_name, arg_value in vars(args).items():
            if arg_value is not None:
                # Handle nested configs (api.*, scoring.*, network.*)
                if '.' in arg_name:
                    config_section, config_key = arg_name.split('.', 1)
                    if hasattr(self, config_section) and hasattr(getattr(self, config_section), config_key):
                        setattr(getattr(self, config_section), config_key, arg_value)
                # Handle top-level configs
                elif hasattr(self, arg_name):
                    setattr(self, arg_name, arg_value)
        
        # Handle special cases for Bittensor and Commune network settings
        if hasattr(args, 'bittensor_netuid') and args.bittensor_netuid is not None:
            self.network.bittensor['netuid'] = args.bittensor_netuid
        
        if hasattr(args, 'bittensor_network') and args.bittensor_network is not None:
            self.network.bittensor['network'] = args.bittensor_network
        
        if hasattr(args, 'wallet_name') and args.wallet_name is not None:
            self.network.bittensor['wallet_name'] = args.wallet_name
        
        if hasattr(args, 'hotkey') and args.hotkey is not None:
            self.network.bittensor['hotkey'] = args.hotkey


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Polaris Validator")
    
    # General settings
    parser.add_argument('--validation_interval', type=int, help='Interval between validations in seconds')
    parser.add_argument('--submission_interval', type=int, help='Interval between weight submissions in seconds')
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')
    parser.add_argument('--log_file', type=str, help='Log file path')
    parser.add_argument('--networks_to_validate', type=str, help='Comma-separated list of networks to validate')
    
    # Bittensor settings
    parser.add_argument('--netuid', type=int, dest='bittensor_netuid', help='Bittensor network UID')
    parser.add_argument('--network', type=str, dest='bittensor_network', help='Bittensor network (e.g., mainnet, testnet)')
    parser.add_argument('--wallet_name', type=str, help='Bittensor wallet name')
    parser.add_argument('--hotkey', type=str, help='Bittensor hotkey name')
    parser.add_argument('--subtensor_endpoint', type=str, help='Custom subtensor endpoint')
    
    # Commune settings
    parser.add_argument('--commune_network', type=str, help='Commune network (e.g., mainnet)')
    parser.add_argument('--commune_wallet_path', type=str, help='Path to Commune wallet')
    parser.add_argument('--commune_module_name', type=str, help='Commune module name')
    
    # Firebase settings
    parser.add_argument('--firebase_credentials_path', type=str, help='Path to Firebase credentials JSON file')
    
    # API settings
    parser.add_argument('--api_base_url', type=str, help='Base URL for Polaris API')
    parser.add_argument('--api_timeout', type=int, help='API request timeout in seconds')
    
    # Scoring settings
    parser.add_argument('--max_weight_value', type=float, help='Maximum weight value to assign')
    parser.add_argument('--min_score_for_weight', type=float, help='Minimum score required to receive a weight')
    
    return parser.parse_args()


def get_config() -> ValidatorConfig:
    """
    Get the validator configuration, initialized from environment variables and command line arguments.
    
    Returns:
        Validator configuration
    """
    # Create default configuration
    config = ValidatorConfig()
    
    # Load from environment variables
    config.load_from_env()
    
    # Parse command line arguments
    args = parse_args()
    
    # Update configuration with command line arguments
    config.load_from_args(args)
    
    return config 