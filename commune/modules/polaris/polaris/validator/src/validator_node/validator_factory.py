import logging
from typing import Optional

from bittensor import Keypair

from validator.src.validator_node.base.validator_base import BaseValidator
from validator.src.validator_node.validators.bittensor_validator import BittensorValidator
from validator.src.validator_node.validators.commune_validator import CommuneValidator
from validator.src.validator_node.settings import ValidatorNodeSettings

logger = logging.getLogger(__name__)

class ValidatorFactory:
    """Factory for creating network-specific validators."""
    
    @staticmethod
    def create_validator(network: str, key: Keypair, settings: ValidatorNodeSettings) -> Optional[BaseValidator]:
        """
        Create and return a validator for the specified network.
        
        Args:
            network: The network to validate on ('bittensor' or 'commune')
            key: The keypair to use for validation
            settings: The validator settings
            
        Returns:
            A validator for the specified network, or None if the network is invalid
        """
        if network.lower() == 'bittensor':
            logger.info("Creating Bittensor validator")
            return BittensorValidator(key, settings)
        elif network.lower() == 'commune':
            logger.info("Creating Commune validator")
            return CommuneValidator(key, settings)
        else:
            logger.error(f"Unknown network: {network}")
            return None
    
    @staticmethod
    def create_validators_for_all_networks(key: Keypair, settings: ValidatorNodeSettings) -> dict[str, BaseValidator]:
        """
        Create and return validators for all supported networks.
        
        Args:
            key: The keypair to use for validation
            settings: The validator settings
            
        Returns:
            A dictionary mapping network names to validators
        """
        validators = {}
        
        try:
            bittensor_validator = BittensorValidator(key, settings)
            validators['bittensor'] = bittensor_validator
            logger.info("Created Bittensor validator")
        except Exception as e:
            logger.error(f"Failed to create Bittensor validator: {e}")
        
        try:
            commune_validator = CommuneValidator(key, settings)
            validators['commune'] = commune_validator
            logger.info("Created Commune validator")
        except Exception as e:
            logger.error(f"Failed to create Commune validator: {e}")
        
        return validators 