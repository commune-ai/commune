"""
Validator factory for the Polaris validator system.

This module provides a factory class to create the appropriate validator
instances based on the configured networks.
"""
import logging
from typing import Dict, Any, List, Optional

from validator.src.config import ValidatorConfig
from validator.src.validators.base_validator import BaseValidator
from validator.src.validators.bittensor_validator import BittensorValidator

# Import other validators as needed
# from validator.src.validators.commune_validator import CommuneValidator

logger = logging.getLogger(__name__)

class ValidatorFactory:
    """
    Factory class for creating validators.
    
    This class creates and manages validator instances for different networks
    based on the provided configuration.
    """
    
    @staticmethod
    def create_validator(network: str, config: ValidatorConfig) -> Optional[BaseValidator]:
        """
        Create a validator instance for the specified network.
        
        Args:
            network: Network name (e.g., 'bittensor', 'commune')
            config: Validator configuration
        
        Returns:
            Validator instance if supported, None otherwise
        """
        logger.info(f"Creating validator for {network} network")
        
        if network.lower() == 'bittensor':
            return BittensorValidator(config)
        
        # Add support for other networks as needed
        # elif network.lower() == 'commune':
        #     return CommuneValidator(config)
        
        else:
            logger.error(f"Unsupported network: {network}")
            return None
    
    @staticmethod
    def create_validators(config: ValidatorConfig) -> Dict[str, BaseValidator]:
        """
        Create validator instances for all configured networks.
        
        Args:
            config: Validator configuration with networks_to_validate list
        
        Returns:
            Dictionary mapping network names to validator instances
        """
        validators = {}
        
        for network in config.networks_to_validate:
            # Skip empty network names
            if not network.strip():
                continue
                
            validator = ValidatorFactory.create_validator(network, config)
            
            if validator:
                validators[network] = validator
            else:
                logger.warning(f"Failed to create validator for {network} network")
        
        logger.info(f"Created {len(validators)} validators: {', '.join(validators.keys())}")
        return validators 