"""
Main entry point for the Polaris validator system.

This module initializes the validator system based on configuration and
runs the validation process.
"""
import logging
import threading
import time
import sys
from typing import Dict, Any, List

from validator.src.config import get_config, ValidatorConfig
from validator.src.validators.validator_factory import ValidatorFactory
from validator.src.validators.base_validator import BaseValidator
from validator.src.utils.logging_utils import configure_logging

logger = logging.getLogger(__name__)

def run_validator(validator: BaseValidator):
    """
    Run a validator in its own thread.
    
    Args:
        validator: Validator instance to run
    """
    try:
        validator.run()
    except Exception as e:
        logger.error(f"Validator {validator.network_name} stopped due to error: {e}")

def main():
    """
    Main entry point for the validator system.
    
    This function:
    1. Loads the configuration
    2. Sets up logging
    3. Creates validator instances
    4. Runs validators in separate threads
    """
    # Load configuration
    config = get_config()
    
    # Configure logging
    configure_logging(
        level=getattr(logging, config.log_level.upper()),
        log_file=config.log_file
    )
    
    logger.info("Starting Polaris validator system")
    logger.info(f"Networks to validate: {', '.join(config.networks_to_validate)}")
    
    # Create validators
    validators = ValidatorFactory.create_validators(config)
    
    if not validators:
        logger.error("No validators created, exiting")
        return
    
    # Start validator threads
    threads = []
    
    for network, validator in validators.items():
        logger.info(f"Starting {network} validator in a separate thread")
        thread = threading.Thread(
            target=run_validator,
            args=(validator,),
            name=f"{network}_validator"
        )
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    # Wait for threads to complete
    try:
        while any(thread.is_alive() for thread in threads):
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down")
        sys.exit(0)
    
    logger.info("All validator threads have exited")

if __name__ == "__main__":
    main() 