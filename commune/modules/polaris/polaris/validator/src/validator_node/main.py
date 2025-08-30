import argparse
import logging
import time
import sys
from typing import List, Optional

from bittensor import Keypair

from validator.src.validator_node.settings import ValidatorNodeSettings
from validator.src.validator_node.validator_factory import ValidatorFactory
from validator.src.validator_node.base.validator_base import BaseValidator

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

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Polaris Compute Subnet Validator")
    
    # Network selection
    parser.add_argument(
        '--network', 
        type=str, 
        default='both',
        choices=['bittensor', 'commune', 'both'], 
        help='Network to validate on (bittensor, commune, or both)'
    )
    
    # Bittensor specific args
    parser.add_argument('--netuid', type=int, default=33, help='Bittensor netuid to validate on')
    parser.add_argument('--wallet_name', type=str, default='default', help='Bittensor wallet name')
    parser.add_argument('--hotkey', type=str, default='default', help='Bittensor hotkey name')
    parser.add_argument('--bittensor_network', type=str, default='local', help='Bittensor network (local/test/finney)')
    
    # Commune specific args
    parser.add_argument('--commune_netuid', type=int, default=0, help='Commune netuid to validate on')
    parser.add_argument('--commune_key', type=str, default='default', help='Commune key name')
    
    # General settings
    parser.add_argument('--max_weight', type=float, default=1.0, help='Maximum weight to assign to any miner')
    parser.add_argument('--validation_interval', type=int, default=900, help='Interval between validations in seconds')
    parser.add_argument('--submission_interval', type=int, default=3600, help='Interval between weight submissions in seconds')
    
    return parser.parse_args()

def run_validation_cycle(validators: List[BaseValidator], cycle_count: int) -> None:
    """Run a single validation cycle for all validators."""
    logger.info(f"Starting validation cycle {cycle_count}")
    
    for validator in validators:
        try:
            validator.track_miner_containers()
        except Exception as e:
            logger.error(f"Error during validation cycle for validator {type(validator).__name__}: {e}")
    
    logger.info(f"Completed validation cycle {cycle_count}")

def submit_weights(validators: List[BaseValidator], cycle_count: int) -> None:
    """Submit weights for all validators."""
    logger.info(f"Starting weight submission cycle {cycle_count}")
    
    for validator in validators:
        try:
            # Get the latest weights from the validator
            weights = validator.miner_data
            if weights:
                logger.info(f"Submitting weights for {len(weights)} miners with {type(validator).__name__}")
                success = validator.submit_weights(weights)
                logger.info(f"Weight submission {'successful' if success else 'failed'}")
            else:
                logger.info(f"No weights to submit for {type(validator).__name__}")
        except Exception as e:
            logger.error(f"Error during weight submission for validator {type(validator).__name__}: {e}")
    
    logger.info(f"Completed weight submission cycle {cycle_count}")

def main() -> None:
    """Main entry point for the validator."""
    args = parse_args()
    
    # Create settings
    settings = ValidatorNodeSettings(
        netuid=args.netuid,
        commune_netuid=args.commune_netuid,
        wallet_name=args.wallet_name,
        hotkey=args.hotkey,
        network=args.bittensor_network,
        max_weight=args.max_weight,
    )
    
    # Load keypair
    try:
        # For simplicity, we'll use the same keypair for both networks
        # In a production environment, you might want separate keys
        import bittensor as bt
        wallet = bt.wallet(name=args.wallet_name, hotkey=args.hotkey)
        key = wallet.hotkey
        logger.info(f"Loaded keypair for {wallet.name}:{wallet.hotkey_str}")
    except Exception as e:
        logger.error(f"Failed to load keypair: {e}")
        return
    
    # Create validators based on network choice
    active_validators = []
    
    if args.network == 'both' or args.network == 'bittensor':
        try:
            bittensor_validator = ValidatorFactory.create_validator('bittensor', key, settings)
            if bittensor_validator:
                active_validators.append(bittensor_validator)
        except Exception as e:
            logger.error(f"Failed to create Bittensor validator: {e}")
    
    if args.network == 'both' or args.network == 'commune':
        try:
            commune_validator = ValidatorFactory.create_validator('commune', key, settings)
            if commune_validator:
                active_validators.append(commune_validator)
        except Exception as e:
            logger.error(f"Failed to create Commune validator: {e}")
    
    if not active_validators:
        logger.error("No validators could be created. Exiting.")
        return
    
    logger.info(f"Starting validation with {len(active_validators)} validator(s)")
    
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
                run_validation_cycle(active_validators, validation_cycle)
                last_validation_time = current_time
            
            # Submit weights if interval has passed
            if current_time - last_submission_time >= args.submission_interval:
                submission_cycle += 1
                submit_weights(active_validators, submission_cycle)
                last_submission_time = current_time
            
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