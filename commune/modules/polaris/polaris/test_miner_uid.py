#!/usr/bin/env python3

import argparse
import logging
import sys
import os
import json
from datetime import datetime

# Add the parent directory to the path so we can import the polaris_cli module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("test_miner_uid.log")
    ]
)
logger = logging.getLogger(__name__)

try:
    from polaris_cli.bittensor_utils import get_uid_from_hotkey
    logger.info("Successfully imported bittensor_utils module")
except ImportError as e:
    logger.error(f"Failed to import bittensor_utils: {str(e)}")
    print(f"Error: Could not import bittensor_utils. Make sure the module exists and is accessible.")
    sys.exit(1)

def test_get_uid(hotkey, netuid, network="finney"):
    """
    Test the get_uid_from_hotkey function.
    
    Args:
        hotkey (str): The SS58 hotkey to test
        netuid (int): The subnet ID to check
        network (str): The network to connect to ('finney' for mainnet, 'test' for testnet)
    """
    print(f"\n{'='*80}")
    print(f"TESTING UID RETRIEVAL")
    print(f"{'='*80}")
    print(f"Hotkey: {hotkey}")
    print(f"Netuid: {netuid}")
    print(f"Network: {network}")
    print(f"{'-'*80}")
    
    # Create test log directory
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_logs')
    os.makedirs(logs_dir, exist_ok=True)
    test_log_file = os.path.join(logs_dir, f'test_uid_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    test_result = {
        'timestamp': datetime.now().isoformat(),
        'hotkey': hotkey,
        'netuid': netuid,
        'network': network,
        'success': False,
        'uid': None,
        'error': None
    }
    
    try:
        # Test if hotkey is a valid SS58 address
        if not hotkey.startswith('5') or len(hotkey) < 40:
            print(f"WARNING: Hotkey '{hotkey}' does not appear to be a valid SS58 address.")
            print(f"Valid SS58 addresses start with '5' and are typically 48-50 characters long.")
            test_result['error'] = "Invalid SS58 address format"
        
        print(f"Retrieving UID for hotkey on network {network}, subnet {netuid}...")
        uid = get_uid_from_hotkey(hotkey, netuid, network)
        
        if uid is not None:
            print(f"\nSUCCESS! Found UID: {uid}")
            test_result['success'] = True
            test_result['uid'] = int(uid)
        else:
            print(f"\nFAILED: No UID found for hotkey on this subnet.")
            test_result['error'] = "UID not found"
        
    except Exception as e:
        error_msg = str(e)
        print(f"\nERROR: {error_msg}")
        logger.error(f"Error during test: {error_msg}")
        test_result['error'] = error_msg
    
    # Write test result to log file
    with open(test_log_file, 'w') as f:
        json.dump(test_result, f, indent=2)
    print(f"\nTest results saved to: {test_log_file}")
    
    return test_result

def main():
    """Main function to parse arguments and run the test."""
    parser = argparse.ArgumentParser(description='Test Bittensor Miner UID Retrieval')
    
    parser.add_argument('--hotkey', type=str, required=True,
                        help='SS58 hotkey address to test')
    parser.add_argument('--netuid', type=int, required=True,
                        help='Subnet ID to check (e.g., 1 for Polaris)')
    parser.add_argument('--network', type=str, choices=['finney', 'test'], default='finney',
                        help='Network to connect to: finney (mainnet) or test (testnet)')
    
    args = parser.parse_args()
    
    print(f"Bittensor Miner UID Retrieval Test")
    print(f"Network: {args.network}")
    print(f"Subnet ID: {args.netuid}")
    
    test_get_uid(args.hotkey, args.netuid, args.network)

if __name__ == "__main__":
    main() 