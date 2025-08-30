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
        logging.FileHandler("test_miner_uid_batch.log")
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

def create_sample_hotkeys_file(output_file):
    """
    Create a sample hotkeys JSON file with example structure.
    
    Args:
        output_file (str): Path to save the sample file
    """
    sample_data = {
        "mainnet": {
            "1": [  # netuid 1
                "5FBo3j6f3mYp6ZWrriSvegts412tRb6rnYmzBjL74RW5RPj4",  # example hotkey 1
                "5FYYECj1YuHXJsRyvArTTXvEJcHScnVPpDmDG3M1YZzFtA97"   # example hotkey 2
            ],
            "11": [  # netuid 11
                "5FA4h2beTLtRmgkoBx3MoNWrWxRNp1nN1u6XQDDCUwZRiuSB"  # example hotkey
            ]
        },
        "testnet": {
            "49": [  # netuid 49
                "5GZnQ2DjxjDrC9v3L5EbYDXQ5hv4zpxCBRWV5oC1eP6Jee6c"  # example hotkey
            ]
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Created sample hotkeys file at: {output_file}")
    print("Edit this file with your actual hotkeys and then run the batch test.")

def test_get_uid_batch(hotkeys_file):
    """
    Test UID retrieval for multiple hotkeys defined in a JSON file.
    
    Args:
        hotkeys_file (str): Path to JSON file with hotkeys
    """
    # Create test logs directory
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Timestamp for all logs in this batch
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_log_file = os.path.join(logs_dir, f'batch_test_results_{timestamp}.json')
    
    # Load hotkeys from file
    try:
        with open(hotkeys_file, 'r') as f:
            hotkey_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load hotkeys file: {str(e)}")
        print(f"ERROR: Could not load hotkeys file: {str(e)}")
        sys.exit(1)
    
    # Initialize results dictionary
    batch_results = {
        'timestamp': datetime.now().isoformat(),
        'results': []
    }
    
    # Test each network
    for network, netuids in hotkey_data.items():
        print(f"\n\n{'='*80}")
        print(f"TESTING NETWORK: {network.upper()}")
        print(f"{'='*80}")
        
        # Convert network name to bittensor format
        bt_network = "finney" if network.lower() == "mainnet" else "test"
        
        # Test each netuid
        for netuid_str, hotkeys in netuids.items():
            netuid = int(netuid_str)
            print(f"\n{'-'*80}")
            print(f"SUBNET ID: {netuid}")
            print(f"{'-'*80}")
            
            # Test each hotkey
            for hotkey in hotkeys:
                print(f"\nTesting hotkey: {hotkey[:10]}...")
                try:
                    uid = get_uid_from_hotkey(hotkey, netuid, bt_network)
                    
                    result = {
                        'network': bt_network,
                        'netuid': netuid,
                        'hotkey': hotkey,
                        'success': uid is not None,
                        'uid': int(uid) if uid is not None else None,
                        'error': None if uid is not None else "UID not found"
                    }
                    
                    if uid is not None:
                        print(f"SUCCESS! Found UID: {uid}")
                    else:
                        print(f"FAILED: No UID found for hotkey on this subnet.")
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"ERROR: {error_msg}")
                    logger.error(f"Error testing hotkey {hotkey[:10]}: {error_msg}")
                    
                    result = {
                        'network': bt_network,
                        'netuid': netuid,
                        'hotkey': hotkey,
                        'success': False,
                        'uid': None,
                        'error': error_msg
                    }
                
                # Add result to batch results
                batch_results['results'].append(result)
    
    # Write batch results to log file
    with open(batch_log_file, 'w') as f:
        json.dump(batch_results, f, indent=2)
    
    print(f"\n\nBatch test complete! Results saved to: {batch_log_file}")
    
    # Print summary
    success_count = sum(1 for r in batch_results['results'] if r['success'])
    total_count = len(batch_results['results'])
    print(f"\nSUMMARY: Successfully retrieved {success_count} out of {total_count} UIDs")
    
    return batch_results

def main():
    """Main function to parse arguments and run the batch test."""
    parser = argparse.ArgumentParser(description='Batch Test Bittensor Miner UID Retrieval')
    
    parser.add_argument('--hotkeys-file', type=str, help='JSON file containing hotkeys to test')
    parser.add_argument('--create-sample', action='store_true', 
                        help='Create a sample hotkeys file template')
    parser.add_argument('--sample-file', type=str, default='sample_hotkeys.json',
                        help='Path for the sample hotkeys file (default: sample_hotkeys.json)')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_hotkeys_file(args.sample_file)
        return
    
    if not args.hotkeys_file:
        parser.print_help()
        print("\nERROR: Please specify either --hotkeys-file or --create-sample")
        sys.exit(1)
    
    print(f"Bittensor Miner UID Batch Retrieval Test")
    print(f"Using hotkeys file: {args.hotkeys_file}")
    
    test_get_uid_batch(args.hotkeys_file)

if __name__ == "__main__":
    main() 