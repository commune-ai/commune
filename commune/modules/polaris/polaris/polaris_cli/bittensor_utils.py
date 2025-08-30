import bittensor as bt
import logging
import os
import json
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

def get_uid_from_hotkey(hotkey: str, netuid: int, network: str = 'finney'):
    """
    Retrieve the UID for a wallet registered on a subnet using its hotkey.

    Args:
        hotkey (str): The public key (hotkey) of the wallet.
        netuid (int): The subnet ID.
        network (str): The network to connect to ('finney' for mainnet, 'test' for testnet)

    Returns:
        int: The UID of the wallet, or None if not found.
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Path for the UID log file
    uid_log_file = os.path.join(logs_dir, 'miner_uid_log.txt')
    
    # Log the attempt to retrieve UID
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'action': 'get_uid_attempt',
        'hotkey': hotkey,
        'netuid': netuid,
        'network': network
    }
    
    try:
        # Connect to the subtensor network
        sub = bt.subtensor(network)  # 'finney' for mainnet, 'test' for testnet
        logger.info(f"Connected to {network} network")

        # Get the metagraph for the subnet
        meta = sub.metagraph(netuid)
        logger.info(f"Retrieved metagraph for subnet {netuid}")
        
        # Ensure we're using the actual SS58 address
        if not hotkey.startswith('5'):
            # If not an SS58 address, log warning
            logger.warning(f"Provided hotkey '{hotkey}' does not appear to be an SS58 address")
            log_entry['error'] = "Not an SS58 address"
            with open(uid_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            return None
            
        # Find the UID for the given hotkey
        uid = next((uid for uid, registered_hotkey in zip(meta.uids, meta.hotkeys) if registered_hotkey == hotkey), None)

        if uid is not None:
            print(f"Miner UID: {uid}")
            logger.info(f"Found UID {uid} for hotkey {hotkey[:10]}...")
            
            # Update log entry with success info
            log_entry['status'] = 'success'
            log_entry['uid'] = int(uid)
            
            # Write to log file
            with open(uid_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
            # Also write to a dedicated file for this specific hotkey
            hotkey_uid_file = os.path.join(logs_dir, f'hotkey_{hotkey[:10]}_uid.txt')
            with open(hotkey_uid_file, 'w') as f:
                f.write(f"Hotkey: {hotkey}\nUID: {uid}\nNetwork: {network}\nNetuid: {netuid}\nTimestamp: {datetime.now().isoformat()}")
        else:
            print("Hotkey not found in the subnet")
            logger.warning(f"Could not find UID for hotkey {hotkey[:10]}... in subnet {netuid}")
            
            # Update log entry with failure info
            log_entry['status'] = 'not_found'
            
            # Write to log file
            with open(uid_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

        return uid
        
    except Exception as e:
        logger.error(f"Error retrieving UID: {str(e)}")
        
        # Update log entry with error info
        log_entry['status'] = 'error'
        log_entry['error_message'] = str(e)
        
        # Write to log file
        with open(uid_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
        return None

# Example Usage
if __name__ == "__main__":
    # Specify the hotkey and subnet ID
    hotkey = "5FBo3j6f3mYp6ZWrriSvegts412tRb6rnYmzBjL74RW5RPj4"  # Replace with the actual SS58 hotkey
    netuid = 1  # Replace with the correct subnet ID
    network = "finney"  # Use 'test' for testnet, 'finney' for mainnet

    # Retrieve the UID
    uid = get_uid_from_hotkey(hotkey, netuid, network)
    print(f"UID: {uid}") 