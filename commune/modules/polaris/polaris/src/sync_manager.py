# src/sync_manager.py

import json
import logging
import os
from typing import Dict, Optional, Tuple

from src.network_sync import NetworkSync
from src.user_manager import UserManager
from src.utils import get_project_root

logger = logging.getLogger(__name__)

class SyncManager:
    def __init__(self):
        self.user_manager = UserManager()
        self.network_sync = NetworkSync()
        
    def load_system_info(self) -> Optional[Dict]:
        """Load current system info from file."""
        try:
            system_info_path = os.path.join(get_project_root(), 'system_info.json')
            if not os.path.exists(system_info_path):
                logger.error("System info file not found")
                return None
                
            with open(system_info_path, 'r') as f:
                data = json.load(f)
                return data[0] if data else None
        except Exception as e:
            logger.error(f"Failed to load system info: {e}")
            return None

    def get_network_info(self) -> Optional[Dict]:
        """Extract network info from system info."""
        system_info = self.load_system_info()
        if not system_info or 'compute_resources' not in system_info:
            return None
            
        resources = system_info['compute_resources']
        if not resources or 'network' not in resources[0]:
            return None
            
        return resources[0]['network']

    def sync_network_info(self) -> bool:
        """
        Sync network information between system info, user info and remote server.
        """
        try:
            # Get current user info - don't show prompt since this is for start command
            skip_reg, user_info = self.user_manager.check_existing_registration(show_prompt=False)
            if not user_info:
                logger.error("No user registration found")
                return False

            # Get latest network info from system
            new_network_info = self.get_network_info()
            if not new_network_info:
                logger.error("No network info found in system info")
                return False

            # Update local user info using existing UserManager method
            if not self.user_manager.update_network_info(new_network_info):
                logger.error("Failed to update local user info")
                return False

            # Sync with remote server
            if not self.network_sync.update_miner_network(user_info['miner_id'], new_network_info):
                logger.error("Failed to sync network info with remote server")
                return False

            logger.info("Network info synced successfully")
            return True

        except Exception as e:
            logger.error(f"Error during network sync: {e}")
            return False

    def verify_sync_status(self) -> Tuple[bool, Dict[str, bool]]:
        """
        Verify sync status across all components.
        """
        status = {
            'system_info': False,
            'user_info': False,
            'remote_sync': False
        }
        
        try:
            # Check system info
            system_network = self.get_network_info()
            status['system_info'] = bool(system_network)
            
            # Check user info
            user_info = self.user_manager.get_user_info()
            status['user_info'] = bool(user_info and user_info.get('network_info'))
            
            # Compare network info if both exist
            if status['system_info'] and status['user_info']:
                system_network_str = json.dumps(system_network, sort_keys=True)
                user_network_str = json.dumps(user_info['network_info'], sort_keys=True)
                if system_network_str != user_network_str:
                    logger.warning("Network info mismatch between system and user info")
                    return False, status
            
            # Verify remote sync by attempting an update
            if user_info and system_network:
                status['remote_sync'] = self.network_sync.update_miner_network(
                    user_info['miner_id'],
                    system_network
                )
            
            return all(status.values()), status
            
        except Exception as e:
            logger.error(f"Error verifying sync status: {e}")
            return False, status