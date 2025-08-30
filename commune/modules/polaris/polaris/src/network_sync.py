# src/network_sync.py

import logging
import os
from typing import Dict

import requests

from src.network_format import convert_ssh_format

logger = logging.getLogger(__name__)
server_url_ = os.getenv('SERVER_URL')

class NetworkSync:
    def __init__(self):
        self.base_url = server_url_
        self.headers = {'Content-Type': 'application/json'}
        
    def update_miner_network(self, miner_id: str, network_info: Dict) -> bool:
        """
        Update miner's network information in the orchestrator.
        
        Args:
            miner_id: The ID of the registered miner
            network_info: Dictionary containing updated network information
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            # Convert SSH format if needed
            if 'ssh' in network_info:
                try:
                    network_info['ssh'] = convert_ssh_format(network_info['ssh'])
                except ValueError as e:
                    logger.error(f"SSH format conversion failed: {e}")
                    return False
                    
            # Prepare request payload matching NetworkUpdateRequest model exactly
            payload = {
                "miner_id": miner_id,
                "network": {
                    "internal_ip": str(network_info["internal_ip"]),
                    "ssh": str(network_info["ssh"]),
                    "username": str(network_info["username"]),
                    "password": str(network_info["password"]),
                    "open_ports": list(map(str, network_info.get("open_ports", ["22"])))
                }
            }
            
            logger.debug(f"Sending network update request with payload: {payload}")
            
            response = requests.post(
                f"{self.base_url}/network_update",
                json=payload,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully updated network info for miner {miner_id}")
                return True
            
            error_msg = f"Failed to update network info. Status: {response.status_code}"
            try:
                error_details = response.json()
                logger.error(f"{error_msg}\nDetails: {error_details}")
            except:
                logger.error(f"{error_msg}\nResponse: {response.text}")
            return False
            
        except requests.RequestException as e:
            logger.error(f"Network sync failed: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during network sync: {str(e)}")
            return False