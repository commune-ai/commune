import re
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Valid SSH key types
VALID_KEY_TYPES: List[str] = [
    'ssh-rsa',
    'ssh-ed25519',
    'ecdsa-sha2-nistp256',
    'ecdsa-sha2-nistp384',
    'ecdsa-sha2-nistp521'
]

def validate_ssh_key(key: str) -> bool:
    """
    Validate SSH public key format.
    
    Args:
        key: SSH public key string
    
    Returns:
        bool: True if key is valid, False otherwise
    """
    try:
        if not key or not isinstance(key, str):
            logger.error("Invalid key: Empty or not a string")
            return False

        # Split key into parts
        key_parts = key.strip().split()
        if len(key_parts) < 2:
            logger.error("Invalid key format: Missing parts")
            return False

        # Validate key type
        key_type = key_parts[0]
        if key_type not in VALID_KEY_TYPES:
            logger.error(f"Invalid key type: {key_type}")
            return False

        # Validate key data (base64)
        key_data = key_parts[1]
        if not re.match(r'^[A-Za-z0-9+/]+={0,2}$', key_data):
            logger.error("Invalid key data format")
            return False

        return True

    except Exception as e:
        logger.error(f"Error validating SSH key: {str(e)}")
        return False

def sanitize_command(command: str) -> Optional[str]:
    """
    Sanitize shell commands to prevent injection.
    
    Args:
        command: Shell command string
    
    Returns:
        Optional[str]: Sanitized command or None if invalid
    """
    try:
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[;&|`$]', '', command)
        
        # Basic command structure validation
        if not re.match(r'^[a-zA-Z0-9\s\-_./"\']+$', sanitized):
            logger.error("Invalid command structure")
            return None
            
        return sanitized
    except Exception as e:
        logger.error(f"Error sanitizing command: {str(e)}")
        return None

def validate_port_number(port: int) -> bool:
    """
    Validate port number is in valid range.
    
    Args:
        port: Port number to validate
    
    Returns:
        bool: True if port is valid, False otherwise
    """
    try:
        return isinstance(port, int) and 1024 <= port <= 65535
    except Exception as e:
        logger.error(f"Error validating port number: {str(e)}")
        return False