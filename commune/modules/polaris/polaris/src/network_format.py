# src/network_format.py

import re
from typing import Optional, Tuple


def convert_ssh_format(ssh_string: str) -> str:
    """
    Convert SSH command string to ssh:// URL format.
    
    Args:
        ssh_string: SSH string in format "ssh user@host -p port"
        
    Returns:
        str: SSH string in format "ssh://user@host:port"
    """
    # If already in correct format, return as is
    if ssh_string.startswith('ssh://'):
        return ssh_string
        
    # Parse traditional SSH command format
    pattern = r'^ssh\s+([^@]+)@([^\s]+)\s+-p\s+(\d+)$'
    match = re.match(pattern, ssh_string)
    
    if match:
        user, host, port = match.groups()
        return f"ssh://{user}@{host}:{port}"
        
    raise ValueError(f"Invalid SSH format: {ssh_string}")

def parse_ssh_url(ssh_url: str) -> Tuple[str, str, str]:
    """
    Parse SSH URL into components.
    
    Args:
        ssh_url: SSH URL string (ssh://user@host:port)
        
    Returns:
        Tuple[str, str, str]: (username, hostname, port)
    """
    try:
        # Remove ssh:// prefix
        ssh_url = ssh_url.replace('ssh://', '')
        
        # Split user@host:port
        user_part, host_part = ssh_url.split('@')
        host, port = host_part.split(':')
        
        return user_part, host, port
    except Exception as e:
        raise ValueError(f"Failed to parse SSH URL: {e}")