import logging
import os
from pathlib import Path
from typing import Any, Dict

from docker.errors import NotFound
from dotenv import load_dotenv

import docker

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class SSHManager:
    def __init__(self):
        """Initialize Docker client and SSH configuration"""
        try:
            self.client = docker.from_env()
            self.username = "devuser"
            # Read HOST from environment variable with no default
            self.host = os.environ.get('HOST')
            if not self.host:
                raise ValueError("HOST environment variable is not set")
            logger.info(f"SSH Manager initialized with host: {self.host}")
        except Exception as e:
            logger.error(f"Failed to initialize SSH Manager: {str(e)}")
            raise

    def setup_ssh_access(self, container_id: str, public_key: str) -> Dict[str, Any]:
        """Setup SSH access for a container"""
        try:
            logger.info(f"Setting up SSH access for container {container_id}")
            logger.debug(f"Using public key: {public_key[:30]}...")

            container = self.client.containers.get(container_id)
            
            # Commands to set up SSH
            commands = [
                'mkdir -p /home/devuser/.ssh',
                'chmod 700 /home/devuser/.ssh',
                f'echo "{public_key}" > /home/devuser/.ssh/authorized_keys',
                'chmod 600 /home/devuser/.ssh/authorized_keys',
                'chown -R devuser:devuser /home/devuser/.ssh'
            ]
            
            for cmd in commands:
                logger.debug(f"Executing: {cmd}")
                exit_code, output = container.exec_run(
                    cmd=f"/bin/bash -c '{cmd}'",
                    user='root'
                )
                if exit_code != 0:
                    raise Exception(f"Command failed: {cmd}\nOutput: {output.decode()}")

            # Get port mapping
            ports = container.ports.get('22/tcp', [])
            if not ports:
                raise Exception("No SSH port mapping found")
            
            ssh_port = int(ports[0]['HostPort'])
            logger.info(f"SSH access configured. Port: {ssh_port}, Host: {self.host}")
            
            return {
                "port": ssh_port,
                "username": self.username,
                "host": self.host
            }

        except Exception as e:
            logger.error(f"Error setting up SSH access: {str(e)}")
            raise

    def remove_ssh_access(self, container_id: str) -> None:
        """Remove SSH access for a container"""
        try:
            container = self.client.containers.get(container_id)
            container.exec_run('rm -f /home/devuser/.ssh/authorized_keys', user='root')
            logger.info(f"SSH access removed for container {container_id}")
        except Exception as e:
            logger.error(f"Error removing SSH access: {str(e)}")
            raise

    def get_ssh_info(self, container_id: str) -> Dict[str, Any]:
        """Get SSH connection information"""
        try:
            container = self.client.containers.get(container_id)
            ports = container.ports.get('22/tcp', [])
            if not ports:
                raise Exception("No SSH port mapping found")
            
            ssh_port = int(ports[0]['HostPort'])
            
            return {
                "port": ssh_port,
                "username": self.username,
                "host": self.host
            }
        except Exception as e:
            logger.error(f"Error getting SSH info: {str(e)}")
            raise