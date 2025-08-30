import logging
import os
import socket
import time
from pathlib import Path
from typing import Any, Dict, List

from docker.errors import DockerException

import docker

logger = logging.getLogger(__name__)

class ContainerManager:
    def __init__(self):
        """Initialize Docker client with explicit connection settings."""
        try:
            # Create Docker client using the Unix socket
            self.client = docker.DockerClient(
                base_url='unix://var/run/docker.sock',
                version='auto',
                timeout=30
            )
            
            # Test connection by retrieving the Docker daemon version
            version = self.client.version()
            logger.info(f"Connected to Docker daemon (version: {version.get('Version', 'unknown')})")
            
            # Compute the absolute path to the build context directory.
            # Starting from this file (container.py), go up three levels to reach "compute_subnet"
            # Then append "docker/dev_container" to get the full path.
            base_path = Path(__file__).resolve().parents[2]
            self.dev_container_path = (base_path / "docker" / "dev_container").resolve()
            logger.info(f"Using build context: {self.dev_container_path}")
            
            # Validate that the build context directory exists.
            if not self.dev_container_path.is_dir():
                raise Exception(f"Build context directory does not exist: {self.dev_container_path}")
            
            # Retrieve SSH port range from environment variables
            self.port_start = int(os.getenv('SSH_PORT_RANGE_START'))
            self.port_end = int(os.getenv('SSH_PORT_RANGE_END'))
            
            # Track recently freed ports with timestamps
            self.recently_freed_ports = {}
            
        except Exception as e:
            logger.error(f"Docker connection failed: {str(e)}")
            raise DockerException(f"Failed to connect to Docker daemon: {str(e)}")

    def _is_port_available(self, port: int) -> bool:
        """
        Check if a port is available by trying to create a socket.
        
        Args:
            port: Port number to check
        Returns:
            True if port is available, False otherwise
        """
        # Check if this port was recently freed
        if port in self.recently_freed_ports:
            time_since_freed = time.time() - self.recently_freed_ports[port]
            # If port was freed less than 3 seconds ago, consider it still in cooldown
            if time_since_freed < 3:
                logger.info(f"Port {port} was recently freed ({time_since_freed:.1f}s ago) and is in cooldown")
                return False
            else:
                # Remove port from recently freed list if cooldown has passed
                del self.recently_freed_ports[port]
        
        # Actually check if port is available by trying to bind
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('0.0.0.0', port))
            return True
        except socket.error:
            return False
        finally:
            sock.close()

    def _wait_for_port_release(self, port: int, max_attempts: int = 5, delay: int = 1) -> bool:
        """
        Wait for a port to be released, with multiple attempts.
        
        Args:
            port: Port number to wait for
            max_attempts: Maximum number of attempts
            delay: Delay in seconds between attempts
        Returns:
            True if port becomes available, False otherwise
        """
        for attempt in range(max_attempts):
            if self._is_port_available(port):
                return True
            logger.info(f"Port {port} still unavailable, waiting... (attempt {attempt+1}/{max_attempts})")
            time.sleep(delay)
        return False

    def _get_next_available_port(self) -> int:
        """Find the next available port within the defined range."""
        try:
            # Check if the start port is available
            if self._is_port_available(self.port_start):
                return self.port_start
                
            # Check if the end port is available
            if self._is_port_available(self.port_end):
                return self.port_end
            
            # If no ports are immediately available, try waiting for release
            if self._wait_for_port_release(self.port_start):
                return self.port_start
                
            if self._wait_for_port_release(self.port_end):
                return self.port_end
                
            # If both ports are in use, raise an exception
            raise Exception(f"Both ports {self.port_start} and {self.port_end} are in use")
        
        except Exception as e:
            logger.error(f"Port availability check failed: {str(e)}")
            raise

    def create_container(self) -> str:
        """Create a new development container."""
        try:
            # Build the container image using the specified build context directory.
            logger.info("Building container image...")
            image, _ = self.client.images.build(
                path=str(self.dev_container_path),
                tag="dev-container:latest",
                forcerm=True
            )

            # Get the next available SSH port.
            port = self._get_next_available_port()
            logger.info(f"Using port {port} for new container")

            # Create and run the container.
            container = self.client.containers.run(
                image="dev-container:latest",
                detach=True,
                ports={'22/tcp': port},
                volumes={
                    '/workspace': {'bind': '/workspace', 'mode': 'rw'}
                },
                environment={
                    'CONTAINER_SSH_PORT': str(port)
                },
                restart_policy={"Name": "unless-stopped"}
            )

            logger.info(f"Container created with ID: {container.id}")
            return container.id

        except Exception as e:
            logger.error(f"Failed to create container: {str(e)}")
            raise

    def remove_container(self, container_id: str) -> None:
        """Remove a container by its ID - equivalent to 'docker stop' followed by 'docker rm'."""
        try:
            container = self.client.containers.get(container_id)
            
            # Get the port before stopping the container
            ports = container.ports.get('22/tcp', [])
            port = int(ports[0]['HostPort']) if ports else None
            
            # First stop the container (equivalent to 'docker stop')
            logger.info(f"Stopping container {container_id}")
            container.stop(timeout=10)
            
            # Then remove the container (equivalent to 'docker rm')
            logger.info(f"Removing container {container_id}")
            container.remove(v=True)  # v=True removes volumes
            
            # Record when this port was freed for cooldown tracking
            if port:
                self.recently_freed_ports[port] = time.time()
                logger.info(f"Port {port} freed from container {container_id}")
            
            logger.info(f"Container {container_id} removed")
            
        except docker.errors.NotFound:
            logger.warning(f"Container {container_id} not found, may have been already removed")
        except Exception as e:
            logger.error(f"Failed to remove container: {str(e)}")
            # If normal removal fails, try force removal
            try:
                logger.info(f"Attempting force removal of container {container_id}")
                self.client.api.remove_container(container_id, force=True, v=True)
                logger.info(f"Force removal successful for container {container_id}")
            except Exception as force_error:
                logger.error(f"Force removal also failed: {str(force_error)}")
                raise

    def get_container_info(self, container_id: str) -> Dict[str, Any]:
        """Retrieve information about a specific container."""
        try:
            container = self.client.containers.get(container_id)
            ports = container.ports.get('22/tcp', [])
            ssh_port = int(ports[0]['HostPort']) if ports else None
            
            return {
                "id": container.id,
                "status": container.status,
                "ssh_port": ssh_port,
                "created": container.attrs['Created'],
                "name": container.name
            }
        except Exception as e:
            logger.error(f"Failed to get container info: {str(e)}")
            raise

    def list_containers(self, all: bool = False) -> List[Dict[str, Any]]:
        """List all containers (with an option to include non-running ones)."""
        try:
            containers = self.client.containers.list(all=all)
            return [
                {
                    "id": container.id,
                    "status": container.status,
                    "name": container.name,
                    "image": container.image.tags[0] if container.image.tags else 'N/A',
                    "ports": container.ports
                }
                for container in containers
            ]
        except Exception as e:
            logger.error(f"Failed to list containers: {str(e)}")
            raise

    def remove_all_containers(self, force: bool = False) -> None:
        """Remove all containers."""
        try:
            containers = self.client.containers.list(all=True)
            for container in containers:
                try:
                    if force:
                        container.stop(timeout=1)
                    container.remove(force=force)
                    logger.info(f"Removed container {container.id}")
                except Exception as inner_e:
                    logger.warning(f"Failed to remove container {container.id}: {str(inner_e)}")
            
            logger.info("All containers removal process completed")
        except Exception as e:
            logger.error(f"Failed to remove all containers: {str(e)}")
            raise
