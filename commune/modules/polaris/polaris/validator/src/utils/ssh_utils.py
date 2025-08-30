"""
SSH utilities for connecting to and verifying miner nodes.

This module provides utilities for establishing SSH connections to miner nodes,
executing commands to verify their resources, and retrieving hardware specifications.
"""
import logging
import json
import re
import time
from typing import Dict, Any, List, Tuple, Optional

import paramiko

from validator.src.utils.logging_utils import exception_handler

logger = logging.getLogger(__name__)

class SSHClient:
    """Client for connecting to miners via SSH and executing commands."""
    
    def __init__(self, 
                 host: str, 
                 username: str, 
                 port: int = 22, 
                 password: Optional[str] = None, 
                 key_path: Optional[str] = None,
                 key_passphrase: Optional[str] = None,
                 connection_timeout: int = 30,
                 command_timeout: int = 60):
        """
        Initialize SSH client for miner connection.
        
        Args:
            host: Hostname or IP address of the miner
            username: SSH username
            port: SSH port number (default: 22)
            password: SSH password (if using password authentication)
            key_path: Path to private key file (if using key-based authentication)
            key_passphrase: Passphrase for private key (if needed)
            connection_timeout: Timeout for SSH connection in seconds
            command_timeout: Timeout for SSH commands in seconds
        """
        self.host = host
        self.username = username
        self.port = port
        self.password = password
        self.key_path = key_path
        self.key_passphrase = key_passphrase
        self.connection_timeout = connection_timeout
        self.command_timeout = command_timeout
        
        self.client = None
        self.connected = False
    
    @exception_handler(fallback_return=False)
    def connect(self) -> bool:
        """
        Establish SSH connection to the miner.
        
        Returns:
            True if connection successful, False otherwise
        """
        if self.connected:
            logger.debug(f"Already connected to {self.host}")
            return True
        
        try:
            # Initialize SSH client
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            connect_kwargs = {
                'hostname': self.host,
                'port': self.port,
                'username': self.username,
                'timeout': self.connection_timeout
            }
            
            # Add authentication method
            if self.password:
                connect_kwargs['password'] = self.password
            elif self.key_path:
                if self.key_passphrase:
                    key = paramiko.RSAKey.from_private_key_file(
                        self.key_path, password=self.key_passphrase
                    )
                else:
                    key = paramiko.RSAKey.from_private_key_file(self.key_path)
                connect_kwargs['pkey'] = key
            else:
                logger.error("No authentication method provided (password or key)")
                return False
            
            # Connect to server
            logger.info(f"Connecting to miner at {self.host}:{self.port}")
            self.client.connect(**connect_kwargs)
            self.connected = True
            
            logger.info(f"Successfully connected to miner at {self.host}")
            return True
            
        except paramiko.AuthenticationException:
            logger.error(f"Authentication failed for {self.host}")
            return False
        except paramiko.SSHException as e:
            logger.error(f"SSH error connecting to {self.host}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to {self.host}: {e}")
            return False
    
    @exception_handler(fallback_return=('', '', -1))
    def execute_command(self, command: str) -> Tuple[str, str, int]:
        """
        Execute a command on the miner via SSH.
        
        Args:
            command: Command to execute
        
        Returns:
            Tuple of (stdout, stderr, exit_code)
        """
        if not self.connected:
            success = self.connect()
            if not success:
                logger.error(f"Cannot execute command: not connected to {self.host}")
                return '', f"Failed to connect to {self.host}", 1
        
        try:
            logger.debug(f"Executing command on {self.host}: {command}")
            
            # Execute command with timeout
            stdin, stdout, stderr = self.client.exec_command(
                command, timeout=self.command_timeout
            )
            
            # Get command output
            stdout_str = stdout.read().decode('utf-8').strip()
            stderr_str = stderr.read().decode('utf-8').strip()
            exit_code = stdout.channel.recv_exit_status()
            
            if exit_code != 0:
                logger.warning(f"Command on {self.host} exited with code {exit_code}")
                logger.warning(f"stderr: {stderr_str}")
            
            return stdout_str, stderr_str, exit_code
            
        except paramiko.SSHException as e:
            logger.error(f"SSH error during command execution on {self.host}: {e}")
            return '', str(e), 1
        except Exception as e:
            logger.error(f"Error executing command on {self.host}: {e}")
            return '', str(e), 1
    
    def close(self):
        """Close the SSH connection."""
        if self.client:
            self.client.close()
            self.connected = False
            logger.debug(f"Closed SSH connection to {self.host}")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


@exception_handler(fallback_return={})
def get_cpu_info(ssh_client: SSHClient) -> Dict[str, Any]:
    """
    Retrieve CPU information from the miner.
    
    Args:
        ssh_client: Connected SSH client
    
    Returns:
        Dictionary with CPU information
    """
    # Get CPU model
    stdout, _, _ = ssh_client.execute_command("lscpu | grep 'Model name'")
    
    cpu_model = ""
    if stdout:
        model_match = re.search(r'Model name:\s+(.*)', stdout)
        if model_match:
            cpu_model = model_match.group(1).strip()
    
    # Get CPU count
    stdout, _, _ = ssh_client.execute_command("nproc")
    
    cpu_count = 0
    if stdout:
        try:
            cpu_count = int(stdout.strip())
        except ValueError:
            logger.warning(f"Could not parse CPU count: {stdout}")
    
    # Get CPU speed
    stdout, _, _ = ssh_client.execute_command(
        "lscpu | grep 'CPU MHz' | awk '{print $3}'"
    )
    
    cpu_speed = 0.0
    if stdout:
        try:
            cpu_speed_mhz = float(stdout.strip())
            cpu_speed = cpu_speed_mhz / 1000.0  # Convert to GHz
        except ValueError:
            logger.warning(f"Could not parse CPU speed: {stdout}")
    
    return {
        'model': cpu_model,
        'cpu_count': cpu_count,
        'cpu_speed': cpu_speed
    }


@exception_handler(fallback_return=[])
def get_gpu_info(ssh_client: SSHClient) -> List[Dict[str, Any]]:
    """
    Retrieve GPU information from the miner.
    This function detects NVIDIA, AMD, and Intel GPUs.
    
    Args:
        ssh_client: Connected SSH client
    
    Returns:
        List of dictionaries with GPU information
    """
    gpus = []
    
    # 1. Try NVIDIA detection first
    stdout, _, exit_code = ssh_client.execute_command("which nvidia-smi")
    
    if exit_code == 0:
        # Get GPU information in CSV format
        stdout, stderr, exit_code = ssh_client.execute_command(
            "nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader"
        )
        
        if exit_code == 0:
            # Parse CSV output
            for line in stdout.strip().split('\n'):
                if not line:
                    continue
                
                parts = [part.strip() for part in line.split(',')]
                
                if len(parts) >= 2:
                    gpu_name = parts[0].strip()
                    
                    # Extract memory (convert MiB to MB)
                    memory_match = re.search(r'(\d+)\s*MiB', parts[1])
                    memory_mb = 0
                    if memory_match:
                        memory_mb = int(memory_match.group(1))
                    
                    # Extract utilization if available
                    utilization = 0
                    if len(parts) >= 3:
                        util_match = re.search(r'(\d+)\s*%', parts[2])
                        if util_match:
                            utilization = int(util_match.group(1))
                    
                    gpus.append({
                        'name': gpu_name,
                        'memory': memory_mb,
                        'utilization': utilization,
                        'type': 'NVIDIA'
                    })
            
            logger.info(f"Found {len(gpus)} NVIDIA GPUs")
        else:
            logger.warning(f"Error getting NVIDIA GPU info: {stderr}")
    
    # 2. Try AMD detection
    stdout, _, exit_code = ssh_client.execute_command("which rocm-smi")
    
    if exit_code == 0:
        # AMD GPU detection via rocm-smi
        stdout, stderr, exit_code = ssh_client.execute_command("rocm-smi --showmeminfo vram --csv")
        
        if exit_code == 0:
            lines = stdout.strip().split('\n')
            # Skip header row
            if len(lines) > 1:
                for line in lines[1:]:
                    parts = line.split(',')
                    if len(parts) >= 3:  # Adjust based on actual output format
                        try:
                            gpu_id = parts[0].strip()
                            # Memory is usually in bytes, convert to MB
                            memory_mb = int(int(parts[2].strip()) / (1024 * 1024))
                            
                            # Get GPU name from additional command
                            name_cmd = f"rocm-smi -d {gpu_id} --showproductname"
                            name_stdout, _, _ = ssh_client.execute_command(name_cmd)
                            gpu_name = name_stdout.strip() if name_stdout else f"AMD GPU {gpu_id}"
                            
                            gpus.append({
                                'name': gpu_name,
                                'memory': memory_mb,
                                'utilization': 0,  # Default value
                                'type': 'AMD'
                            })
                        except Exception as e:
                            logger.warning(f"Error parsing AMD GPU info: {e}")
            
            logger.info(f"Found {len(gpus) - len([g for g in gpus if g.get('type') == 'NVIDIA'])} AMD GPUs")
        else:
            logger.warning(f"Error getting AMD GPU info: {stderr}")
    
    # 3. Use lspci as a fallback to detect any GPUs not captured above
    stdout, stderr, exit_code = ssh_client.execute_command("which lspci")
    
    if exit_code == 0:
        # Use lspci to detect GPUs (will detect all types, including Intel)
        stdout, stderr, exit_code = ssh_client.execute_command("lspci -v -nn | grep -E 'VGA|3D|Display'")
        
        if exit_code == 0:
            # Extract GPU names and check if they're already in our list
            existing_gpu_names = {gpu['name'].lower() for gpu in gpus}
            
            for line in stdout.strip().split('\n'):
                if not line:
                    continue
                
                # Extract GPU type and name
                gpu_type = 'Unknown'
                if 'nvidia' in line.lower():
                    gpu_type = 'NVIDIA'
                elif 'amd' in line.lower() or 'radeon' in line.lower() or 'ati' in line.lower():
                    gpu_type = 'AMD'
                elif 'intel' in line.lower():
                    gpu_type = 'Intel'
                
                # Check if we already have this GPU type in our list
                gpu_name = line.strip()
                if not any(gpu_type.lower() in existing_name for existing_name in existing_gpu_names):
                    gpus.append({
                        'name': gpu_name,
                        'memory': 0,  # Unknown memory
                        'utilization': 0,
                        'type': gpu_type
                    })
            
            logger.info(f"Total GPUs found with lspci: {len(stdout.strip().split('\n'))}")
    
    # If still no GPUs found, try a more aggressive lspci search
    if not gpus:
        stdout, stderr, exit_code = ssh_client.execute_command("lspci | grep -i 'vga\\|3d\\|display\\|graphic'")
        
        if exit_code == 0 and stdout.strip():
            for line in stdout.strip().split('\n'):
                if not line:
                    continue
                
                gpu_type = 'Unknown'
                if 'nvidia' in line.lower():
                    gpu_type = 'NVIDIA'
                elif 'amd' in line.lower() or 'radeon' in line.lower() or 'ati' in line.lower():
                    gpu_type = 'AMD'
                elif 'intel' in line.lower():
                    gpu_type = 'Intel'
                
                gpus.append({
                    'name': line.strip(),
                    'memory': 0,  # Unknown memory
                    'utilization': 0,
                    'type': gpu_type
                })
    
    return gpus


@exception_handler(fallback_return={})
def get_memory_info(ssh_client: SSHClient) -> Dict[str, Any]:
    """
    Retrieve memory information from the miner.
    
    Args:
        ssh_client: Connected SSH client
    
    Returns:
        Dictionary with memory information
    """
    # Get total memory
    stdout, _, _ = ssh_client.execute_command(
        "free -b | grep 'Mem:' | awk '{print $2}'"
    )
    
    total_memory_bytes = 0
    if stdout:
        try:
            total_memory_bytes = int(stdout.strip())
        except ValueError:
            logger.warning(f"Could not parse total memory: {stdout}")
    
    # Convert to GB
    total_memory_gb = total_memory_bytes / (1024**3)
    
    return {
        'memory': round(total_memory_gb, 1)
    }


@exception_handler(fallback_return={})
def get_storage_info(ssh_client: SSHClient) -> Dict[str, Any]:
    """
    Retrieve storage information from the miner.
    
    Args:
        ssh_client: Connected SSH client
    
    Returns:
        Dictionary with storage information
    """
    # Get total storage
    stdout, _, _ = ssh_client.execute_command(
        "df -BG --total | grep 'total' | awk '{print $2}'"
    )
    
    total_storage_gb = 0
    if stdout:
        try:
            # Remove the 'G' suffix and convert to number
            storage_str = stdout.strip().replace('G', '')
            total_storage_gb = float(storage_str)
        except ValueError:
            logger.warning(f"Could not parse total storage: {stdout}")
    
    return {
        'storage': total_storage_gb
    }


@exception_handler(fallback_return={})
def get_network_info(ssh_client: SSHClient) -> Dict[str, Any]:
    """
    Retrieve network information from the miner.
    
    Args:
        ssh_client: Connected SSH client
    
    Returns:
        Dictionary with network information
    """
    # This is a simplified approach - in reality, you might want to use
    # speedtest-cli or a similar tool to measure actual bandwidth
    
    # Check if speedtest-cli is available and install if not
    stdout, _, exit_code = ssh_client.execute_command("which speedtest-cli")
    
    if exit_code != 0:
        logger.info("speedtest-cli not found, attempting to install...")
        ssh_client.execute_command("pip install speedtest-cli")
    
    # Run a speed test (this may take a while)
    stdout, stderr, exit_code = ssh_client.execute_command(
        "speedtest-cli --simple"
    )
    
    if exit_code != 0:
        logger.warning(f"Error running speed test: {stderr}")
        return {'bandwidth': 100}  # Default value if test fails
    
    # Parse output
    download_speed = 0
    upload_speed = 0
    
    for line in stdout.strip().split('\n'):
        if 'Download' in line:
            match = re.search(r'Download:\s+([\d.]+)\s+Mbit/s', line)
            if match:
                download_speed = float(match.group(1))
        elif 'Upload' in line:
            match = re.search(r'Upload:\s+([\d.]+)\s+Mbit/s', line)
            if match:
                upload_speed = float(match.group(1))
    
    # Use the lower of download and upload speeds as the bandwidth
    bandwidth = min(download_speed, upload_speed) if download_speed and upload_speed else max(download_speed, upload_speed)
    
    return {
        'bandwidth': bandwidth
    }


@exception_handler(fallback_return={})
def get_docker_info(ssh_client: SSHClient) -> Dict[str, Any]:
    """
    Retrieve Docker information from the miner.
    
    Args:
        ssh_client: Connected SSH client
    
    Returns:
        Dictionary with Docker information
    """
    # Check if Docker is installed
    stdout, _, exit_code = ssh_client.execute_command("which docker")
    
    if exit_code != 0:
        logger.info("Docker not found on miner")
        return {'installed': False}
    
    # Get Docker version
    stdout, _, _ = ssh_client.execute_command("docker --version")
    
    docker_version = stdout.strip() if stdout else "Unknown"
    
    # Get running containers
    stdout, _, _ = ssh_client.execute_command(
        "docker ps --format '{{.ID}}|{{.Image}}|{{.Status}}|{{.Names}}'"
    )
    
    containers = []
    if stdout:
        for line in stdout.strip().split('\n'):
            if not line:
                continue
            
            parts = line.split('|')
            if len(parts) >= 4:
                container_id, image, status, name = parts[:4]
                containers.append({
                    'id': container_id,
                    'image': image,
                    'status': status,
                    'name': name
                })
    
    return {
        'installed': True,
        'version': docker_version,
        'containers': containers
    }


def get_hardware_specifications(ssh_client: SSHClient) -> Dict[str, Any]:
    """
    Retrieve comprehensive hardware specifications from the miner.
    
    Args:
        ssh_client: Connected SSH client
    
    Returns:
        Dictionary with all hardware specifications
    """
    # Ensure we're connected
    if not ssh_client.connected and not ssh_client.connect():
        logger.error("Failed to connect to miner for hardware verification")
        return {}
    
    try:
        # Get CPU info
        cpu_info = get_cpu_info(ssh_client)
        logger.info(f"Retrieved CPU info: {cpu_count} cores")
        
        # Get GPU info
        gpus = get_gpu_info(ssh_client)
        logger.info(f"Retrieved info for {len(gpus)} GPUs")
        
        # Get memory info
        memory_info = get_memory_info(ssh_client)
        memory_gb = memory_info.get('memory', 0)
        logger.info(f"Retrieved memory info: {memory_gb:.1f} GB")
        
        # Get storage info
        storage_info = get_storage_info(ssh_client)
        storage_gb = storage_info.get('storage', 0)
        logger.info(f"Retrieved storage info: {storage_gb:.1f} GB")
        
        # Get network info
        network_info = get_network_info(ssh_client)
        bandwidth = network_info.get('bandwidth', 0)
        logger.info(f"Retrieved network info: {bandwidth:.1f} Mbps")
        
        # Get Docker info
        docker_info = get_docker_info(ssh_client)
        docker_installed = docker_info.get('installed', False)
        container_count = len(docker_info.get('containers', []))
        logger.info(f"Docker installed: {docker_installed}, Running containers: {container_count}")
        
        # Combine all information
        return {
            **cpu_info,
            'gpus': gpus,
            **memory_info,
            **storage_info,
            **network_info,
            'docker': docker_info,
            'timestamp': time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting hardware specifications: {e}")
        return {}


def create_ssh_client_from_miner_data(miner_data: Dict[str, Any], 
                                    connection_timeout: int = 30,
                                    command_timeout: int = 60) -> Optional[SSHClient]:
    """
    Create an SSH client from miner registration data.
    
    Args:
        miner_data: Miner data from registration
        connection_timeout: SSH connection timeout in seconds
        command_timeout: SSH command timeout in seconds
    
    Returns:
        SSHClient instance if successful, None otherwise
    """
    try:
        # Extract SSH connection details from miner data
        ssh_info = miner_data.get('ssh', {})
        
        if not ssh_info:
            logger.error(f"No SSH information found for miner {miner_data.get('id', 'unknown')}")
            return None
        
        host = ssh_info.get('host')
        port = int(ssh_info.get('port', 22))
        username = ssh_info.get('username')
        
        # Get authentication method (password or key)
        password = ssh_info.get('password')
        key_path = ssh_info.get('key_path')
        key_passphrase = ssh_info.get('key_passphrase')
        
        if not host or not username:
            logger.error(f"Missing required SSH connection info for miner {miner_data.get('id', 'unknown')}")
            return None
        
        if not password and not key_path:
            logger.error(f"No authentication method provided for miner {miner_data.get('id', 'unknown')}")
            return None
        
        # Create SSH client
        ssh_client = SSHClient(
            host=host,
            username=username,
            port=port,
            password=password,
            key_path=key_path,
            key_passphrase=key_passphrase,
            connection_timeout=connection_timeout,
            command_timeout=command_timeout
        )
        
        return ssh_client
        
    except Exception as e:
        logger.error(f"Error creating SSH client: {e}")
        return None 