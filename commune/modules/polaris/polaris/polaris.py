#!/usr/bin/env python3
"""
Polaris Interface Class

A unified interface for the Polaris compute subnet system that handles:
- User registration and authentication
- Network synchronization and monitoring
- SSH server configuration and management
- System information collection
- Validator and miner operations
- Process management and monitoring
"""

import os
import sys
import json
import time
import logging
import threading
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'validator/src'))

# Import Polaris components
try:
    from src.user_manager import UserManager
    from src.network_sync import NetworkSync
    from src.ssh_manager import SSHManager
    from src.config import (
        ORCHESTRATOR_URL, SSH_PASSWORD, USE_SSH_KEY,
        NGROK_CONFIG_PATH, LOG_DIR, SSH_CONFIG_PATH
    )
    from src.pid_manager import (
        create_pid_file, remove_pid_file, read_pid,
        create_pid_file_for_process, remove_pid_file_for_process
    )
    from src.main import (
        configure_ssh_server, setup_firewall, get_system_info,
        save_system_info, sync_system_info, add_server_public_key,
        verify_user_registration, start_heartbeat, setup_cloud_logging
    )
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    # Continue with limited functionality

# Import validator components if available
try:
    from validator.src.config import ValidatorConfig, get_config
    from validator.src.main import main as validator_main
except ImportError:
    ValidatorConfig = None
    validator_main = None


class Polaris:
    """
    Main interface class for the Polaris compute subnet system.
    
    This class provides a unified API for all Polaris functionality including:
    - Miner and validator operations
    - User registration and authentication
    - Network management and synchronization
    - SSH server configuration
    - System monitoring and logging
    """
    
    def __init__(self, config_path: Optional[str] = None, mode: str = 'miner'):
        """
        Initialize the Polaris interface.
        
        Args:
            config_path: Optional path to configuration file
            mode: Operating mode ('miner' or 'validator')
        """
        self.mode = mode.lower()
        self.config_path = config_path or os.path.expanduser('~/.polaris/config.json')
        self.logger = self._setup_logging()
        
        # Initialize components
        self.user_manager = None
        self.network_sync = None
        self.ssh_manager = None
        self.validator_config = None
        
        # State tracking
        self.is_running = False
        self.heartbeat_thread = None
        self.monitor_thread = None
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components based on mode
        self._initialize_components()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        os.makedirs(LOG_DIR, exist_ok=True)
        
        logger = logging.getLogger('Polaris')
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(
            os.path.join(LOG_DIR, f'polaris_{self.mode}.log')
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load config: {e}")
        
        # Default configuration
        default_config = {
            'mode': self.mode,
            'orchestrator_url': ORCHESTRATOR_URL,
            'ssh_port': 22,
            'use_ssh_key': USE_SSH_KEY,
            'enable_heartbeat': True,
            'heartbeat_interval': 30,
            'enable_monitoring': True,
            'monitor_interval': 60,
            'networks': ['bittensor', 'commune'],
            'validator': {
                'validation_interval': 300,
                'container_timeout': 600,
                'hardware_timeout': 30
            }
        }
        
        # Save default config
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
        
    def _initialize_components(self):
        """Initialize Polaris components based on mode."""
        try:
            # Common components for both miner and validator
            self.user_manager = UserManager()
            self.ssh_manager = SSHManager()
            
            if self.mode == 'miner':
                self.network_sync = NetworkSync()
            elif self.mode == 'validator' and ValidatorConfig:
                self.validator_config = get_config() if not self.config_path else ValidatorConfig()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            
    def setup(self, ssh_port: Optional[int] = None, force: bool = False) -> bool:
        """
        Set up the Polaris system.
        
        Args:
            ssh_port: SSH port to use (default from config)
            force: Force setup even if already configured
            
        Returns:
            bool: True if setup successful
        """
        self.logger.info(f"Setting up Polaris in {self.mode} mode...")
        
        try:
            # Check if already set up
            if not force and os.path.exists(os.path.join(LOG_DIR, '.setup_complete')):
                self.logger.info("Polaris already set up. Use force=True to reconfigure.")
                return True
                
            # Set up SSH server for miners
            if self.mode == 'miner':
                ssh_port = ssh_port or self.config.get('ssh_port', 22)
                
                self.logger.info("Configuring SSH server...")
                if not configure_ssh_server(ssh_port):
                    self.logger.error("Failed to configure SSH server")
                    return False
                    
                self.logger.info("Setting up firewall...")
                setup_firewall()
                
            # Create necessary directories
            self._setup_directories()
            
            # Mark setup as complete
            Path(os.path.join(LOG_DIR, '.setup_complete')).touch()
            
            self.logger.info("Setup completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Setup failed: {e}")
            return False
            
    def _setup_directories(self):
        """Create necessary directories."""
        directories = [
            os.path.expanduser('~/.polaris'),
            os.path.expanduser('~/.polaris/logs'),
            os.path.expanduser('~/.polaris/data'),
            os.path.expanduser('~/.polaris/pid'),
            LOG_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    def register(self, wallet_name: Optional[str] = None, 
                 wallet_path: Optional[str] = None,
                 network: str = 'bittensor') -> bool:
        """
        Register with the Polaris network.
        
        Args:
            wallet_name: Name of the wallet to use
            wallet_path: Path to wallet directory
            network: Network to register on ('bittensor', 'commune', or 'independent')
            
        Returns:
            bool: True if registration successful
        """
        self.logger.info(f"Registering on {network} network...")
        
        try:
            # Verify user registration first
            if not verify_user_registration():
                self.logger.error("User not registered with Polaris")
                return False
                
            # Handle network-specific registration
            if network == 'bittensor':
                # TODO: Implement Bittensor registration
                self.logger.info("Bittensor registration not yet implemented")
                return False
            elif network == 'commune':
                # TODO: Implement Commune registration
                self.logger.info("Commune registration not yet implemented")
                return False
            elif network == 'independent':
                # Independent miners just need to be registered with Polaris
                return True
            else:
                self.logger.error(f"Unknown network: {network}")
                return False
                
        except Exception as e:
            self.logger.error(f"Registration failed: {e}")
            return False
            
    def start(self, daemon: bool = False) -> bool:
        """
        Start Polaris services.
        
        Args:
            daemon: Run in background as daemon
            
        Returns:
            bool: True if started successfully
        """
        if self.is_running:
            self.logger.warning("Polaris is already running")
            return True
            
        self.logger.info(f"Starting Polaris {self.mode}...")
        
        try:
            # Create PID file
            create_pid_file()
            
            # Start mode-specific services
            if self.mode == 'miner':
                success = self._start_miner(daemon)
            elif self.mode == 'validator':
                success = self._start_validator(daemon)
            else:
                self.logger.error(f"Unknown mode: {self.mode}")
                return False
                
            if success:
                self.is_running = True
                self.logger.info("Polaris started successfully")
            else:
                remove_pid_file()
                
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to start: {e}")
            remove_pid_file()
            return False
            
    def _start_miner(self, daemon: bool) -> bool:
        """Start miner services."""
        try:
            # Start SSH server
            if self.ssh_manager:
                self.logger.info("Starting SSH server...")
                self.ssh_manager.start_server()
                
            # Sync system information
            self.logger.info("Syncing system information...")
            system_info = get_system_info()
            save_system_info(system_info)
            sync_system_info(system_info)
            
            # Start heartbeat
            if self.config.get('enable_heartbeat', True):
                self.logger.info("Starting heartbeat...")
                self.heartbeat_thread = start_heartbeat(
                    self.config.get('heartbeat_interval', 30)
                )
                
            # Start monitoring
            if self.config.get('enable_monitoring', True):
                self.logger.info("Starting monitoring...")
                self._start_monitoring()
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start miner: {e}")
            return False
            
    def _start_validator(self, daemon: bool) -> bool:
        """Start validator services."""
        try:
            if not validator_main:
                self.logger.error("Validator module not available")
                return False
                
            # Run validator in a separate thread if daemon mode
            if daemon:
                validator_thread = threading.Thread(
                    target=validator_main,
                    daemon=True
                )
                validator_thread.start()
            else:
                validator_main()
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start validator: {e}")
            return False
            
    def _start_monitoring(self):
        """Start system monitoring."""
        def monitor_loop():
            while self.is_running:
                try:
                    # Update system info
                    system_info = get_system_info()
                    save_system_info(system_info)
                    
                    # Sync with orchestrator
                    if self.network_sync:
                        self.network_sync.update_miner_network_info()
                        
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
                    
                time.sleep(self.config.get('monitor_interval', 60))
                
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop(self) -> bool:
        """
        Stop Polaris services.
        
        Returns:
            bool: True if stopped successfully
        """
        if not self.is_running:
            self.logger.warning("Polaris is not running")
            return True
            
        self.logger.info("Stopping Polaris...")
        
        try:
            self.is_running = False
            
            # Stop SSH server
            if self.mode == 'miner' and self.ssh_manager:
                self.ssh_manager.stop_server()
                
            # Wait for threads to finish
            if self.heartbeat_thread and self.heartbeat_thread.is_alive():
                self.heartbeat_thread.join(timeout=5)
                
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
                
            # Remove PID file
            remove_pid_file()
            
            self.logger.info("Polaris stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop: {e}")
            return False
            
    def status(self) -> Dict[str, Any]:
        """
        Get current status of Polaris services.
        
        Returns:
            dict: Status information
        """
        status = {
            'mode': self.mode,
            'running': self.is_running,
            'pid': os.getpid() if self.is_running else None,
            'uptime': None,
            'services': {}
        }
        
        try:
            # Check PID file
            pid = read_pid()
            if pid:
                status['pid'] = pid
                # TODO: Calculate uptime from PID file creation time
                
            # Check SSH server status (miner only)
            if self.mode == 'miner' and self.ssh_manager:
                # TODO: Implement SSH server status check
                status['services']['ssh'] = 'unknown'
                
            # Check heartbeat status
            if self.heartbeat_thread:
                status['services']['heartbeat'] = (
                    'running' if self.heartbeat_thread.is_alive() else 'stopped'
                )
                
            # Check monitoring status
            if self.monitor_thread:
                status['services']['monitoring'] = (
                    'running' if self.monitor_thread.is_alive() else 'stopped'
                )
                
            # Get system info
            if os.path.exists(os.path.expanduser('~/.polaris/data/system_info.json')):
                with open(os.path.expanduser('~/.polaris/data/system_info.json'), 'r') as f:
                    status['system_info'] = json.load(f)
                    
        except Exception as e:
            self.logger.error(f"Error getting status: {e}")
            status['error'] = str(e)
            
        return status
        
    def get_logs(self, lines: int = 100, follow: bool = False) -> Union[str, None]:
        """
        Get Polaris logs.
        
        Args:
            lines: Number of lines to retrieve
            follow: Follow log output (like tail -f)
            
        Returns:
            str: Log content or None if error
        """
        log_file = os.path.join(LOG_DIR, f'polaris_{self.mode}.log')
        
        try:
            if not os.path.exists(log_file):
                return "No log file found"
                
            if follow:
                # TODO: Implement log following
                return "Log following not yet implemented"
            else:
                with open(log_file, 'r') as f:
                    log_lines = f.readlines()
                    return ''.join(log_lines[-lines:])
                    
        except Exception as e:
            self.logger.error(f"Error reading logs: {e}")
            return None
            
    def update(self) -> bool:
        """
        Update Polaris to the latest version.
        
        Returns:
            bool: True if update successful
        """
        self.logger.info("Updating Polaris...")
        
        try:
            # TODO: Implement update logic
            # This would typically:
            # 1. Check for updates from git repository
            # 2. Download and apply updates
            # 3. Restart services if needed
            
            self.logger.warning("Update functionality not yet implemented")
            return False
            
        except Exception as e:
            self.logger.error(f"Update failed: {e}")
            return False
            
    def add_ssh_key(self, public_key: str) -> bool:
        """
        Add an SSH public key for authentication.
        
        Args:
            public_key: SSH public key string
            
        Returns:
            bool: True if key added successfully
        """
        if self.mode != 'miner':
            self.logger.error("SSH keys are only used in miner mode")
            return False
            
        try:
            return add_server_public_key(public_key)
        except Exception as e:
            self.logger.error(f"Failed to add SSH key: {e}")
            return False
            
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get current system information.
        
        Returns:
            dict: System information
        """
        try:
            return get_system_info()
        except Exception as e:
            self.logger.error(f"Failed to get system info: {e}")
            return {}
            
    def sync_network_info(self) -> bool:
        """
        Manually sync network information with orchestrator.
        
        Returns:
            bool: True if sync successful
        """
        if self.mode != 'miner':
            self.logger.error("Network sync is only available in miner mode")
            return False
            
        try:
            if self.network_sync:
                self.network_sync.update_miner_network_info()
                return True
            else:
                self.logger.error("Network sync not initialized")
                return False
        except Exception as e:
            self.logger.error(f"Network sync failed: {e}")
            return False
            
    def configure_validator(self, **kwargs) -> bool:
        """
        Configure validator settings.
        
        Args:
            **kwargs: Validator configuration options
            
        Returns:
            bool: True if configuration successful
        """
        if self.mode != 'validator':
            self.logger.error("This method is only available in validator mode")
            return False
            
        try:
            # Update validator config
            if 'validation_interval' in kwargs:
                self.config['validator']['validation_interval'] = kwargs['validation_interval']
            if 'container_timeout' in kwargs:
                self.config['validator']['container_timeout'] = kwargs['container_timeout']
            if 'hardware_timeout' in kwargs:
                self.config['validator']['hardware_timeout'] = kwargs['hardware_timeout']
                
            # Save updated config
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to configure validator: {e}")
            return False
            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.is_running:
            self.stop()
            

    # Convenience functions for quick usage
    @staticmethod
    def create_miner(config_path: Optional[str] = None) -> Polaris:
        """Create a Polaris miner instance."""
        return Polaris(config_path=config_path, mode='miner')
        
    @staticmethod
    def create_validator(config_path: Optional[str] = None) -> Polaris:
        """Create a Polaris validator instance."""
        return Polaris(config_path=config_path, mode='validator')
    

if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Polaris Interface')
    parser.add_argument('--mode', choices=['miner', 'validator'], 
                        default='miner', help='Operating mode')
    parser.add_argument('--setup', action='store_true', 
                        help='Run setup')
    parser.add_argument('--start', action='store_true', 
                        help='Start services')
    parser.add_argument('--stop', action='store_true', 
                        help='Stop services')
    parser.add_argument('--status', action='store_true', 
                        help='Show status')
    parser.add_argument('--daemon', action='store_true',
                        help='Run in daemon mode')
    
    args = parser.parse_args()
    
    # Create Polaris instance
    polaris = Polaris(mode=args.mode)
    
    # Execute requested action
    if args.setup:
        polaris.setup()
    elif args.start:
        polaris.start(daemon=args.daemon)
        if not args.daemon:
            # Keep running until interrupted
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                polaris.stop()
    elif args.stop:
        polaris.stop()
    elif args.status:
        status = polaris.status()
        print(json.dumps(status, indent=2))
