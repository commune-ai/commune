import argparse
import json
import logging
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# Add the parent directory to sys.path to enable imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import requests
from cloud_logging import CloudLogger
# Use direct imports
from pid_manager import PID_FILE, create_pid_file, remove_pid_file
from ssh_manager import SSHManager
from sync_manager import SyncManager
from system_info import get_system_info
from user_manager import UserManager

from utils import configure_logging, get_local_ip, get_project_root

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Password is optional, if not provided we'll use public key authentication
if "SSH_PASSWORD" not in os.environ:
    logger.warning("SSH_PASSWORD environment variable not set. Using public key authentication.")

user_manager = UserManager()

def is_admin():
    """Check if the script is running with administrative privileges."""
    if platform.system().lower() == 'windows':
        try:
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except AttributeError:
            return False
    else:
        # For Linux/Unix systems
        return os.geteuid() == 0

def run_as_admin():
    """
    Relaunch the current script with administrative privileges.
    """
    if platform.system().lower() == 'windows':
        try:
            script_path = os.path.abspath(__file__)
            params = f'"{script_path}"'
            
            import ctypes
            ctypes.windll.shell32.ShellExecuteW(
                None,
                "runas",
                sys.executable,
                params,
                None,
                1
            )
            logger.info("Script relaunched with administrative privileges.")
            return True
        except Exception as e:
            logger.exception(f"Failed to get admin rights: {e}")
            return False
    else:
        # For Linux/Unix systems
        if os.geteuid() != 0:
            try:
                # Get password from environment variable
                password = os.getenv('SSH_PASSWORD')
                if password:
                    logger.info("Restarting with sudo using SSH_PASSWORD...")
                    
                    # Construct the sudo command
                    cmd = ['sudo', '-S'] + [sys.executable] + sys.argv
                    
                    # Use subprocess to pipe the password
                    process = subprocess.Popen(
                        cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True
                    )
                    
                    # Send password to stdin
                    stdout, stderr = process.communicate(input=password + '\n')
                    
                    if process.returncode != 0:
                        logger.error(f"Sudo failed: {stderr}")
                        return False
                    return True
            
                logger.error("SSH_PASSWORD environment variable not set")
                return False
                
            except Exception as e:
                logger.exception(f"Failed to restart with sudo: {e}")
                return False
        return True

def configure_ssh():
    """Configure SSH server for the current platform."""
    system = platform.system().lower()
    if system == 'windows':
        return configure_ssh_windows()
    elif system == 'darwin':  # macOS
        return configure_ssh_macos()
    else:
        return configure_ssh_linux()

def configure_ssh_linux():
    """Configure SSH server on Linux."""
    try:
        # Check if running as root
        is_root = os.geteuid() == 0
        logger.info(f"Running as root: {is_root}")
        
        # Check if SSH server is installed and running
        logger.info("Checking SSH server status...")
        
        # Try service command first
        try:
            cmd = ['service', 'ssh', 'status']
            if not is_root:
                cmd.insert(0, 'sudo')
            result = subprocess.run(cmd, capture_output=True, text=True)
            using_service = True
            using_systemd = False
            logger.info("Using service command for SSH management")
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Try systemctl as fallback
            try:
                cmd = ['systemctl', 'status', 'ssh']
                if not is_root:
                    cmd.insert(0, 'sudo')
                result = subprocess.run(cmd, capture_output=True, text=True)
                using_service = False
                using_systemd = True
                logger.info("Using systemctl for SSH management")
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Final fallback - check if sshd process is running
                result = subprocess.run(['pgrep', 'sshd'], capture_output=True, text=True)
                using_service = False
                using_systemd = False
                logger.info("Neither service nor systemctl available, using direct commands")
                
                if result.returncode != 0:
                    # Assume SSH is not installed or not running
                    logger.info("SSH server appears to be not running or not installed.")
                    # Install SSH server if not present
                    logger.info("Installing SSH server...")
                    try:
                        cmd_update = ['apt-get', 'update']
                        cmd_install = ['apt-get', 'install', '-y', 'openssh-server']
                        if not is_root:
                            cmd_update.insert(0, 'sudo')
                            cmd_install.insert(0, 'sudo')
                        subprocess.run(cmd_update, check=True)
                        subprocess.run(cmd_install, check=True)
                    except subprocess.CalledProcessError:
                        try:
                            # Try with yum for Red Hat-based systems
                            cmd_install = ['yum', 'install', '-y', 'openssh-server']
                            if not is_root:
                                cmd_install.insert(0, 'sudo')
                            subprocess.run(cmd_install, check=True)
                        except subprocess.CalledProcessError as e:
                            logger.error(f"Failed to install SSH server: {e}")
                            return False
        
        # Start and enable SSH service based on available commands
        logger.info("Starting SSH service...")
        if using_service:
            try:
                cmd = ['service', 'ssh', 'start']
                if not is_root:
                    cmd.insert(0, 'sudo')
                subprocess.run(cmd, check=True)
                # For service command, we need to ensure it starts on boot
                # Check for common init systems
                if Path('/etc/init.d').exists():
                    cmd = ['update-rc.d', 'ssh', 'defaults']
                    if not is_root:
                        cmd.insert(0, 'sudo')
                    try:
                        subprocess.run(cmd, check=True)
                        logger.info("SSH service configured to start on boot")
                    except subprocess.CalledProcessError:
                        logger.warning("Could not configure SSH to start on boot with update-rc.d")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to start SSH with service command: {e}")
                using_service = False
        
        if not using_service and using_systemd:
            try:
                cmd_start = ['systemctl', 'start', 'ssh']
                cmd_enable = ['systemctl', 'enable', 'ssh']
                if not is_root:
                    cmd_start.insert(0, 'sudo')
                    cmd_enable.insert(0, 'sudo')
                subprocess.run(cmd_start, check=True)
                subprocess.run(cmd_enable, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to start SSH with systemctl: {e}")
                using_systemd = False
        
        if not using_service and not using_systemd:
            # Direct sshd invocation
            try:
                cmd = ['/usr/sbin/sshd']
                if not is_root:
                    cmd.insert(0, 'sudo')
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to start SSH service using direct command: {e}")
                return False
            
            # For startup on boot without systemd
            # Add to /etc/rc.local if it exists
            rc_local = Path('/etc/rc.local')
            if rc_local.exists():
                content = rc_local.read_text()
                if '/usr/sbin/sshd' not in content:
                    # Need to use sudo to write to rc.local if not root
                    if is_root:
                        with open('/etc/rc.local', 'a') as f:
                            f.write('\n# Start SSH server\n/usr/sbin/sshd\n')
                    else:
                        with open('/tmp/rc_local_append', 'w') as f:
                            f.write('\n# Start SSH server\n/usr/sbin/sshd\n')
                        subprocess.run(['sudo', 'bash', '-c', 'cat /tmp/rc_local_append >> /etc/rc.local'], check=True)
                        subprocess.run(['rm', '/tmp/rc_local_append'], check=True)
                    
                    cmd = ['chmod', '+x', '/etc/rc.local']
                    if not is_root:
                        cmd.insert(0, 'sudo')
                    subprocess.run(cmd, check=True)
        
        logger.info("SSH server configured successfully on Linux.")
        return True
    except Exception as e:
        logger.error(f"Failed to configure SSH on Linux: {e}")
        return False

def configure_ssh_windows():
    """Configure SSH server on Windows."""
    if not create_ssh_directory_windows():
        logger.error("Failed to create SSH directory.")
        return False
    
    commands = [
        'powershell "Get-WindowsCapability -Online | Where-Object Name -like \'OpenSSH.Server*\'"',
        'powershell "Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0"',
        'powershell "Start-Service sshd"',
        'powershell "Set-Service -Name sshd -StartupType Automatic"'
    ]
    
    success = True
    for cmd in commands:
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0 and not "already installed" in result.stdout:
                logger.error(f"Failed to execute SSH command: {cmd}\nError: {result.stderr}")
                success = False
                break
        except Exception as e:
            logger.error(f"Error executing SSH command: {cmd}\nError: {e}")
            success = False
            break

    if success:
        logger.info("SSH server configured successfully.")
        return True
    else:
        logger.error("SSH server configuration failed.")
        return False

def create_ssh_directory_windows():
    """Create SSH directory on Windows with administrative privileges."""
    ssh_dir = r'C:\ProgramData\ssh'
    
    # Try using PowerShell commands first
    try:
        ps_command = f'powershell -Command "New-Item -ItemType Directory -Force -Path \'{ssh_dir}\'"'
        subprocess.run(ps_command, check=True, shell=True, capture_output=True)
        logger.info(f"SSH directory created at {ssh_dir} using PowerShell")
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(f"PowerShell directory creation failed: {e}")
    
    # Try using cmd.exe as fallback
    try:
        cmd_command = f'cmd /c mkdir "{ssh_dir}" 2>nul'
        subprocess.run(cmd_command, check=True, shell=True, capture_output=True)
        logger.info(f"SSH directory created at {ssh_dir} using cmd")
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(f"CMD directory creation failed: {e}")
    
    # Try using os.makedirs as final fallback
    try:
        os.makedirs(ssh_dir, exist_ok=True)
        logger.info(f"SSH directory created at {ssh_dir} using os.makedirs")
        return True
    except Exception as e:
        logger.error(f"Failed to create SSH directory: {e}")
        return False

def configure_ssh_macos():
    """Configure SSH server on macOS."""
    try:
        # Check if SSH server is already running
        result = subprocess.run(['sudo', 'launchctl', 'list', 'com.openssh.sshd'], capture_output=True, text=True)
        
        # Start SSH service if not running
        if result.returncode != 0:
            logger.info("Starting SSH server on macOS...")
            subprocess.run(['sudo', 'launchctl', 'load', '-w', '/System/Library/LaunchDaemons/ssh.plist'], check=True)
        
        # Ensure SSH is enabled in System Preferences (using touch command as trigger)
        subprocess.run(['sudo', 'touch', '/etc/ssh/sshd_config'], check=True)
        
        # Update SSH configuration if needed
        ssh_port = os.environ.get('SSH_PORT', '22')
        ssh_config_path = '/etc/ssh/sshd_config'
        
        # Create backup of original config
        subprocess.run(['sudo', 'cp', ssh_config_path, f'{ssh_config_path}.bak'], check=True)
        
        # Read current config
        result = subprocess.run(['sudo', 'cat', ssh_config_path], capture_output=True, text=True, check=True)
        current_config = result.stdout
        
        # Ensure necessary settings are present
        required_settings = [
            f'Port {ssh_port}',
            'PermitRootLogin yes',
            'PasswordAuthentication yes'
        ]
        
        new_config = current_config
        for setting in required_settings:
            key = setting.split()[0]
            if key in new_config:
                # Replace existing setting
                pattern = rf'{key}\s+\w+'
                new_config = re.sub(pattern, setting, new_config)
            else:
                # Add new setting
                new_config += f'\n{setting}'
        
        # Write updated config
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write(new_config)
            temp_path = temp_file.name
        
        subprocess.run(['sudo', 'cp', temp_path, ssh_config_path], check=True)
        subprocess.run(['sudo', 'chmod', '644', ssh_config_path], check=True)
        os.unlink(temp_path)
        
        # Restart SSH service to apply changes
        subprocess.run(['sudo', 'launchctl', 'unload', '/System/Library/LaunchDaemons/ssh.plist'], check=True)
        subprocess.run(['sudo', 'launchctl', 'load', '-w', '/System/Library/LaunchDaemons/ssh.plist'], check=True)
        
        logger.info("SSH server configured successfully on macOS.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to configure SSH server on macOS: {e}")
        return False
    except Exception as e:
        logger.error(f"Error configuring SSH server on macOS: {e}")
        return False

def setup_firewall():
    """Configure firewall based on platform."""
    system = platform.system().lower()
    if system == 'windows':
        return setup_firewall_windows()
    elif system == 'darwin':  # macOS
        return setup_firewall_macos()
    else:
        return setup_firewall_linux()

def setup_firewall_linux():
    """Configure Linux firewall (ufw) to allow SSH connections."""
    try:
        # Check if running as root
        is_root = os.geteuid() == 0
        
        # Check if ufw is installed
        result = subprocess.run(['which', 'ufw'], capture_output=True)
        if result.returncode != 0:
            logger.info("Installing ufw...")
            cmd_update = ['apt-get', 'update']
            cmd_install = ['apt-get', 'install', '-y', 'ufw']
            
            if not is_root:
                cmd_update.insert(0, 'sudo')
                cmd_install.insert(0, 'sudo')
                
            subprocess.run(cmd_update, check=True)
            subprocess.run(cmd_install, check=True)
        
        # Configure ufw
        cmd_allow = ['ufw', 'allow', 'ssh']
        cmd_enable = ['ufw', '--force', 'enable']
        
        if not is_root:
            cmd_allow.insert(0, 'sudo')
            cmd_enable.insert(0, 'sudo')
            
        subprocess.run(cmd_allow, check=True)
        subprocess.run(cmd_enable, check=True)
        
        logger.info("Linux firewall configured to allow SSH connections.")
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to configure Linux firewall: {e}")
        # Don't fail the entire process if firewall setup fails
        # SSH might still work, especially in container environments
        logger.warning("Could not configure firewall. SSH access might be blocked.")
        return False

def setup_firewall_windows():
    """Configure Windows Firewall to allow SSH connections."""
    firewall_cmd = 'netsh advfirewall firewall add rule name="OpenSSH" dir=in action=allow protocol=TCP localport=22'
    try:
        subprocess.run(firewall_cmd, shell=True, check=True, capture_output=True)
        logger.info("Windows Firewall configured to allow SSH connections on port 22.")
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to configure Windows Firewall: {e}")
        return False
    except Exception as e:
        logger.warning(f"Error configuring firewall: {e}")
        return False

def setup_firewall_macos():
    """Configure macOS firewall to allow SSH connections."""
    try:
        ssh_port = os.environ.get('SSH_PORT', '22')
        
        # Enable macOS firewall
        logger.info("Configuring macOS firewall...")
        
        # Check if firewall is already enabled
        result = subprocess.run(
            ['sudo', '/usr/libexec/ApplicationFirewall/socketfilterfw', '--getglobalstate'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # If firewall is not enabled, enable it
        if "disabled" in result.stdout.lower():
            subprocess.run(
                ['sudo', '/usr/libexec/ApplicationFirewall/socketfilterfw', '--setglobalstate', 'on'],
                check=True
            )
        
        # Allow SSH server
        subprocess.run(
            ['sudo', '/usr/libexec/ApplicationFirewall/socketfilterfw', '--add', '/usr/sbin/sshd'],
            check=True
        )
        
        # Allow incoming connections
        subprocess.run(
            ['sudo', '/usr/libexec/ApplicationFirewall/socketfilterfw', '--unblockapp', '/usr/sbin/sshd'],
            check=True
        )
        
        # Update firewall rules
        subprocess.run(['sudo', 'defaults', 'write', '/Library/Preferences/com.apple.alf', 'globalstate', '-int', '1'], check=True)
        
        # Restart firewall to apply changes
        subprocess.run(['sudo', 'pkill', '-HUP', 'socketfilterfw'], check=True)
        
        logger.info("macOS firewall configured to allow SSH connections.")
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to configure macOS firewall: {e}")
        return False
    except Exception as e:
        logger.warning(f"Error configuring firewall: {e}")
        return False

def format_network_info(network_info: Dict[str, Any]) -> Dict[str, Any]:
    """Format network information for display."""
    return {
        "internal_ip": network_info.get("internal_ip"),
        "ssh": network_info.get("ssh"),
        "open_ports": network_info.get("open_ports", []),
        "username": network_info.get("username"),
        "auth_type": "public_key"  # Default to public key authentication
    }

def save_and_sync_info(system_info, filename='system_info.json'):
    """Save system info and sync network details."""
    try:
        root_dir = get_project_root()
        abs_path = os.path.join(root_dir, filename)
        with open(abs_path, 'w') as f:
            json.dump([system_info], f, indent=4)
        logger.debug(f"System information saved to {abs_path}")

        user_manager = UserManager()
        has_registration, user_info = user_manager.check_existing_registration(show_prompt=False)
        
        if has_registration and user_info:
            sync_manager = SyncManager()
            if sync_manager.sync_network_info():
                logger.info("Network information synchronized successfully")
                
                overall_status, component_status = sync_manager.verify_sync_status()
                if not overall_status:
                    logger.warning("Sync verification failed:")
                    for component, status in component_status.items():
                        logger.warning(f"- {component}: {'Success' if status else 'Failed'}")
            else:
                logger.warning("Failed to synchronize network information")

        return abs_path
    except Exception as e:
        logger.exception(f"Failed to save and sync system info: {e}")
        return None

def handle_shutdown(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info("Received shutdown signal. Cleaning up...")
    sys.exit(0)
    
def main():
    """Main function to start the system service."""
    parser = argparse.ArgumentParser(description='Polaris Cloud System Service')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Starting Polaris Cloud System Service on {platform.system()} {platform.release()}")

    # Generate system information
    logger.info("Generating system information...")
    sys_info = generate_system_info()
    if not sys_info:
        logger.error("Failed to generate system information")
        logger.warning("Continuing without system information - some features may not work correctly")
    else:
        # Save system information to system_info.json
        logger.info("Saving system information to system_info.json...")
        file_path = save_and_sync_info(sys_info)
        if file_path:
            logger.info(f"System information saved to: {file_path}")
        else:
            logger.error("Failed to save system information")
            logger.warning("Continuing without saved system information - some features may not work correctly")

    # Check if user has registered
    has_registered, user_info = user_manager.check_existing_registration()
    
    if not has_registered:
        logger.warning("No registration found. Please register first using 'polaris register'")
        return

    miner_id = user_info.get('miner_id')
    network_info = user_info.get('network_info', {})
    server_public_key = user_info.get('server_public_key')
    
    # If server's public key is provided, add it to authorized_keys
    if server_public_key:
        logger.info("Adding server's public key to authorized_keys file")
        if add_public_key_to_authorized_keys(server_public_key):
            logger.info("Server's public key added successfully")
        else:
            logger.warning("Failed to add server's public key")

    logger.info(f"Registered as miner: {miner_id}")
    
    # Set up cloud logging
    cloud_logger = CloudLogger(miner_id)
    cloud_logger.start()
    
    # Start sending heartbeats in the background
    import threading

    from heartbeat import start_heartbeat
    
    heartbeat_thread = threading.Thread(
        target=start_heartbeat,
        args=(miner_id,),
        daemon=True
    )
    heartbeat_thread.start()
    
    # Keep the main thread alive (and handle graceful shutdown)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        cloud_logger.stop()
        sys.exit(0)

def add_public_key_to_authorized_keys(server_public_key: str) -> bool:
    """Add the server's public key to the authorized_keys file."""
    try:
        # Determine the user's home directory and .ssh directory
        home_dir = os.path.expanduser("~")
        ssh_dir = os.path.join(home_dir, ".ssh")
        
        # Create .ssh directory if it doesn't exist
        if not os.path.exists(ssh_dir):
            os.makedirs(ssh_dir, mode=0o700)
            logger.info("Created .ssh directory")
        
        # Path to authorized_keys file
        auth_keys_path = os.path.join(ssh_dir, "authorized_keys")
        
        # Check if authorized_keys already contains this key
        existing_keys = ""
        if os.path.exists(auth_keys_path):
            with open(auth_keys_path, 'r') as f:
                existing_keys = f.read()
        
        # Avoid duplicates by checking if key already exists
        if server_public_key in existing_keys:
            logger.info("Server public key already exists in authorized_keys")
            return True
        
        # Append the new key
        with open(auth_keys_path, 'a+') as f:
            # Add a newline if the file doesn't end with one
            if existing_keys and not existing_keys.endswith('\n'):
                f.write('\n')
            f.write(f"{server_public_key}\n")
        
        # Set correct permissions
        os.chmod(auth_keys_path, 0o600)
        
        logger.info("Successfully added server public key to authorized_keys")
        return True
    except Exception as e:
        logger.error(f"Failed to add server public key: {str(e)}")
        return False

def generate_system_info():
    """Generate system information using the system_info module."""
    try:
        # We already import get_system_info at the top of the file
        logger.info("Calling get_system_info function...")
        return get_system_info()
    except Exception as e:
        logger.error(f"Error generating system information: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    main()
