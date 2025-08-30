import getpass
import logging
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import paramiko
import sshtunnel

# Add the parent directory to sys.path to enable imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import config
import utils

logger = logging.getLogger(__name__)

class SSHManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.is_windows = platform.system().lower() == 'windows'
        self.is_macos = platform.system().lower() == 'darwin'
        # Check if we're running as root
        self.is_root = os.geteuid() == 0 if not self.is_windows else False
        self.logger.info(f"Running as root: {self.is_root}")
        
        # Check if common service management commands are available
        if not self.is_windows:
            # Check for service command
            try:
                subprocess.run(['which', 'service'], check=True, capture_output=True)
                self.has_service_cmd = True
                self.logger.info("Service command is available")
            except subprocess.CalledProcessError:
                self.has_service_cmd = False
            
            # Check for systemd
            try:
                subprocess.run(['which', 'systemctl'], check=True, capture_output=True)
                self.has_systemd = True
                self.logger.info("Systemctl command is available")
            except subprocess.CalledProcessError:
                self.has_systemd = False
                
            if not self.has_service_cmd and not self.has_systemd:
                self.logger.info("Neither service nor systemctl available, will use direct commands")
        else:
            self.has_service_cmd = False
            self.has_systemd = False
        
    def _check_linux_ssh_installed(self):
        """Check if OpenSSH is already installed on Linux"""
        try:
            result = subprocess.run(
                ['dpkg', '-s', 'openssh-server'],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception:
            return False
            
    def _check_macos_ssh_installed(self):
        """Check if OpenSSH is already installed on macOS"""
        try:
            # On macOS, SSH is pre-installed, just check if the service file exists
            result = subprocess.run(
                ['ls', '/System/Library/LaunchDaemons/ssh.plist'],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception:
            return False

    def _install_linux_ssh(self):
        """Install OpenSSH on Linux with error handling"""
        try:
            # Try installing without update first
            self.logger.info("Attempting to install OpenSSH server...")
            subprocess.run(
                ['sudo', 'apt-get', 'install', '-y', 'openssh-server'],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError:
            self.logger.warning("Direct installation failed, attempting with apt update...")
            try:
                # If that fails, try updating apt (but handle errors gracefully)
                update_result = subprocess.run(
                    ['sudo', 'apt-get', 'update'],
                    capture_output=True,
                    text=True
                )
                if update_result.returncode != 0:
                    self.logger.warning(f"Apt update had issues but continuing: {update_result.stderr}")
                
                # Attempt installation again
                subprocess.run(
                    ['sudo', 'apt-get', 'install', '-y', 'openssh-server'],
                    check=True,
                    capture_output=True,
                    text=True
                )
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Failed to install OpenSSH: {e}")
                raise RuntimeError("Failed to install OpenSSH server") from e
                
    def _install_macos_ssh(self):
        """Setup OpenSSH on macOS with error handling"""
        try:
            # On macOS, SSH is pre-installed, just need to ensure the service is enabled
            self.logger.info("Enabling SSH service on macOS...")
            subprocess.run(
                ['sudo', 'launchctl', 'load', '-w', '/System/Library/LaunchDaemons/ssh.plist'],
                check=True,
                capture_output=True,
                text=True
            )
            # Create SSH directory if it doesn't exist
            subprocess.run(['sudo', 'mkdir', '-p', '/etc/ssh'], check=True)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to enable SSH service on macOS: {e}")
            raise RuntimeError("Failed to setup SSH server on macOS") from e

    def create_sshd_config(self, port):
        if self.is_windows:
            config_content = f"""
# SSH Server Configuration
Port {port}
PermitRootLogin yes
AuthorizedKeysFile .ssh/authorized_keys
PasswordAuthentication yes
PermitEmptyPasswords no
ChallengeResponseAuthentication no
UsePAM yes
Subsystem sftp sftp-server.exe
"""
        elif self.is_macos:
            config_content = f"""
# SSH Server Configuration
Port {port}
PermitRootLogin yes
AuthorizedKeysFile .ssh/authorized_keys
PasswordAuthentication yes
PermitEmptyPasswords no
ChallengeResponseAuthentication no
UsePAM yes
Subsystem sftp /usr/libexec/sftp-server
"""
        else:
            config_content = f"""
# SSH Server Configuration
Port {port}
PermitRootLogin yes
AuthorizedKeysFile .ssh/authorized_keys
PasswordAuthentication yes
PermitEmptyPasswords no
ChallengeResponseAuthentication no
UsePAM yes
Subsystem sftp /usr/lib/openssh/sftp-server
"""
        try:
            # Write to temp file first
            temp_config = config.HOME_DIR / 'sshd_config_temp'
            with open(temp_config, 'w', encoding='utf-8') as f:
                f.write(config_content.strip())
            
            if self.is_windows:
                utils.run_elevated(f'copy /Y "{temp_config}" "{config.SSH_CONFIG_PATH}"')
            else:
                subprocess.run(['sudo', 'cp', str(temp_config), '/etc/ssh/sshd_config'], check=True)
                subprocess.run(['sudo', 'chmod', '644', '/etc/ssh/sshd_config'], check=True)
            
            temp_config.unlink(missing_ok=True)
        except Exception as e:
            self.logger.error(f"Failed to create SSH config: {e}")
            raise

    def setup_server(self, port):
        """Setup SSH server with the specified port"""
        try:
            if self.is_windows:
                utils.run_elevated('powershell -Command "Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0"')
                utils.run_elevated('mkdir "C:\\ProgramData\\ssh" 2>NUL')
            elif self.is_macos:
                if not self._check_macos_ssh_installed():
                    self._install_macos_ssh()
                subprocess.run(['sudo', 'mkdir', '-p', '/etc/ssh'], check=True)
            else:
                if not self._check_linux_ssh_installed():
                    self._install_linux_ssh()
                subprocess.run(['sudo', 'mkdir', '-p', '/etc/ssh'], check=True)
            
            # Stop the service
            self.stop_server()
            
            # Create and copy new config
            self.create_sshd_config(port)
            
            # Start the service
            self.start_server()
            
            if self.is_windows:
                utils.run_elevated('powershell -Command "Set-Service -Name sshd -StartupType Automatic"')
            elif self.is_macos:
                # On macOS, we already enabled the service with launchctl load -w
                pass
            else:
                # Configure to start on boot based on available commands
                if self.has_service_cmd and Path('/etc/init.d').exists():
                    cmd = ['update-rc.d', 'ssh', 'defaults']
                    if not self.is_root:
                        cmd.insert(0, 'sudo')
                    try:
                        subprocess.run(cmd, check=True)
                        self.logger.info("SSH service configured to start on boot using update-rc.d")
                        return
                    except subprocess.CalledProcessError:
                        self.logger.warning("Could not configure SSH to start on boot with update-rc.d")
                
                if self.has_systemd:
                    cmd = ['systemctl', 'enable', 'ssh']
                    if not self.is_root:
                        cmd.insert(0, 'sudo')
                    try:
                        subprocess.run(cmd, check=True)
                        self.logger.info("SSH service enabled to start on boot using systemctl")
                        return
                    except subprocess.CalledProcessError as e:
                        self.logger.warning(f"Could not enable SSH with systemctl: {e}")
                
                # Alternative to enable SSH at startup using rc.local
                rc_local = Path('/etc/rc.local')
                if rc_local.exists():
                    content = rc_local.read_text()
                    if '/usr/sbin/sshd' not in content:
                        try:
                            if self.is_root:
                                with open('/etc/rc.local', 'a') as f:
                                    f.write('\n# Start SSH server\n/usr/sbin/sshd\n')
                            else:
                                with open('/tmp/rc_local_append', 'w') as f:
                                    f.write('\n# Start SSH server\n/usr/sbin/sshd\n')
                                subprocess.run(['sudo', 'bash', '-c', 'cat /tmp/rc_local_append >> /etc/rc.local'], check=True)
                                subprocess.run(['rm', '/tmp/rc_local_append'], check=True)
                            
                            cmd = ['chmod', '+x', '/etc/rc.local']
                            if not self.is_root:
                                cmd.insert(0, 'sudo')
                            subprocess.run(cmd, check=True)
                            self.logger.info("Added SSH server startup to /etc/rc.local")
                        except Exception as e:
                            self.logger.warning(f"Could not update rc.local: {e}")
                
        except Exception as e:
            self.logger.error(f"Failed to setup SSH server: {e}")
            raise

    def setup_user(self):
        username = getpass.getuser()
        password = config.SSH_PASSWORD
        
        self.logger.info(f"Configuring user {username} for SSH access...")
        
        try:
            if self.is_windows:
                enable_cmd = f'wmic UserAccount where Name="{username}" set PasswordExpires=false'
                utils.run_elevated(enable_cmd)
                
                commands = [
                    f'net user {username} "{password}"',
                    f'powershell -Command "$password = ConvertTo-SecureString \'{password}\' -AsPlainText -Force; Set-LocalUser -Name \'{username}\' -Password $password"',
                    f'net user {username} /active:yes',
                    'powershell -Command "Set-ItemProperty -Path HKLM:\\SOFTWARE\\OpenSSH -Name DefaultShell -Value C:\\Windows\\System32\\cmd.exe -Force"'
                ]
                
                for cmd in commands:
                    utils.run_elevated(cmd)
                    time.sleep(1)
            else:
                # Set password using chpasswd for Linux or dscl for macOS
                if self.is_macos:
                    # On macOS, use dscl to set password
                    try:
                        # The dscl command requires the password as the last parameter, not passing it with path
                        dscl_cmd = ['sudo', 'dscl', '.', '-passwd', f'/Users/{username}', password]
                        self.logger.debug(f"Running dscl command: {dscl_cmd}")
                        subprocess.run(
                            dscl_cmd,
                            check=True,
                            capture_output=True,
                            text=True
                        )
                    except subprocess.CalledProcessError as e:
                        self.logger.warning(f"Failed to set password with dscl: {e}. Trying with passwd command.")
                        # Alternative approach using the passwd command
                        passwd_proc = subprocess.Popen(
                            ['sudo', 'passwd', username],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True
                        )
                        # Pass the password twice (new password and confirmation)
                        passwd_proc.communicate(input=f"{password}\n{password}\n")
                else:
                    # On Linux, use chpasswd
                    chpasswd_proc = subprocess.Popen(
                        ['sudo', 'chpasswd'],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    chpasswd_proc.communicate(input=f'{username}:{password}\n')
                
                # Ensure .ssh directory exists with correct permissions
                ssh_dir = Path.home() / '.ssh'
                ssh_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
                
                # Set ownership
                if self.is_macos:
                    # On macOS, we need to use the 'staff' group or get the user's primary group
                    try:
                        # Get the user's primary group
                        group_result = subprocess.run(
                            ['id', '-gn', username],
                            capture_output=True,
                            text=True
                        )
                        group_name = group_result.stdout.strip()
                        if not group_name:
                            group_name = 'staff'  # Default to 'staff' if we can't get the group
                        
                        subprocess.run(['sudo', 'chown', '-R', f'{username}:{group_name}', str(ssh_dir)], check=True)
                    except Exception as e:
                        self.logger.warning(f"Failed to set ownership with primary group: {e}. Trying with 'staff' group.")
                        subprocess.run(['sudo', 'chown', '-R', f'{username}:staff', str(ssh_dir)], check=True)
                else:
                    # On Linux
                    subprocess.run(['sudo', 'chown', '-R', f'{username}:{username}', str(ssh_dir)], check=True)
                cmd = ['chown', '-R', f'{username}:{username}', str(ssh_dir)]
                if not self.is_root:
                    cmd.insert(0, 'sudo')
                subprocess.run(cmd, check=True)
            
            self.logger.info("User configured successfully")
            return username, password
            
        except Exception as e:
            self.logger.error(f"Failed to configure user: {e}")
            raise

    def stop_server(self):
        """Stop the SSH server"""
        try:
            if self.is_windows:
                stop_cmd = subprocess.Popen(
                    ['net', 'stop', 'sshd'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stop_cmd.communicate(input='y\n')
            elif self.is_macos:
                subprocess.run(['sudo', 'launchctl', 'unload', '/System/Library/LaunchDaemons/ssh.plist'], check=True)
            else:
                # Try service command first
                if self.has_service_cmd:
                    cmd = ['service', 'ssh', 'stop']
                    if not self.is_root:
                        cmd.insert(0, 'sudo')
                    try:
                        subprocess.run(cmd, check=True)
                        self.logger.info("SSH server stopped using service command")
                        return
                    except subprocess.CalledProcessError as e:
                        self.logger.warning(f"Failed to stop SSH with service command: {e}")
                
                # Try systemctl if service failed or not available
                if self.has_systemd:
                    cmd = ['systemctl', 'stop', 'ssh']
                    if not self.is_root:
                        cmd.insert(0, 'sudo')
                    try:
                        subprocess.run(cmd, check=True)
                        self.logger.info("SSH server stopped using systemctl")
                        return
                    except subprocess.CalledProcessError as e:
                        self.logger.warning(f"Failed to stop SSH with systemctl: {e}")
                
                # Fallback to killing the process
                try:
                    # Get the PID of the sshd process
                    ps_result = subprocess.run(['ps', '-ef', '|', 'grep', 'sshd', '|', 'grep', '-v', 'grep'], 
                                              shell=True, capture_output=True, text=True)
                    if ps_result.stdout:
                        # Extract the PID and kill the process
                        lines = ps_result.stdout.strip().split('\n')
                        for line in lines:
                            parts = line.split()
                            if len(parts) > 1:
                                pid = parts[1]
                                cmd = ['kill', pid]
                                if not self.is_root:
                                    cmd.insert(0, 'sudo')
                                subprocess.run(cmd, check=True)
                                self.logger.info(f"Killed SSH process with PID {pid}")
                except subprocess.CalledProcessError as e:
                    self.logger.warning(f"Could not kill SSH process: {e}")
            time.sleep(2)
        except subprocess.CalledProcessError:
            self.logger.warning("SSH service was not running or could not be stopped")

    def start_server(self):
        """Start the SSH server"""
        try:
            if self.is_windows:
                start_cmd = subprocess.Popen(
                    ['net', 'start', 'sshd'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                start_cmd.communicate(input='y\n')
            elif self.is_macos:
                subprocess.run(['sudo', 'launchctl', 'load', '-w', '/System/Library/LaunchDaemons/ssh.plist'], check=True)
            else:
                # Try service command first
                if self.has_service_cmd:
                    cmd = ['service', 'ssh', 'start']
                    if not self.is_root:
                        cmd.insert(0, 'sudo')
                    try:
                        subprocess.run(cmd, check=True)
                        self.logger.info("SSH server started using service command")
                        return
                    except subprocess.CalledProcessError as e:
                        self.logger.warning(f"Failed to start SSH with service command: {e}")
                
                # Try systemctl if service failed or not available
                if self.has_systemd:
                    cmd = ['systemctl', 'start', 'ssh']
                    if not self.is_root:
                        cmd.insert(0, 'sudo')
                    try:
                        subprocess.run(cmd, check=True)
                        self.logger.info("SSH server started using systemctl")
                        return
                    except subprocess.CalledProcessError as e:
                        self.logger.warning(f"Failed to start SSH with systemctl: {e}")
                
                # Direct startup as last resort
                cmd = ['/usr/sbin/sshd']
                if not self.is_root:
                    cmd.insert(0, 'sudo')
                subprocess.run(cmd, check=True)
                self.logger.info("SSH server started directly using /usr/sbin/sshd")
            time.sleep(2)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to start SSH server: {e}")
            raise