
import subprocess
import platform
import time
import shutil

def check_and_start_docker():
    """
    Check if Docker is installed and start it if it's not running.
    Returns: tuple (success: bool, message: str)
    """
    
    # Check if Docker is installed
    if not is_docker_installed():
        return False, "Docker is not installed on this system"
    
    # Check if Docker is running
    if is_docker_running():
        return True, "Docker is already running"
    
    # Try to start Docker
    success = start_docker()
    
    if success:
        # Wait a bit for Docker to fully start
        print("Waiting for Docker to start...")
        for i in range(30):  # Wait up to 30 seconds
            if is_docker_running():
                return True, "Docker started successfully"
            time.sleep(1)
        return False, "Docker started but may not be fully ready"
    else:
        return False, "Failed to start Docker"

def is_docker_installed():
    """Check if Docker is installed."""
    return shutil.which('docker') is not None

def is_docker_running():
    """Check if Docker daemon is running."""
    try:
        result = subprocess.run(
            ['docker', 'info'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def start_docker():
    """Start Docker daemon based on the operating system."""
    system = platform.system()
    
    try:
        if system == "Darwin":  # macOS
            # Try to start Docker Desktop
            subprocess.run(['open', '-a', 'Docker'], check=True)
            return True
            
        elif system == "Linux":
            # Try systemctl first (systemd-based systems)
            try:
                subprocess.run(['sudo', 'systemctl', 'start', 'docker'], check=True)
                return True
            except subprocess.CalledProcessError:
                # Try service command (older systems)
                try:
                    subprocess.run(['sudo', 'service', 'docker', 'start'], check=True)
                    return True
                except subprocess.CalledProcessError:
                    return False
                    
        elif system == "Windows":
            # Try to start Docker Desktop on Windows
            try:
                # Try to start Docker Desktop
                subprocess.run(['powershell', '-Command', 
                              'Start-Process "C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe"'], 
                              check=True)
                return True
            except subprocess.CalledProcessError:
                # Alternative method
                try:
                    subprocess.run(['cmd', '/c', 'start', '', 
                                  'C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe'], 
                                  check=True)
                    return True
                except subprocess.CalledProcessError:
                    return False
                    
    except Exception as e:
        print(f"Error starting Docker: {e}")
        return False