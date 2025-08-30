# src/utils.py

import logging
import os
import platform
import random
import socket
import subprocess
import sys
from pathlib import Path

# Add the parent directory to sys.path to enable imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dotenv import load_dotenv

load_dotenv()

# Now this import should work
import config  # Ensure config.py exists in the src/ directory

logger = logging.getLogger(__name__)


def configure_logging():
    """
    Configures the logging settings.
    """
    logger = logging.getLogger('polaris_cli')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    log_file = os.path.join(get_project_root(), 'logs', 'polaris.log')  # Corrected path
    os.makedirs(os.path.dirname(log_file), exist_ok=True)  # Ensure log directory exists
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger


def run_elevated(cmd):
    if platform.system().lower() == "windows":
        cmd = cmd.replace('"', '""')
        vbs_content = (
            'Set UAC = CreateObject("Shell.Application")\n'
            f'UAC.ShellExecute "cmd.exe", "/c {cmd}", "", "runas", 1'
        )
        
        vbs_path = config.HOME_DIR / 'temp_elevate.vbs'
        with open(vbs_path, 'w', encoding='utf-8') as f:
            f.write(vbs_content)
        
        subprocess.run(['cscript', '//Nologo', str(vbs_path)], check=True)
        vbs_path.unlink(missing_ok=True)
    else:
        subprocess.run(['sudo', cmd], shell=True, check=True)


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            # Fallback method
            hostname = socket.gethostname()
            return socket.gethostbyname(hostname)
        except Exception:
            return '127.0.0.1'


def get_project_root():
    """
    Determines the root directory of the project.
    
    Returns:
        str: Absolute path to the project root.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
