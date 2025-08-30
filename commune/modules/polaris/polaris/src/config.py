import os
import string
from pathlib import Path

from dotenv import load_dotenv

# Load .env file
load_dotenv()

NGROK_AUTH_TOKEN = os.getenv('NGROK_AUTH_TOKEN')
SSH_CONFIG_PATH = "C:\\ProgramData\\ssh\\sshd_config"
SSH_PASSWORD = os.getenv('SSH_PASSWORD')  # Get password from .env
USE_SSH_KEY = os.getenv('USE_SSH_KEY', 'false').lower() == 'true'

# Only check for SSH_PASSWORD if not using SSH key authentication
if not SSH_PASSWORD and not USE_SSH_KEY:
    raise ValueError("SSH_PASSWORD not found in .env file and USE_SSH_KEY is not set to true. Configure either password or key authentication.")

PASSWORD_CHARS = string.ascii_letters + string.digits
HOME_DIR = Path.home()
NGROK_CONFIG_DIR = HOME_DIR / '.ngrok2'
LOG_DIR = HOME_DIR / '.remote-access'