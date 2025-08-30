import logging
import platform
import subprocess
import time
from pathlib import Path

import requests
import yaml

from . import config, utils


class NgrokManager:
    def __init__(self):
        self.logger = logging.getLogger('remote_access')
        self.is_windows = platform.system().lower() == "windows"
    
    def kill_existing(self):
        try:
            if self.is_windows:
                subprocess.run("taskkill /F /IM ngrok.exe 2>nul", shell=True, check=False)
            else:
                subprocess.run("pkill ngrok", shell=True, check=False)
            time.sleep(2)
        except Exception as e:
            self.logger.error(f"Error killing ngrok: {e}")

    def create_config(self, port):
        config.NGROK_CONFIG_DIR.mkdir(exist_ok=True)
        config_path = config.NGROK_CONFIG_DIR / 'ngrok.yml'
        
        # Basic tunnel configuration
        tunnel_config = {
            'proto': 'tcp',
            'addr': str(port)  # Ensure port is string
        }
        
        ngrok_config = {
            'version': '2',
            'authtoken': config.NGROK_AUTH_TOKEN,
            'tunnels': {
                'ssh': tunnel_config
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(ngrok_config, f)
        
        return config_path

    def start_tunnel(self, port):
        self.kill_existing()
        config_path = self.create_config(port)
        
        if self.is_windows:
            cmd = ["ngrok", "start", "--all"]
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            subprocess.Popen(cmd, startupinfo=startupinfo)
        else:
            # On Linux, we need to specify the tunnel type explicitly
            cmd = ["ngrok", "tcp", str(port)]
            try:
                # Start ngrok with explicit tunnel type
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                
                # Check for immediate errors
                time.sleep(1)
                if process.poll() is not None:
                    _, stderr = process.communicate()
                    if stderr:
                        self.logger.error(f"Ngrok failed to start: {stderr}")
                        raise Exception(f"Ngrok failed to start: {stderr}")
            except Exception as e:
                self.logger.error(f"Error starting ngrok: {e}")
                raise
        
        time.sleep(3)
        
        # Wait for tunnel to be ready and get URL
        for _ in range(30):
            try:
                r = requests.get("http://localhost:4040/api/tunnels", timeout=5)
                if r.status_code == 200:
                    tunnels = r.json().get("tunnels", [])
                    if tunnels:
                        tunnel = tunnels[0]
                        url = tunnel["public_url"]
                        raw_url = url.replace("tcp://", "")
                        host, port = raw_url.split(":")
                        self.logger.info(f"Ngrok tunnel established at {host}:{port}")
                        return host, port
            except requests.exceptions.RequestException:
                time.sleep(1)
                continue
        
        raise Exception("Failed to establish tunnel after 30 seconds")

    def check_tunnel_status(self):
        """Check if ngrok tunnel is running and return its details"""
        try:
            r = requests.get("http://localhost:4040/api/tunnels", timeout=5)
            if r.status_code == 200:
                tunnels = r.json().get("tunnels", [])
                if tunnels:
                    return True, tunnels[0]["public_url"]
            return False, None
        except requests.exceptions.RequestException:
            return False, None