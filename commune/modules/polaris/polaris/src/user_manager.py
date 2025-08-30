# src/user_manager.py

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

console = Console()

class UserManager:
    def __init__(self):
        self.project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.user_file = self.project_root / 'user_info.json'

    def save_user_info(self, miner_id: str, username: str, network_info: Dict, server_public_key: str = None) -> bool:
        """
        Save user registration information.
        """
        try:
            user_data = {
                'miner_id': miner_id,
                'username': username,
                'network_info': network_info
            }
            
            # Add server_public_key if provided
            if server_public_key:
                user_data['server_public_key'] = server_public_key
                
            with open(self.user_file, 'w') as f:
                json.dump(user_data, f, indent=4)
            return True
        except Exception as e:
            console.print(f"[red]Failed to save user information: {e}[/red]")
            return False

    def get_user_info(self) -> Optional[Dict]:
        """
        Retrieve saved user information.
        """
        try:
            if self.user_file.exists():
                with open(self.user_file, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            console.print(f"[red]Failed to read user information: {e}[/red]")
            return None

    def check_existing_registration(self, show_prompt: bool = True) -> Tuple[bool, Optional[Dict]]:
        """
        Check if there's an existing registration and handle user choice.
        
        Args:
            show_prompt: If True, shows the prompt for existing registration (used during registration)
                       If False, just checks existence (used during start)
        """
        user_info = self.get_user_info()
        if user_info:
            if show_prompt:
                console.print(Panel(
                    f"[yellow]Existing registration found:[/yellow]\n"
                    f"Miner ID: {user_info['miner_id']}\n"
                    f"Username: {user_info['username']}",
                    title="⚠️ Existing Registration",
                    border_style="yellow"
                ))
                
                if not Confirm.ask("Do you want to proceed with a new registration?", default=False):
                    return True, user_info
                return False, None
            else:
                # For start command, just return the existing registration
                return True, user_info
        return False, None

    def update_network_info(self, network_info: Dict) -> bool:
        """
        Update network information for existing user.
        """
        user_info = self.get_user_info()
        if user_info:
            user_info['network_info'] = network_info
            return self.save_user_info(
                user_info['miner_id'],
                user_info['username'],
                network_info
            )
        return False

    def clear_user_info(self) -> bool:
        """
        Clear saved user information.
        """
        try:
            if self.user_file.exists():
                os.remove(self.user_file)
            return True
        except Exception as e:
            console.print(f"[red]Failed to clear user information: {e}[/red]")
            return False