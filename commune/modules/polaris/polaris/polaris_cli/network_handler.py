# polaris_cli/network_handler.py
import argparse
import datetime
import getpass
import json
import logging
import os
import platform
import random
import re
import string
import subprocess
import sys
import threading
import time
from enum import Enum
from time import sleep
from typing import Any, Dict, List, Optional, Tuple

import requests
from communex.client import CommuneClient
from communex.compat.key import classic_load_key
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.emoji import Emoji
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (BarColumn, Progress, SpinnerColumn,
                           TaskProgressColumn, TextColumn)
from rich.prompt import Confirm, Prompt
from rich.spinner import Spinner
from rich.style import Style
from rich.table import Table
from rich.text import Text
from rich.syntax import Syntax

# Import custom bittensor utility
from polaris_cli.bittensor_utils import get_uid_from_hotkey

if platform.system() == "Windows":
    import msvcrt
else:
    import termios
    import tty

console = Console()
logger = logging.getLogger(__name__)

server_url = os.getenv('SERVER_URL')

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all logs; adjust as needed
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("polaris_cli.log")  
    ]
)


class NetworkType(Enum):
    COMMUNE = "commune"
    BITTENSOR = "bittensor"
    NORMAL = "normal"


class CrossPlatformMenu:
    def __init__(self, options, title="Select an option"):
        self.options = options
        self.title = title
        self.selected = 0

    def _get_char_windows(self):
        """Get character input for Windows."""
        char = msvcrt.getch()
        if char in [b'\xe0', b'\x00']:  # Arrow keys prefix
            char = msvcrt.getch()
            return {
                b'H': 'up',
                b'P': 'down',
                b'\r': 'enter'
            }.get(char, None)
        elif char == b'\r':
            return 'enter'
        return None

    def _get_char_unix(self):
        """Get character input for Unix-like systems."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                sys.stdin.read(1)  # skip '['
                ch = sys.stdin.read(1)
                return {
                    'A': 'up',
                    'B': 'down'
                }.get(ch, None)
            elif ch == '\r':
                return 'enter'
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return None

    def _get_char(self):
        """Get character input in a cross-platform way."""
        if platform.system() == "Windows":
            return self._get_char_windows()
        return self._get_char_unix()

    def _clear_screen(self):
        """Clear the screen in a cross-platform way."""
        console.clear()

    def show(self):
        """Display the menu and handle user input."""
        while True:
            self._clear_screen()

            # Display title in a panel
            title_panel = Panel(
                Text(self.title, justify="center", style="bold cyan"),
                box=box.ROUNDED,
                border_style="blue",
                padding=(1, 2)
            )
            console.print(Align.center(title_panel))
            console.print()

            # Display options with enhanced styling
            options_panel = Panel(
                Group(
                    *[
                        Text(
                            f"{'➜ ' if i == self.selected else '  '}{option}",
                            style="bold blue" if i == self.selected else "white"
                        )
                        for i, option in enumerate(self.options)
                    ]
                ),
                box=box.ROUNDED,
                border_style="cyan",
                padding=(1, 2)
            )
            console.print(Align.center(options_panel))

            key = self._get_char()

            if key == 'up':
                self.selected = (self.selected - 1) % len(self.options)
            elif key == 'down':
                self.selected = (self.selected + 1) % len(self.options)
            elif key == 'enter':
                return self.selected


class NetworkSelectionHandler:
    def __init__(self):
        self.console = Console()
        self.api_test_url = 'https://polaris-test-server.onrender.com/api/v1'
        self.api_base_url = server_url
        self.created_miner_id = None

    def set_miner_id(self, miner_id: str):
        """Store the miner ID after successful compute registration."""
        if not miner_id:
            raise ValueError("Miner ID cannot be empty")
        self.created_miner_id = miner_id
        logger.info(f"Set miner ID to: {miner_id}")

    def select_network(self):
        """Display enhanced network options with cross-platform arrow key selection."""
        welcome_panel = Panel(
            Group(
                Text("🌟 Welcome to", style="cyan"),
                Text("POLARIS COMPUTE NETWORK",
                     style="bold white", justify="center"),
                Text("Choose your registration network", style="cyan"),
            ),
            box=box.HEAVY,
            border_style="blue",
            padding=(1, 2)
        )
        self.console.print(Align.center(welcome_panel))
        self.console.print()

        options = [
            "🌐 Commune Network",
            "🔗 Bittensor Network",
            "📡 Normal Provider"
        ]

        menu = CrossPlatformMenu(
            options,
            title="Select Registration Network"
        )

        selected_index = menu.show()

        if selected_index == 0:
            return NetworkType.COMMUNE
        elif selected_index == 1:
            return NetworkType.BITTENSOR
        elif selected_index == 2:
            return NetworkType.NORMAL
        else:
            self.console.print(Panel(
                "[yellow]Registration cancelled.[/yellow]",
                border_style="yellow"
            ))
            sys.exit(0)

    def handle_commune_registration(self, key_name=None, netuid=33):
        """Handle Commune network registration process with enhanced UI.
        
        Args:
            key_name (str, optional): Name of the Commune key to use. If None, user will be prompted.
            netuid (int, optional): Network UID to register with. Defaults to 33 (Mainnet).
        
        Returns:
            tuple: (wallet_name, commune_uid, ss58_address) if successful, None otherwise
        """
        # Step 1: Confirm Registration as Commune Miner
        registration_panel = Panel(
            Group(
                # Registration Header
                Text.assemble(
                    ("You are about to register as a ", "bright_blue"),
                    ("POLARIS COMMUNE MINER", "bold white")
                ),
                Text(""),
                Text("This will:", style="cyan"),
                # Benefits
                Text("• Connect you to our Polaris Commune Network", style="white"),
                Text("• Enable you to earn rewards", style="white"),
                Text("• Join the decentralized compute ecosystem", style="white"),
                Text(""),
                Text("────────────────────────────────────", style="dim"),
                Text(""),
                # Key Requirements
                Text("Requirements:", style="cyan"),
                Text("• Must have registered Commune key", style="white"),
                Text(
                    f"• Key must be registered under our Polaris subnet on Commune (netuid={netuid})", style="white"
                ),
                Text(""),
                # Information for those without key
                Text("If you don't have a registered Commune key, visit:",
                     style="yellow"),
                Text("https://communeai.org/docs/working-with-keys/key-basics",
                     style="blue underline"),
                Text(""),
                # Key information if provided
                Text(f"Using key: {key_name}" if key_name else "Please enter your Commune wallet name below", style="cyan"),
            ),
            box=box.HEAVY,
            border_style="green",
            padding=(1, 3),
            title="[bold green]🔒 Commune Miner Registration[/bold green]",
            subtitle="[dim]Please confirm your details[/dim]"
        )
        self.console.print(Align.left(registration_panel))

        # Get Wallet Name - either use provided key_name or ask user
        wallet_name = key_name
        if not wallet_name:
            wallet_name = Prompt.ask("\n[bold cyan]Enter your wallet name[/bold cyan]")

        if not wallet_name.strip():
            self.console.print(
                Panel("[red]Wallet name cannot be empty.[/red]", border_style="red")
            )
            return None

        # Step 4: Retrieve Commune UID and SS58 Address
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                expand=True
            ) as progress:
                task = progress.add_task(
                    f"[cyan]Retrieving Commune UID for netuid={netuid}...", total=100
                )

                key = classic_load_key(wallet_name)
                ss58_address = key.ss58_address
                commune_uid = self._get_commune_uid(wallet_name, netuid)

                while not progress.finished:
                    progress.update(task, advance=1)
                    sleep(0.01)

            if not commune_uid:
                self.console.print(Panel(
                    f"[red]Failed to retrieve Commune UID for netuid={netuid}[/red]\n"
                    f"Please ensure your key '{wallet_name}' is registered to mine Polaris subnet.",
                    title="❌ Error",
                    border_style="red"
                ))
                return None

            success_panel = Panel(
                Group(
                    Text("✅ Successfully retrieved wallet information!",
                         style="green"),
                    Text(f"\nCommune UID: {commune_uid}", style="cyan"),
                    Text(
                        f"Wallet Address: {ss58_address[:10]}...{ss58_address[-8:]}", style="cyan"
                    ),
                    Text(
                        f"Network: {'Mainnet' if netuid == 33 else 'Testnet'} (netuid={netuid})", style="cyan"
                    ),
                ),
                box=box.ROUNDED,
                border_style="green",
                title="[bold green]Wallet Verified[/bold green]"
            )
            self.console.print(Align.center(success_panel))
            return wallet_name, commune_uid, ss58_address

        except Exception as e:
            self.console.print(Panel(
                f"[red]Error during Commune registration: {str(e)}[/red]",
                title="❌ Error",
                border_style="red"
            ))
            logger.error(f"Error during Commune registration: {e}")
            return None
            
    def _get_commune_uid(self, wallet_name, netuid=33):
        """Retrieve Commune UID for the given wallet."""
        try:
            key = classic_load_key(wallet_name)
            commune_node_url = "wss://api.communeai.net/"
            client = CommuneClient(commune_node_url)
            modules_keys = client.query_map_key(netuid)
            val_ss58 = key.ss58_address
            miner_uid = next((uid for uid, address in modules_keys.items() 
                           if address == val_ss58), None)
            
            if miner_uid is not None:
                logger.info(f"Retrieved miner UID: {miner_uid} for wallet: {wallet_name}")
                return miner_uid
            else:
                logger.error(f"Miner's SS58 address not found in the network for netuid {netuid}")
                return None
            
        except StopIteration:
            logger.error("Miner's SS58 address not found in the network.")
            return None
        except Exception as e:
            logger.error(f"Error retrieving miner UID: {e}")
            return None

    def get_created_miner_id(self) -> str:
        """Get the miner ID that was created during the compute registration process."""
        # This is where you would get the miner ID that was created earlier
        # For now, we'll return it from a class variable that should be set during miner creation
        if self.created_miner_id:
            return self.created_miner_id
        raise ValueError("Miner ID not found. Please ensure miner was created successfully.")

    def display_success_message(self, miner_id: str, commune_uid: str, wallet_name: str):
        """Display a beautiful success message indicating the node is live."""
        # Optional: Add a brief animation
        with Live(console=self.console, refresh_per_second=10) as live:
            for i in range(3, 0, -1):
                countdown = f"[bold green]Launching your node in {i}...[/bold green] 🚀"
                live.update(Align.center(Text(countdown, justify="center")))
                sleep(1)

        success_message = f"""
## 🎉 **Congratulations!**

Your node is now **live** on the **Polaris Commune Network**.

---

**Miner ID:** `{miner_id}`

**Commune UID:** `{commune_uid}`

**Wallet Name:** `{wallet_name}`

You are now part of a decentralized compute ecosystem. 🚀

**What's Next?**
- **Monitor your node's status:** Use `polaris status` to check if Polaris and Compute Subnet are running.
- **View your compute resources:** Use `polaris view-compute` to see your pod compute resources.
- **Monitor miner heartbeat:** Use `polaris monitor` to keep an eye on your miner's heartbeat signals in real-time.

Thank you for joining us! 🌟
"""

        panel = Panel(
            Align.center(Markdown(success_message)),
            box=box.ROUNDED,
            border_style="bold green",
            padding=(2, 4),
            title="[bold green]🚀 Node Live on Polaris Commune Network[/bold green]",
            subtitle="[dim]Your node is now active and contributing to the network[/dim]"
        )

        self.console.print(panel)

    def register_commune_miner(self, wallet_name: str, commune_uid: str, wallet_address: str):
        """Register miner with Commune network."""
        try:
            miner_id = self.get_created_miner_id()
            api_url = f'{self.api_base_url}/commune/register'

            payload = {
                'miner_id': miner_id,
                'commune_uid': str(commune_uid),
                'wallet_name': wallet_name,
                'wallet_address': wallet_address,
                'netuid': 33  # Using Mainnet as default
            }

            # Mask sensitive information before logging
            masked_wallet_address = f"{wallet_address[:10]}...{wallet_address[-8:]}"
            masked_payload = {**payload, 'wallet_address': masked_wallet_address}

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                expand=True
            ) as progress:
                task = progress.add_task(
                    "[cyan]Registering with Commune network...", total=100
                )
                
                self.console.print(Panel(
                    f"[green]Server Url: {api_url}[/green]",
                    title="Server Url",
                    border_style="green"
                ))

                response = requests.post(api_url, json=payload)
                response.raise_for_status()

                while not progress.finished:
                    progress.update(task, advance=1)
                    sleep(0.01)

                result = response.json()

            if result['status'] == 'success':
                # Display the beautiful success message
                self.display_success_message(miner_id, commune_uid, wallet_name)
                return result
            else:
                # Display server-provided error message
                error_message = result.get('message', 'Unknown error')
                self.console.print(Panel(
                    f"[red]Registration failed: {error_message}[/red]",
                    title="❌ Error",
                    border_style="red"
                ))
                logger.error(f"Registration failed: {error_message}")
                return None

        except requests.HTTPError as http_err:
            # Attempt to extract error details from response
            try:
                error_details = http_err.response.json()
                error_message = error_details.get('message', str(http_err))
            except (json.JSONDecodeError, AttributeError):
                error_message = str(http_err)

            # Log the error
            logger.error(f"HTTPError during Commune registration: {error_message}")

            self.console.print(Panel(
                f"[red]Failed to register with Commune network: {error_message}[/red]",
                title="❌ Error",
                border_style="red"
            ))
            return None

        except Exception as e:
            # Log the unexpected error
            logger.error(f"Unexpected error during Commune registration: {e}")

            self.console.print(Panel(
                f"[red]Failed to register with Commune network: {str(e)}[/red]",
                title="❌ Error",
                border_style="red"
            ))
            return None

    def verify_commune_status(self, miner_id: str):
        """Verify Commune registration status."""
        try:
            api_url = f'{self.api_base_url}/commune/miner/{miner_id}/verify'

            # Log the verification URL
            logger.debug(f"Verifying Commune status with URL: {api_url}")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                expand=True
            ) as progress:
                task = progress.add_task(
                    "[cyan]Verifying Commune registration...", total=100
                )

                response = requests.get(api_url)
                response.raise_for_status()
                result = response.json()

                while not progress.finished:
                    progress.update(task, advance=1)
                    sleep(0.01)

            if result.get('status') == 'success':
                self.console.print(Panel(
                    f"[green]Commune registration verified successfully![/green]\n"
                    f"[cyan]Miner ID: {miner_id}[/cyan]\n"
                    f"[cyan]Status: {result.get('message', 'Active')}[/cyan]",
                    title="✅ Verification Successful",
                    border_style="green"
                ))
                return True
            else:
                self.console.print(Panel(
                    f"[yellow]Commune verification returned non-success status.[/yellow]\n"
                    f"[yellow]Status: {result.get('status', 'unknown')}[/yellow]\n"
                    f"[yellow]Message: {result.get('message', 'No message')}[/yellow]",
                    title="⚠️ Verification Warning",
                    border_style="yellow"
                ))
                return False
        except Exception as e:
            self.console.print(Panel(
                f"[red]Failed to verify Commune registration: {str(e)}[/red]",
                title="❌ Verification Error",
                border_style="red"
            ))
            return False

    def register_bittensor_miner(self, wallet_name: str, hotkey: str, netuid: int, wallet_address: str = "unknown", hotkey_address: str = "unknown", coldkey_address: str = "unknown"):
        """Register miner with Bittensor network API."""
        try:
            # Use the existing miner ID from independent registration if it exists
            if self.created_miner_id:
                miner_id = self.created_miner_id
                logger.info(f"Using existing miner ID from independent registration: {miner_id}")
            else:
                # Fallback to generating a unique miner ID if none exists
                import hashlib
                import time
                
                # Generate a more unique miner ID using hash of wallet addresses and timestamp
                timestamp = int(time.time())
                unique_str = f"{wallet_name}-{hotkey}-{netuid}-{timestamp}"
                hash_obj = hashlib.md5(unique_str.encode())
                short_hash = hash_obj.hexdigest()[:8]
                
                # New unique miner ID format with hash
                miner_id = f"bt-{wallet_name}-{hotkey}-{netuid}-{short_hash}"
                logger.info(f"No existing miner ID found, generated new one: {miner_id}")
                
                # Store the created miner ID for later retrieval
                self.created_miner_id = miner_id
            
            # Get miner UID from hotkey
            network = "finney" if netuid == 49 else "test"
            miner_uid = None
            
            # Determine which addresses to use - ALWAYS prefer actual SS58 addresses over names
            # Check if the addresses are valid SS58 format (they start with 5 and are ~48 chars long)
            is_valid_ss58 = lambda addr: addr.startswith('5') and len(addr) > 40 and len(addr) < 50
            
            # For hotkey - use actual SS58 address if available
            if hotkey_address != "unknown" and is_valid_ss58(hotkey_address):
                actual_hotkey = hotkey_address
                logger.info(f"Using actual SS58 hotkey address for UID lookup: {hotkey_address[:10]}...")
            else:
                actual_hotkey = hotkey
                logger.warning(f"No valid SS58 hotkey address provided for UID lookup. Using name instead: {hotkey}")
                # Since we don't have a valid SS58 address, we can't get the UID
                self.console.print("[yellow]No valid SS58 hotkey address provided. Cannot retrieve miner UID.[/yellow]")
                
            # Try to retrieve UID from hotkey using our utility function - ONLY if we have a valid SS58 address
            if is_valid_ss58(actual_hotkey):
                self.console.print(f"[cyan]Retrieving miner UID from hotkey on network {network}...[/cyan]")
                try:
                    miner_uid = get_uid_from_hotkey(actual_hotkey, netuid, network)
                    if miner_uid is not None:
                        self.console.print(f"[green]Found miner UID: {miner_uid}[/green]")
                    else:
                        self.console.print("[yellow]Could not find miner UID for the provided hotkey on this subnet.[/yellow]")
                except Exception as e:
                    logger.error(f"Error retrieving miner UID: {str(e)}")
                    self.console.print(f"[yellow]Error retrieving miner UID: {str(e)}[/yellow]")
            
            network_name = "mainnet" if netuid == 49 else "testnet"
            api_url = f'{self.api_base_url}/bittensor/register'

            # For hotkey - use actual SS58 address if available (we checked earlier but setting again for clarity)
            if hotkey_address != "unknown" and is_valid_ss58(hotkey_address):
                actual_hotkey = hotkey_address
                logger.info(f"Using actual SS58 hotkey address for registration: {hotkey_address[:10]}...")
            else:
                actual_hotkey = hotkey
                logger.warning(f"Using hotkey name instead of SS58 address for registration: {hotkey}")
            
            # For coldkey - use actual SS58 address if available
            if coldkey_address != "unknown" and is_valid_ss58(coldkey_address):
                actual_coldkey = coldkey_address
                logger.info(f"Using actual SS58 coldkey address: {coldkey_address[:10]}...")
            else:
                actual_coldkey = wallet_name
                logger.warning(f"Using coldkey name instead of SS58 address: {wallet_name}")
            
            # Prepare payload
            payload = {
                'miner_id': miner_id,
                'wallet_name': wallet_name,
                'hotkey': actual_hotkey,  # Will be actual SS58 hotkey address when available
                'network': network_name,
                'netuid': netuid,
                'wallet_address': wallet_address,
                'subnet_uid': str(netuid),  # Use str(netuid) as required by the API
                'coldkey': actual_coldkey   # Will be actual SS58 coldkey address when available
            }
            
            # Add miner_uid to payload if we found it
            if miner_uid is not None:
                payload['miner_uid'] = int(miner_uid)
                self.console.print(f"[green]Including miner UID in registration: {miner_uid}[/green]")
                
                # Also log to a dedicated miner_uid file
                try:
                    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
                    os.makedirs(logs_dir, exist_ok=True)
                    miner_uid_file = os.path.join(logs_dir, f'miner_id_{miner_id}_uid.txt')
                    with open(miner_uid_file, 'w') as f:
                        f.write(f"Miner ID: {miner_id}\nMiner UID: {miner_uid}\nHotkey: {actual_hotkey}\nNetwork: {network_name}\nNetuid: {netuid}\nTimestamp: {datetime.datetime.now().isoformat()}")
                    logger.info(f"Wrote miner UID information to {miner_uid_file}")
                except Exception as e:
                    logger.error(f"Failed to write miner UID file: {str(e)}")
            else:
                logger.warning(f"No miner UID found for miner_id {miner_id}")

            # Log request details (with sensitive data masked)
            def mask_address(address):
                if address != "unknown" and is_valid_ss58(address) and len(address) > 20:
                    return f"{address[:10]}...{address[-8:]}"
                return address
                
            masked_payload = {
                **payload, 
                'wallet_address': mask_address(wallet_address),
                'hotkey': mask_address(actual_hotkey),
                'coldkey': mask_address(actual_coldkey)
            }
            
            logger.info(f"Registering Bittensor miner with payload: {masked_payload}")

            self.console.print(Panel(
                f"[cyan]Registering Bittensor miner with API[/cyan]\n"
                f"[green]Miner ID: {miner_id}[/green]\n"
                f"[green]Network: {network_name} (netuid {netuid})[/green]",
                title="🌐 Bittensor Registration",
                border_style="cyan"
            ))
            
            self.console.print(f"[dim cyan]API URL: {api_url}[/dim cyan]")

            # Send the registration request
            response = requests.post(api_url, json=payload)
            
            # Log response details
            logger.info(f"Registration response status: {response.status_code}")
            
            if response.status_code == 200 or response.status_code == 201:
                result = response.json()
                
                # Log the full Bittensor response in a clearly formatted way
                logger.info("========== BITTENSOR REGISTRATION RESPONSE BEGIN ==========")
                try:
                    # Format as pretty JSON if possible
                    formatted_result = json.dumps(result, indent=2)
                    logger.info(f"\n{formatted_result}")
                except Exception as e:
                    # Fall back to simple logging if JSON formatting fails
                    logger.info(f"Raw response: {result}")
                    logger.error(f"Error formatting response: {str(e)}")
                logger.info("========== BITTENSOR REGISTRATION RESPONSE END ==========")
                
                # Also print to console for immediate visibility during registration
                self.console.print("[bold cyan]Bittensor API Response:[/bold cyan]")
                try:
                    # Use Rich's syntax highlighting for JSON
                    json_str = json.dumps(result, indent=2)
                    syntax = Syntax(json_str, "json", theme="monokai", word_wrap=True)
                    self.console.print(syntax)
                except Exception as e:
                    self.console.print(f"[dim]{result}[/dim]")
                    logger.error(f"Error displaying JSON response: {str(e)}")
                
                # Extract and log the miner UID from the response
                miner_uid = None
                
                # Try different possible locations for the miner UID in the response
                if 'uid' in result:
                    miner_uid = result.get('uid')
                elif 'miner_uid' in result:
                    miner_uid = result.get('miner_uid')
                elif 'registration' in result and isinstance(result.get('registration'), dict):
                    registration = result.get('registration', {})
                    if 'uid' in registration:
                        miner_uid = registration.get('uid')
                    elif 'miner_uid' in registration:
                        miner_uid = registration.get('miner_uid')
                    elif 'miner_id' in registration:
                        miner_uid = registration.get('miner_id')
                
                # Log the full response for debugging
                logger.info(f"Full Bittensor registration response: {result}")
                
                if miner_uid:
                    logger.info(f"Bittensor registration successful - Miner UID: {miner_uid}")
                    self.console.print(f"[green]Miner UID from Bittensor response: {miner_uid}[/green]")
                else:
                    logger.warning("Could not find miner UID in Bittensor response")
                    # Log keys in the response at top level
                    logger.info(f"Response keys at top level: {list(result.keys())}")
                    if 'registration' in result and isinstance(result.get('registration'), dict):
                        logger.info(f"Registration keys: {list(result.get('registration').keys())}")
                
                if result.get('status') == 'success':
                    self.console.print(Panel(
                        f"[green]Successfully registered with Bittensor API![/green]\n"
                        f"[cyan]Miner ID: {miner_id}[/cyan]\n"
                        f"[cyan]Miner UID: {miner_uid or 'Not found in response'}[/cyan]\n"
                        f"[cyan]Network: {network_name} (netuid {netuid})[/cyan]",
                        title="✅ Registration Successful",
                        border_style="green"
                    ))
                    # Return both the result and the extracted miner UID
                    result['miner_uid'] = miner_uid
                    return result
                else:
                    # Display server-provided error message
                    error_message = result.get('message', 'Unknown error')
                    self.console.print(Panel(
                        f"[yellow]Registration returned non-success status: {error_message}[/yellow]",
                        title="⚠️ Registration Warning",
                        border_style="yellow"
                    ))
                    return result
            else:
                self.console.print(Panel(
                    f"[red]Registration failed with status code: {response.status_code}[/red]\n"
                    f"[red]Response: {response.text}[/red]",
                    title="❌ Registration Error",
                    border_style="red"
                ))
                return None

        except Exception as e:
            # Log the error
            logger.error(f"Error during Bittensor registration: {str(e)}")
            self.console.print(Panel(
                f"[red]Failed to register with Bittensor API: {str(e)}[/red]",
                title="❌ Error",
                border_style="red"
            ))
            return None

    def verify_bittensor_status(self, miner_id: str):
        """Verify Bittensor registration status."""
        try:
            api_url = f'{self.api_base_url}/bittensor/miner/{miner_id}/verify'

            # Log the verification URL
            logger.debug(f"Verifying Bittensor status with URL: {api_url}")
            self.console.print(f"[dim cyan]Verifying registration status: {api_url}[/dim cyan]")

            response = requests.get(api_url)
            
            # Log response details
            logger.info(f"Verification response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Verification response: {result}")
                
                if result.get('status') == 'success':
                    self.console.print(Panel(
                        f"[green]Bittensor registration verified successfully![/green]\n"
                        f"[cyan]Miner ID: {miner_id}[/cyan]\n"
                        f"[cyan]Status: {result.get('message', 'Active')}[/cyan]",
                        title="✅ Verification Successful",
                        border_style="green"
                    ))
                    return True
                else:
                    self.console.print(Panel(
                        f"[yellow]Bittensor verification returned non-success status.[/yellow]\n"
                        f"[yellow]Status: {result.get('status', 'unknown')}[/yellow]\n"
                        f"[yellow]Message: {result.get('message', 'No message')}[/yellow]",
                        title="⚠️ Verification Warning",
                        border_style="yellow"
                    ))
                    return False
            else:
                self.console.print(Panel(
                    f"[red]Verification failed with status code: {response.status_code}[/red]\n"
                    f"[red]Response: {response.text}[/red]",
                    title="❌ Verification Error",
                    border_style="red"
                ))
                return False
        except Exception as e:
            self.console.print(Panel(
                f"[red]Failed to verify Bittensor registration: {str(e)}[/red]",
                title="❌ Verification Error",
                border_style="red"
            ))
            return False

    def handle_bittensor_registration(self):
        """Handle Bittensor network registration."""
        panel = Panel(
            Group(
                Text("Bittensor Network registration coming soon!",
                     style="bold yellow"),
                Text(
                    "We're working hard to bring you Bittensor integration.", style="italic"),
            ),
            box=box.ROUNDED,
            border_style="yellow",
            title="[bold yellow]🚧 Coming Soon[/bold yellow]"
        )
        self.console.print(Align.center(panel))
        logger.info("Bittensor registration is not yet implemented.")
        return None

    def run_registration_flow(self):
        """Run the complete registration flow with enhanced UI."""
        try:
            # Show welcome banner
            welcome_panel = Panel(
                Group(
                    Text("🌟 Welcome to", style="cyan"),
                    Text("POLARIS COMPUTE NETWORK",
                         style="bold white", justify="center"),
                    Text("\nPlease select your registration network", style="cyan"),
                ),
                box=box.HEAVY,
                border_style="blue",
                padding=(1, 2)
            )
            self.console.print(Align.center(welcome_panel))
            self.console.print()

            network = self.select_network()

            if network == NetworkType.COMMUNE:
                registration_details = self.handle_commune_registration()
                if registration_details:
                    wallet_name, commune_uid, ss58_address = registration_details
                    # miner_id should be set externally via set_miner_id
                    if not self.created_miner_id:
                        self.console.print(Panel(
                            "[red]Miner ID not set. Please ensure compute registration is completed before Commune registration.[/red]",
                            title="❌ Error",
                            border_style="red"
                        ))
                        logger.error("Miner ID not set before Commune registration.")
                        sys.exit(1)
                    self.register_commune_miner(wallet_name, commune_uid, ss58_address)
            elif network == NetworkType.BITTENSOR:
                self.handle_bittensor_registration()
            elif network == NetworkType.NORMAL:
                self.console.print(Panel(
                    "[bold green]Normal Provider registration is not implemented yet.[/bold green]",
                    border_style="green"
                ))
                logger.info("Normal Provider registration is not implemented.")
            else:
                self.console.print(Panel(
                    "[yellow]Unknown network selected. Exiting.[/yellow]",
                    border_style="yellow"
                ))
                logger.warning("Unknown network selected.")
                sys.exit(0)

        except KeyboardInterrupt:
            self.console.print("\n")
            self.console.print(Panel(
                "[yellow]Registration process cancelled by user.[/yellow]",
                title="ℹ️ Cancelled",
                border_style="yellow"
            ))
            logger.info("Registration process cancelled by user.")
        except Exception as e:
            self.console.print(Panel(
                f"[red]An unexpected error occurred: {str(e)}[/red]",
                title="❌ Error",
                border_style="red"
            ))
            logger.error(f"Unexpected error in registration flow: {e}")