import ast
import copy
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import questionary
import requests
from click_spinner import spinner
from communex.client import CommuneClient
from communex.compat.key import classic_load_key
from polaris_cli.network_handler import NetworkSelectionHandler, NetworkType
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn)
from rich.prompt import Confirm, Prompt
from rich.table import Table

from src.pid_manager import PID_FILE
from src.user_manager import UserManager
from src.utils import configure_logging

logger = configure_logging()
console = Console()
server_url_ = "https://api.polaris.com"

logger = configure_logging()
console = Console()
server_url_ = os.getenv('SERVER_URL')

def load_system_info(json_path='system_info.json') -> Dict[str, Any]:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    system_info_full_path = os.path.join(project_root, json_path)
    if not os.path.exists(system_info_full_path):
        console.print(Panel(
            "[red]System information file not found.[/red]\n"
            "Please ensure that 'polaris start' is running.",
            title="Error",
            border_style="red"
        ))
        sys.exit(1)
    try:
        with open(system_info_full_path, 'r') as f:
            data = json.load(f)
        logger.info("System information loaded successfully.")
        return data[0]
    except json.JSONDecodeError as e:
        console.print(Panel(
            f"[red]Failed to parse the system information file: {e}[/red]",
            title="Error",
            border_style="red"
        ))
        sys.exit(1)
    except Exception as e:
        console.print(Panel(
            f"[red]Error reading system information: {e}[/red]",
            title="Error",
            border_style="red"
        ))
        sys.exit(1)

def display_system_info(system_info: Dict[str, Any]) -> None:
    table = Table(title="System Information", box=box.ROUNDED)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    def flatten_dict(d: Dict[str, Any], parent_key: str = '') -> list:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key))
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                for i, item in enumerate(v):
                    items.extend(flatten_dict(item, f"{new_key}[{i}]"))
            else:
                items.append((new_key, v))
        return items

    flattened = flatten_dict(system_info)
    for key, value in flattened:
        if isinstance(value, list):
            value = ', '.join(map(str, value))
        if 'password' in key.lower():
            value = '*' * 8
        table.add_row(key, str(value))
    console.print(table)

def submit_registration(submission: Dict[str, Any]) -> Dict[str, Any]:
    try:
        console.print("[cyan]Connecting to registration server...[/cyan]")
        api_url = f'{server_url_}/miners/'
        headers = {'Content-Type': 'application/json'}
        
        # Log the request details
        console.print(f"[cyan]Sending registration data to {api_url}...[/cyan]")
        console.print(f"[dim cyan]Request Data (miner_id if available): {submission.get('miner_id', 'Not specified')}[/dim cyan]")
        console.print(f"[dim cyan]Request Data (username/wallet): {submission.get('username', submission.get('wallet_name', 'Not specified'))}[/dim cyan]")
        
        response = requests.post(api_url, json=submission, headers=headers, timeout=10)
        
        # Log the response details
        console.print(f"[cyan]Server response status: {response.status_code}[/cyan]")
        try:
            response_json = response.json()
            console.print(f"[dim cyan]Response Data: {json.dumps(response_json, indent=2)}[/dim cyan]")
            if 'miner_id' in response_json:
                console.print(f"[green]Received miner_id from server: {response_json['miner_id']}[/green]")
        except Exception:
            console.print(f"[yellow]Response Text: {response.text}[/yellow]")
        
        response.raise_for_status()
        
        console.print("[green]Registration submitted successfully![/green]")
        result = response.json()
        return result
    except requests.RequestException as e:
        console.print(Panel(
            f"[red]Failed to connect to registration server: {str(e)}[/red]",
            title="❌ Connection Error",
            border_style="red"
        ))
        sys.exit(1)
    except Exception as e:
        console.print(Panel(
            f"[red]Registration failed: {str(e)}[/red]",
            title="❌ Registration Error",
            border_style="red"
        ))
        sys.exit(1)

def display_registration_success(result: Dict[str, Any]) -> None:
    miner_id = result.get('miner_id', 'N/A')
    message = result.get('message', 'Registration successful')
    added_resources = result.get('added_resources', [])
    console.print(Panel(
        f"[green]{message}[/green]\n\n"
        f"Miner ID: [bold cyan]{miner_id}[/bold cyan]\n"
        f"Added Resources: [cyan]{', '.join(added_resources) if added_resources else 'None'}[/cyan]\n\n"
        "[yellow]Important: Save your Miner ID - you'll need it to manage your compute resources.[/yellow]",
        title="✅ Registration Complete",
        border_style="green"
    ))

def check_commune_balance(key_name: str, required: float = 10.0):
    import re
    from subprocess import run
    result = run(["comx", "key", "balance", key_name], capture_output=True, text=True)
    balance = 0.0
    if result.returncode == 0:
        match = re.search(r"([\d\.]+)", result.stdout)
        if match:
            balance = float(match.group(1))
    return (balance, balance >= required)

def add_server_key_to_authorized_keys(server_public_key):
    """Add the server's public key to the authorized_keys file."""
    try:
        # Determine the user's home directory and .ssh directory
        home_dir = os.path.expanduser("~")
        ssh_dir = os.path.join(home_dir, ".ssh")
        
        # Create .ssh directory if it doesn't exist
        if not os.path.exists(ssh_dir):
            os.makedirs(ssh_dir, mode=0o700)
            console.print("[green]Created .ssh directory[/green]")
        
        # Path to authorized_keys file
        auth_keys_path = os.path.join(ssh_dir, "authorized_keys")
        
        # Check if authorized_keys already contains this key
        existing_keys = ""
        if os.path.exists(auth_keys_path):
            with open(auth_keys_path, 'r') as f:
                existing_keys = f.read()
        
        # Avoid duplicates by checking if key already exists
        if server_public_key in existing_keys:
            console.print("[yellow]Server public key already exists in authorized_keys[/yellow]")
            return True
        
        # Append the new key
        with open(auth_keys_path, 'a+') as f:
            # Add a newline if the file doesn't end with one
            if existing_keys and not existing_keys.endswith('\n'):
                f.write('\n')
            f.write(f"{server_public_key}\n")
        
        # Set correct permissions
        os.chmod(auth_keys_path, 0o600)
        
        console.print("[green]Successfully added server public key to authorized_keys[/green]")
        return True
    except Exception as e:
        console.print(f"[error]Failed to add server public key: {str(e)}[/error]")
        return False

def process_compute_resource(resource: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Ensure environment variables are loaded from .env
        from dotenv import load_dotenv
        load_dotenv(override=True)
        
        resource_id = resource.get('id', 'unknown')
        
        # Get SSH values directly from .env without fallbacks
        env_host = os.environ.get('SSH_HOST')
        env_ssh_user = os.environ.get('SSH_USER')
        env_ssh_port = os.environ.get('SSH_PORT')
        
        # Log the environment variables being used
        console.print(f"[dim cyan]Using SSH_HOST: {env_host}[/dim cyan]")
        console.print(f"[dim cyan]Using SSH_USER: {env_ssh_user}[/dim cyan]")
        console.print(f"[dim cyan]Using SSH_PORT: {env_ssh_port}[/dim cyan]")
        
        # Create a new resource structure for API submission
        processed_resource = {
            'id': resource_id,
            'resource_type': resource.get('resource_type', 'CPU'),
            'location': resource.get('location', 'Unknown'),
            'hourly_price': resource.get('hourly_price', 1.0),
            'ram': resource.get('ram', '16GB'),
            'storage': resource.get('storage', {'type': 'SSD', 'capacity': '500GB'}),
            'network': {
                'internal_ip': resource.get('network', {}).get('internal_ip', env_host),
                'ssh': resource.get('network', {}).get('ssh', f"ssh://{env_ssh_user}@{env_host}:{env_ssh_port}"),
                'open_ports': resource.get('network', {}).get('open_ports', ['22']),
                'username': resource.get('network', {}).get('username', env_ssh_user),
                'auth_type': resource.get('network', {}).get('auth_type', 'public_key')
            }
        }
        
        # Add CPU specs if available
        if 'cpu_specs' in resource:
            processed_resource['cpu_specs'] = resource['cpu_specs']
        
        # Add GPU specs if available
        if 'gpu_specs' in resource:
            processed_resource['gpu_specs'] = resource['gpu_specs']
        
        return processed_resource
    except Exception as e:
        console.print(f"[error]Error processing compute resource {resource.get('id', 'unknown')}: {str(e)}[/error]")
        return None

def process_stepping(stepping: Any, resource_id: str) -> int:
    if stepping is None:
        return 1
    if not isinstance(stepping, int):
        console.print(Panel(
            f"[red]Invalid 'stepping' value for resource {resource_id}.[/red]\n"
            "It must be an integer.",
            title="Validation Error",
            border_style="red"
        ))
        sys.exit(1)
    return stepping

def process_online_cpus(online_cpus: Any, resource_id: str) -> str:
    # Just return a safe default format if there are any issues
    try:
        # If it's already a list of integers, use that
        if isinstance(online_cpus, list) and all(isinstance(cpu, int) for cpu in online_cpus) and online_cpus:
            return f"{min(online_cpus)}-{max(online_cpus)}"
            
        # If it's a string in list format, try to parse it
        if isinstance(online_cpus, str) and online_cpus.startswith('[') and online_cpus.endswith(']'):
            try:
                cpu_list = ast.literal_eval(online_cpus)
                if isinstance(cpu_list, list) and all(isinstance(cpu, int) for cpu in cpu_list) and cpu_list:
                    return f"{min(cpu_list)}-{max(cpu_list)}"
            except:
                pass
                
        # If it's already in the right format, use that
        if isinstance(online_cpus, str) and '-' in online_cpus:
            try:
                start, end = map(int, online_cpus.strip("'\"").split('-'))
                if start <= end:
                    return f"{start}-{end}"
            except:
                pass
    except Exception as e:
        console.print(f"[yellow]Warning: Error processing online_cpus: {e}. Using default value.[/yellow]")
        
    # Default fallback value
    return "0-15"

def process_ssh(ssh_str: str) -> str:
    if not ssh_str:
        return ""
    ssh_str = ssh_str.strip()
    if ssh_str.startswith('ssh://'):
        return ssh_str
    pattern = r'^ssh\s+([^@]+)@([\w\.-]+)\s+-p\s+(\d+)$'
    match = re.match(pattern, ssh_str)
    if match:
        user, host, port = match.groups()
        return f"ssh://{user}@{host}:{port}"
    raise ValueError(f"Invalid SSH format: {ssh_str}")

def register_miner(skip_existing_check=False):
    try:
        from src.user_manager import UserManager
        user_manager = UserManager()
        if not skip_existing_check:
            skip_registration, user_info = user_manager.check_existing_registration(show_prompt=True)
            if skip_registration:
                console.print("[yellow]Using existing registration.[/yellow]")
                return

        # Display welcome message and explanation
        console.print(Panel(
            "[bold cyan]Welcome to the Polaris Compute Subnet Registration![/bold cyan]\n\n"
            "[cyan]This process will register your node as a Commune miner with the Polaris subnet.[/cyan]\n"
            "[yellow]Please provide accurate information about your system to ensure proper registration.[/yellow]",
            title="🌟 Polaris Miner Registration",
            border_style="cyan"
        ))
        
        # Load and display system information
        console.print("[cyan]Loading system information...[/cyan]")
        system_info = load_system_info()
        if system_info:
            display_system_info(system_info)
        else:
            console.print("[error]Failed to load system information.[/error]")
            return

        # Validate compute resources (may exit if validation fails)
        console.print("[cyan]Validating compute resources...[/cyan]")
        validate_compute_resources(system_info.get('compute_resources', []))
        console.print("[green]Compute resources validated successfully![/green]")

        # Get username input
        username = questionary.text(
            "Enter username:",
            style=custom_style
        ).ask()
        
        if not username or username.strip() == "":
            console.print("[error]Username cannot be empty.[/error]")
            return
            
        console.print(f"[green]Username set to: {username}[/green]")
            
        # Get commune key for registration
        console.print("[cyan]Checking for commune key...[/cyan]")
        commune_key = questionary.text(
            "Enter commune key: (press enter to generate a new one)",
            style=custom_style
        ).ask()

        if not commune_key or commune_key.strip() == "":
            console.print("[cyan]Generating new commune key...[/cyan]")
            try:
                result = subprocess.run(['commune', 'key', 'new'], capture_output=True, text=True, check=True)
                commune_key = result.stdout.strip()
                console.print(f"[green]Generated new commune key: {commune_key}[/green]")
            except Exception as e:
                console.print(f"[error]Failed to generate commune key: {str(e)}[/error]")
                return
        else:
            # Validate commune key
            try:
                if not commune_key.startswith("0x") or len(commune_key) != 66:
                    console.print("[error]Invalid commune key format. Key should start with 0x and be 66 characters long.[/error]")
                    return
                console.print(f"[green]Using commune key: {commune_key}[/green]")
                check_commune_balance(commune_key)
            except Exception as e:
                console.print(f"[error]Error validating commune key: {str(e)}[/error]")
                return

        # Display available networks
        console.print(Panel(
            "[cyan]Network Selection[/cyan]\n"
            "[yellow]Select the network your miner will connect to[/yellow]",
            title="Network Setup",
            border_style="cyan"
        ))
        
        networks = ['mainnet', 'testnet', 'devnet']
        network = questionary.select(
            "Select network:",
            choices=networks,
            style=custom_style
        ).ask()
        
        if not network:
            console.print("[error]No network selected. Exiting.[/error]")
            return

        console.print(f"[green]Selected network: {network}[/green]")

        # Process the compute resources
        console.print("[cyan]Processing compute resources...[/cyan]")
        processed_resources = []
        for resource in system_info.get('compute_resources', []):
            processed_resource = process_compute_resource(resource)
            if processed_resource:
                processed_resources.append(processed_resource)

        if not processed_resources:
            console.print("[error]No valid compute resources found.[/error]")
            return

        console.print(f"[green]Processed {len(processed_resources)} compute resources successfully![/green]")

        # Prepare registration data
        console.print("[cyan]Preparing registration submission...[/cyan]")
        submission = {
            'username': username,
            'commune_key': commune_key,
            'network': network,
            'compute_resources': processed_resources
        }

        # Submit registration
        console.print("[cyan]Submitting registration to server...[/cyan]")
        result = submit_registration(submission)

        if result:
            miner_id = result.get('miner_id', 'unknown')
            server_public_key = result.get('server_public_key')
            
            # Display registration success
            display_registration_success(result)
            
            # Handle the server's public key if provided
            if server_public_key:
                console.print("[cyan]Server provided its public key. Adding to authorized_keys...[/cyan]")
                add_server_key_to_authorized_keys(server_public_key)
            
            # Save user information
            console.print("[cyan]Saving user information...[/cyan]")
            # Extract network info from the first compute resource for simplicity
            network_info = processed_resources[0].get('network', {})
            user_manager.save_user_info(miner_id, commune_key, network_info, server_public_key)
            console.print(f"[green]User information saved with Miner ID: {miner_id}[/green]")
            
            # Ask if user wants to start services
            start_services = questionary.confirm(
                "Do you want to start Polaris services now?",
                style=custom_style
            ).ask()
            
            if start_services:
                console.print("[cyan]Starting Polaris services...[/cyan]")
                
                from .start import start_polaris, start_system
                
                if not start_system():
                    console.print("[error]Failed to start system process.[/error]")
                    return
                    
                if not start_polaris():
                    console.print("[error]Failed to start API process.[/error]")
                    from .start import stop_system
                    stop_system()
                    return
                    
                console.print("[success]Polaris services started successfully![/success]")
        else:
            console.print("[error]Registration failed. Please try again later.[/error]")

    except Exception as e:
        console.print(f"[error]Registration failed: {str(e)}[/error]")
        import traceback
        console.print(traceback.format_exc())

def register_independent_miner(skip_existing_check=False):
    import socket

    import questionary
    import requests
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm

    from .start import start_polaris
    console = Console()
    from src.user_manager import UserManager
    user_manager = UserManager()
    if not skip_existing_check:
        skip_registration, user_info = user_manager.check_existing_registration()
        if skip_registration:
            miner_id = user_info.get('miner_id', 'Unknown')
            username = user_info.get('username', 'Unknown')
            console.print(Panel(
                f"[yellow]Existing registration found:[/yellow]\n"
                f"Miner ID: [bold cyan]{miner_id}[/bold cyan]\n"
                f"Username: [bold cyan]{username}[/bold cyan]",
                title="⚠️ Existing Registration",
                border_style="yellow"
            ))
            if not Confirm.ask("Do you want to proceed with a new registration?", default=False):
                console.print("[yellow]Using existing registration.[/yellow]")
                return
    console.print(Panel(
        "[cyan]Independent Miner Registration[/cyan]\n"
        "[yellow]You will register your miner and compute resources without joining any network.[/yellow]",
        title="Independent Miner",
        border_style="cyan"
    ))
    system_info = load_system_info()
    if system_info:
        display_system_info(system_info)
        if Confirm.ask("Do you want to proceed with registration?", default=True):
            username = questionary.text("Enter your desired username:").ask()
            if not username:
                console.print("[error]Username is required for registration.[/error]")
                return
            location = system_info.get("location")
            if not location:
                try:
                    response = requests.get('https://ipinfo.io/json', timeout=3)
                    if response.status_code == 200:
                        ip_data = response.json()
                        if 'country' in ip_data and 'region' in ip_data:
                            location = f"{ip_data['region']}, {ip_data['country']}"
                        elif 'country' in ip_data:
                            location = ip_data['country']
                except:
                    try:
                        hostname = socket.gethostname()
                        if hostname:
                            location = f"Host: {hostname}"
                    except:
                        pass
            if not location:
                location = "Polaris HQ"
            submission = {
                "name": username,
                "location": location,
                "description": "Independent Miner registered via Polaris CLI tool",
                "compute_resources": [],
                "registration_type": "independent"
            }
            compute_resources = system_info.get("compute_resources", [])
            if isinstance(compute_resources, dict):
                compute_resources = [compute_resources]
            if not compute_resources:
                console.print("[error]No compute resources found.[/error]")
                return
            for resource in compute_resources:
                try:
                    resource_copy = resource.copy() if isinstance(resource, dict) else {}
                    if not resource_copy.get("location"):
                        resource_copy["location"] = location
                    processed_resource = process_compute_resource(resource_copy)
                    submission["compute_resources"].append(processed_resource)
                except Exception as e:
                    console.print(Panel(
                        f"[red]Error processing compute resource:[/red]\n{str(e)}",
                        title="Error",
                        border_style="red"
                    ))
                    return
            try:
                result = submit_registration(submission)
                if not result or not result.get('miner_id'):
                    console.print("[error]Failed to register with server.[/error]")
                    return
                miner_id = result.get('miner_id')
                network_info = submission['compute_resources'][0]['network']
                user_manager.save_user_info(miner_id, username, network_info)
                
                # Handle the server's public key if provided in the response
                if "server_public_key" in result and result["server_public_key"]:
                    console.print("[cyan]Server provided its public key. Adding to authorized_keys...[/cyan]")
                    add_server_key_to_authorized_keys(result["server_public_key"])
                
                console.print(Panel(
                    "[green]Independent miner registered successfully![/green]\n\n"
                    f"Miner ID: [bold cyan]{miner_id}[/bold cyan]\n"
                    f"Username: [bold cyan]{username}[/bold cyan]\n\n"
                    "[yellow]Important: Save your Miner ID - you'll need it to manage your compute resources.[/yellow]\n"
                    "[yellow]Note: As an independent miner, you are not connected to any network.[/yellow]",
                    title="✅ Registration Complete",
                    border_style="green"
                ))
                from .start import start_polaris
                if start_polaris():
                    console.print("[success]Polaris services started successfully![/success]")
                else:
                    console.print("[error]Failed to start Polaris services.[/error]")
            except Exception as e:
                console.print(Panel(
                    f"[red]Error during registration: {str(e)}[/red]",
                    title="Error",
                    border_style="red"
                ))
        else:
            console.print("[yellow]Registration cancelled.[/yellow]")

def validate_compute_resources(compute_resources) -> None:
    if not compute_resources:
        console.print(Panel(
            "[red]No compute resources found in system info.[/red]",
            title="Error",
            border_style="red"
        ))
        sys.exit(1)
    
    # Only require essential fields, be more lenient with others
    essential_fields = [
        "id", "resource_type", "location",
        "ram", "storage.type", "storage.capacity"
    ]
    
    # Non-essential fields - warn but don't fail if missing
    recommended_fields = [
        "hourly_price", "storage.read_speed", "storage.write_speed",
        "cpu_specs.vendor_id", "cpu_specs.cpu_name", 
        "network.internal_ip", "network.ssh", "network.password",
        "network.username"
    ]
    
    # If CPU fields are missing, we'll provide defaults later
    cpu_fields = [
        "cpu_specs.op_modes", "cpu_specs.total_cpus",
        "cpu_specs.online_cpus"
    ]
    
    def check_field(resource, field):
        path = field.split('.')
        value = resource
        for key in path:
            if not isinstance(value, dict) or key not in value:
                return False
            value = value[key]
        return value is not None
    
    for resource in (compute_resources if isinstance(compute_resources, list) else [compute_resources]):
        # Check essential fields - these cause errors if missing
        missing_essential = [field for field in essential_fields if not check_field(resource, field)]
        if missing_essential:
            console.print(Panel(
                f"[red]Missing required fields for resource {resource.get('id', 'unknown')}:[/red]\n"
                f"{', '.join(missing_essential)}",
                title="Validation Error",
                border_style="red"
            ))
            sys.exit(1)
            
        # Check recommended fields - these only show warnings
        missing_recommended = [field for field in recommended_fields if not check_field(resource, field)]
        if missing_recommended:
            console.print(
                f"[yellow]Warning: Missing recommended fields for resource {resource.get('id', 'unknown')}:[/yellow]\n"
                f"[yellow]{', '.join(missing_recommended)}[/yellow]"
            )
            
        # Check CPU fields - add defaults if missing
        # If any CPU fields are missing, add CPU specs object with defaults
        missing_cpu = [field for field in cpu_fields if not check_field(resource, field)]
        if missing_cpu and "resource_type" in resource and resource["resource_type"].upper() == "CPU":
            console.print(f"[yellow]Warning: Adding default CPU specs for resource {resource.get('id', 'unknown')}[/yellow]")
            if "cpu_specs" not in resource:
                resource["cpu_specs"] = {}
            
            if "op_modes" not in resource["cpu_specs"]:
                resource["cpu_specs"]["op_modes"] = "32-bit, 64-bit"
                
            if "total_cpus" not in resource["cpu_specs"]:
                resource["cpu_specs"]["total_cpus"] = 16
                
            if "online_cpus" not in resource["cpu_specs"]:
                resource["cpu_specs"]["online_cpus"] = "0-15"

def validate_ram_format(ram_value: str, resource_id: str) -> None:
    if not isinstance(ram_value, str) or not ram_value.endswith("GB"):
        console.print(Panel(
            f"[red]Invalid RAM format for resource {resource_id}.[/red]\n"
            "Expected format: '16.0GB' or similar",
            title="Validation Error",
            border_style="red"
        ))
        sys.exit(1)
    try:
        float(ram_value[:-2])
    except ValueError:
        console.print(Panel(
            f"[red]Invalid RAM value for resource {resource_id}.[/red]\n"
            "RAM value must be a number followed by 'GB'",
            title="Validation Error",
            border_style="red"
        ))
        sys.exit(1)

def register_independent_miner_and_return_id(username=None, skip_existing_check=False):
    """
    Register an independent miner and return the miner ID.
    
    Args:
        username (str, optional): Username to use for registration. If None, a unique one will be generated.
        skip_existing_check (bool): Whether to skip checking for existing registration.
        
    Returns:
        str: The miner ID if registration was successful, None otherwise.
    """
    import socket
    import time
    import uuid

    import questionary
    import requests
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt
    console = Console()
    from src.user_manager import UserManager
    user_manager = UserManager()
    
    # Check for existing registration if not skipping
    if not skip_existing_check:
        skip_registration, user_info = user_manager.check_existing_registration()
        if skip_registration:
            miner_id = user_info.get('miner_id', 'Unknown')
            existing_username = user_info.get('username', 'Unknown')
            console.print(Panel(
                f"[yellow]Existing registration found:[/yellow]\n"
                f"Miner ID: [bold cyan]{miner_id}[/bold cyan]\n"
                f"Username: [bold cyan]{existing_username}[/bold cyan]",
                title="⚠️ Existing Registration",
                border_style="yellow"
            ))
            if not Confirm.ask("Do you want to proceed with a new registration?", default=False):
                console.print("[yellow]Using existing registration.[/yellow]")
                return miner_id
    
    console.print(Panel(
        "[cyan]Independent Miner Registration[/cyan]\n"
        "[yellow]You will register your miner and compute resources without joining any network.[/yellow]",
        title="Independent Miner",
        border_style="cyan"
    ))
    
    system_info = load_system_info()
    if system_info:
        display_system_info(system_info)
        if Confirm.ask("Do you want to proceed with registration?", default=True):
            # Use provided username or generate a unique one
            if not username:
                # Generate a unique username if none provided
                unique_id = str(uuid.uuid4())[:8]  # First 8 characters of a UUID
                timestamp = int(time.time()) % 10000  # Last 4 digits of current timestamp
                username = f"polaris-{timestamp}-{unique_id}"
                console.print(f"[green]Generated unique username: {username}[/green]")
            
            if not username:
                console.print("[error]Username is required for registration.[/error]")
                return None
                
            location = system_info.get("location")
            if not location:
                try:
                    response = requests.get('https://ipinfo.io/json', timeout=3)
                    if response.status_code == 200:
                        ip_data = response.json()
                        if 'country' in ip_data and 'region' in ip_data:
                            location = f"{ip_data['region']}, {ip_data['country']}"
                        elif 'country' in ip_data:
                            location = ip_data['country']
                except:
                    try:
                        hostname = socket.gethostname()
                        if hostname:
                            location = f"Host: {hostname}"
                    except:
                        pass
                        
            if not location:
                location = "Polaris HQ"
                
            submission = {
                "name": username,
                "location": location,
                "description": "Independent Miner registered via Polaris CLI tool",
                "compute_resources": [],
                "registration_type": "independent"
            }
            
            compute_resources = system_info.get("compute_resources", [])
            if isinstance(compute_resources, dict):
                compute_resources = [compute_resources]
                
            if not compute_resources:
                console.print("[error]No compute resources found.[/error]")
                return None
                
            for resource in compute_resources:
                try:
                    resource_copy = resource.copy() if isinstance(resource, dict) else {}
                    if not resource_copy.get("location"):
                        resource_copy["location"] = location
                    processed_resource = process_compute_resource(resource_copy)
                    submission["compute_resources"].append(processed_resource)
                except Exception as e:
                    console.print(Panel(
                        f"[red]Error processing compute resource:[/red]\n{str(e)}",
                        title="Error",
                        border_style="red"
                    ))
                    return None
                    
            try:
                result = submit_registration(submission)
                if not result or not result.get('miner_id'):
                    console.print("[error]Failed to register with server.[/error]")
                    return None
                    
                miner_id = result.get('miner_id')
                network_info = submission['compute_resources'][0]['network']
                user_manager.save_user_info(miner_id, username, network_info)
                
                # Handle the server's public key if provided in the response
                if "server_public_key" in result and result["server_public_key"]:
                    console.print("[cyan]Server provided its public key. Adding to authorized_keys...[/cyan]")
                    add_server_key_to_authorized_keys(result["server_public_key"])
                
                console.print(Panel(
                    "[green]Independent miner registered successfully![/green]\n\n"
                    f"Miner ID: [bold cyan]{miner_id}[/bold cyan]\n"
                    f"Username: [bold cyan]{username}[/bold cyan]\n\n"
                    "[yellow]Important: Save your Miner ID - you'll need it to manage your compute resources.[/yellow]\n"
                    "[yellow]Note: As an independent miner, you are not connected to any network.[/yellow]",
                    title="✅ Registration Complete",
                    border_style="green"
                ))
                
                # Return the miner ID instead of starting services
                return miner_id
                
            except Exception as e:
                console.print(Panel(
                    f"[red]Error during registration: {str(e)}[/red]",
                    title="Error",
                    border_style="red"
                ))
                return None
    else:
        console.print("[error]Failed to load system information.[/error]")
        return None

if __name__ == "__main__":
    register_miner()
