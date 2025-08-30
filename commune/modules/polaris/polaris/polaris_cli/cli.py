import json
import os
import subprocess
import sys
from pathlib import Path
import requests

import click
import questionary
from questionary import Style
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from .heartbeat_monitor import monitor_heartbeat
from .log_monitor import check_main
from .repo_manager import update_repository
from .start import (check_status, start_polaris, start_system, stop_polaris,
                    stop_system)
from .bittensor_miner import start_bittensor_miner, stop_bittensor_miner, is_bittensor_running
from .register import load_system_info, display_system_info, register_miner as commune_register, register_independent_miner, register_independent_miner_and_return_id

custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red",
    "success": "green"
})

console = Console(theme=custom_theme)

custom_style = Style([
    ('qmark', 'fg:#ff9d00 bold'),
    ('question', 'bold'),
    ('answer', 'fg:#00ff00 bold'),
    ('pointer', 'fg:#ff9d00 bold'),
    ('highlighted', 'fg:#ff9d00 bold'),
    ('selected', 'fg:#00ff00'),
    ('separator', 'fg:#cc5454'),
    ('instruction', ''),
    ('text', ''),
    ('disabled', 'fg:#858585 italic')
])

POLARIS_HOME = Path.home() / '.polaris'
BITTENSOR_CONFIG_PATH = POLARIS_HOME / 'bittensor'
SERVER_ENDPOINT = "https://polaris-test-server.onrender.com/api/v1/miners"

def setup_directories():
    """Create necessary directories if they don't exist"""
    POLARIS_HOME.mkdir(exist_ok=True)
    BITTENSOR_CONFIG_PATH.mkdir(exist_ok=True)
    (BITTENSOR_CONFIG_PATH / 'pids').mkdir(exist_ok=True)
    (BITTENSOR_CONFIG_PATH / 'logs').mkdir(exist_ok=True)

def display_dashboard():
    """Display the dashboard with fixed-width panels, matching the screenshot as closely as possible."""
    polaris_logo = r"""
      ____        __            _     
     / __ \____  / /___ _______(_)____
    / /_/ / __ \/ / __ `/ ___/ / ___/
   / ____/ /_/ / / /_/ / /  / (__  ) 
  /_/    \____/_/\__,_/_/  /_/____/  
    """
    header_panel = Panel(
        f"[cyan]{polaris_logo}[/cyan]\n"
        "[bold white]♦ The Best Place to List Your GPUs ♦[/bold white]\n\n"
        "[purple]Welcome to the Polaris Compute Subnet![/purple]\n\n"
        "[bold white]♦ Our Mission is to be the Best Place on This Planet to List Your GPUs – We're just getting started! ♦[/bold white]",
        title="[bold cyan]POLARIS SUBNET[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
        width=100
    )
    console.print(header_panel, justify="center")
    console.print("[cyan]Powering GPU Computation[/cyan]", justify="center")
    table = Table(show_header=False, show_lines=True, box=box.ROUNDED, width=150)
    table.add_column(justify="left")
    table.add_column(justify="left")
    setup_commands = (
        "[bold cyan]Setup Commands[/bold cyan]\n"
        "• [bold]register[/bold] – Register as a new miner (required before starting)\n"
        "• [bold]update subnet[/bold] – Update the Polaris repository"
    )
    service_management = (
        "[bold cyan]Service Management[/bold cyan]\n"
        "• [bold]start[/bold] – Start Polaris and selected compute processes\n"
        "• [bold]stop[/bold] – Stop running processes\n"
        "• [bold]status[/bold] – Check if services are running"
    )
    monitoring_logs = (
        "[bold cyan]Monitoring & Logs[/bold cyan]\n"
        "• [bold]logs[/bold] – View logs without process monitoring\n"
        "• [bold]monitor[/bold] – Monitor miner heartbeat signals in real-time\n"
        "• [bold]check-main[/bold] – Check if main process is running and view its logs\n"
        "• [bold]view-compute[/bold] – View pod compute resources"
    )
    bittensor_integration = (
        "[bold cyan]Bittensor Integration[/bold cyan]\n"
        "Polaris integrates with Bittensor to provide a decentralized compute subnet\n"
        "• [bold]Wallet Management[/bold] – Create or use existing Bittensor wallets\n"
        "• [bold]Validator Mode[/bold] – Run as a Bittensor subnet validator\n"
        "• [bold]Network Registration[/bold] – Register with Bittensor network (netuid 12)\n"
        "• [bold]Heartbeat Service[/bold] – Maintain connection with the Bittensor network"
    )
    table.add_row(setup_commands, service_management)
    table.add_row(monitoring_logs, bittensor_integration)
    combined_panel = Panel(table, border_style="cyan", box=box.ROUNDED, width=150)
    console.print(combined_panel, justify="center")
    bottom_panel = Panel(
    "1. First register as a miner\n"
    "2. Then start your preferred service type\n"
    "3. Check status to verify everything is running\n"
    "4. Use logs to monitor operation\n"
    "5. Use stop when you want to shut down services\n\n"
    "[bold white]Examples:[/bold white]\n"
    "$ [magenta]polaris register[/magenta] – Register as a new miner\n"
    "$ [magenta]polaris start[/magenta] – Start the Polaris services\n"
    "$ [magenta]polaris status[/magenta] – Check which services are running\n"
    "$ [magenta]polaris stop[/magenta] – Stop running services\n"
    "$ [magenta]polaris logs[/magenta] – View service logs",
    border_style="cyan",
    box=box.ROUNDED,
    width=150,
    title="[bold cyan]Quick Start Guide[/bold cyan]",
    title_align="center"
    )
    console.print(bottom_panel, justify="start")

def setup_directories():
    POLARIS_HOME.mkdir(exist_ok=True)
    BITTENSOR_CONFIG_PATH.mkdir(exist_ok=True)
    (BITTENSOR_CONFIG_PATH / 'pids').mkdir(exist_ok=True)
    (BITTENSOR_CONFIG_PATH / 'logs').mkdir(exist_ok=True)

def display_registration_summary(wallet_name, hotkey, network_name, netuid):
    console.print(Panel(
        f"[cyan]Registration Summary[/cyan]\n\n"
        f"Wallet: [bold green]{wallet_name}[/bold green]\n"
        f"Hotkey: [bold green]{hotkey}[/bold green]\n"
        f"Network: [bold green]{network_name} (netuid {netuid})[/bold green]\n\n"
        "[yellow]Proceeding to start Polaris services...[/yellow]",
        title="✅ Registration Complete",
        border_style="green"
    ))

def select_registration_type():
    choices = [
        'Commune Miner Node',
        'Bittensor Miner Node',
        'Polaris Miner Node (Coming Soon)',
        'Independent Miner'
    ]
    answer = questionary.select(
        "Select registration type:",
        choices=choices,
        style=custom_style,
        qmark="🔑"
    ).ask()
    return answer.lower() if answer else ""

@click.group()
def cli():
    setup_directories()
    pass

@cli.command()
def register():
    from src.user_manager import UserManager
    user_manager = UserManager()
    skip_registration, user_info = user_manager.check_existing_registration(show_prompt=True)
    if skip_registration:
        console.print("[yellow]Using existing registration.[/yellow]")
        return
    reg_type = select_registration_type()
    if "bittensor" in reg_type:
        handle_bittensor_registration()
    elif "commune" in reg_type:
        commune_register(skip_existing_check=True)
    elif "polaris" in reg_type:
        console.print(Panel(
            "[yellow]Polaris Miner Node is coming soon![/yellow]",
            title="🚧 Coming Soon",
            border_style="yellow"
        ))
    elif "independent" in reg_type:
        register_independent_miner(skip_existing_check=True)
    else:
        console.print("[error]Invalid registration type selected.[/error]")

def handle_bittensor_registration():
    console.print(Panel(
        "[cyan]Bittensor Wallet Configuration[/cyan]\n"
        "[yellow]You'll need a wallet to participate in the Bittensor subnet[/yellow]",
        box=box.ROUNDED,
        title="Bittensor Setup"
    ))

    # Check if user already has a wallet
    has_wallet = questionary.confirm(
        "Do you already have a Bittensor wallet?",
        style=custom_style
    ).ask()

    if has_wallet:
        try:
            console.print("[cyan]Fetching wallet list from btcli...[/cyan]")
            result = subprocess.run(
                ['btcli', 'wallet', 'list'],
                capture_output=True,
                text=True,
                check=True
            )
            wallets = parse_wallet_list(result.stdout)
            
            if not wallets:
                console.print("[yellow]No wallets found. Creating a new wallet...[/yellow]")
                selected_wallet_name = create_new_wallet()
                if not selected_wallet_name:
                    return
                hotkeys = ['default']  # New wallets have a default hotkey
                selected_hotkey = 'default'
            else:
                console.print(f"[green]Found {len(wallets)} wallet(s)![/green]")
                
                # Let user select a wallet
                selected_wallet_name = questionary.select(
                    "Select your wallet (cold key):",
                    choices=wallets,
                    style=custom_style
                ).ask()
                
                if not selected_wallet_name:
                    console.print("[yellow]Wallet selection cancelled.[/yellow]")
                    return
                
                # Get hotkeys for the selected wallet
                console.print(f"[cyan]Getting hotkeys for wallet '{selected_wallet_name}'...[/cyan]")
                hotkeys_result = subprocess.run(
                    ['btcli', 'wallet', 'hotkeys', '--wallet.name', selected_wallet_name],
                    capture_output=True,
                    text=True
                )
                
                # Parse hotkeys (simple approach - extract lines that don't start with common headers)
                hotkeys = []
                for line in hotkeys_result.stdout.split('\n'):
                    line = line.strip()
                    if line and not line.startswith(('Wallet', 'Hotkeys', '-', '=')):
                        hotkeys.append(line)
                
                if not hotkeys:
                    console.print("[yellow]No hotkeys found for this wallet. Using 'default'.[/yellow]")
                    hotkeys = ['default']
                
                console.print(f"[green]Found {len(hotkeys)} hotkey(s) for wallet '{selected_wallet_name}'![/green]")
                
                # Let user select a hotkey
                selected_hotkey = questionary.select(
                    "Select a hotkey:",
                    choices=hotkeys,
                    style=custom_style
                ).ask()
                
                if not selected_hotkey:
                    console.print("[yellow]Hotkey selection cancelled.[/yellow]")
                    return
            
            console.print(f"[green]Using wallet '{selected_wallet_name}' with hotkey '{selected_hotkey}'[/green]")
        except subprocess.CalledProcessError as e:
            console.print(f"[error]Error fetching wallet information: {e}[/error]")
            console.print("[yellow]Creating a new wallet...[/yellow]")
            selected_wallet_name = create_new_wallet()
            if not selected_wallet_name:
                return
            selected_hotkey = 'default'  # New wallets have a default hotkey
    else:
        console.print("[cyan]Creating a new wallet...[/cyan]")
        selected_wallet_name = create_new_wallet()
        if not selected_wallet_name:
            return
        selected_hotkey = 'default'  # New wallets have a default hotkey

    # Network selection
    console.print(Panel(
        "[cyan]Network Selection[/cyan]\n"
        "[yellow]Select the Bittensor network you want to register on[/yellow]",
        box=box.ROUNDED,
        title="Network Setup"
    ))
    
    network_choices = [
        "Mainnet (netuid 49)",
        "Testnet (netuid 100)"
    ]
    
    selected_network = questionary.select(
        "Select the network to register on:",
        choices=network_choices,
        style=custom_style
    ).ask()
    
    if not selected_network:
        console.print("[yellow]Network selection cancelled.[/yellow]")
        return
    
    console.print(f"[green]Selected {selected_network}[/green]")
    
    # Parse the netuid from the selection
    netuid = 100 if "Testnet" in selected_network else 49
    
    console.print(Panel(
        f"[cyan]Registering on {selected_network}[/cyan]\n"
        "[yellow]This may take a few minutes...[/yellow]",
        box=box.ROUNDED,
        title="Network Registration"
    ))

    # Attempt registration with Bittensor network
    console.print("[cyan]Starting Bittensor registration process...[/cyan]")
    
    wallet, message = register_wallet(selected_wallet_name, selected_hotkey, netuid)

    if not wallet:
        console.print(f"[error]Registration failed: {message}[/error]")
        console.print("[red]Stopping execution.[/red]")
        return  # Stop execution immediately
    
    console.print("[green]Successfully registered with Bittensor network![/green]")
    
    # Generate a unique username based on wallet name (no user prompt)
    import uuid
    import time
    unique_id = str(uuid.uuid4())[:8]  # First 8 characters of a UUID
    timestamp = int(time.time()) % 10000  # Last 4 digits of current timestamp
    generated_username = f"{selected_wallet_name}-{timestamp}-{unique_id}"
    
    # First register as an independent miner to get a miner ID
    console.print("[cyan]Registering with Polaris as an independent miner first...[/cyan]")
    # Pass the generated username instead of prompting
    miner_id = register_independent_miner_and_return_id(username=generated_username, skip_existing_check=True)
    
    if not miner_id:
        console.print("[error]Failed to get miner ID from independent registration. Cannot proceed.[/error]")
        return
    
    console.print(f"[green]Successfully registered as independent miner with ID: {miner_id}[/green]")
    
    # Load system info
    console.print("[cyan]Loading system information...[/cyan]")
    system_info = load_system_info()
    if system_info:
        display_system_info(system_info)

        # Now register the miner with the Bittensor API using NetworkSelectionHandler
        console.print("[cyan]Now registering with Bittensor API...[/cyan]")
        try:
            # Use 'btcli w list' to get the actual SS58 addresses
            try:
                console.print(f"[cyan]Retrieving SS58 addresses for wallet '{selected_wallet_name}' with hotkey '{selected_hotkey}'...[/cyan]")
                wallet_list_result = subprocess.run([
                    'btcli', 'w', 'list'
                ], capture_output=True, text=True, check=True)
                
                # Process the output to find the SS58 addresses
                lines = wallet_list_result.stdout.split('\n')
                coldkey_address = "unknown"
                hotkey_address = "unknown"
                
                # Flag to track if we're processing the selected wallet
                processing_selected_wallet = False
                
                for line in lines:
                    line = line.strip()
                    
                    # Check if this is the coldkey line for our selected wallet
                    if "Coldkey" in line and selected_wallet_name in line and "ss58_address" in line:
                        processing_selected_wallet = True
                        # Extract the ss58 address for the coldkey
                        parts = line.split("ss58_address")
                        if len(parts) > 1:
                            coldkey_address = parts[1].strip()
                            console.print(f"[green]Found coldkey SS58 address: {coldkey_address}[/green]")
                    
                    # If we're processing the selected wallet and find the selected hotkey
                    elif processing_selected_wallet and "Hotkey" in line and selected_hotkey in line and "ss58_address" in line:
                        # Extract the ss58 address for the hotkey
                        parts = line.split("ss58_address")
                        if len(parts) > 1:
                            hotkey_address = parts[1].strip()
                            console.print(f"[green]Found hotkey SS58 address: {hotkey_address}[/green]")
                    
                    # If we encounter a new coldkey, reset the flag
                    elif "Coldkey" in line and selected_wallet_name not in line:
                        processing_selected_wallet = False
                
                if coldkey_address != "unknown" and hotkey_address != "unknown":
                    console.print(f"[green]Successfully retrieved SS58 addresses for wallet '{selected_wallet_name}' with hotkey '{selected_hotkey}'![/green]")
                    # Use the hotkey address as the wallet address for registration
                    wallet_address = hotkey_address
                else:
                    console.print("[yellow]Could not find SS58 addresses from wallet list. Using alternative methods...[/yellow]")
                    wallet_address = "unknown"
            except Exception as e:
                console.print(f"[yellow]Error retrieving SS58 addresses: {str(e)}[/yellow]")
                console.print("[yellow]Using alternative methods to try to retrieve wallet addresses...[/yellow]")
                coldkey_address = "unknown"
                hotkey_address = "unknown"
                wallet_address = "unknown"
            
            # Register with the Bittensor API using the miner_id from independent registration
            from .network_handler import NetworkSelectionHandler
            network_handler = NetworkSelectionHandler()
            
            # IMPORTANT: Set the miner ID that we got from independent registration
            network_handler.set_miner_id(miner_id)
            console.print(f"[green]Using miner ID for Bittensor registration: {miner_id}[/green]")
            
            # Register with the Bittensor endpoint
            registration_result = network_handler.register_bittensor_miner(
                wallet_name=selected_wallet_name,
                hotkey=selected_hotkey,
                netuid=netuid,
                wallet_address=wallet_address,
                hotkey_address=hotkey_address,
                coldkey_address=coldkey_address
            )
            
            if registration_result:
                # Skip verification as requested and show final success message
                console.print(Panel(
                    "[green]Registration process completed successfully![/green]\n\n"
                    f"Miner ID: [bold cyan]{miner_id}[/bold cyan]\n"
                    f"Wallet: [bold cyan]{selected_wallet_name}[/bold cyan]\n"
                    f"Hotkey: [bold cyan]{selected_hotkey}[/bold cyan]\n"
                    f"Network: [bold cyan]{'Mainnet' if netuid == 49 else 'Testnet'} (netuid={netuid})[/bold cyan]\n\n"
                    "[yellow]Your compute resources are pending verification. You will be notified when verification is complete.[/yellow]\n"
                    "[yellow]You can now use your miner ID to manage your compute resources and check status.[/yellow]",
                    title="✅ Registration Complete",
                    border_style="green"
                ))
            else:
                console.print(Panel(
                    "[yellow]Registration with Bittensor API encountered an issue, but the process completed.[/yellow]\n\n"
                    f"Miner ID: [bold cyan]{miner_id}[/bold cyan]\n"
                    f"Wallet: [bold cyan]{selected_wallet_name}[/bold cyan]\n"
                    f"Hotkey: [bold cyan]{selected_hotkey}[/bold cyan]\n"
                    f"Network: [bold cyan]{'Mainnet' if netuid == 49 else 'Testnet'} (netuid={netuid})[/bold cyan]\n\n"
                    "[yellow]Your compute resources are pending verification. You will be notified when verification is complete.[/yellow]",
                    title="⚠️ Registration Partially Complete",
                    border_style="yellow"
                ))
        except Exception as e:
            console.print(f"[error]Error during Bittensor API registration: {str(e)}[/error]")
            # Show a modified success message
            console.print(Panel(
                "[yellow]Registration completed with some issues.[/yellow]\n\n"
                f"Miner ID: [bold cyan]{miner_id}[/bold cyan]\n"
                f"Wallet: [bold cyan]{selected_wallet_name}[/bold cyan]\n"
                f"Hotkey: [bold cyan]{selected_hotkey}[/bold cyan]\n"
                f"Network: [bold cyan]{'Mainnet' if netuid == 49 else 'Testnet'} (netuid={netuid})[/bold cyan]\n\n"
                "[yellow]Your compute resources are pending verification. You will be notified when verification is complete.[/yellow]",
                title="⚠️ Registration Partially Complete",
                border_style="yellow"
            ))
    else:
        console.print("[error]Failed to load system information. Registration cannot proceed.[/error]")
        
    # Start Polaris services
    console.print("[cyan]Managing Polaris services...[/cyan]")
    from .start import stop_polaris, start_polaris, status_api
    
    # Check if services are already running and stop them first
    console.print("[cyan]Checking for existing Polaris services...[/cyan]")
    services_running = status_api()
    
    if services_running:
        console.print("[cyan]Stopping existing Polaris services before restart...[/cyan]")
        stop_result = stop_polaris()
        if stop_result:
            console.print("[green]Successfully stopped existing Polaris services.[/green]")
        else:
            console.print("[yellow]Warning: Could not fully stop existing services.[/yellow]")
    
    # Start fresh services
    console.print("[cyan]Starting Polaris services...[/cyan]")
    start_result = start_polaris()
    
    if start_result:
        console.print("[green]Polaris services started successfully![/green]")
    else:
        console.print("[yellow]Warning: Polaris services may not have started properly.[/yellow]")
        console.print("[yellow]Check 'polaris status' for more information.[/yellow]")

def parse_wallet_list(wallet_list_output):
    wallets = {}
    current_wallet = None
    for line in wallet_list_output.splitlines():
        line = line.strip()
        if not line or line == "Wallets":
            continue
        if "Coldkey" in line:
            parts = line.split("Coldkey")
            if len(parts) > 1:
                parts = parts[1].strip().split()
                if parts:
                    wallet_name = parts[0]
                    current_wallet = wallet_name
                    wallets[current_wallet] = []
        elif "Hotkey" in line and current_wallet:
            parts = line.split("Hotkey")
            if len(parts) > 1:
                parts = parts[1].strip().split()
                if parts:
                    hotkey_name = parts[0]
                    wallets[current_wallet].append(hotkey_name)
    return wallets

def create_new_wallet():
    console.print(Panel(
        "[cyan]Creating a New Bittensor Wallet[/cyan]\n"
        "[yellow]You will need to provide a name for your new wallet.[/yellow]",
        box=box.ROUNDED,
        title="Wallet Creation"
    ))
    
    while True:
        wallet_name = questionary.text(
            "Enter a name for your new wallet:",
            style=custom_style
        ).ask()
        
        if not wallet_name or not wallet_name.strip():
            console.print("[error]Wallet name cannot be empty. Please enter a valid name.[/error]")
            continue
            
        console.print(f"\n[info]Creating new coldkey with name: {wallet_name}...[/info]")
        
        try:
            console.print("[cyan]Creating coldkey (this may take a moment)...[/cyan]")
            subprocess.run([
                'btcli', 'wallet', 'new_coldkey',
                '--wallet.name', wallet_name
            ], check=True)
            console.print("[green]✓[/green] [info]Coldkey created successfully![/info]")
        except subprocess.CalledProcessError as e:
            console.print(f"[error]Failed to create coldkey: {str(e)}[/error]")
            return None
        
        console.print("[info]Creating new hotkey (default)...[/info]")
        
        try:
            console.print("[cyan]Creating hotkey (this may take a moment)...[/cyan]")
            subprocess.run([
                'btcli', 'wallet', 'new_hotkey',
                '--wallet.name', wallet_name,
                '--wallet.hotkey', 'default'
            ], check=True)
            console.print("[green]✓[/green] [info]Hotkey created successfully![/info]")
        except subprocess.CalledProcessError as e:
            console.print(f"[error]Failed to create hotkey: {str(e)}[/error]")
            return None
                
        # Provide visual confirmation of successful wallet creation
        console.print(Panel(
            f"[green]Wallet '{wallet_name}' with default hotkey has been created successfully![/green]",
            title="✅ Wallet Creation Complete",
            border_style="green"
        ))
        
        return wallet_name

def register_wallet(wallet_name, hotkey, netuid):
    network_name = "Mainnet" if netuid == 49 else "Testnet"

    console.print(f"\n[info]Registering on {network_name} subnet (netuid={netuid}) (this may take a few minutes)...[/info]")

    # Use specific commands for testnet and mainnet
    try:
        if netuid == 100:  # Testnet
            command = [
                'btcli', 'subnets', 'register',
                '--wallet-name', wallet_name,
                '--hotkey', hotkey,
                '--netuid', '100',
                '--network', 'test'
            ]
            console.print("[yellow]Make sure your local subtensor node is running on port 9944 or use a public testnet node.[/yellow]")
        else:  # Mainnet (netuid 49)
            command = [
                'btcli', 'subnets', 'register',
                '--wallet-name', wallet_name,
                '--hotkey', hotkey,
                '--netuid', '49',
                '--network', 'finney'
            ]
        
        console.print(Panel(
            "[cyan]Starting registration process...[/cyan]\n\n"
            "[bold yellow]IMPORTANT: You will be prompted for input during this process.[/bold yellow]\n"
            "[bold yellow]Watch carefully for prompts like 'Do you want to continue?' and 'Enter your password:'[/bold yellow]",
            title="⚠️ Interactive Process",
            border_style="yellow"
        ))
        
        console.print("[cyan]Executing registration command. Please respond to prompts as they appear:[/cyan]")
        
        # Use subprocess.run with direct interaction instead of capturing output
        result = subprocess.run(command, check=False)
        
        if result.returncode != 0:
            console.print(Panel(
                f"[red]Registration process exited with code {result.returncode}[/red]",
                title="❌ Registration Failed",
                border_style="red"
            ))
            return None, f"Process exited with code {result.returncode}"

        # Check if registration was successful by querying the network
        console.print("[cyan]Verifying registration status...[/cyan]")
        
        # Check if the wallet is registered by running another btcli command
        verify_cmd = [
            'btcli', 'wallet', 'balance',
            '--wallet.name', wallet_name,
            '--wallet.hotkey', hotkey
        ]
        
        if netuid == 100:  # Testnet
            verify_cmd.extend(['--network', 'test'])
        else:  # Mainnet
            verify_cmd.extend(['--network', 'finney'])
            
        try:
            verify_result = subprocess.run(verify_cmd, capture_output=True, text=True, check=True)
            console.print("[green]Wallet balance check successful.[/green]")
            
            console.print(Panel(
                "[green]Successfully registered on subnet![/green]\n"
                f"[cyan]Wallet: {wallet_name}[/cyan]\n"
                f"[cyan]Hotkey: {hotkey}[/cyan]\n"
                f"[cyan]Network: {network_name} (netuid={netuid})[/cyan]",
                title="✅ Registration Successful",
                border_style="green"
            ))
            return wallet_name, "Success"
        except subprocess.CalledProcessError:
            console.print(Panel(
                "[yellow]Could not verify registration status.[/yellow]\n"
                "[yellow]Registration may still be processing on the network.[/yellow]",
                title="⚠️ Registration Status Unknown",
                border_style="yellow"
            ))
            return wallet_name, "Unknown status"

    except Exception as e:
        console.print(Panel(
            f"[red]Failed to register on subnet: {str(e)}[/red]",
            title="❌ Registration Error",
            border_style="red"
        ))
        return None, str(e)

def select_start_mode():
    choices = [
        'Miner',
        'Validator'
    ]
    answer = questionary.select(
        "Select mode:",
        choices=choices,
        style=custom_style,
        qmark="🚀"
    ).ask()
    return answer.lower() if answer else ""

@cli.command()
def start():
    mode = select_start_mode()
    if mode == 'validator':
        if is_bittensor_running():
            console.print("[warning]Bittensor miner is already running.[/warning]")
            return
        wallet_name = handle_bittensor_registration()
        if wallet_name:
            if start_bittensor_miner(wallet_name):
                console.print("[success]Bittensor miner started successfully![/success]")
                display_dashboard()
            else:
                console.print("[error]Failed to start Bittensor miner.[/error]")
    elif mode == 'miner':
        console.print("\n[info]Starting Miner program...[/info]")
        if not start_system():
            console.print("[error]Failed to start system process.[/error]")
            return
        if not start_polaris():
            console.print("[error]Failed to start API process.[/error]")
            stop_system()
            return
        console.print("[success]Miner processes started successfully![/success]")
        display_dashboard()
    else:
        console.print("[error]Unknown mode selected.[/error]")

@cli.command()
def stop():
    if is_bittensor_running():
        if stop_bittensor_miner():
            console.print("[success]Bittensor miner stopped successfully.[/success]")
        else:
            console.print("[error]Failed to stop Bittensor miner.[/error]")
    else:
        if stop_polaris():
            console.print("[success]Miner processes stopped successfully![/success]")
        else:
            console.print("[error]Failed to stop miner processes.[/error]")

@cli.command(name='status')
def status():
    if is_bittensor_running():
        if (BITTENSOR_CONFIG_PATH / 'pids' / 'miner.pid').exists():
            console.print("[success]Bittensor miner is running.[/success]")
        else:
            console.print("[warning]Bittensor miner is not running.[/warning]")
    else:
        check_status()

@cli.command(name='monitor')
def monitor():
    monitor_heartbeat()

@cli.group(name='update')
def update():
    pass

@update.command(name='subnet')
def update_subnet():
    if update_repository():
        console.print("[success]Repository update completed successfully.[/success]")
    else:
        console.print("[error]Failed to update repository.[/error]")
        exit(1)

@cli.command(name='check-main')
def check_main_command():
    check_main()

@cli.command(name='logs')
def view_logs():
    if is_bittensor_running():
        log_file = BITTENSOR_CONFIG_PATH / 'logs' / 'miner.log'
        if not log_file.exists():
            console.print("[warning]No Bittensor miner logs found.[/warning]")
            return
        try:
            subprocess.run(['tail', '-f', str(log_file)], check=True)
        except KeyboardInterrupt:
            pass
    else:
        from .log_monitor import monitor_process_and_logs
        monitor_process_and_logs()

if __name__ == "__main__":
    cli()
