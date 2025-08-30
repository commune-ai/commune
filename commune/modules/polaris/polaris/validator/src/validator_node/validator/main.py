import json
import os
import signal
import sys
import time
import asyncio
import subprocess
from pathlib import Path
from typing import Optional

import typer
from communex.compat.key import classic_load_key
from loguru import logger
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from substrateinterface import Keypair

from validator.src.validator_node._config import ValidatorSettings
from validator.src.validator_node.base.utils import get_netuid
from validator.src.validator_node.validator_ import ValidatorNode

app = typer.Typer()
console = Console()

# Constants for status tracking
STATUS_FILE = Path.home() / '.polaris' / 'validator' / 'status.json'

def setup_logging():
    """Setup validator logging"""
    log_dir = os.path.join(os.path.expanduser('~'), '.polaris', 'validator', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'validator.log')
    
    logger.remove()  # Remove default handler
    logger.add(sys.stdout, level="INFO")
    logger.add(log_file, rotation="500 MB", level="DEBUG")
    return log_file

def update_status(status: str, error: str = None):
    """Update validator status file
    
    Args:
        status (str): Current status of the validator
        error (str, optional): Error message if any. Defaults to None.
    """
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    status_data = {
        "status": status,
        "timestamp": time.time(),
        "error": error
    }
    STATUS_FILE.write_text(json.dumps(status_data))

def cleanup(signal, frame):
    """Cleanup handler for graceful shutdown"""
    logger.info("Initiating cleanup process...")
    update_status("stopped")
    sys.exit(0)

def initialize_client(settings: ValidatorSettings):
    """Initialize the substrate client based on settings
    
    Args:
        settings (ValidatorSettings): Validator settings
        
    Returns:
        SubstrateInterface: Initialized substrate client
    """
    # This is a placeholder - implement actual client initialization
    # based on your specific needs
    return None

@app.command()
def main(
    wallet: str = typer.Option(..., help="Wallet name"),
    testnet: bool = typer.Option(False, help="Use testnet endpoints"),
    log_level: str = typer.Option("INFO", help="Logging level"),
    host: str = typer.Option("0.0.0.0", help="Host address to bind to"),
    port: int = typer.Option(8000, help="Port to listen on"),
    iteration_interval: int = typer.Option(800, help="Interval between validation iterations"),
    network: str = typer.Option("commune", help="Network to validate for (commune or bittensor)"),
    netuid: int = typer.Option(None, help="Network UID to validate (defaults to 33 for Commune, 49 for Bittensor)")
):
    """Main validator process
    
    This is the entry point for the validator node. It handles:
    - Logging setup
    - Key management
    - Validator initialization
    - Main validation loop
    - Graceful shutdown
    """
    # Setup signal handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    # Validate network parameter
    if network not in ["commune", "bittensor"]:
        console.print(Panel(
            f"[red]Invalid network '{network}'. Must be 'commune' or 'bittensor'.[/red]", 
            title="Error", 
            border_style="red",
            box=box.HEAVY
        ))
        return
    
    # Initialize logging
    log_file = setup_logging()
    logger.info(f"Initialized logging to {log_file}")
    update_status("starting")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
        transient=False
    ) as progress:
        try:
            # Task for key loading
            key_task = progress.add_task("[cyan]Loading key...", total=1)
            
            # Load key
            try:
                # Load network-specific key
                if network == "commune":
                    logger.info(f"Loading Commune key for wallet: {wallet}")
                    key = classic_load_key(wallet)
                else:  # bittensor
                    logger.info(f"Loading Bittensor key for wallet: {wallet}")
                    # For now, we'll still use the Commune key format internally
                    # This would need to be replaced with proper Bittensor key loading
                    key = classic_load_key(wallet)
                
                logger.info(f"Loaded key for wallet: {wallet}")
                progress.update(key_task, completed=1)
            except Exception as e:
                error_msg = f"Failed to load wallet key: {str(e)}"
                logger.error(error_msg)
                update_status("failed", error_msg)
                console.print(Panel(
                    f"[red]{error_msg}[/red]", 
                    title="Error", 
                    border_style="red",
                    box=box.HEAVY
                ))
                return

            # Task for validator initialization
            init_task = progress.add_task("[cyan]Initializing validator node...", total=1)
            
            try:
                # Set netuid based on network if not provided
                if netuid is None:
                    if network == "commune":
                        netuid = 33  # Default for Commune
                    else:  # bittensor
                        netuid = 49  # Default for Bittensor Polaris
                
                # Initialize settings with command line parameters
                settings = ValidatorSettings(
                    use_testnet=testnet,
                    logging_level=log_level,
                    host=host,
                    port=port,
                    iteration_interval=iteration_interval,
                    max_allowed_weights=420  # Default value
                )
                
                # Add network-specific settings
                if network == "commune":
                    # Additional Commune settings
                    pass
                else:  # bittensor
                    # Add Bittensor-specific settings
                    setattr(settings, 'bittensor_netuid', netuid)
                    setattr(settings, 'bittensor_network', 'finney' if not testnet else 'test')
                    
                # Initialize the appropriate validator
                if network == "commune":
                    logger.info(f"Initializing Commune validator for netuid {netuid}")
                    from validator.src.validator_node.commune_validator import CommuneValidator
                    validator = CommuneValidator(
                        key=key,
                        settings=settings
                    )
                else:  # bittensor
                    logger.info(f"Initializing Bittensor validator for netuid {netuid}")
                    from validator.src.validator_node.bittensor_validator import BittensorValidator
                    validator = BittensorValidator(
                        key=key,
                        settings=settings
                    )
                
                progress.update(init_task, completed=1)
                
            except Exception as e:
                error_msg = f"Failed to initialize validator: {str(e)}"
                logger.error(error_msg)
                update_status("failed", error_msg)
                console.print(Panel(
                    f"[red]{error_msg}[/red]", 
                    title="Error", 
                    border_style="red",
                    box=box.HEAVY
                ))
                return

            # Task for startup
            startup_task = progress.add_task(f"[cyan]Starting {network} validator...", total=1)
            
            try:
                # Initialize the network connection
                asyncio.run(validator.initialize_network())
                
                # Start validator API
                command = f'uvicorn validator.src.validator_node.api:app --host {host} --port {port} --log-level warning'
                logger.info(f"Starting API server: {command}")
                subprocess.Popen(command, shell=True)
                
                # Start validation loop in a separate thread
                validator.start_validation_loop()
                
                progress.update(startup_task, completed=1)
                
                # Update status file
                update_status("running")
                
                # Display success message
                console.print(Panel(
                    f"[green]Validator started successfully on {host}:{port}[/green]\n"
                    f"[white]Network: {network.upper()}[/white]\n"
                    f"[white]NetUID: {netuid}[/white]\n"
                    f"[white]Testnet: {'Yes' if testnet else 'No'}[/white]\n"
                    f"[white]Validation Interval: {iteration_interval}s[/white]",
                    title="✅ Validator Running",
                    border_style="green",
                    box=box.HEAVY
                ))
                
                # Wait for termination signal
                signal.pause()
                
            except Exception as e:
                error_msg = f"Failed to start validator: {str(e)}"
                logger.error(error_msg)
                update_status("failed", error_msg)
                console.print(Panel(
                    f"[red]{error_msg}[/red]", 
                    title="Error", 
                    border_style="red",
                    box=box.HEAVY
                ))
                return
                
        except KeyboardInterrupt:
            console.print("\n")
            console.print(Panel(
                "[yellow]Validator process cancelled by user.[/yellow]",
                title="ℹ️ Cancelled",
                border_style="yellow"
            ))
            update_status("stopped")
            logger.info("Validator process cancelled by user.")

if __name__ == "__main__":
    app()
