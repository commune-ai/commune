# polaris_cli/view_pod.py

import json
import os
import sys

import requests
from rich import box
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from src.pid_manager import PID_FILE
from src.utils import configure_logging

logger = configure_logging()
console = Console()
server_url_ = os.getenv('SERVER_URL')

def fetch_pod_data(miner_id):
    """
    Fetches pod data from the backend API.
    """
    try:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("Fetching pod data...", total=None)
            api_url = f'{server_url_}/miners/{miner_id}'
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            pod_data = response.json()
            progress.update(task, completed=True)
            return pod_data
    except requests.HTTPError as http_err:
        console.print(f"[red]Failed to fetch pod data: {http_err}[/red]")
        sys.exit(1)
    except requests.Timeout:
        console.print("[red]Request timed out while fetching pod data.[/red]")
        sys.exit(1)
    except Exception as err:
        console.print(f"[red]Error fetching pod data: {err}[/red]")
        sys.exit(1)

def display_pod_info(pod_data):
    """
    Displays pod details and compute resources in tabular format using rich.
    """
    # Pod Details
    pod_info_table = Table(title="Pod Details", box=box.ROUNDED)
    pod_info_table.add_column("Field", style="cyan", no_wrap=True)
    pod_info_table.add_column("Value", style="magenta")
    
    pod_details = {
        "ID": pod_data.get('id', 'N/A'),
        "Name": pod_data.get('name', 'N/A'),
        "Location": pod_data.get('location', 'N/A'),
        "Description": pod_data.get('description', 'N/A'),
        "Created At": pod_data.get('created_at', 'N/A'),
        "Updated At": pod_data.get('updated_at', 'N/A'),
    }
    
    for key, value in pod_details.items():
        pod_info_table.add_row(key, str(value))
    
    console.print(pod_info_table)
    
    # Compute Resources
    compute_resources = pod_data.get('compute_resources', [])
    if not compute_resources:
        console.print("[yellow]No compute resources found for this pod.[/yellow]")
        return
    
    compute_table = Table(title="Compute Resources", box=box.ROUNDED)
    compute_table.add_column("ID", style="cyan", no_wrap=True)
    compute_table.add_column("Type", style="magenta")
    compute_table.add_column("Location", style="green")
    compute_table.add_column("Price/Hr", style="yellow")
    compute_table.add_column("RAM", style="blue")
    compute_table.add_column("Storage", style="magenta")
    compute_table.add_column("Specs", style="green")
    
    for resource in compute_resources:
        storage = resource.get('storage', {})
        storage_info = f"{storage.get('type', 'N/A')} {storage.get('capacity', 'N/A')}\nRead: {storage.get('read_speed', 'N/A')}\nWrite: {storage.get('write_speed', 'N/A')}"
        
        specs = ""
        if resource.get('resource_type', '').upper() == 'CPU':
            cpu_specs = resource.get('cpu_specs', {})
            specs = (
                f"CPU Name: {cpu_specs.get('cpu_name', 'N/A')}\n"
                f"Total CPUs: {cpu_specs.get('total_cpus', 'N/A')}\n"
                f"Threads/Core: {cpu_specs.get('threads_per_core', 'N/A')}\n"
                f"Max MHz: {cpu_specs.get('cpu_max_mhz', 'N/A')}"
            )
        elif resource.get('resource_type', '').upper() == 'GPU':
            gpu_specs = resource.get('gpu_specs', {})
            specs = (
                f"GPU Name: {gpu_specs.get('gpu_name', 'N/A')}\n"
                f"Memory: {gpu_specs.get('memory_size', 'N/A')}\n"
                f"CUDA Cores: {gpu_specs.get('cuda_cores', 'N/A')}\n"
                f"Clock Speed: {gpu_specs.get('clock_speed', 'N/A')}"
            )
        
        compute_table.add_row(
            resource.get('id', 'N/A'),
            resource.get('resource_type', 'N/A').upper(),
            resource.get('location', 'N/A'),
            f"${resource.get('hourly_price', 'N/A')}",
            resource.get('ram', 'N/A'),
            storage_info,
            specs
        )
    
    console.print(compute_table)

def view_pod():
    """
    Orchestrates the pod viewing process.
    """
    miner_id = Prompt.ask("Enter the Miner ID", default="")

    if not miner_id:
        console.print("[red]Miner ID is required.[/red]")
        sys.exit(1)
    
    pod_data = fetch_pod_data(miner_id)
    display_pod_info(pod_data)
