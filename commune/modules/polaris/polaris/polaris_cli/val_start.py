# polaris_cli/val_start.py
import ctypes
import json
import logging
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
from communex.compat.key import classic_load_key
from loguru import logger
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

# Initialize logging and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('validator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
console = Console()

# Constants
DETACHED_PROCESS = 0x00000008 if platform.system() == 'Windows' else 0
VALIDATOR_PID_DIR = os.path.join(os.path.expanduser('~'), '.polaris', 'validator', 'pids')
VALIDATOR_LOG_DIR = os.path.join(os.path.expanduser('~'), '.polaris', 'validator', 'logs')
STATUS_FILE = Path.home() / '.polaris' / 'validator' / 'status.json'

def get_validator_script_path():
    """Get the correct path to the validator main script"""
    # Start from the current file's directory
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent  # polaris-subnet directory
    
    # Try multiple possible paths
    possible_paths = [
        project_root / 'validator' / 'src' / 'validator_node' / 'validator' / 'main.py',
        project_root / 'src' / 'validator_node' / 'validator' / 'main.py',
        project_root / 'validator_node' / 'validator' / 'main.py'
    ]

    for path in possible_paths:
        if path.exists():
            return str(path)

    return None

def setup_directories():
    """Ensure validator directories exist"""
    os.makedirs(VALIDATOR_PID_DIR, exist_ok=True)
    os.makedirs(VALIDATOR_LOG_DIR, exist_ok=True)

def get_pid_file():
    """Get path to validator PID file"""
    return os.path.join(VALIDATOR_PID_DIR, "validator.pid")

def create_pid_file(pid):
    """Create PID file for validator"""
    try:
        with open(get_pid_file(), 'w') as f:
            f.write(str(pid))
        logger.info(f"Validator PID file created with PID: {pid}")
        return True
    except Exception as e:
        logger.error(f"Failed to create validator PID file: {e}")
        return False

def read_pid():
    """Read validator PID from PID file"""
    try:
        with open(get_pid_file(), 'r') as f:
            return int(f.read().strip())
    except:
        return None

def remove_pid_file():
    """Remove validator PID file"""
    try:
        os.remove(get_pid_file())
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        logger.error(f"Failed to remove validator PID file: {e}")
        return False

def update_status(status: str, error: str = None):
    """Update validator status file"""
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    status_data = {
        "status": status,
        "timestamp": time.time(),
        "error": error
    }
    STATUS_FILE.write_text(json.dumps(status_data))

def wait_for_validator_start(timeout=60):
    """Wait for validator to start and show progress"""
    start_time = time.time()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("[cyan]Starting validator...", total=100)
        
        while (time.time() - start_time) < timeout:
            if STATUS_FILE.exists():
                try:
                    status_data = json.loads(STATUS_FILE.read_text())
                    current_status = status_data.get("status")
                    
                    if current_status == "running":
                        progress.update(task, completed=100)
                        return True
                    elif current_status == "failed":
                        error_msg = status_data.get("error", "Unknown error")
                        console.print(Panel(
                            f"[red]{error_msg}[/red]",
                            title="Validator Start Failed",
                            border_style="red",
                            box=box.HEAVY
                        ))
                        return False
                except json.JSONDecodeError:
                    pass
            
            progress.update(task, advance=2)
            time.sleep(1)
        
        console.print("[red]Timeout waiting for validator to start[/red]")
        return False

def start_validator(wallet_name: str, testnet: bool = False):
    """Start validator process"""
    setup_directories()
    
    # Check if already running
    pid = read_pid()
    if pid and psutil.pid_exists(pid):
        console.print(f"[yellow]Validator is already running with PID {pid}.[/yellow]")
        return False

    # Get validator script path
    validator_path = get_validator_script_path()
    if not validator_path:
        console.print("[red]Validator script not found. Please check your installation.[/red]")
        console.print("[yellow]Expected locations:[/yellow]")
        console.print("- validator/src/validator_node/validator/main.py")
        console.print("- src/validator_node/validator/main.py")
        console.print("- validator_node/validator/main.py")
        return False

    # Setup log files
    stdout_log = os.path.join(VALIDATOR_LOG_DIR, 'validator_stdout.log')
    stderr_log = os.path.join(VALIDATOR_LOG_DIR, 'validator_stderr.log')

    # Load key to verify it exists
    try:
        key = classic_load_key(wallet_name)
    except Exception as e:
        console.print(Panel(
            f"[red]Failed to load wallet key: {str(e)}[/red]",
            title="Error",
            border_style="red",
            box=box.HEAVY
        ))
        return False

    # Clear any existing status file
    if STATUS_FILE.exists():
        STATUS_FILE.unlink()

    try:
        with open(stdout_log, 'a') as stdout_f, open(stderr_log, 'a') as stderr_f:
            cmd = [
                sys.executable,
                validator_path,
                "--wallet", wallet_name
            ]
            if testnet:
                cmd.append("--testnet")

            logger.info(f"Starting validator with command: {' '.join(cmd)}")
            
            if platform.system() == 'Windows':
                process = subprocess.Popen(
                    cmd,
                    stdout=stdout_f,
                    stderr=stderr_f,
                    creationflags=DETACHED_PROCESS,
                    close_fds=True
                )
            else:
                process = subprocess.Popen(
                    cmd,
                    stdout=stdout_f,
                    stderr=stderr_f,
                    start_new_session=True,
                    close_fds=True
                )

            if create_pid_file(process.pid):
                # Wait for validator to fully start
                if wait_for_validator_start():
                    console.print(Panel(
                        "[green]Validator started successfully![/green]\n"
                        f"[blue]PID: [white]{process.pid}[/white][/blue]\n"
                        f"[blue]Logs:[/blue]\n"
                        f"[white]stdout: {stdout_log}[/white]\n"
                        f"[white]stderr: {stderr_log}[/white]\n\n"
                        "[yellow]Use command [white]polaris val-logs[/white] to monitor the validator in real-time[/yellow]",
                        title="Success",
                        border_style="green",
                        box=box.HEAVY
                    ))
                    return True
                else:
                    process.kill()
                    remove_pid_file()
                    return False
            else:
                process.kill()
                return False

    except Exception as e:
        logger.error(f"Failed to start validator: {e}")
        console.print(Panel(
            f"[red]Failed to start validator: {str(e)}[/red]",
            title="Error",
            border_style="red",
            box=box.HEAVY
        ))
        return False

def stop_validator():
    """Stop validator process"""
    pid = read_pid()
    if not pid:
        console.print("[yellow]No validator process found.[/yellow]")
        return True

    try:
        process = psutil.Process(pid)
        console.print(f"[yellow]Terminating validator (PID {pid})...[/yellow]")

        # Try graceful shutdown
        process.terminate()
        try:
            process.wait(timeout=10)
            console.print(Panel(
                "[green]Validator stopped successfully.[/green]",
                title="Success",
                border_style="green",
                box=box.HEAVY
            ))
        except psutil.TimeoutExpired:
            # Force kill if graceful shutdown fails
            if platform.system() == 'Windows':
                subprocess.run(['taskkill', '/F', '/PID', str(pid)], check=True)
            else:
                os.kill(pid, signal.SIGKILL)
            console.print(Panel(
                "[yellow]Validator forcefully stopped.[/yellow]",
                title="Warning",
                border_style="yellow",
                box=box.HEAVY
            ))

        remove_pid_file()
        update_status("stopped")
        return True

    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        console.print(Panel(
            f"[red]Error stopping validator: {str(e)}[/red]",
            title="Error",
            border_style="red",
            box=box.HEAVY
        ))
        remove_pid_file()
        return False

def check_validator_status():
    """Check if validator is running"""
    pid = read_pid()
    if pid and psutil.pid_exists(pid):
        try:
            process = psutil.Process(pid)
            if process.is_running():
                # Check status file for more details
                if STATUS_FILE.exists():
                    try:
                        status_data = json.loads(STATUS_FILE.read_text())
                        status = status_data.get("status", "unknown")
                        console.print(Panel(
                            f"[green]Validator is running[/green]\n"
                            f"[blue]PID: [white]{pid}[/white][/blue]\n"
                            f"[blue]Status: [white]{status}[/white][/blue]",
                            title="Validator Status",
                            border_style="green",
                            box=box.HEAVY
                        ))
                    except json.JSONDecodeError:
                        console.print(f"[green]Validator is running with PID {pid}.[/green]")
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    console.print(Panel(
        "[yellow]Validator is not running.[/yellow]",
        title="Validator Status",
        border_style="yellow",
        box=box.HEAVY
    ))
    remove_pid_file()
    return False