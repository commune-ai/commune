#!/usr/bin/env python3
import ctypes
import logging
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
from dotenv import load_dotenv
from rich.console import Console

from polaris_cli.repo_manager import (ensure_repository_exists,
                                      update_repository)

# Initialize logging and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('polaris.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
console = Console()

# Constants
DETACHED_PROCESS = 0x00000008 if platform.system() == 'Windows' else 0
PID_DIR = os.path.join(os.path.expanduser('~'), '.polaris', 'pids')

# ---------------- Utility Functions ----------------

def is_admin():
    """Check if the current process has admin privileges."""
    try:
        if platform.system() == 'Windows':
            return ctypes.windll.shell32.IsUserAnAdmin()
        else:
            return os.geteuid() == 0
    except Exception:
        return False

def elevate_privileges():
    """Restart the current script with elevated privileges."""
    if platform.system() == 'Windows':
        script = os.path.abspath(sys.argv[0])
        params = ' '.join(sys.argv[1:])
        try:
            if not is_admin():
                ctypes.windll.shell32.ShellExecuteW(
                    None,
                    "runas",
                    sys.executable,
                    f'"{script}" {params}',
                    None,
                    1
                )
                sys.exit()
        except Exception as e:
            logger.error(f"Failed to elevate privileges: {e}")
            return False
    else:
        if not is_admin():
            script = os.path.abspath(sys.argv[0])
            params = ' '.join(sys.argv[1:])
            try:
                cmd = ['sudo', sys.executable, script]
                if params:
                    cmd.extend(params.split())
                subprocess.run(cmd, check=True)
                sys.exit()
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to elevate privileges: {e}")
                return False
    return True

def get_project_root():
    """
    Get the project root directory.
    (Assuming this launcher is two levels deep from the root.)
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def ensure_pid_directory():
    """Ensure PID directory exists."""
    os.makedirs(PID_DIR, exist_ok=True)

def get_pid_file(process_name):
    """Get path to PID file for a given process name."""
    return os.path.join(PID_DIR, f"{process_name}.pid")

def create_pid_file(process_name, pid):
    """Create PID file for a process."""
    try:
        with open(get_pid_file(process_name), 'w') as f:
            f.write(str(pid))
        logger.info(f"PID file for '{process_name}' created with PID: {pid}")
        return True
    except Exception as e:
        logger.error(f"Failed to create PID file for {process_name}: {e}")
        return False

def read_pid(process_name):
    """Read PID from PID file."""
    try:
        with open(get_pid_file(process_name), 'r') as f:
            return int(f.read().strip())
    except Exception:
        return None

def remove_pid_file(process_name):
    """Remove PID file for a process."""
    try:
        os.remove(get_pid_file(process_name))
        return True
    except FileNotFoundError:
        logger.warning(f"PID file for '{process_name}' does not exist. Nothing to remove.")
        return False
    except Exception as e:
        logger.error(f"Failed to remove PID file for {process_name}: {e}")
        return False

def stop_process(pid, process_name, force=False):
    """Stop a single process with privilege handling."""
    try:
        process = psutil.Process(pid)
        console.print(f"[yellow]Terminating {process_name} (PID {pid})...[/yellow]")
        
        # Check if this is a heartbeat process
        is_heartbeat = False
        try:
            cmd = " ".join(process.cmdline())
            if 'heartbeat_service.py' in cmd:
                is_heartbeat = True
                console.print(f"[yellow]This is a heartbeat process. Using enhanced termination...[/yellow]")
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
            
        try:
            # First try sending SIGTERM
            process.terminate()
            
            # Heartbeat processes may need extra attention
            if is_heartbeat:
                # Wait up to 3 seconds for process to terminate
                gone, alive = psutil.wait_procs([process], timeout=3)
                if process in alive:
                    console.print(f"[yellow]{process_name} (PID {pid}) did not terminate after 3 seconds, killing forcefully...[/yellow]")
                    process.kill()
                    gone, alive = psutil.wait_procs([process], timeout=3)
                    if process in alive:
                        console.print(f"[red]WARNING: {process_name} (PID {pid}) could not be killed![/red]")
                        if platform.system() != 'Windows':
                            # Last resort - use SIGKILL directly
                            os.kill(pid, signal.SIGKILL)
                            console.print(f"[yellow]Sent SIGKILL directly to process {pid}[/yellow]")
                    else:
                        console.print(f"[green]{process_name} (PID {pid}) forcefully killed.[/green]")
                else:
                    console.print(f"[green]{process_name} (PID {pid}) terminated gracefully.[/green]")
                return True
            else:
                # For non-heartbeat processes, use the original logic
                try:
                    process.wait(timeout=10)
                    console.print(f"[green]{process_name} (PID {pid}) stopped successfully.[/green]")
                    return True
                except psutil.TimeoutExpired:
                    if force:
                        if platform.system() == 'Windows':
                            subprocess.run(['taskkill', '/F', '/PID', str(pid)], check=True)
                        else:
                            os.kill(pid, signal.SIGKILL)
                        console.print(f"[green]{process_name} forcefully stopped.[/green]")
                        return True
                    return False
        except psutil.AccessDenied:
            if not is_admin():
                console.print("[yellow]Requesting elevated privileges...[/yellow]")
                if elevate_privileges():
                    return stop_process(pid, process_name, force)
            return False
    except psutil.NoSuchProcess:
        console.print(f"[yellow]Process {pid} not found.[/yellow]")
        return True
    except Exception as e:
        logger.error(f"Failed to stop {process_name}: {e}")
        # Last resort for heartbeat - try direct kill command
        if process_name == 'heartbeat':
            try:
                console.print(f"[yellow]Attempting emergency process kill for {process_name}...[/yellow]")
                if platform.system() == 'Windows':
                    subprocess.run(['taskkill', '/F', '/PID', str(pid)], check=True, capture_output=True)
                else:
                    result = subprocess.run(['kill', '-9', str(pid)], check=False, capture_output=True)
                    if result.returncode == 0:
                        console.print(f"[green]Emergency kill successful for {process_name}.[/green]")
                        return True
                    else:
                        console.print(f"[red]Emergency kill failed: {result.stderr.decode()}[/red]")
            except Exception as kill_error:
                console.print(f"[red]Failed emergency kill: {kill_error}[/red]")
        return False

def stop_all(process_names):
    """Stop all processes in the given list of process names and any unicorn processes on port 8000."""
    if "unicorn" not in process_names:
        process_names = process_names + ["unicorn"]
        console.print("[blue]Adding unicorn to processes to stop...[/blue]")
    
    # Check for heartbeat specifically
    heartbeat_cleanup_needed = "heartbeat" in process_names
    
    success = True
    for name in process_names:
        pid = read_pid(name)
        if not pid:
            console.print(f"[yellow]{name} is not running (no PID file found).[/yellow]")
            continue
        if not stop_process(pid, name, force=False):
            console.print(f"[yellow]Attempting forced shutdown of {name}...[/yellow]")
            if not stop_process(pid, name, force=True):
                console.print(f"[red]Failed to stop {name}.[/red]")
                success = False
                continue
        remove_pid_file(name)
    
    # Special handling for heartbeat - try to find and kill any heartbeat process even without PID file
    if heartbeat_cleanup_needed:
        # Multiple approaches to ensure heartbeat is truly dead
        try:
            # First try pgrep to find any Python processes with heartbeat_service.py
            console.print("[blue]Checking for any rogue heartbeat processes...[/blue]")
            result = subprocess.run(['pgrep', '-f', 'heartbeat_service.py'], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid_str in pids:
                    try:
                        pid = int(pid_str.strip())
                        console.print(f"[yellow]Found rogue heartbeat process: {pid}. Attempting to terminate...[/yellow]")
                        # First try polite SIGTERM
                        os.kill(pid, signal.SIGTERM)
                        time.sleep(2)  # Give it 2 seconds to terminate gracefully
                        
                        # Check if it's still running
                        if psutil.pid_exists(pid):
                            console.print(f"[yellow]Process {pid} still running, using SIGKILL...[/yellow]")
                            os.kill(pid, signal.SIGKILL)
                            time.sleep(1)  # Give it a second to be killed
                            
                            # Final check
                            if psutil.pid_exists(pid):
                                console.print(f"[red]WARNING: Process {pid} could not be killed![/red]")
                                success = False
                            else:
                                console.print(f"[green]Successfully terminated heartbeat process {pid}[/green]")
                        else:
                            console.print(f"[green]Successfully terminated heartbeat process {pid}[/green]")
                    except Exception as e:
                        console.print(f"[red]Failed to kill heartbeat process {pid_str}: {e}[/red]")
                        success = False
            else:
                console.print("[green]No rogue heartbeat processes found via pgrep.[/green]")
                
            # As a backup, try ps + grep to find any missed processes
            console.print("[blue]Double-checking with ps + grep...[/blue]")
            ps_result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            if ps_result.returncode == 0:
                lines = ps_result.stdout.splitlines()
                for line in lines:
                    if 'heartbeat_service.py' in line and 'grep' not in line:
                        parts = line.split()
                        if len(parts) > 1:
                            try:
                                pid = int(parts[1])
                                console.print(f"[yellow]Found additional heartbeat process via ps: {pid}[/yellow]")
                                os.kill(pid, signal.SIGKILL)
                                console.print(f"[green]Sent SIGKILL to process {pid}[/green]")
                            except Exception as e:
                                console.print(f"[red]Failed to kill process: {e}[/red]")
                                success = False
            
            # Final safety - use pkill as a backstop
            console.print("[blue]Final safety check - using pkill...[/blue]")
            pkill_result = subprocess.run(['pkill', '-9', '-f', 'heartbeat_service.py'], capture_output=True, text=True)
            if pkill_result.returncode == 0:
                console.print("[green]pkill cleanup successful[/green]")
            else:
                console.print("[yellow]pkill found no processes to kill (good)[/yellow]")
                
        except Exception as e:
            console.print(f"[red]Error during heartbeat cleanup: {e}[/red]")
            
        # Verify no heartbeat processes remain
        time.sleep(1)  # Brief pause to let system catch up
        try:
            verify_result = subprocess.run(['pgrep', '-f', 'heartbeat_service.py'], capture_output=True, text=True)
            if verify_result.returncode == 0 and verify_result.stdout.strip():
                console.print("[red]WARNING: Some heartbeat processes still remain after cleanup![/red]")
                success = False
            else:
                console.print("[green]Verified: All heartbeat processes terminated.[/green]")
        except Exception as e:
            console.print(f"[red]Error during final verification: {e}[/red]")
    
    try:
        # Check for unicorn processes on port 8000
        result = subprocess.run(["lsof", "-i", ":8000", "-t"], capture_output=True, text=True)
        if result.returncode == 0:
            port_pids = result.stdout.strip().split("\n")
            for port_pid in port_pids:
                if port_pid.strip():
                    # Verify it's a unicorn process
                    proc_check = subprocess.run(["ps", "-p", port_pid, "-o", "comm="], capture_output=True, text=True)
                    if "unicorn" in proc_check.stdout.lower():
                        console.print(f"[yellow]Found unicorn process {port_pid} running on port 8000.[/yellow]")
                        if not stop_process(int(port_pid), "unicorn", force=False):
                            console.print(f"[yellow]Attempting forced shutdown of unicorn on port 8000...[/yellow]")
                            if not stop_process(int(port_pid), "unicorn", force=True):
                                console.print(f"[red]Failed to stop unicorn process {port_pid} on port 8000.[/red]")
                                success = False
    except Exception as e:
        console.print(f"[red]Error checking for unicorn processes on port 8000: {e}[/red]")
    
    return success

def check_status_for(process_names):
    """Check if processes are running for the given list of names."""
    all_running = True
    for name in process_names:
        pid = read_pid(name)
        if pid and psutil.pid_exists(pid):
            try:
                process = psutil.Process(pid)
                if process.is_running():
                    console.print(f"[green]{name} is running with PID {pid}.[/green]")
                    continue
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        console.print(f"[yellow]{name} is not running.[/yellow]")
        all_running = False
        remove_pid_file(name)
    return all_running

# ---------------- Mode-specific Start Functions ----------------

def start_api():
    """Start the API server (FastAPI app) and optionally the Heartbeat service."""
    ensure_pid_directory()
    project_root = get_project_root()  # e.g., /home/tang/polaris-subnet
    env_path = os.path.join(project_root, '.env')
    load_dotenv(dotenv_path=env_path)
    ensure_repository_exists()

    # Setup common log directory
    log_dir = os.path.join(project_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # --- Start API Server (Uvicorn) with nohup ---
    api_cwd = os.path.join(project_root, 'compute_subnet')
    api_log = os.path.join(log_dir, 'api_server.log')
    
    # Build the nohup command
    api_cmd = [
        'nohup', 'python3',
        '-m', 'uvicorn',
        'src.main:app',
        '--reload',
        '--host', '0.0.0.0',
        '--port', '8000',
        '>', api_log, '2>&1', '&'
    ]
    
    try:
        # Use shell=True to properly interpret the redirection operators
        process = subprocess.Popen(
            ' '.join(api_cmd),
            cwd=api_cwd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait briefly for the process to start
        time.sleep(2)
        
        # Find the actual PID of the uvicorn process, as nohup will create a child process
        uvicorn_pid = None
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmd = proc.info.get('cmdline', [])
                if cmd and 'uvicorn' in ' '.join(cmd) and 'src.main:app' in ' '.join(cmd):
                    uvicorn_pid = proc.info['pid']
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if not uvicorn_pid:
            console.print("[yellow]Warning: Started API server but couldn't determine its PID.[/yellow]")
            # Try to use the process group ID instead
            try:
                # Get process group ID from the shell process
                pgid = os.getpgid(process.pid)
                console.print(f"[yellow]Using process group ID {pgid} instead.[/yellow]")
                uvicorn_pid = pgid
            except Exception as e:
                console.print(f"[red]Failed to get process group ID: {e}[/red]")
                return False
        
        logger.info(f"API server started with nohup and PID: {uvicorn_pid}")
        console.print("[blue]API server started with nohup...[/blue]")
    except Exception as e:
        console.print(f"[red]Failed to start API server: {e}[/red]")
        return False

    if create_pid_file('api', uvicorn_pid):
        console.print(f"[green]API server running on PID {uvicorn_pid}[/green]")
        console.print(f"[blue]API logs: {api_log}[/blue]")
    else:
        console.print("[red]Failed to create PID file for API server.[/red]")
        return False

    # --- (Optional) Start Heartbeat Service with nohup ---
    heartbeat_log = os.path.join(log_dir, 'heartbeat.log')
    heartbeat_cmd = [
        'nohup', 'python3',
        os.path.join(project_root, 'polaris_cli', 'heartbeat_service.py'),
        '>', heartbeat_log, '2>&1', '&'
    ]
    
    try:
        # Use shell=True to properly interpret the redirection operators
        process = subprocess.Popen(
            ' '.join(heartbeat_cmd),
            cwd=project_root,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait briefly for the process to start
        time.sleep(2)
        
        # Find the actual PID of the heartbeat process
        heartbeat_pid = None
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmd = proc.info.get('cmdline', [])
                if cmd and 'heartbeat_service.py' in ' '.join(cmd):
                    heartbeat_pid = proc.info['pid']
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if not heartbeat_pid:
            console.print("[yellow]Warning: Started Heartbeat service but couldn't determine its PID.[/yellow]")
            # Try to use the process group ID instead
            try:
                pgid = os.getpgid(process.pid)
                console.print(f"[yellow]Using process group ID {pgid} instead.[/yellow]")
                heartbeat_pid = pgid
            except Exception as e:
                console.print(f"[yellow]Failed to get process group ID for heartbeat: {e}[/yellow]")
                logger.warning("Heartbeat service started but PID tracking failed")
                
                # IMPORTANT FIX: Instead of continuing silently, try a more aggressive search for any python process running heartbeat
                console.print("[yellow]Trying alternative method to find heartbeat process...[/yellow]")
                try:
                    result = subprocess.run(['pgrep', '-f', 'heartbeat_service.py'], capture_output=True, text=True)
                    if result.returncode == 0 and result.stdout.strip():
                        heartbeat_pid = int(result.stdout.strip())
                        console.print(f"[green]Found heartbeat process using pgrep: {heartbeat_pid}[/green]")
                    else:
                        console.print("[red]Could not find heartbeat process using pgrep.[/red]")
                        # Don't return here - warn the user but continue
                except Exception as alt_e:
                    console.print(f"[red]Alternative method failed: {alt_e}[/red]")
                
                # Even if we can't track it, warn clearly
                if not heartbeat_pid:
                    console.print("[red]WARNING: Heartbeat service may be running but couldn't be tracked.[/red]")
                    console.print("[red]You may need to manually kill it later with 'pkill -f heartbeat_service.py'[/red]")
        
        if heartbeat_pid:
            logger.info(f"Heartbeat service started with nohup and PID: {heartbeat_pid}")
            console.print("[blue]Heartbeat service started with nohup...[/blue]")
            
            # Create PID file for heartbeat
            if create_pid_file('heartbeat', heartbeat_pid):
                console.print(f"[green]Heartbeat service running on PID {heartbeat_pid}[/green]")
                console.print(f"[blue]Heartbeat logs: {heartbeat_log}[/blue]")
            else:
                console.print("[yellow]Warning: Failed to create PID file for Heartbeat service.[/yellow]")
                return True  # Continue even if PID file creation fails
        else:
            # If we couldn't find the PID, warn about it specifically
            console.print("[red]WARNING: Heartbeat service may be running but couldn't be tracked![/red]")
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to start Heartbeat service: {e}[/yellow]")
        logger.error(f"Failed to start Heartbeat service: {e}")
    
    return True

def start_system():
    """Start the System tasks process (runs the system main script)."""
    ensure_pid_directory()
    project_root = get_project_root()  # e.g., /home/tang/polaris-subnet
    env_path = os.path.join(project_root, '.env')
    load_dotenv(dotenv_path=env_path)
    ensure_repository_exists()

    # Setup common log directory
    log_dir = os.path.join(project_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    sys_stdout_log = os.path.join(log_dir, 'system_main.log')
    sys_stderr_log = os.path.join(log_dir, 'system_main_error.log')
    # The system main is located at /home/tang/polaris-subnet/src/main.py
    sys_cmd = [
        'python3',
        os.path.join(project_root, 'src', 'main.py')
    ]
    try:
        with open(sys_stdout_log, 'a') as stdout, open(sys_stderr_log, 'a') as stderr:
            sys_proc = subprocess.Popen(
                sys_cmd,
                cwd=project_root,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True
            )
        logger.info(f"System process started with command: {' '.join(sys_cmd)}")
        console.print("[blue]System process started...[/blue]")
    except Exception as e:
        console.print(f"[red]Failed to start system process: {e}[/red]")
        sys.exit(1)

    if create_pid_file('system', sys_proc.pid):
        console.print(f"[green]System process running on PID {sys_proc.pid}[/green]")
        console.print(f"[blue]System logs: {sys_stdout_log} and {sys_stderr_log}[/blue]")
    else:
        sys_proc.kill()
        sys.exit(1)

    return True

def stop_api():
    """Stop the API-related processes (API server and heartbeat)."""
    names = ['api', 'heartbeat']
    return stop_all(names)

def stop_system():
    """Stop the System tasks process."""
    return stop_all(['system'])

def status_api():
    """Check status of the API-related processes."""
    return check_status_for(['api', 'heartbeat'])

def status_system():
    """Check status of the System tasks process."""
    return check_status_for(['system'])

# ---------------- Main Dispatcher ----------------

def main():
    """
    Usage:
      polaris [api|system] [start|stop|status]

    - 'api' mode starts the FastAPI server (and heartbeat service) from compute_subnet.
    - 'system' mode starts the system tasks process from src/main.py.
    """
    if len(sys.argv) != 3:
        console.print("[red]Usage: polaris [api|system] [start|stop|status][/red]")
        sys.exit(1)

    mode = sys.argv[1].lower()
    command = sys.argv[2].lower()

    if not is_admin():
        console.print("[yellow]Not running with administrative privileges. Attempting to elevate...[/yellow]")
        if elevate_privileges():
            sys.exit(0)
        else:
            console.print("[red]Failed to obtain administrative privileges.[/red]")
            sys.exit(1)

    if mode == "api":
        if command == "start":
            start_api()
        elif command == "stop":
            if not stop_api():
                sys.exit(1)
        elif command == "status":
            if not status_api():
                sys.exit(1)
        else:
            console.print(f"[red]Unknown command: {command}[/red]")
            sys.exit(1)
    elif mode == "system":
        if command == "start":
            start_system()
        elif command == "stop":
            if not stop_system():
                sys.exit(1)
        elif command == "status":
            if not status_system():
                sys.exit(1)
        else:
            console.print(f"[red]Unknown command: {command}[/red]")
            sys.exit(1)
    else:
        console.print("[red]Unknown mode. Use 'api' or 'system'.[/red]")
        sys.exit(1)

start_polaris = start_api
stop_polaris = stop_api
check_status = status_api

if __name__ == "__main__":
    main()
