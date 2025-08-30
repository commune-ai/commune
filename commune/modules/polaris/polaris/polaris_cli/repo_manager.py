import os
import shutil
import subprocess
import sys
from pathlib import Path

import git
from rich.console import Console

from src.utils import get_project_root

console = Console()

# Commented out as we're now using the local directory instead of cloning
# REPO_URL = "https://github.com/BANADDA/cloudserver.git"
REPO_FOLDER_NAME = "compute_subnet"

def get_repo_path():
    """Get the path where the repository should be."""
    project_root = get_project_root()
    return os.path.join(project_root, REPO_FOLDER_NAME)

def ensure_repository_exists():
    """
    Check if repository exists locally.
    
    Returns:
        tuple: (success: bool, main_py_path: str or None)
    """
    try:
        repo_path = get_repo_path()
        main_py_path = os.path.join(repo_path, "src", "main.py")
        
        if not os.path.exists(main_py_path):
            console.print(f"[red]Required files not found at {main_py_path}.[/red]")
            console.print("[yellow]Make sure compute_subnet is properly set up in your repository.[/yellow]")
            return False, None
        else:
            console.print("[green]Local compute_subnet directory found.[/green]")
            return True, main_py_path
    
    except Exception as e:
        console.print(f"[red]Failed to check repository: {e}[/red]")
        return False, None

def update_repository():
    """
    Since we're now using the local directory integrated with the main repo,
    this function is mainly a placeholder that just checks if files exist.
    
    Returns:
        bool: True if files exist, False otherwise.
    """
    try:
        repo_path = get_repo_path()
        main_py_path = os.path.join(repo_path, "src", "main.py")
        
        if not os.path.exists(main_py_path):
            console.print(f"[red]Required files not found at {main_py_path}.[/red]")
            return False
            
        console.print("[green]Local compute_subnet files are available.[/green]")
        return True
            
    except Exception as e:
        console.print(f"[red]Failed to check files: {e}[/red]")
        return False

def start_server(env_vars=None):
    """
    Start the uvicorn server with the correct configuration.
    
    Args:
        env_vars: Optional dictionary of environment variables
    
    Returns:
        subprocess.Popen: The server process
    """
    try:
        repo_path = get_repo_path()
        main_py_path = os.path.join(repo_path, "src", "main.py")
        
        if not os.path.exists(main_py_path):
            console.print(f"[red]Server entry point not found at {main_py_path}[/red]")
            return None

        # Prepare environment
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)

        # Start uvicorn server
        cmd = [
            'uvicorn',
            'src.main:app',
            '--reload',
            '--host', '0.0.0.0',
            '--port', '8000'
        ]

        process = subprocess.Popen(
            cmd,
            cwd=repo_path,  # Set working directory to repository root
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        console.print("[green]Server started successfully[/green]")
        return process

    except Exception as e:
        console.print(f"[red]Failed to start server: {e}[/red]")
        return None