# src/pid_manager.py

import logging
import os

from src.utils import get_project_root

logger = logging.getLogger("polaris_cli.pid_manager")

# Define the default PID file path using the project root
PID_FILE = os.path.join(get_project_root(), 'polaris.pid')


def create_pid_file():
    """
    Creates the default PID file to ensure only one instance runs.

    Returns:
        bool: True if the PID file was created successfully, False otherwise.
    """
    try:
        pid = os.getpid()
        with open(PID_FILE, 'w') as f:
            f.write(str(pid))
        logger.debug(f"Attempting to create PID file at: {PID_FILE}")
        logger.info(f"PID file created with PID: {pid}")
        return True
    except Exception as e:
        logger.exception(f"Failed to create PID file at {PID_FILE}: {e}")
        return False


def remove_pid_file():
    """
    Removes the default PID file if it exists.
    """
    try:
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
            logger.debug(f"Attempting to remove PID file at: {PID_FILE}")
            logger.info("PID file removed.")
        else:
            logger.debug(f"No PID file found at: {PID_FILE} to remove.")
    except Exception as e:
        logger.exception(f"Failed to remove PID file at {PID_FILE}: {e}")


# --- New Functions for Process-Specific PID Management ---

def get_pid_file(process_name):
    """
    Constructs the PID file path for a given process.

    Args:
        process_name (str): The name of the process.

    Returns:
        str: The full path to the PID file for the process.
    """
    return os.path.join(get_project_root(), f"{process_name}.pid")


def create_pid_file_for_process(process_name, pid):
    """
    Creates a PID file for a specific process.

    Args:
        process_name (str): The name of the process.
        pid (int): The process ID.

    Returns:
        bool: True if the PID file was created successfully, False otherwise.
    """
    pid_file = get_pid_file(process_name)
    try:
        with open(pid_file, 'w') as f:
            f.write(str(pid))
        logger.debug(f"Attempting to create PID file at: {pid_file}")
        logger.info(f"PID file for '{process_name}' created with PID: {pid}")
        return True
    except Exception as e:
        logger.exception(f"Failed to create PID file at {pid_file}: {e}")
        return False


def read_pid(process_name):
    """
    Reads the PID from the PID file of a specific process.

    Args:
        process_name (str): The name of the process.

    Returns:
        int or None: The PID if found and valid, else None.
    """
    pid_file = get_pid_file(process_name)
    if not os.path.exists(pid_file):
        logger.debug(f"PID file for '{process_name}' does not exist at: {pid_file}")
        return None

    try:
        with open(pid_file, 'r') as f:
            pid_str = f.read().strip()
            pid = int(pid_str)
            logger.debug(f"Read PID {pid} for '{process_name}' from {pid_file}")
            return pid
    except Exception as e:
        logger.exception(f"Failed to read PID from {pid_file}: {e}")
        return None


def remove_pid_file_for_process(process_name):
    """
    Removes the PID file for the specified process.
    
    Args:
        process_name (str): The name of the process ('polaris' or 'compute_subnet').
    
    Returns:
        bool: True if the file was removed successfully or does not exist, False otherwise.
    """
    pid_dir = os.path.join(get_project_root(), 'pids', process_name)
    pid_file = os.path.join(pid_dir, 'pid.txt')
    
    try:
        if os.path.exists(pid_file):
            os.remove(pid_file)
            logger.info(f"PID file for '{process_name}' removed.")
            return True
        else:
            logger.warning(f"PID file for '{process_name}' does not exist. Nothing to remove.")
            return False
    except Exception as e:
        logger.error(f"Failed to remove PID file for {process_name}: {e}")
        return False

