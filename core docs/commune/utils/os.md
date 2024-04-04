# Operating System and Network Monitoring Utilities

This module provides utilities for handling operating system tasks and network monitoring which includes checking for the existence of a Unix process ID, termination of a process, running an OS command, ensuring a directory path exists and, monitoring network upload and download metrics. Consistency in output (especially for machine learning models) can be achieved by using the seed function in this module.

## Functions:

### `check_pid(pid: int) -> bool`
- Checks if a Unix process ID exists which is useful when monitoring system processes.
  
### `kill_process(pid: int)`
- Terminates a process which can help in resource management and process control.

### `run_command(command: str)`
- Runs an operating system command, this function is a helper for executing shell commands programmatically.

### `path_exists(path: str) -> bool`
- Checks if a directory path exists, useful for file and path management operations.

### `ensure_path(path: str) -> str`
- Ensures that a directory exists, otherwise, it creates it; helpful in file manipulation tasks to avoid `FileNotFound` errors.

### `seed_everything(seed: int)`
- Sets the seed for generating random numbers for numpy, torch, python's built in random library & `PYTHONHASHSEED` environment. The provided seed maintains reproducibility across multiple runs.


## Class:
### `NetworkMonitor`
- A context manager class for monitoring network uploads and downloads. 

## Usage:
- These utility functions are useful in a wide range of programming contexts, particularly for system administrative tasks, file management, and network monitoring. 

## Note:
The `kill_pid` function in the module utilizes the `kill_process` function which serves the same functionality.