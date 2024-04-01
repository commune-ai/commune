# Readme

This script creates a Python module for managing sub-processes. By instantiating `SubprocessModule`, users can add, remove, and list sub-processes. The state of the subprocess can be cached for later use. The script also provides socket connection status checking functionality.

## Prerequisites

- Python 3
- Access to Unix-like command-line shell
- Commune library installed
- shlex library installed
- subprocess library installed
- socket library installed

## Code Explanation

- `SubprocessModule` class extends `Module` class from commune package. It overloads the constructor to initialize a cache path and also defines a custom reduce method for pickling.
- `subprocess_map` property provides access to the process cache.
- `rm_subprocess` or `rm` method removes a specified sub-process, it also removes it from the cache if its there.
- `rm_all` method removes all listed sub-processes from the cache.
- `add_subprocess`, `add`, or `submit` method adds sub-processes to run in the background. It accepts a string command to execute in the shell.
- `ls`, `ls_keys`, `list_keys` or `list` method lists all keys in the sub-process map.
- `portConnection` property checks the connection status of a socket port on the host.
- `streamlit` class method writes the sub-process map to the Streamlit interface.
- When script is run as main, it runs the SubprocessModule.

## Quick Start

```python
subprocess_module = SubprocessModule()

# Add a subprocess
subprocess_module.add_subprocess(command='ls -l')

# List all subprocesses
print(subprocess_module.ls())

# Remove a subprocess
removed_pid = subprocess_module.rm('12345')    # Replace '12345' with actual PID
print(f"Removed process with PID: {removed_pid}")

# Check socket connection on port 8080
is_connected = subprocess_module.portConnection(8080)
print(f"Connection on port 8080: {is_connected}")
```

## Note

Make sure to handle all necessary exceptions while managing sub-processes. Do not share your `SubprocessModule` instances over an untrustworthy network as it may lead to security vulnerabilities.
