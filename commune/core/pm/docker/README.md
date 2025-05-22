 # start of file
# Docker Module for Commune

This module provides a comprehensive interface for managing Docker containers, similar to how PM2 manages processes.

## Features

- Build and run Docker containers with advanced configuration options
- Manage container lifecycle (start, stop, restart, delete)
- Monitor container resource usage
- Save and load container configurations
- Execute commands in running containers
- View container logs
- List and manage Docker images

## Usage

```python
import commune as c

# Initialize the Docker module
docker = c.module('docker')()

# Start a container
docker.start('my_container', 'python:3.8', 
             cmd='python -m http.server',
             ports={'8000': 8000})

# List running containers
containers = docker.list()
print(containers)

# Monitor container resource usage
stats = docker.monitor()
print(stats)

# Execute a command in a container
result = docker.exec('my_container', 'ls -la')
print(result)

# View container logs
logs = docker.logs('my_container')
print(logs)

# Stop a container
docker.stop('my_container')

# Remove a container
docker.delete('my_container')

# Save current container configuration
docker.save('my_setup')

# Load a saved configuration
docker.load('my_setup')
```

## PM2-like Interface

The Docker module provides a PM2-like interface for managing containers:

- `start(name, image, **kwargs)`: Start a container
- `stop(name)`: Stop a container
- `restart(name)`: Restart a container
- `delete(name)`: Remove a container
- `list(all=False)`: List containers
- `monitor()`: Monitor container resource usage
- `save(config_name)`: Save current container configuration
- `load(config_name)`: Load a saved configuration

## Advanced Options

The module supports advanced Docker features:

- GPU configuration
- Volume mapping
- Port mapping
- Environment variables
- Network configuration
- Shared memory size
- Custom entrypoints and commands
