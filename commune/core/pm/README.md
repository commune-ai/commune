# Commune Process Manager (PM)

The Process Manager (PM) module in Commune provides tools for managing and orchestrating processes across your distributed system. It integrates with popular process management solutions like PM2 and Docker to give you flexible control over your application processes.

## Features

- **Multiple Process Management Options**: Support for PM2 and Docker-based process management
- **Unified Interface**: Common API for managing processes regardless of the underlying technology
- **Process Monitoring**: Tools to monitor and maintain process health
- **Configuration Management**: Easy configuration of process parameters
- **Scalability**: Designed to work with Commune's distributed architecture

## Components

### PM2 Integration

The PM module integrates with [PM2](https://pm2.keymetrics.io/), a popular production process manager for Node.js applications, but can be used for any application type. PM2 provides:

- Process monitoring
- Automatic restarts
- Log management
- Load balancing

### Docker Integration

The Docker integration allows you to manage containerized processes, providing:

- Isolation
- Consistent environments
- Resource management
- Scalability

## Usage

```python
from commune.core.pm import PM

# Initialize the Process Manager
pm = PM()

# Start a process
pm.start(name="my_process", script="path/to/script.py")

# Stop a process
pm.stop("my_process")

# Restart a process
pm.restart("my_process")

# Get process status
status = pm.status("my_process")
```

## Configuration

The PM module can be configured through the Commune configuration system. Example configuration:

```yaml
pm:
  default_manager: "pm2"  # or "docker"
  pm2:
    bin_path: "/usr/local/bin/pm2"
  docker:
    compose_path: "./docker-compose.yml"
```

## Installation

The PM module is included in the Commune framework. If you're using it standalone, ensure you have the required dependencies:

```bash
# For PM2 integration
npm install pm2 -g

# For Docker integration
pip install docker-compose
```

## Contributing

Contributions to the PM module are welcome! Please follow the Commune contribution guidelines when submitting changes.

## License

This module is part of the Commune framework and is released under the same license terms as the main project.
