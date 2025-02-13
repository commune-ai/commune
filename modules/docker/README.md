
# Docker Module for Commune

A powerful and flexible Docker management module that provides a high-level interface for Docker operations.

## Features

- 🐳 Docker container management
- 🏗️ Image building and handling
- 📝 Log management
- 📊 Container statistics
- 🔄 Resource pruning
- 🖥️ Multi-container operations

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/commune.git

# Install dependencies
pip install pandas
```

## Usage

### Basic Examples

```python
import commune as c

# Initialize Docker module
docker = c.Docker()

# Build a Docker image
docker.build(path="./my_dockerfile", tag="my_app:latest")

# Run a container
docker.run(
    path="my_app:latest",
    volumes=["/host/path:/container/path"],
    gpus=[0,1],  # Use specific GPUs
    env_vars={"KEY": "VALUE"}
)

# Get container logs
logs = docker.logs(name="my_container", follow=True)

# View container statistics
stats = docker.stats()

# Kill containers
docker.kill("my_container")
docker.kill_all()  # Kill all containers
```

## API Reference

### Core Methods

#### `build(path, tag, sudo=False, verbose=True, no_cache=False, env={})`
Builds a Docker image from a Dockerfile.

#### `run(path, cmd=None, volumes=None, name=None, gpus=False, ...)`
Runs a Docker container with extensive configuration options.

#### `kill(name, sudo=False, verbose=True, prune=False)`
Kills and removes a specific container.

#### `kill_all(sudo=False, verbose=True)`
Kills all running containers.

#### `logs(name, follow=False, tail=100, since=None)`
Retrieves container logs with various filtering options.

#### `stats(container=None)`
Gets resource usage statistics for containers.

#### `prune(all=False)`
Cleans up unused Docker resources.

### Utility Methods

#### `file(path='./')`
Gets content of the first Dockerfile found in path.

#### `files(path='./')`
Finds all Dockerfiles in given path.

#### `images(to_records=True)`
Lists all Docker images in system.

## Configuration

- Default shared memory size: 100GB
- Default network mode: host
- Supports both GPU and CPU configurations
- Flexible volume mounting
- Customizable environment variables
- Port mapping support

## Requirements

- Python 3.6+
- Docker installed and running
- pandas library
- commune framework

## License

[Your License Here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

