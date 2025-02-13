

## Scripts Overview

### build.sh
Builds a Docker image for the project.
```bash
./run/build.sh [name]  # name is optional, defaults to repository name
```

### enter.sh
Enters a running Docker container in interactive mode.
```bash
./run/enter.sh   # name is optional, defaults to repository name
```

### install.sh
Sets up the development environment by installing required dependencies:
- npm
- pm2
- Python3
- pip3
- Installs the project as a Python package

```bash
./run/install.sh
```

### start.sh
Starts a Docker container with the following features:
- Host network mode
- Auto-restart unless stopped
- Privileged mode
- 4GB shared memory
- Mounted volumes for app and configuration
- Docker socket access

```bash
./run/start.sh [name]  # name is optional, defaults to repository name
```

### stop.sh
Stops and removes a running Docker container.
```bash
./run/stop.sh   # name is optional, defaults to repository name
```

### test.sh
Runs project tests in a temporary Docker container.
```bash
./run/test.sh
```

## Features

- Automatic repository name detection
- Cross-platform support (Linux, MacOS, Windows)
- Docker container management
- Development environment setup
- Test automation

## Requirements

- Docker
- bash shell
- Package managers (apt, brew, or choco depending on OS)

## Usage

1. Clone the repository
2. Run `./run/install.sh` to set up the development environment
3. Use other scripts as needed for building, starting, stopping, or testing

## Notes

- All scripts use the repository name as the default container/image name
- Custom names can be provided as arguments to most scripts
- The project is automatically installed as a Python package during setup