# Commune Installation Guide

## Overview
Commune is a global toolbox that allows you to connect and share any tool (module). This guide will help you install and set up Commune on your system.

## Prerequisites
- Python 3.8 - 3.12
- Docker (optional, for containerized deployment)
- npm (for certain features)

## Installation Methods

### Method 1: Quick Install (Recommended)
```bash
# Clone the repository
git clone https://github.com/commune-ai/commune.git
cd commune

# Run the install script
make install
# OR
./scripts/install.sh
```

This will automatically:
- Install Python 3 and pip if not present
- Install npm if not present
- Install Docker and Docker Compose if not present
- Install the Commune Python package

### Method 2: Manual Python Installation
```bash
# Clone the repository
git clone https://github.com/commune-ai/commune.git
cd commune

# Install as a Python package
pip install -e .
```

### Method 3: Docker Installation
```bash
# Clone the repository
git clone https://github.com/commune-ai/commune.git
cd commune

# Build and start with Docker
make build
make start

# Enter the container
make enter
```

## Verify Installation
```bash
# Check if commune is installed
c key
```

## Usage

### Basic Commands
```bash
# Start commune
make start

# Stop commune
make stop

# Restart commune
make restart

# Enter Docker container
make enter
```

### Python Usage
```python
import commune as c

# Example usage
c.print('Hello Commune!', color='green')
```

## System Requirements
- **Operating Systems**: Linux, macOS, Windows (WSL)
- **Python**: 3.8, 3.9, 3.10, 3.11, or 3.12
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: At least 2GB free space

## Dependencies
The main dependencies include:
- FastAPI for web framework
- PyTorch for machine learning
- Rich for terminal formatting
- Various cryptographic libraries
- And many more (see setup.py for full list)

## Troubleshooting

### Docker Issues
If Docker fails to start:
```bash
# Install Docker manually
./scripts/install_docker.sh
```

### Python Issues
If Python is not found:
```bash
# Install Python manually
./scripts/install_python.sh
```

### npm Issues
If npm is not found:
```bash
# Install npm manually
./scripts/install_npm.sh
```

## Uninstallation
To uninstall Commune:
```bash
./scripts/uninstall.sh
```

## Support
- Homepage: https://communeai.org/
- Repository: https://github.com/commune-ai/commune
- Issues: https://github.com/commune-ai/commune/issues

## License
MIT License
