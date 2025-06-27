
<p align="center">
  <picture>
    <img alt="commune" src="https://raw.githubusercontent.com/commune-ai/commune/refs/heads/main/.github/assets/banner.svg" style="max-width: 100%;padding:0px 0px 20px 0px;">
  </picture>
</p>

<p align="center">
    <a href="https://github.com/commune-ai/commune/blob/main/LICENCE.md"><img alt="GitHub" src="https://img.shields.io/badge/licence-MIT Licence-blue"></a>
    <!-- Uncomment when release to pypi -->
    <a href="https://pypi.org/project/commune/"><img alt="PyPI" src="https://img.shields.io/pypi/v/commune"></a>
    <a href="https://pypi.org/project/commune/"><img alt="Python Version" src="https://img.shields.io/pypi/pyversions/commune?logo=python"></a>
</p>




Commune is a comprehensive, modular framework for building distributed applications and services. It provides a seamless way to create, manage, serve, and interact with Python modules across networks.

## Key Features

- **Module Management**: Create, import, and manage Python modules with ease
- **Distributed Computing**: Connect modules across networks with built-in server/client capabilities
- **Blockchain Integration**: Native support for Substrate-based blockchains
- **Key Management**: Secure cryptographic key generation and management
- **Validation System**: Score and validate modules in distributed networks
- **Storage Solutions**: Simple key-value storage with file system persistence
- **CLI Interface**: Intuitive command-line interface for module interaction
- **Docker Support**: Container management for deployment and scaling

## Installation

### Prerequisites

- Python 3.10+
- Node.js 14+
- npm 6+

### Standard Installation

```bash
# Clone the repository
git clone https://github.com/commune-ai/commune.git

# Install Commune
cd commune
pip install -e ./

# Install npm and pm2 (required for the webserver)
chmod +x ./run/*
sudo ./run/install_npm_env.sh

# Test the installation
c test
```

### Docker Installation

```bash
# Clone the repository
git clone https://github.com/commune-ai/commune.git
cd commune

# Start Docker container
make start

# Enter the container
make enter

# Test the installation
c test
```

## Core Modules

### Module System

The heart of Commune is its module system, allowing you to create, manage, and serve Python modules:

```python
import commune as c

# Create a new module
c.new_module('my_module')

# Access a module
my_module = c.mod('my_module')

# Serve a module
c.serve('my_module', port=8000)
```

### Server

The Server module provides a robust API framework with authentication, middleware, and process management:

```python
# Serve a module
c.serve('my_module', port=8000)

# Make client requests
result = c.call('my_module/function', arg1=1, arg2=2)

# Virtual client
module = c.connect('my_module')
result = module.function(arg1=1, arg2=2)
```

### Blockchain

The Chain module offers a Python interface for interacting with Substrate-based blockchains:

```python
from commune import Chain, Keypair

# Connect to a network
chain = Chain(network='main')

# Generate a keypair
keypair = Keypair.create_from_uri('//Alice')

# Query balance
balance = chain.balance(keypair.ss58_address)
```

### Key Management

The Key module provides cryptographic key management with multi-crypto support:

```python
import commune as c

# Create a new key
key = c.key()

# Generate a key with specific crypto type
sr25519_key = c.key(crypto_type='sr25519')
ed25519_key = c.key(crypto_type='ed25519')
ecdsa_key = c.key(crypto_type='ecdsa')
```

### Validator

The Vali module enables validation and scoring of modules in distributed networks:

```python
from commune.vali import Vali

# Create a validator with a custom scoring function
def my_score_function(client):
    return client.info().get('score', 0)

validator = Vali(
    network='local',
    score=my_score_function,
    tempo=10
)
```

### Storage

The Store module provides key-value storage with file system persistence:

```python
from commune import Store

# Initialize a store
store = Store()

# Store data
store.put('config', {'api_key': '12345', 'timeout': 30})

# Retrieve data
config = store.get('config')
```

### Docker Management

The Docker module offers container management similar to PM2:

```python
import commune as c

# Initialize Docker module
docker = c.mod('pm')()

# Start a container
docker.start('my_container', 'python:3.8', 
             cmd='python -m http.server',
             ports={'8000': 8000})
```

## CLI Usage

Commune provides an intuitive command-line interface:

```bash
# Create a new module
c new_module my_module

# Get module code
c code my_module

# Serve a module
c serve my_module

# Call a module function
c call my_module/function arg1 arg2 kwarg1=value1
```

## Port Configuration

```bash
# Check the port range
c port_range

# Set the port range
c set_port_range 8000 9000
```

## Scripts

Commune includes several utility scripts in the `run/` directory:

- `build.sh`: Build a Docker image
- `enter.sh`: Enter a running Docker container
- `install.sh`: Set up the development environment
- `start.sh`: Start a Docker container
- `stop.sh`: Stop and remove a running Docker container
- `test.sh`: Run project tests

## Contributing

Contributions are welcome! Please ensure that any changes maintain backward compatibility and include appropriate tests.

## License

MIT License
