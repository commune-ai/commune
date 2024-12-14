# Commune: Incentivizing Applications in a Multichain World

Commune is a modular consensus system designed to incentivize applications to run on networks through innovative token models and bonding curves. It provides a flexible framework for building and managing distributed applications across multiple chains.

## 🚀 Quick Start

### Installation

**Requirements:**
- Python 3.10+
- NodeJS 14+
- npm 6+

```bash
# Clone the repository
git clone https://github.com/commune-ai/commune.git

# Install the package
cd commune
pip install -e ./

# Install npm dependencies
chmod +x ./scripts/*
sudo ./scripts/install_npm_env.sh
```

### Basic Usage

```python
import commune as c

# Create a new module
c.new_module('agi')

# Serve a module
c.serve('agi')

# Call a module
c.call('agi/forward', "hello")
```

## 🏗️ Core Concepts

### Modules

A module is a collection of functions and state variables. Modules are the building blocks of Commune and can be:
- Served as HTTP endpoints
- Called remotely
- Managed through the CLI
- Inherited and extended

### Networks

Networks in Commune consist of:
- **Namespace**: Maps server names to addresses
- **Servers**: Modules exposed as network endpoints
- **Validators**: Nodes that validate and secure the network

### Keys

Commune uses SR25519 keys for:
- Signing messages
- Encrypting/decrypting data
- Cross-chain compatibility
- Creating temporary tokens

## 💻 CLI Interface

Commune provides a Pythonic CLI for interacting with the system:

```bash
# Create a new module
c new_module agi

# Serve a module
c serve agi

# List modules
c modules

# Get module info
c module_info agi
```

## 🔗 Network Features

### Validator System

- **Staking**: Token-based staking with rewards
- **Anti-Rug Mechanism**: Prevents pump and dump schemes
- **Price Controls**: Min/max pricing and stop loss mechanisms
- **Subnet Management**: Create and manage subnets with locked liquidity

### Token Economics

- Bonding curves for price discovery
- Token emission vs native emission
- Cross-chain token locking
- User-to-user token lending

## 🛠️ Development

### Creating a Module

```python
import commune as c

class MyModule(c.Module):
    def __init__(self, param1=1):
        self.set_config(locals())
    
    def forward(self, x):
        return x * self.config.param1
```

### Serving a Module

```python
# Serve on local network
c.serve('mymodule', network='local')

# Serve on Subspace network
c.serve('mymodule', network='subspace', netuid=0)
```

## 📚 Documentation

For more detailed documentation, please see:
- [Module Basics](docs/modules.md)
- [Network Architecture](docs/network.md)
- [Key Management](docs/keys.md)
- [Validator System](docs/validators.md)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
