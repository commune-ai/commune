<p align="center">
 <img src="https://communeai.org/logo-asci.svg" width="352" height="59" alt="Commune AI" style="max-width: 100%;">
</p>

# Commune: Modular Consensus System

Commune AI provides a powerful framework for building distributed AI applications with modular consensus mechanisms. The system enables seamless integration of AI models, distributed computing, and blockchain technology.

## Features

These components can be applied to:

* 🔗 **Networks** - Create and manage distributed networks with custom consensus mechanisms
* 🤖 **AI Models** - Deploy and serve AI models in a distributed environment
* 💫 **Modules** - Build modular components that can be composed into larger systems
* 🔑 **Keys** - Manage cryptographic keys and identities securely
* 🖥️ **Servers** - Deploy and monitor distributed services
* ✨ **Validation** - Implement custom validation rules and scoring mechanisms

## Quick Tour

To immediately start using Commune, you can create and serve a module with just a few lines of code:

```python
import commune as c

# Create a simple module
class MyModule(c.Module):
    def __init__(self, param1=1):
        self.set_config(locals())
        
    def forward(self, x):
        return x * self.param1

# Serve the module
module = c.serve('MyModule')
```

Basic CLI usage:

```bash
# Create a new module
>>>c new_module my_module

# Serve a module
>>>c serve my_module

# Call a module function
>>>c call my_module/forward 10
```

## Installation

### With pip

```bash
pip install commune
```

### From source

```bash
git clone https://github.com/commune-ai/commune.git
cd commune
pip install -e .
```

```bash
pip install commune
```

### Development setup

```bash
chmod +x ./run/***
sudo ./run/install.sh  # Install development environment
```

## Core Components

### 1. Module System
Modules are the building blocks of Commune applications:

```python
import commune as c

class CustomModule(c.Module):
    def __init__(self, config=None):
        self.set_config(config)
    
    def process(self, data):
        return self.model(data)
```

### 2. Network Management
Create and manage distributed networks:

```python
# Start a local network
c.serve('mymodule', network='local')

# Join a subnet
c.serve('mymodule', network='subspace', netuid=0)
```

### 3. Validation System
Implement custom validation rules:

```python
class Validator(c.Module):
    def validate(self, module):
        score = self.evaluate_performance(module)
        return score
```

## CLI Reference

Commune provides an intuitive CLI interface:

```bash
# Module operations
c new_module <name>        # Create new module
c serve <module>           # Serve module
c call <module>/<function> # Call module function
c modules                  # List modules

# Key management
c add_key <name>          # Generate new key
c keys                    # List keys
c sign <message>          # Sign message

# Server management
c servers                 # List servers
c connect <module>        # Connect to server
c logs <module>           # View logs
```

## Why Commune?

1. **Modular Design**:
   - Build complex systems from simple components
   - Easy-to-use module system
   - Flexible composition of services

2. **Distributed Computing**:
   - Scale across multiple nodes
   - Built-in consensus mechanisms
   - Robust networking layer

3. **AI Integration**:
   - Deploy AI models as services
   - Distributed training support
   - Model validation and scoring

4. **Security**:
   - Built-in key management
   - Secure communication
   - Validation mechanisms

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

[MIT License](LICENSE)