# Commune: A Modular Consensus System

Commune is a modular consensus system designed for building and managing distributed applications with innovative token models and bonding curves.

## 🚀 Quick Start

### Requirements
- Python 3.10+
- NodeJS 14+
- npm 6+

### Installation

```bash
# Clone repository
git clone https://github.com/commune-ai/commune.git
# Install package
pip install -e ./commune


# Install npm dependencies
chmod +x ./run/*
sudo ./run/install.sh # to install the environments
```

## 🏗️ Core Components

### 1. Module System
- Create modules: `c new_module <name>`
- Serve modules: `c serve <module>`
- Call modules: `c call <module>/<function>`
- View modules: `c modules`
- [Learn More About Modules](docs/modules.md)

### 2. Key Management
- Generate keys: `c add_key <name>`
- List keys: `c keys`
- Sign messages: `c sign <message>`
- Create tickets: `c ticket <name>`
- [Learn More About Keys](core/key.md)

### 3. Server Management
- Start server: `c serve <module>`
- View servers: `c servers`
- Connect to server: `c connect <module>`
- View logs: `c logs <module>`
- [Learn More About Servers](core/server.md)

### 4. Validator System
- Monitor network: `c vali`
- View scoreboard: `c scoreboard`
- Score modules: `c score <module>`
- [Learn More About Validation](core/vali.md)

## 💻 CLI Interface

Commune provides a Pythonic CLI:
```bash
# Module format
c <module>/<function> *args **kwargs

# Direct format
c <function> *args **kwargs
```

[Learn More About CLI](core/cli.md)

## 🔗 Network Features

### Validator System
- Token-based staking
- Anti-rug mechanisms
- Price controls
- Subnet management

### Token Economics
- Bonding curves
- Token emission
- Cross-chain locking
- User lending

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
# Local network
c.serve('mymodule', network='local')

# Subspace network
c.serve('mymodule', network='subspace', netuid=0)
```

## 📚 Documentation
- [Installation Guide](1_install.md)
- [Module Basics](0_intro.md)
- [CLI Reference](core/cli.md)
- [Server Guide](core/server.md)
- [Key Management](core/key.md)
- [Validator System](core/vali.md)

## 🤝 Contributing
Contributions welcome! Please submit Pull Requests.

## 📄 License
[Add License Information]
```

This README provides:
1. Clear installation instructions
2. Overview of core components
3. Links to detailed documentation
4. Code examples
5. CLI usage
6. Network features
7. Development guide

All references to core modules and documentation are included with proper links. The structure is clean and easy to navigate while providing comprehensive information about the system's capabilities.



config details 

