# Installation Guide

This guide walks you through the process of installing Commune and its dependencies.

## Prerequisites

### System Requirements
- Linux (Ubuntu 20.04+ recommended)
- Python 3.10+
- Node.js 20+
- Rust toolchain

### Required Dependencies
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    build-essential \
    curl \
    git \
    libssl-dev \
    pkg-config
```

## Installation Methods

### 1. From PyPI (Recommended)
```bash
pip install commune
```

### 2. From Source
```bash
# Clone the repository
git clone https://github.com/commune-ai/commune.git

# Navigate to the directory
cd commune

# Install Python dependencies
pip install -e .

# Install Node.js dependencies
chmod +x ./scripts/*
./scripts/install_npm_env.sh
```

## Environment Setup

### 1. Configuration
Create a `.env` file in your project root:
```bash
OPENAI_API_KEY=your_api_key  # Optional: For AI integration
```

### 2. Verify Installation
```bash
# Check CLI installation
c --version

# Run basic test
c test basic
```

## Common Issues and Solutions

### 1. Node.js Installation
If you encounter issues with Node.js:
```bash
# Install nvm (Node Version Manager)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash

# Reload shell configuration
source ~/.bashrc

# Install and use Node.js 20
nvm install 20
nvm use 20
```

### 2. Python Version Conflicts
If you have multiple Python versions:
```bash
# Create a virtual environment
python3.10 -m venv venv

# Activate the environment
source venv/bin/activate
```

### 3. Rust Toolchain
If Rust installation is needed:
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Add Rust to PATH
source $HOME/.cargo/env
```

## Next Steps

After installation:
1. Follow the [Quick Start Tutorial](02-Quick-Start.md)
2. Read about [Core Concepts](10-Module-System.md)
3. Try creating your [First Module](20-First-Module.md)

## Troubleshooting

If you encounter any issues:
1. Check our [GitHub Issues](https://github.com/commune-ai/commune/issues)
2. Join our [Discord](https://discord.gg/commune) for community support
3. Ensure all dependencies are properly installed
4. Check system logs for error messages
