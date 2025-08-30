#!/bin/bash

# activate_polaris.sh - Activate the Polaris Compute Subnet environment

# Activate the virtual environment
source venv/bin/activate

# Ensure Rust is in PATH
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

echo "Polaris environment activated! You can now run polaris commands."
echo "Try: polaris --help"
