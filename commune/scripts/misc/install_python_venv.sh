#!/bin/bash

# Check if python3-venv is installed
if ! dpkg -s python3-venv &> /dev/null; then
    echo "python3-venv package not found. Installing..."
    sudo apt-get install python3-venv -y
fi

# Create a virtual environment
python3 -m venv env

# Activate the virtual environment
source env/bin/activate

./scripts/install_python_env.sh