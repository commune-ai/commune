#!/bin/bash

# # Check if python3-venv is installed
if ! dpkg -s python3-pip &> /dev/null; then
    echo "python3-venv package not found. Installing..."
    sudo apt install python3-pip
fi

# # Create a virtual environment
# python3 -m venv env

# # Activate the virtual environment
# source env/bin/activate
pip install --upgrade pip

# Deactivate the virtual environment
echo "Setup is complete. To activate the virtual environment, run 'source myenv/bin/activate'"


# Install dependencies


