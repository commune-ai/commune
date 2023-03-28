#!/bin/bash

# Check if python3-venv is installed
if ! dpkg -s python3-venv &> /dev/null; then
    echo "python3-venv package not found. Installing..."
    sudo apt-get install python3-venv -y
fi

# Create a virtual environment
python3 -m venv env

# Activate the virtual environment
source myenv/bin/activate
pip install --upgrade pip

# install bittensor
pip install bittensor
pip install https://github.com/opentensor/cubit/releases/download/v1.1.2/cubit-1.1.2-cp310-cp310-linux_x86_64.whl[test]

# install commune
pip install -e .

# Deactivate the virtual environment
echo "Setup is complete. To activate the virtual environment, run 'source myenv/bin/activate'"



# Install dependencies


