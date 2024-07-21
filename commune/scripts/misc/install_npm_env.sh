#!/bin/bash

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Node.js is not installed. Installing Node.js..."
    
    # Add NodeSource repository
    curl -sL https://deb.nodesource.com/setup_16.x | sudo -E bash -

    # Install Node.js
    sudo apt-get install -y nodejs
else
    echo "Node.js is already installed."
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "npm is not installed. Installing npm..."

    # Install npm
    sudo apt-get install -y npm
else
    echo "npm is already installed."
fi

# Verify the installation
node -v
npm -v

# install pm2 
npm install -g pm2


