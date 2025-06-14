#!/bin/bash

# Script to install npm and Node.js
OS_NAME=$(uname)
echo "Installing npm on OS: $OS_NAME"

# Check and install npm (which comes with Node.js)
if ! command -v npm &> /dev/null
then
    echo "npm not found. Installing Node.js and npm..."
    if [ "$OS_NAME" == "Linux" ]; then
        echo "Installing Node.js and npm on Linux"
        # Update package manager
        sudo apt update
        
        # Install Node.js and npm from NodeSource repository for latest version
        curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
        sudo apt install nodejs -y
        
        # Alternative: Install from default repository (older version)
        # sudo apt install nodejs npm -y
    elif [ "$OS_NAME" == "Darwin" ]; then
        echo "Installing Node.js and npm on Mac"
        # Check if Homebrew is installed
        if ! command -v brew &> /dev/null; then
            echo "Homebrew not found. Installing Homebrew first..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        brew install node
    elif [[ "$OS_NAME" == MINGW* ]] || [[ "$OS_NAME" == CYGWIN* ]]; then
        echo "Installing Node.js and npm on Windows"
        if command -v choco &> /dev/null; then
            choco install nodejs -y
        else
            echo "Chocolatey not found. Please install Node.js manually from https://nodejs.org/"
            exit 1
        fi
    else
        echo "Unsupported OS: $OS_NAME"
        exit 1
    fi
else
    echo "npm is already installed: $(npm --version)"
fi

# Verify Node.js installation (npm comes with Node.js)
if ! command -v node &> /dev/null
then
    echo "Warning: Node.js not found but npm might be installed"
else
    echo "Node.js is installed: $(node --version)"
fi

# Update npm to latest version
if command -v npm &> /dev/null; then
    echo "Updating npm to latest version..."
    npm install -g npm@latest
fi

# Verify installations
echo ""
echo "=== npm Installation Summary ==="
if command -v node &> /dev/null; then
    echo "Node.js version: $(node --version)"
else
    echo "Node.js: NOT INSTALLED"
fi

if command -v npm &> /dev/null; then
    echo "npm version: $(npm --version)"
else
    echo "npm: NOT INSTALLED"
fi

echo ""
echo "npm installation script completed!"