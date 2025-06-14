#!/bin/bash

# Script to install Python and pip
OS_NAME=$(uname)
echo "Installing Python on OS: $OS_NAME"

# Check and install python3
if ! command -v python3 &> /dev/null
then
    echo "Python3 not found. Installing..."
    if [ "$OS_NAME" == "Linux" ]; then
        echo "Installing Python3 on Linux"
        sudo apt update
        sudo apt install python3 python3-pip python3-venv -y
    elif [ "$OS_NAME" == "Darwin" ]; then
        echo "Installing Python3 on Mac"
        # Check if Homebrew is installed
        if ! command -v brew &> /dev/null; then
            echo "Homebrew not found. Please install Homebrew first."
            exit 1
        fi
        brew install python3
    elif [[ "$OS_NAME" == MINGW* ]] || [[ "$OS_NAME" == CYGWIN* ]]; then
        echo "Installing Python3 on Windows"
        if command -v choco &> /dev/null; then
            choco install python3 -y
        else
            echo "Chocolatey not found. Please install Chocolatey or Python manually."
            exit 1
        fi
    else
        echo "Unsupported OS: $OS_NAME"
        exit 1
    fi
else
    echo "Python3 is already installed: $(python3 --version)"
fi

# Check and install pip3
if ! command -v pip3 &> /dev/null
then
    echo "Pip3 not found. Installing..."
    if [ "$OS_NAME" == "Linux" ]; then
        echo "Installing pip3 on Linux"
        sudo apt update
        sudo apt install python3-pip -y
    elif [ "$OS_NAME" == "Darwin" ]; then
        echo "Installing pip3 on Mac"
        # pip3 usually comes with python3 on Mac via Homebrew
        python3 -m ensurepip --upgrade
    elif [[ "$OS_NAME" == MINGW* ]] || [[ "$OS_NAME" == CYGWIN* ]]; then
        echo "Installing pip3 on Windows"
        python3 -m ensurepip --upgrade
    fi
else
    echo "Pip3 is already installed: $(pip3 --version)"
fi

# Verify installations
echo ""
echo "=== Python Installation Summary ==="
if command -v python3 &> /dev/null; then
    echo "Python3 version: $(python3 --version)"
else
    echo "Python3: NOT INSTALLED"
fi

if command -v pip3 &> /dev/null; then
    echo "Pip3 version: $(pip3 --version)"
else
    echo "Pip3: NOT INSTALLED"
fi

echo ""
echo "Python installation script completed!"