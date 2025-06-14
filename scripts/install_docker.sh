#!/bin/bash

# Docker Installation Script
# Supports: Linux, macOS, and Windows (WSL)
# Author: Auto-generated
# Date: $(date +%Y-%m-%d)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            OS=$NAME
            VER=$VERSION_ID
        fi
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        echo "macos"
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        # Windows
        echo "windows"
    else
        echo "unknown"
    fi
}

# Function to check if Docker is already installed
check_docker_installed() {
    if command -v docker &> /dev/null; then
        print_message "$GREEN" "Docker is already installed!"
        docker --version
        return 0
    else
        return 1
    fi
}

# Function to install Docker on Linux
install_docker_linux() {
    print_message "$YELLOW" "Installing Docker on Linux..."
    
    # Update package index
    sudo apt-get update -y || sudo yum update -y || sudo dnf update -y
    
    # Install prerequisites
    if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        sudo apt-get install -y \
            ca-certificates \
            curl \
            gnupg \
            lsb-release
        
        # Add Docker's official GPG key
        sudo mkdir -p /etc/apt/keyrings
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        
        # Set up the repository
        echo \
          "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
          $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        
        # Install Docker Engine
        sudo apt-get update -y
        sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
        
    elif command -v yum &> /dev/null; then
        # RHEL/CentOS/Fedora
        sudo yum install -y yum-utils
        sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
        sudo yum install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
        
    elif command -v dnf &> /dev/null; then
        # Fedora
        sudo dnf -y install dnf-plugins-core
        sudo dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo
        sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    else
        print_message "$RED" "Unsupported Linux distribution"
        exit 1
    fi
    
    # Start and enable Docker
    sudo systemctl start docker
    sudo systemctl enable docker
    
    # Add current user to docker group
    sudo usermod -aG docker $USER
    
    print_message "$GREEN" "Docker installed successfully on Linux!"
    print_message "$YELLOW" "Please log out and back in for group changes to take effect."
}

# Function to install Docker on macOS
install_docker_macos() {
    print_message "$YELLOW" "Installing Docker on macOS..."
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        print_message "$YELLOW" "Homebrew not found. Installing Homebrew first..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    # Install Docker Desktop using Homebrew
    brew install --cask docker
    
    print_message "$GREEN" "Docker Desktop installed successfully on macOS!"
    print_message "$YELLOW" "Please open Docker Desktop from Applications to complete setup."
}

# Function to install Docker on Windows (WSL)
install_docker_windows() {
    print_message "$YELLOW" "Detected Windows environment..."
    
    # Check if running in WSL
    if grep -qi microsoft /proc/version 2>/dev/null; then
        print_message "$YELLOW" "Running in WSL. Installing Docker in WSL..."
        install_docker_linux
        
        print_message "$YELLOW" "Note: For best experience, install Docker Desktop for Windows."
        print_message "$YELLOW" "Download from: https://www.docker.com/products/docker-desktop"
    else
        print_message "$RED" "Native Windows detected. Please use Docker Desktop for Windows."
        print_message "$YELLOW" "Download from: https://www.docker.com/products/docker-desktop"
        print_message "$YELLOW" "After installation, you can use Docker from WSL or PowerShell."
        exit 1
    fi
}

# Main installation function
main() {
    print_message "$GREEN" "=== Docker Installation Script ==="
    
    # Check if Docker is already installed
    if check_docker_installed; then
        read -p "Docker is already installed. Do you want to reinstall? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_message "$YELLOW" "Installation cancelled."
            exit 0
        fi
    fi
    
    # Detect OS and install accordingly
    OS=$(detect_os)
    
    case $OS in
        "linux")
            install_docker_linux
            ;;
        "macos")
            install_docker_macos
            ;;
        "windows")
            install_docker_windows
            ;;
        *)
            print_message "$RED" "Unsupported operating system: $OSTYPE"
            exit 1
            ;;
    esac
    
    # Verify installation
    if command -v docker &> /dev/null; then
        print_message "$GREEN" "\nDocker installation completed successfully!"
        docker --version
        
        # Test Docker installation
        print_message "$YELLOW" "\nTesting Docker installation..."
        if docker run hello-world &> /dev/null; then
            print_message "$GREEN" "Docker is working correctly!"
        else
            print_message "$YELLOW" "Docker is installed but requires additional setup."
            print_message "$YELLOW" "You may need to start Docker or log out and back in."
        fi
    else
        print_message "$RED" "Docker installation verification failed."
        print_message "$YELLOW" "Please check the installation logs above."
    fi
}

# Run main function
main