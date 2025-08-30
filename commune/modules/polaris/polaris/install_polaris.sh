#!/bin/bash

# install_polaris.sh - Automated installer for Polaris Compute Subnet

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Display banner
echo -e "${BLUE}===============================================${NC}"
echo -e "${BLUE}    Polaris Compute Subnet - Auto Installer    ${NC}"
echo -e "${BLUE}===============================================${NC}"

# Check if we're in the right directory
if [ ! -f "setup.py" ] || [ ! -d "polaris_cli" ]; then
    echo -e "${RED}Error: This script must be run from the root of the polaris-subnet repository.${NC}"
    echo -e "${YELLOW}Please clone the repository and run this script from there:${NC}"
    echo -e "git clone https://github.com/bigideainc/polaris-subnet.git"
    echo -e "cd polaris-subnet"
    echo -e "./install_polaris.sh"
    exit 1
fi

# Function to handle errors
handle_error() {
    echo -e "${RED}Error: $1${NC}"
    echo -e "${YELLOW}Installation failed. Please check the error message above.${NC}"
    exit 1
}

# Function to check and install system dependencies
install_dependencies() {
    echo -e "${BLUE}Checking and installing system dependencies...${NC}"
    
    # Detect OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        echo -e "${BLUE}Detected macOS system${NC}"
        
        # Check if Homebrew is installed
        if ! command -v brew &> /dev/null; then
            echo -e "${YELLOW}Homebrew not found. We recommend installing it for better dependency management.${NC}"
            echo -e "${YELLOW}Visit https://brew.sh for installation instructions.${NC}"
        fi
        
        # Install Rust if needed
        if ! command -v rustc &> /dev/null; then
            echo -e "${BLUE}Installing Rust...${NC}"
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y || handle_error "Failed to install Rust"
            source "$HOME/.cargo/env"
        else
            echo -e "${GREEN}Rust is already installed.${NC}"
        fi
        
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        echo -e "${BLUE}Detected Linux system${NC}"
        
        # Check for apt (Debian/Ubuntu)
        if command -v apt &> /dev/null; then
            echo -e "${BLUE}Installing system dependencies...${NC}"
            sudo apt update || handle_error "Failed to update apt repositories"
            sudo apt install -y g++ rustc cargo build-essential python3-dev || handle_error "Failed to install system dependencies"
        else
            echo -e "${YELLOW}Warning: Could not determine package manager. You may need to install these dependencies manually:${NC}"
            echo -e "${YELLOW}- g++${NC}"
            echo -e "${YELLOW}- rustc${NC}"
            echo -e "${YELLOW}- cargo${NC}"
            echo -e "${YELLOW}- build-essential${NC}"
            echo -e "${YELLOW}- python3-dev${NC}"
            
            # Still try to install Rust if needed
            if ! command -v rustc &> /dev/null; then
                echo -e "${BLUE}Installing Rust...${NC}"
                curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y || handle_error "Failed to install Rust"
                source "$HOME/.cargo/env"
            fi
        fi
    else
        echo -e "${YELLOW}Warning: Unsupported OS detected. You may need to install dependencies manually.${NC}"
    fi
    
    # Ensure Rust is in PATH
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
}

# Function to set up virtual environment
setup_venv() {
    echo -e "${BLUE}Setting up Python virtual environment...${NC}"
    
    # Check if venv exists
    if [ -d "venv" ]; then
        echo -e "${YELLOW}Virtual environment already exists. Creating a fresh one...${NC}"
        rm -rf venv
    fi
    
    # Create and activate virtual environment
    python3 -m venv venv || handle_error "Failed to create virtual environment"
    source venv/bin/activate || handle_error "Failed to activate virtual environment"
    
    # Verify activation
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        handle_error "Virtual environment activation failed"
    fi
    
    echo -e "${GREEN}Virtual environment created and activated.${NC}"
}

# Function to switch to the correct branch
switch_branch() {
    echo -e "${BLUE}Switching to cli_changes branch...${NC}"
    
    # Check current branch
    current_branch=$(git rev-parse --abbrev-ref HEAD)
    if [ "$current_branch" == "cli_changes" ]; then
        echo -e "${GREEN}Already on cli_changes branch.${NC}"
        return
    fi
    
    # Fetch all branches (in case cli_changes is remote-only)
    git fetch --all || handle_error "Failed to fetch branches"
    
    # Try to checkout, handling both local and remote branch possibilities
    if git checkout cli_changes 2>/dev/null; then
        echo -e "${GREEN}Successfully switched to cli_changes branch.${NC}"
    elif git checkout -b cli_changes origin/cli_changes; then
        echo -e "${GREEN}Successfully created and switched to cli_changes branch from remote.${NC}"
    else
        handle_error "Could not find cli_changes branch"
    fi
}

# Function to install Python dependencies
install_python_deps() {
    echo -e "${BLUE}Installing Python dependencies...${NC}"
    
    # Upgrade pip
    pip install --upgrade pip || echo -e "${YELLOW}Warning: Failed to upgrade pip. Continuing with installation...${NC}"
    
    # Install from requirements.txt
    echo -e "${BLUE}Installing packages from requirements.txt...${NC}"
    pip install -r requirements.txt || handle_error "Failed to install requirements"
    
    # Install network-specific packages
    echo -e "${BLUE}Installing network-specific packages...${NC}"
    pip install bittensor bittensor-cli || handle_error "Failed to install bittensor"
    pip install communex==0.1.36.4 || handle_error "Failed to install communex"
    
    # Install the package in development mode
    echo -e "${BLUE}Installing Polaris in development mode...${NC}"
    pip install -e . || handle_error "Failed to install Polaris"
    
    echo -e "${GREEN}All Python dependencies installed successfully.${NC}"
}

# Function to set up the environment configuration
setup_env_config() {
    echo -e "${BLUE}Setting up environment configuration...${NC}"
    
    # Check if .env file already exists
    if [ -f ".env" ]; then
        echo -e "${YELLOW}A .env file already exists.${NC}"
        read -p "Do you want to replace it with a new configuration? (y/n) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}Keeping existing .env file. Please ensure it has all required configuration values.${NC}"
            return
        fi
    fi
    
    # Check if .env.example exists
    if [ ! -f ".env.example" ]; then
        echo -e "${YELLOW}No .env.example found. Creating a basic template...${NC}"
        cat > .env.example << EOL
# Polaris Compute Subnet - Environment Configuration Example
# Replace these example values with your actual information

# Your machine's public IP address or domain name
HOST=your_public_ip_address

# Port for the Polaris API server
API_PORT=8000

# SSH configuration for remote connections
SSH_PORT_RANGE_START=11000
SSH_PORT_RANGE_END=11002
SSH_PASSWORD=your_secure_password
SSH_USER=your_ssh_username
SSH_HOST=your_ssh_host_ip
SSH_PORT=11000

# Optional: Ngrok authentication token if using Ngrok for tunneling
# NGROK_AUTH_TOKEN=your_ngrok_auth_token

# Server URL for the orchestrator
SERVER_URL=https://polaris-test-server.onrender.com/api/v1
EOL
    fi
    
    # Create a new .env file with default values
    echo -e "${BLUE}Creating .env file with default values...${NC}"
    cat > .env << EOL
# Polaris Compute Subnet - Environment Configuration
# Generated by install_polaris.sh

# Network configuration
HOST=auto_detect
API_PORT=8000

# SSH configuration
SSH_PORT_RANGE_START=11000
SSH_PORT_RANGE_END=11002
SSH_PASSWORD=change_me
SSH_USER=change_me
SSH_HOST=auto_detect
SSH_PORT=11000

# Server configuration
SERVER_URL=https://polaris-test-server.onrender.com/api/v1
EOL
    
    # Get the user's public IP
    echo -e "${BLUE}Detecting your public IP address...${NC}"
    public_ip=$(curl -s ifconfig.me)
    if [ -n "$public_ip" ]; then
        echo -e "${GREEN}Detected public IP: $public_ip${NC}"
        # Replace the placeholder with actual IP
        sed -i.bak "s/HOST=auto_detect/HOST=$public_ip/g" .env
        sed -i.bak "s/SSH_HOST=auto_detect/SSH_HOST=$public_ip/g" .env
        rm -f .env.bak  # Remove backup file
    else
        echo -e "${YELLOW}Could not detect your public IP. You'll need to update this manually in the .env file.${NC}"
    fi
    
    # Prompt directly for SSH username and password
    echo -e "${YELLOW}===== REQUIRED CONFIGURATION =====${NC}"
    echo -e "${BLUE}Please provide your SSH credentials:${NC}"
    
    # Get SSH username
    echo -e "${GREEN}SSH Username${NC} (your system username to connect via SSH):"
    read -p "> " ssh_username
    while [ -z "$ssh_username" ]; do
        echo -e "${RED}SSH username cannot be empty. Please try again:${NC}"
        read -p "> " ssh_username
    done
    
    # Get SSH password
    echo -e "${GREEN}SSH Password${NC} (used for secure connections):"
    read -s -p "> " ssh_password
    echo ""
    while [ -z "$ssh_password" ]; do
        echo -e "${RED}SSH password cannot be empty. Please try again:${NC}"
        read -s -p "> " ssh_password
        echo ""
    done
    
    # Update the .env file with the provided credentials
    sed -i.bak "s/SSH_USER=change_me/SSH_USER=$ssh_username/g" .env
    sed -i.bak "s/SSH_PASSWORD=change_me/SSH_PASSWORD=$ssh_password/g" .env
    rm -f .env.bak  # Remove backup file
    
    echo -e "${GREEN}Environment configuration successfully updated with your credentials.${NC}"
    echo -e "${BLUE}All other configuration values have been set to recommended defaults.${NC}"
    
    # Ask if user wants to customize any other settings
    echo ""
    read -p "Would you like to customize any other settings in the .env file? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Determine which editor to use
        if command -v nano &> /dev/null; then
            nano .env
        elif command -v vim &> /dev/null; then
            vim .env
        elif command -v vi &> /dev/null; then
            vi .env
        else
            echo -e "${YELLOW}No suitable text editor found. If you need to customize other settings, edit the .env file manually after installation.${NC}"
        fi
    fi
    
    echo -e "${GREEN}Environment configuration setup complete.${NC}"
}

# Function to verify installation
verify_installation() {
    echo -e "${BLUE}Verifying installation...${NC}"
    
    if command -v polaris &> /dev/null; then
        echo -e "${GREEN}Polaris is installed and available!${NC}"
        polaris --help > /dev/null || handle_error "Polaris command failed"
        echo -e "${GREEN}Verification complete! Polaris is ready to use.${NC}"
    else
        handle_error "Polaris command not found in PATH"
    fi
}

# Function to create an activation script
create_activation_script() {
    echo -e "${BLUE}Creating activation script...${NC}"
    
    cat > activate_polaris.sh << EOL
#!/bin/bash

# activate_polaris.sh - Activate the Polaris Compute Subnet environment

# Activate the virtual environment
source venv/bin/activate

# Ensure Rust is in PATH
if [ -f "\$HOME/.cargo/env" ]; then
    source "\$HOME/.cargo/env"
fi

echo "Polaris environment activated! You can now run polaris commands."
echo "Try: polaris --help"
EOL

    chmod +x activate_polaris.sh
    echo -e "${GREEN}Activation script created: activate_polaris.sh${NC}"
}

# Main installation process
main() {
    echo -e "${BLUE}Starting Polaris Compute Subnet installation process...${NC}"
    
    # Run all installation steps
    switch_branch
    install_dependencies
    setup_venv
    install_python_deps
    setup_env_config
    verify_installation
    create_activation_script
    
    echo -e "${GREEN}====================================${NC}"
    echo -e "${GREEN}    Installation Complete!         ${NC}"
    echo -e "${GREEN}====================================${NC}"
    echo -e "${YELLOW}To use Polaris Compute Subnet:${NC}"
    echo -e "${BLUE}1. Run the activation script:${NC}"
    echo -e "   ${GREEN}source ./activate_polaris.sh${NC}"
    echo -e "${BLUE}2. Start Polaris:${NC}"
    echo -e "   ${GREEN}polaris start${NC}"
    echo -e "${BLUE}3. For more options:${NC}"
    echo -e "   ${GREEN}polaris --help${NC}"
    
    # Remind about .env file
    echo -e "${YELLOW}NOTE: Make sure your .env file is properly configured before starting Polaris.${NC}"
    
    # Ask if user wants to start Polaris now
    echo ""
    read -p "Would you like to start Polaris now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Starting Polaris...${NC}"
        polaris start
    fi
}

# Execute the main function
main 