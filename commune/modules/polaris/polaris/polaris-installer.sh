#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
MAGENTA='\033[0;35m'
BG_RED='\033[41m'
BG_GREEN='\033[42m'

# Error handling function
error_exit() {
    print_error "$1"
    if [ -n "$2" ] && [ "$2" -eq 1 ]; then
        exit 1
    fi
    return 1
}

# Function to check command status and handle errors
check_command() {
    if [ $? -ne 0 ]; then
        error_exit "$1" "$2"
        return 1
    fi
    return 0
}

# Function to print colored output
print_status() {
    echo -e "${BLUE}[*]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[+]${NC} $1"
}

print_error() {
    echo -e "${RED}[-]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Function to install WSL automatically
install_wsl_automatically() {
    clear
    echo -e "${BG_GREEN}${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BG_GREEN}${BOLD}║            Automatic WSL Installation Helper                 ║${NC}"
    echo -e "${BG_GREEN}${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo
    echo -e "${BLUE}${BOLD}This will guide you through installing WSL automatically:${NC}"
    echo
    echo -e "${YELLOW}Step 1: Launching PowerShell to install WSL${NC}"
    echo -e "${YELLOW}(You may see UAC prompts requesting administrator permissions)${NC}"
    echo
    
    # Create a PowerShell script to install WSL
    cat > install_wsl.ps1 << 'EOF'
# Check if running as administrator
if (-NOT ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Warning "This script requires administrator privileges. Attempting to restart as administrator..."
    Start-Process powershell.exe "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs
    exit
}

Write-Host "Starting WSL installation..." -ForegroundColor Green

# Install WSL
Write-Host "Running: wsl --install" -ForegroundColor Yellow
wsl --install

Write-Host "`nWSL installation process started!" -ForegroundColor Green
Write-Host "After your computer restarts, you'll need to:" -ForegroundColor Yellow
Write-Host "1. Install Ubuntu from Microsoft Store" -ForegroundColor Yellow
Write-Host "2. Launch Ubuntu and set up your username/password" -ForegroundColor Yellow
Write-Host "3. Copy the Polaris script to your WSL environment" -ForegroundColor Yellow
Write-Host "`nPress any key to exit. Your computer will need to restart to complete the WSL installation." -ForegroundColor Cyan

$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
EOF

    echo -e "${GREEN}About to launch PowerShell as administrator to install WSL...${NC}"
    echo -e "${YELLOW}If you see a Windows security prompt, select 'Yes' to continue.${NC}"
    echo -e "${YELLOW}Your computer will need to restart after this process to complete the WSL installation.${NC}"
    echo
    read -p "Press Enter to continue or Ctrl+C to cancel..."
    
    # Execute the PowerShell script
    powershell.exe -ExecutionPolicy Bypass -File install_wsl.ps1
    
    echo
    echo -e "${BLUE}${BOLD}After your system restarts:${NC}"
    echo -e "1. Install Ubuntu from Microsoft Store"
    echo -e "2. Launch Ubuntu from Start Menu"
    echo -e "3. Copy this script to your WSL environment using:"
    echo -e "${MAGENTA}   cp /mnt/c/Users/$USERNAME/Desktop/polaris_manager.sh ~/${NC}"
    echo -e "4. Make it executable: ${MAGENTA}chmod +x ~/polaris_manager.sh${NC}"
    echo -e "5. Run it: ${MAGENTA}./polaris_manager.sh${NC}"
    echo
    echo -e "${RED}Your system will need to restart to complete the WSL installation.${NC}"
    echo
    read -p "Press Enter to exit. Please restart your computer after this..."
    exit 0
}

# Function to show WSL setup instructions
show_wsl_instructions() {
    clear
    echo -e "${BG_GREEN}${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BG_GREEN}${BOLD}║       Windows Subsystem for Linux (WSL) Setup Guide          ║${NC}"
    echo -e "${BG_GREEN}${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo
    echo -e "${RED}${BOLD}⚠️  IMPORTANT: Polaris requires a Linux environment  ⚠️${NC}"
    echo -e "${YELLOW}This script can only run on Linux or Windows with WSL (Windows Subsystem for Linux)${NC}"
    echo
    echo -e "${BLUE}${BOLD}Follow these steps to set up WSL on Windows:${NC}"
    echo
    echo -e "${GREEN}${BOLD}Step 1:${NC} ${BLUE}Open PowerShell as Administrator and run:${NC}"
    echo -e "${MAGENTA}   wsl --install${NC}"
    echo
    echo -e "${GREEN}${BOLD}Step 2:${NC} ${BLUE}Restart your computer${NC}"
    echo -e "${YELLOW}   (This is required to complete the WSL installation)${NC}"
    echo
    echo -e "${GREEN}${BOLD}Step 3:${NC} ${BLUE}After restart, open Microsoft Store and install Ubuntu${NC}"
    echo -e "${MAGENTA}   • Search for 'Ubuntu' in Microsoft Store${NC}"
    echo -e "${MAGENTA}   • Click 'Get' or 'Install'${NC}"
    echo
    echo -e "${GREEN}${BOLD}Step 4:${NC} ${BLUE}Launch Ubuntu from Start Menu${NC}"
    echo -e "${MAGENTA}   • Set up your username and password when prompted${NC}"
    echo -e "${YELLOW}   • Remember these credentials as you'll need them later${NC}"
    echo
    echo -e "${GREEN}${BOLD}Step 5:${NC} ${BLUE}Update Ubuntu packages:${NC}"
    echo -e "${MAGENTA}   sudo apt update && sudo apt upgrade -y${NC}"
    echo
    echo -e "${GREEN}${BOLD}Step 6:${NC} ${BLUE}Copy this script to your WSL environment:${NC}"
    echo -e "${MAGENTA}   1. In Windows, copy this script to a location like C:\\Users\\YourUsername\\${NC}"
    echo -e "${MAGENTA}   2. In WSL terminal, access it with: cp /mnt/c/Users/YourUsername/polaris_manager.sh ~/${NC}"
    echo -e "${MAGENTA}   3. Make it executable: chmod +x ~/polaris_manager.sh${NC}"
    echo -e "${MAGENTA}   4. Run it: ./polaris_manager.sh${NC}"
    echo
    echo -e "${CYAN}${BOLD}Docker on WSL:${NC}"
    echo -e "${BLUE}• For optimal performance on WSL, install Docker Desktop for Windows${NC}"
    echo -e "${BLUE}• Enable WSL 2 integration in Docker Desktop settings${NC}"
    echo -e "${BLUE}• This configuration provides better performance and reliability${NC}"
    echo
    echo -e "${YELLOW}${BOLD}For detailed instructions, visit:${NC}"
    echo -e "${BLUE}• WSL Installation: https://learn.microsoft.com/en-us/windows/wsl/install${NC}"
    echo -e "${BLUE}• Docker with WSL: https://docs.docker.com/desktop/wsl/${NC}"
    echo
    echo -e "${GREEN}${BOLD}Press Enter to continue...${NC}"
    read -p ""
}

# Function to check system compatibility
check_system_compatibility() {
    local os_name=$(uname -s)
    local is_wsl=false

    if [ -f /proc/version ] && grep -qi microsoft /proc/version; then
        is_wsl=true
    fi

    if [ "$os_name" != "Linux" ] && [ "$is_wsl" = false ]; then
        clear
        echo -e "${BG_RED}${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${BG_RED}${BOLD}║                 SYSTEM COMPATIBILITY ERROR                   ║${NC}"
        echo -e "${BG_RED}${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
        echo
        echo -e "${RED}${BOLD}⚠️  This script requires a Linux-based environment! ⚠️${NC}"
        echo -e "${YELLOW}Current system detected: ${BOLD}$os_name${NC}"
        echo
        echo -e "${BLUE}${BOLD}You have two options:${NC}"
        echo -e "${GREEN}${BOLD}1.${NC} ${BLUE}Run this script in a Linux environment (native Linux installation)${NC}"
        echo -e "${GREEN}${BOLD}2.${NC} ${BLUE}Use Windows Subsystem for Linux (WSL) if you're on Windows${NC}"
        echo
        echo -e "${YELLOW}${BOLD}Would you like me to help you install WSL automatically? (y/n)${NC}"
        read -p "${CYAN}> ${NC}" install_wsl_auto
        
        if [[ $install_wsl_auto =~ ^[Yy]$ ]]; then
            install_wsl_automatically
        else
            echo -e "${YELLOW}${BOLD}Would you like to see detailed WSL setup instructions instead? (y/n)${NC}"
            read -p "${CYAN}> ${NC}" show_wsl
            if [[ $show_wsl =~ ^[Yy]$ ]]; then
                show_wsl_instructions
            fi
        fi
        exit 1
    fi

    # Check Linux distribution if needed
    if [ "$is_wsl" = true ]; then
        print_success "Running in WSL environment - ${GREEN}Compatible ✓${NC}"
        echo -e "${YELLOW}Note: Docker in WSL may require special configuration.${NC}"
        echo -e "${YELLOW}Select option 5 from the menu for more information.${NC}"
    else
        # Check for systemd support
        if pidof systemd >/dev/null; then
            print_success "Running in Linux environment with systemd - ${GREEN}Compatible ✓${NC}"
        else
            print_warning "Running in Linux environment without systemd"
            echo -e "${YELLOW}Some features may require manual configuration${NC}"
        fi
    fi
}

# Function to check prerequisites
check_prerequisites() {
    local missing_prereqs=false
    local missing_packages=""
    
    print_status "Checking system prerequisites..."
    
    # Check if sudo is available
    if ! command_exists sudo; then
        print_error "sudo is not installed or not in PATH"
        missing_prereqs=true
        missing_packages+=" sudo"
    fi
    
    # Check for core utilities
    for cmd in curl wget git; do
        if ! command_exists $cmd; then
            print_error "$cmd is not installed"
            missing_prereqs=true
            missing_packages+=" $cmd"
        fi
    done
    
    # Check for python3
    if ! command_exists python3; then
        print_error "python3 is not installed"
        missing_prereqs=true
        missing_packages+=" python3"
    fi
    
    if [ "$missing_prereqs" = true ]; then
        echo
        print_warning "Missing prerequisites: ${missing_packages}"
        echo -e "${YELLOW}These packages are required for Polaris installation.${NC}"
        read -p "Would you like to install them now? (y/n): " install_prereqs
        if [[ $install_prereqs =~ ^[Yy]$ ]]; then
            print_status "Installing prerequisites..."
            sudo apt-get update
            sudo apt-get install -y $missing_packages
            echo
            print_success "Prerequisites installed successfully!"
        else
            print_error "Cannot continue without required prerequisites."
            exit 1
        fi
    else
        print_success "All basic prerequisites are installed!"
    fi
}

# Function to display the welcome banner
show_welcome_banner() {
    clear
    echo -e "${CYAN}${BOLD}"
    echo "██████╗  ██████╗ ██╗      █████╗ ██████╗ ██╗███████╗"
    echo "██╔══██╗██╔═══██╗██║     ██╔══██╗██╔══██╗██║██╔════╝"
    echo "██████╔╝██║   ██║██║     ███████║██████╔╝██║███████╗"
    echo "██╔═══╝ ██║   ██║██║     ██╔══██║██╔══██╗██║╚════██║"
    echo "██║     ╚██████╔╝███████╗██║  ██║██║  ██║██║███████║"
    echo "╚═╝      ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚══════╝"
    echo -e "${NC}"
    echo -e "${BOLD}           Compute Subnet Management Tool${NC}"
    echo -e "${YELLOW}═══════════════════════════════════════════════════════${NC}"
    echo
    echo -e "${CYAN}${BOLD}About Polaris:${NC}"
    echo -e "${BLUE}Polaris is a modern development workspace manager for distributed compute resources.${NC}"
    echo -e "${BLUE}It simplifies managing compute resources, monitoring their status, and${NC}"
    echo -e "${BLUE}automating key tasks in a distributed environment.${NC}"
    echo
    echo -e "${BG_RED}${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BG_RED}${BOLD}║               ⚠️  LINUX ENVIRONMENT REQUIRED  ⚠️             ║${NC}"
    echo -e "${BG_RED}${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo -e "${YELLOW}${BOLD}This tool requires a Linux-based environment to run properly${NC}"
    echo -e "${YELLOW}If you're on Windows, please use WSL (Windows Subsystem for Linux)${NC}"
    echo -e "${YELLOW}Select option 5 from the menu for detailed WSL setup instructions${NC}"
    echo
    # Check system compatibility
    check_system_compatibility
    
    # Check prerequisites only if system is compatible
    check_prerequisites
    echo
}

# Function to check if Polaris is installed
check_polaris_installation() {
    # First check if the repository exists
    if [ ! -d "polaris-subnet" ]; then
        return 1 # Not installed
    fi

    # Change into the repository directory
    cd polaris-subnet || return 1

    # Check if virtual environment exists and activate it
    if [ ! -d "venv" ] || [ ! -f "venv/bin/activate" ]; then
        cd ..
        return 1 # Virtual environment not found
    fi

    # Try to activate virtual environment and check polaris
    source venv/bin/activate
    if ! command -v polaris &>/dev/null || ! polaris --help &>/dev/null; then
        deactivate
        cd ..
        return 1 # Polaris command not working
    fi

    # Everything worked, clean up and return success
    deactivate
    cd ..
    return 0
}

# Function to backup Polaris configuration
backup_polaris_config() {
    if [ -d "polaris-subnet" ] && [ -f "polaris-subnet/.env" ]; then
        print_status "Creating backup of Polaris configuration..."
        
        local backup_dir="polaris_backup_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$backup_dir"
        
        # Backup .env file
        cp polaris-subnet/.env "$backup_dir/env_backup"
        
        # Backup other important configuration files if they exist
        if [ -d "polaris-subnet/config" ]; then
            cp -r polaris-subnet/config "$backup_dir/"
        fi
        
        # Check if successful
        if [ -f "$backup_dir/env_backup" ]; then
            print_success "Configuration backup created successfully at: $backup_dir"
            return 0
        else
            print_error "Failed to create backup"
            rm -rf "$backup_dir" 2>/dev/null
            return 1
        fi
    else
        print_error "No Polaris configuration found to backup"
        return 1
    fi
}

# Function to enter Polaris environment
enter_polaris_environment() {
    # First check if the repository exists
    if [ ! -d "polaris-subnet" ]; then
        print_error "Polaris repository not found!"
        return 1
    fi

    # Change into the repository directory
    cd polaris-subnet || return 1

    # Check if virtual environment exists
    if [ ! -d "venv" ] || [ ! -f "venv/bin/activate" ]; then
        print_error "Virtual environment not found!"
        cd ..
        return 1
    fi

    # Activate the environment
    print_status "Activating Polaris environment..."
    source venv/bin/activate

    # Check if polaris command is available
    if ! command -v polaris &>/dev/null; then
        print_error "Polaris command not found in environment!"
        deactivate
        cd ..
        return 1
    fi

    # Show available commands
    clear
    echo -e "${CYAN}${BOLD}Welcome to Polaris Environment${NC}"
    echo -e "${GREEN}─────────────────────────────────────────────────────${NC}"
    echo -e "${YELLOW}Available Polaris Commands:${NC}"
    echo -e "• ${CYAN}polaris start${NC}     - Start Polaris services"
    echo -e "• ${CYAN}polaris stop${NC}      - Stop Polaris services"
    echo -e "• ${CYAN}polaris status${NC}    - Check service status"
    echo -e "• ${CYAN}polaris logs${NC}      - View service logs"
    echo -e "• ${CYAN}polaris register${NC}  - Register as a new miner"
    echo -e "• ${CYAN}polaris --help${NC}    - Show all available commands"
    echo
    echo -e "${YELLOW}Current Status:${NC}"
    polaris status
    echo
    echo -e "${GREEN}─────────────────────────────────────────────────────${NC}"
    echo -e "${YELLOW}You are now in the Polaris environment.${NC}"
    echo -e "${YELLOW}Type 'exit' to leave this environment.${NC}"
    echo

    # Start an interactive shell in the environment
    $SHELL

    # When shell exits, deactivate the environment
    deactivate
    cd ..
    echo -e "${GREEN}Exited Polaris environment.${NC}"
}

# Function to show menu and get user choice
show_menu() {
    local is_installed=$(check_polaris_installation && echo true || echo false)
    
    echo -e "${YELLOW}Available Options:${NC}"
    echo -e "${GREEN}─────────────────────────────────────────────────────${NC}"
    echo -e "1) ${CYAN}Install Polaris${NC}"
    if [ "$is_installed" = true ]; then
        echo -e "2) ${YELLOW}Reinstall Polaris${NC}"
        echo -e "3) ${GREEN}Enter Polaris Environment${NC}"
    fi
    echo -e "4) ${RED}Uninstall Polaris${NC}"
    echo -e "5) ${BLUE}Check Installation Status${NC}"
    echo -e "6) ${MAGENTA}Show WSL Setup Instructions${NC}"
    echo -e "7) ${GREEN}Backup Polaris Configuration${NC}"
    echo -e "8) ${YELLOW}Advanced Options${NC}"
    echo -e "9) ${RED}Exit${NC}"
    echo -e "${GREEN}─────────────────────────────────────────────────────${NC}"
    echo
    read -p "Please select an option [1-9]: " choice
    echo

    case $choice in
        1)
            if [ "$is_installed" = true ]; then
                print_warning "Polaris is already installed!"
                echo -e "Choose ${YELLOW}Reinstall${NC} option if you want to install again."
                sleep 2
                return 0
            else
                install_polaris
            fi
            ;;
        2)
            if [ "$is_installed" = true ]; then
                print_warning "This will reinstall Polaris. Your current installation will be removed."
                read -p "Do you want to continue? (y/n): " confirm
                if [[ $confirm =~ ^[Yy]$ ]]; then
                    # Ask for backup before reinstall
                    read -p "Do you want to backup your configuration first? (y/n): " backup_confirm
                    if [[ $backup_confirm =~ ^[Yy]$ ]]; then
                        backup_polaris_config
                    fi
                    uninstall_polaris
                    install_polaris
                fi
            else
                print_error "Polaris is not installed. Choose Install option instead."
                sleep 2
            fi
            ;;
        3)
            if [ "$is_installed" = true ]; then
                enter_polaris_environment
            else
                print_error "Invalid option"
                sleep 1
            fi
            ;;
        4)
            if [ "$is_installed" = true ]; then
                # Ask for backup before uninstall
                read -p "Do you want to backup your configuration before uninstalling? (y/n): " backup_confirm
                if [[ $backup_confirm =~ ^[Yy]$ ]]; then
                    backup_polaris_config
                fi
                uninstall_polaris
            else
                print_error "Polaris is not installed."
                sleep 2
            fi
            ;;
        5)
            if [ "$is_installed" = true ]; then
                print_success "Polaris is installed and configured."
                echo -e "${YELLOW}To use Polaris, select option 3 to enter Polaris environment.${NC}"
                if command -v polaris &> /dev/null; then
                    echo -e "${BLUE}Current Polaris status:${NC}"
                    polaris status
                fi
            else
                print_warning "Polaris is not installed on this system."
            fi
            echo
            read -p "Press Enter to continue..."
            ;;
        6)
            show_wsl_instructions
            ;;
        7)
            if [ "$is_installed" = true ]; then
                backup_polaris_config
                read -p "Press Enter to continue..."
            else
                print_error "Polaris is not installed. Nothing to backup."
                sleep 2
            fi
            ;;
        8)
            show_advanced_options
            ;;
        9)
            echo -e "${YELLOW}Thank you for using Polaris Manager!${NC}"
            exit 0
            ;;
        *)
            print_error "Invalid option"
            sleep 1
            ;;
    esac
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check and install Docker
install_docker() {
    if command_exists docker; then
        print_success "Docker is already installed"
        # Start Docker if not running
        if pidof systemd >/dev/null && ! systemctl is-active --quiet docker; then
            print_status "Starting Docker service..."
            sudo systemctl start docker
        fi
    else
        print_status "Installing Docker..."
        
        # Check if running in WSL
        local is_wsl=false
        if [ -f /proc/version ] && grep -qi microsoft /proc/version; then
            is_wsl=true
            print_warning "Installing Docker in WSL environment..."
            echo -e "${YELLOW}For optimal performance, consider using Docker Desktop for Windows with WSL2 integration.${NC}"
            echo -e "${YELLOW}See https://docs.docker.com/desktop/wsl/ for more information.${NC}"
            sleep 3
        fi
        
        # Remove old versions if they exist
        for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do
            sudo apt-get remove -y $pkg >/dev/null 2>&1
        done

        # Install Docker prerequisites
        sudo apt-get update
        sudo apt-get install -y ca-certificates curl gnupg

        # Add Docker's official GPG key
        sudo install -m 0755 -d /etc/apt/keyrings
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        sudo chmod a+r /etc/apt/keyrings/docker.gpg

        # Add the repository to Apt sources
        echo \
            "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
            "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
            sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

        # Install Docker Engine
        sudo apt-get update
        sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

        # Start Docker service
        if pidof systemd >/dev/null; then
            sudo systemctl start docker
        else
            print_warning "Could not start Docker service automatically."
            print_warning "You may need to start it manually using appropriate commands for your system."
        fi

        # Add current user to docker group
        sudo usermod -aG docker $USER
        print_success "Docker installed successfully"
        
        # Special instructions for WSL
        if [ "$is_wsl" = true ]; then
            print_warning "In WSL, you may need to start Docker manually:"
            echo -e "${YELLOW}  sudo service docker start${NC}"
            echo
            print_warning "Or consider using Docker Desktop for Windows with WSL2 integration"
            read -p "Press Enter to continue..."
        fi
    fi
}

# Function to check and install SSH server
install_ssh() {
    if ! command_exists sshd; then
        print_status "Installing SSH server..."
        sudo apt-get update
        sudo apt-get install -y openssh-server
        
        if pidof systemd >/dev/null; then
            sudo systemctl enable ssh
            sudo systemctl start ssh
        else
            print_warning "Could not start SSH service automatically."
            print_warning "You may need to start it manually using appropriate commands for your system."
            echo -e "${YELLOW}Try: sudo service ssh start${NC}"
        fi
        
        print_success "SSH server installed"
    else
        print_success "SSH server is already installed"
        # Ensure SSH is running if systemd is available
        if pidof systemd >/dev/null && ! systemctl is-active --quiet ssh; then
            print_status "Starting SSH service..."
            sudo systemctl start ssh
        elif [ ! -z "$(ps -e | grep sshd)" ]; then
            print_success "SSH service is running"
        else
            print_warning "SSH service is not running."
            echo -e "${YELLOW}Try: sudo service ssh start${NC}"
        fi
    fi
}

# Function to install Python requirements
install_python_requirements() {
    print_status "Installing Python requirements..."
    sudo apt-get update
    sudo apt-get install -y python3-venv python3-pip g++ rustc cargo build-essential python3-dev
    print_success "Python requirements installed"
}

# Function to get valid port range
get_port_range() {
    while true; do
        read -p "Enter starting port number for SSH (recommended range 11000-65000): " port_start
        read -p "Enter ending port number for SSH (must be greater than start port): " port_end
        
        if [[ "$port_start" =~ ^[0-9]+$ ]] && [[ "$port_end" =~ ^[0-9]+$ ]] && \
           [ "$port_start" -ge 1024 ] && [ "$port_end" -le 65535 ] && \
           [ "$port_end" -gt "$port_start" ]; then
            break
        else
            print_error "Invalid port range. Please enter valid numbers (start: 1024-65534, end: start+1-65535)"
        fi
    done
}

# Function to uninstall Polaris
uninstall_polaris() {
    print_warning "This will remove all Polaris components from your system."
    read -p "Are you sure you want to continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Uninstallation cancelled."
        return 0
    fi

    # Stop Polaris processes
    print_status "Stopping Polaris processes..."
    if [ -d "polaris-subnet" ]; then
        cd polaris-subnet
        if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
            source venv/bin/activate
            if command -v polaris &> /dev/null; then
                polaris status 2>/dev/null | grep -q "is running" && polaris stop 
            fi
            deactivate
        fi
        cd ..
    fi

    # Remove virtual environment and files
    if [ -d "polaris-subnet" ]; then
        cd polaris-subnet
        if [ -d "venv" ]; then
            print_status "Removing virtual environment..."
            rm -rf venv
        fi
        
        # Remove configuration files
        if [ -f ".env" ]; then
            print_status "Removing configuration files..."
            rm .env
        fi

        # Clean Python cache
        print_status "Cleaning Python cache..."
        find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find . -type f -name "*.pyc" -delete 2>/dev/null || true
        find . -type f -name "*.pyo" -delete 2>/dev/null || true
        find . -type f -name "*.pyd" -delete 2>/dev/null || true
        find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
        find . -type d -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true
        
        cd ..
        print_status "Removing Polaris directory..."
        rm -rf polaris-subnet
    fi

    print_success "Polaris has been uninstalled successfully!"
    sleep 2
}

# Function to install Polaris
install_polaris() {
    clear
    echo -e "${BG_GREEN}${BOLD}                 Polaris Installation                 ${NC}"
    echo -e "${YELLOW}═══════════════════════════════════════════════════════${NC}"
    echo

    # Check and install prerequisites
    print_status "Checking and installing prerequisites..."
    
    # Install required packages
    sudo apt-get update
    sudo apt-get install -y curl wget git

    # Check Python version compatibility
    print_status "Checking Python version compatibility..."
    if command_exists python3; then
        python_version=$(python3 --version | cut -d ' ' -f 2)
        python_major=$(echo "$python_version" | cut -d '.' -f 1)
        python_minor=$(echo "$python_version" | cut -d '.' -f 2)
        
        if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 8 ]); then
            print_error "Python version $python_version is too old. Polaris requires Python 3.8 or higher."
            read -p "Would you like to install a compatible Python version? (y/n): " install_python
            if [[ $install_python =~ ^[Yy]$ ]]; then
                print_status "Installing Python 3.10..."
                sudo apt-get install -y software-properties-common
                sudo add-apt-repository -y ppa:deadsnakes/ppa
                sudo apt-get update
                sudo apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip
                print_success "Python 3.10 installed successfully!"
                print_warning "Some commands in this script will continue to use the system Python."
                print_warning "However, Polaris will be installed using Python 3.10."
            else
                print_error "Cannot continue without a compatible Python version."
                return 1
            fi
        else
            print_success "Python version $python_version is compatible."
        fi
    else
        print_error "Python 3 is not installed."
        read -p "Would you like to install Python 3.10? (y/n): " install_python
        if [[ $install_python =~ ^[Yy]$ ]]; then
            print_status "Installing Python 3.10..."
            sudo apt-get install -y software-properties-common
            sudo add-apt-repository -y ppa:deadsnakes/ppa
            sudo apt-get update
            sudo apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip
            print_success "Python 3.10 installed successfully!"
        else
            print_error "Cannot continue without Python 3."
            return 1
        fi
    fi

    # Install Docker
    install_docker

    # Install SSH server
    install_ssh

    # Install system dependencies first
    print_status "Installing system dependencies..."
    sudo apt-get install -y g++ rustc cargo build-essential python3-dev
    check_command "Failed to install system dependencies" 0

    # Get user inputs
    print_status "Please provide the following information:"
    echo
    
    # Get public IP
    while true; do
        read -p "Enter your public IP address: " public_ip
        if [[ $public_ip =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            break
        else
            print_error "Invalid IP address format. Please try again."
        fi
    done

    # Get SSH username
    read -p "Enter SSH username: " ssh_user

    # Get SSH password
    while true; do
        read -s -p "Enter SSH password: " ssh_password
        echo
        read -s -p "Confirm SSH password: " ssh_password_confirm
        echo
        if [ "$ssh_password" = "$ssh_password_confirm" ]; then
            print_warning "Note: This password will be stored in plaintext in the .env file."
            print_warning "For production environments, consider using SSH keys instead."
            echo -e "${YELLOW}Do you want to continue with password authentication? (y/n)${NC}"
            read -p "${CYAN}> ${NC}" use_password
            if [[ $use_password =~ ^[Yy]$ ]]; then
                break
            else
                echo -e "${CYAN}Would you like to set up SSH key authentication instead? (y/n)${NC}"
                read -p "${CYAN}> ${NC}" setup_ssh_key
                if [[ $setup_ssh_key =~ ^[Yy]$ ]]; then
                    setup_ssh_key_auth
                    use_ssh_key=true
                    break
                fi
            fi
        else
            print_error "Passwords do not match. Please try again."
        fi
    done

    # Get port range
    get_port_range

    # Create project directory and setup virtual environment
    print_status "Setting up Polaris..."
    
    # Clone repository
    git clone https://github.com/bigideainc/polaris-subnet.git
    if [ $? -ne 0 ]; then
        print_error "Failed to clone the repository. Please check your internet connection and try again."
        return 1
    fi
    
    cd polaris-subnet

    # Create virtual environment
    print_status "Creating Python virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        print_error "Failed to create virtual environment. Please check your Python installation."
        cd ..
        return 1
    fi
    
    source venv/bin/activate

    # Install packages in the correct order to avoid dependency conflicts
    print_status "Installing Python dependencies in the correct order..."
    
    # First upgrade pip
    pip install --upgrade pip
    
    # Install network-specific packages in the correct order
    print_status "Installing Bittensor and related packages..."
    pip install bittensor
    check_command "Failed to install bittensor" 0
    
    pip install bittensor-cli
    check_command "Failed to install bittensor-cli" 0
    
    pip install communex==0.1.36.4
    check_command "Failed to install communex" 0

    # Now install the rest of the requirements
    print_status "Installing remaining dependencies..."
    pip install -r requirements.txt
    check_command "Failed to install requirements" 0

    # Install Polaris in development mode
    pip install -e .
    check_command "Failed to install Polaris" 0

    # Create .env file
    print_status "Creating .env file..."
    
    # Check if SSH key authentication is being used
    if [ "$use_ssh_key" = true ]; then
        # Create .env file without SSH_PASSWORD
        cat > .env << EOF
HOST=$public_ip
API_PORT=8000
SSH_PORT_RANGE_START=$port_start
SSH_PORT_RANGE_END=$port_end
SSH_USER=$ssh_user
SSH_HOST=$public_ip
SSH_PORT=$port_start
SERVER_URL=https://polaris-test-server.onrender.com/api/v1
USE_SSH_KEY=true
EOF
    else
        # Create .env file with SSH_PASSWORD
        cat > .env << EOF
HOST=$public_ip
API_PORT=8000
SSH_PORT_RANGE_START=$port_start
SSH_PORT_RANGE_END=$port_end
SSH_PASSWORD=$ssh_password
SSH_USER=$ssh_user
SSH_HOST=$public_ip
SSH_PORT=$port_start
SERVER_URL=https://polaris-test-server.onrender.com/api/v1
EOF
    fi

    print_success "Polaris setup completed successfully!"
    print_status "Starting Polaris..."
    
    # Start Polaris
    polaris start

    echo
    print_success "Setup complete! Polaris is now running."
    echo
    print_warning "Important: Please log out and log back in for Docker group changes to take effect."
    echo
    print_status "You can use the following commands:"
    echo "  - polaris status : Check service status"
    echo "  - polaris logs   : View logs"
    echo "  - polaris stop   : Stop the service"
    
    read -p "Press Enter to continue..."
}

# Function to set up SSH key authentication
setup_ssh_key_auth() {
    print_status "Setting up SSH key authentication..."
    
    # Check if SSH key exists
    if [ ! -f ~/.ssh/id_rsa ]; then
        print_status "Generating SSH key pair..."
        ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""
        check_command "Failed to generate SSH key pair" 0
    fi
    
    print_success "SSH key setup complete!"
    echo
    print_status "Your public key is:"
    cat ~/.ssh/id_rsa.pub
    echo
    print_warning "You'll need to add this public key to authorized_keys on your servers"
    print_warning "or distribute it to your clients."
    echo
}

# Function to show advanced options
show_advanced_options() {
    clear
    echo -e "${CYAN}${BOLD}Advanced Options:${NC}"
    echo -e "${GREEN}─────────────────────────────────────────────────────${NC}"
    echo -e "1) ${BLUE}Update Polaris to Latest Version${NC}"
    echo -e "2) ${YELLOW}Repair Docker Installation${NC}"
    echo -e "3) ${CYAN}Configure Firewall for Polaris${NC}"
    echo -e "4) ${MAGENTA}Troubleshoot Common Issues${NC}"
    echo -e "5) ${GREEN}Back to Main Menu${NC}"
    echo -e "${GREEN}─────────────────────────────────────────────────────${NC}"
    echo
    read -p "Please select an option [1-5]: " adv_choice
    echo

    case $adv_choice in
        1)
            update_polaris
            ;;
        2)
            repair_docker
            ;;
        3)
            configure_firewall
            ;;
        4)
            troubleshoot_issues
            ;;
        5)
            return 0
            ;;         *)
            print_error "Invalid option"
            sleep 1
            show_advanced_options
            ;;
    esac
}

# Function to update Polaris to latest version
update_polaris() {
    if ! check_polaris_installation; then
        print_error "Polaris is not installed. Please install it first."
        read -p "Press Enter to continue..."
        return 1
    fi

    print_status "Updating Polaris to the latest version..."
    
    # Backup before update
    print_status "Creating backup before update..."
    backup_polaris_config
    
    # Update the code
    cd polaris-subnet
    print_status "Pulling latest changes from repository..."
    git pull
    check_command "Failed to pull latest changes" 0
    
    # Update dependencies
    print_status "Updating dependencies..."
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt --upgrade
    check_command "Failed to update dependencies" 0
    
    # Reinstall the package
    pip install -e . --upgrade
    check_command "Failed to reinstall the package" 0
    
    print_status "Restarting Polaris services..."
    polaris restart
    check_command "Failed to restart Polaris services" 0
    
    deactivate
    cd ..
    
    print_success "Polaris has been updated successfully!"
    read -p "Press Enter to continue..."
}

# Function to repair Docker installation
repair_docker() {
    print_status "Checking Docker installation..."
    
    if ! command_exists docker; then
        print_warning "Docker is not installed. Installing Docker..."
        install_docker
        read -p "Press Enter to continue..."
        return
    fi
    
    # Check if Docker daemon is running
    if ! docker info &>/dev/null; then
        print_warning "Docker daemon is not running. Attempting to start..."
        if pidof systemd >/dev/null; then
            sudo systemctl start docker
            if ! docker info &>/dev/null; then
                print_error "Failed to start Docker daemon. Trying to reinstall..."
                sudo apt-get remove -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
                install_docker
            else
                print_success "Docker daemon started successfully!"
            fi
        else
            print_warning "No systemd detected. Trying to start Docker service manually..."
            sudo service docker start
            if ! docker info &>/dev/null; then
                print_error "Failed to start Docker service. Trying to reinstall..."
                sudo apt-get remove -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
                install_docker
            else
                print_success "Docker service started successfully!"
            fi
        fi
    else
        print_success "Docker is installed and running correctly!"
        
        # Verify user is in docker group
        if ! groups | grep -q docker; then
            print_warning "Your user is not in the docker group. Adding..."
            sudo usermod -aG docker $USER
            print_warning "You need to log out and log back in for this change to take effect."
        fi
    fi
    
    read -p "Press Enter to continue..."
}

# Function to configure firewall for Polaris
configure_firewall() {
    print_status "Configuring firewall for Polaris..."
    
    # Check if ufw is installed
    if ! command_exists ufw; then
        print_warning "UFW (Uncomplicated Firewall) is not installed."
        read -p "Do you want to install UFW? (y/n): " install_ufw
        if [[ $install_ufw =~ ^[Yy]$ ]]; then
            sudo apt-get update
            sudo apt-get install -y ufw
            check_command "Failed to install UFW" 0
        else
            print_warning "Firewall configuration aborted."
            read -p "Press Enter to continue..."
            return
        fi
    fi
    
    # Check if the .env file exists to get port ranges
    if [ -f "polaris-subnet/.env" ]; then
        source polaris-subnet/.env
        
        # Allow SSH ports
        if [ ! -z "$SSH_PORT_RANGE_START" ] && [ ! -z "$SSH_PORT_RANGE_END" ]; then
            print_status "Allowing SSH port range $SSH_PORT_RANGE_START to $SSH_PORT_RANGE_END..."
            for port in $(seq $SSH_PORT_RANGE_START $SSH_PORT_RANGE_END); do
                sudo ufw allow $port/tcp
            done
        fi
        
        # Allow API port
        if [ ! -z "$API_PORT" ]; then
            print_status "Allowing API port $API_PORT..."
            sudo ufw allow $API_PORT/tcp
        fi
        
        # Enable UFW if it's not already enabled
        if ! sudo ufw status | grep -q "Status: active"; then
            print_warning "UFW is not enabled. Enabling..."
            print_warning "This might disconnect your SSH session if port 22 is not allowed."
            read -p "Are you sure you want to enable UFW? (y/n): " enable_ufw
            if [[ $enable_ufw =~ ^[Yy]$ ]]; then
                # Make sure SSH is allowed to prevent lockout
                sudo ufw allow 22/tcp
                sudo ufw --force enable
            fi
        fi
        
        print_success "Firewall configured successfully!"
    else
        print_error "Could not find Polaris configuration (.env file)."
        print_error "Please install Polaris first or restore a configuration backup."
    fi
    
    read -p "Press Enter to continue..."
}

# Function to troubleshoot common issues
troubleshoot_issues() {
    clear
    echo -e "${CYAN}${BOLD}Troubleshooting Common Issues:${NC}"
    echo -e "${GREEN}─────────────────────────────────────────────────────${NC}"
    echo -e "1) ${RED}Docker permission issues${NC}"
    echo -e "2) ${RED}SSH connection problems${NC}"
    echo -e "3) ${RED}Polaris not starting${NC}"
    echo -e "4) ${RED}Python environment issues${NC}"
    echo -e "5) ${GREEN}Back to Advanced Options${NC}"
    echo -e "${GREEN}─────────────────────────────────────────────────────${NC}"
    echo
    read -p "Please select an issue to troubleshoot [1-5]: " issue_choice
    echo

    case $issue_choice in
        1)
            troubleshoot_docker_permissions
            ;;
        2)
            troubleshoot_ssh_connection
            ;;
        3)
            troubleshoot_polaris_startup
            ;;
        4)
            troubleshoot_python_env
            ;;
        5)
            show_advanced_options
            return 0
            ;;
        *)
            print_error "Invalid option"
            sleep 1
            troubleshoot_issues
            ;;
    esac
}

# Template function for troubleshooting sections
troubleshoot_docker_permissions() {
    clear
    echo -e "${CYAN}${BOLD}Troubleshooting Docker Permissions:${NC}"
    echo -e "${GREEN}─────────────────────────────────────────────────────${NC}"
    echo -e "${YELLOW}Common Docker permission issues:${NC}"
    echo -e "1. User not in docker group"
    echo -e "2. Docker daemon not running"
    echo -e "3. Socket permission issues"
    echo
    
    print_status "Checking if user is in docker group..."
    if groups | grep -q docker; then
        print_success "User is in docker group."
    else
        print_error "User is NOT in docker group."
        print_status "Adding user to docker group..."
        sudo usermod -aG docker $USER
        print_warning "You need to log out and log back in for this change to take effect."
    fi
    
    print_status "Checking Docker daemon status..."
    if docker info &>/dev/null; then
        print_success "Docker daemon is running."
    else
        print_error "Docker daemon is NOT running."
        print_status "Starting Docker daemon..."
        if pidof systemd >/dev/null; then
            sudo systemctl start docker
        else
            sudo service docker start
        fi
    fi
    
    print_status "Checking Docker socket permissions..."
    if [ -S /var/run/docker.sock ]; then
        socket_perms=$(ls -la /var/run/docker.sock)
        echo -e "Socket permissions: ${YELLOW}$socket_perms${NC}"
        
        if [[ $socket_perms == *"root docker"* ]]; then
            print_success "Socket permissions appear correct."
        else
            print_warning "Socket permissions may be incorrect."
            print_status "Correcting socket permissions..."
            sudo chown root:docker /var/run/docker.sock
        fi
    else
        print_error "Docker socket does not exist at /var/run/docker.sock"
    fi
    
    echo
    read -p "Press Enter to return to the troubleshooting menu..."
    troubleshoot_issues
}

# Troubleshooting functions for SSH connection issues
troubleshoot_ssh_connection() {
    clear
    echo -e "${CYAN}${BOLD}Troubleshooting SSH Connection Issues:${NC}"
    echo -e "${GREEN}─────────────────────────────────────────────────────${NC}"
    echo -e "${YELLOW}Common SSH connection issues:${NC}"
    echo -e "1. SSH service not running"
    echo -e "2. Firewall blocking SSH ports"
    echo -e "3. Incorrect SSH configuration"
    echo -e "4. Authentication issues"
    echo
    
    # Check if SSH service is running
    print_status "Checking if SSH service is running..."
    if pidof systemd >/dev/null; then
        if systemctl is-active --quiet ssh || systemctl is-active --quiet sshd; then
            print_success "SSH service is running."
        else
            print_error "SSH service is NOT running."
            print_status "Starting SSH service..."
            sudo systemctl start ssh || sudo systemctl start sshd
        fi
    else
        if pgrep sshd >/dev/null; then
            print_success "SSH daemon is running."
        else
            print_error "SSH daemon is NOT running."
            print_status "Starting SSH daemon..."
            sudo service ssh start || sudo service sshd start
        fi
    fi
    
    # Check SSH configuration
    print_status "Checking SSH configuration..."
    if [ -f /etc/ssh/sshd_config ]; then
        if grep -q "^Port " /etc/ssh/sshd_config; then
            ssh_port=$(grep "^Port " /etc/ssh/sshd_config | awk '{print $2}')
            echo -e "SSH port configured in sshd_config: ${YELLOW}$ssh_port${NC}"
        else
            echo -e "SSH is using the default port: ${YELLOW}22${NC}"
        fi
        
        # Check if password authentication is enabled
        if grep -q "^PasswordAuthentication yes" /etc/ssh/sshd_config; then
            print_success "Password authentication is enabled."
        else
            if grep -q "^PasswordAuthentication no" /etc/ssh/sshd_config; then
                print_warning "Password authentication is disabled."
                print_warning "Make sure your SSH keys are properly set up."
            else
                print_warning "Password authentication setting not found in sshd_config."
                print_warning "Default is usually to allow password authentication."
            fi
        fi
    else
        print_error "SSH configuration file not found at /etc/ssh/sshd_config"
    fi
    
    # Check Polaris SSH settings if available
    if [ -f "polaris-subnet/.env" ]; then
        print_status "Checking Polaris SSH configuration..."
        source polaris-subnet/.env
        
        if [ ! -z "$SSH_PORT_RANGE_START" ] && [ ! -z "$SSH_PORT_RANGE_END" ]; then
            echo -e "Polaris SSH port range: ${YELLOW}$SSH_PORT_RANGE_START - $SSH_PORT_RANGE_END${NC}"
        fi
        
        if [ ! -z "$SSH_USER" ]; then
            echo -e "Polaris SSH user: ${YELLOW}$SSH_USER${NC}"
        fi
        
        if [ ! -z "$SSH_HOST" ]; then
            echo -e "Polaris SSH host: ${YELLOW}$SSH_HOST${NC}"
        fi
    fi
    
    # Check firewall settings
    print_status "Checking firewall settings..."
    if command_exists ufw; then
        if sudo ufw status | grep -q "Status: active"; then
            echo -e "UFW firewall is active."
            echo -e "SSH ports allowed:"
            sudo ufw status | grep -E "^[0-9]+/tcp.*ALLOW" | grep -v "^80/tcp"
        else
            print_success "UFW firewall is inactive. No firewall blocking SSH connections."
        fi
    elif command_exists iptables; then
        echo -e "Checking iptables rules for SSH ports..."
        sudo iptables -L -n | grep -i ssh
    fi
    
    echo
    print_status "Testing SSH connection locally..."
    if command_exists ssh; then
        ssh_port=22
        if [ -f "polaris-subnet/.env" ] && [ ! -z "$SSH_PORT" ]; then
            ssh_port=$SSH_PORT
        fi
        
        # Test SSH connection to localhost
        echo -e "Attempting to connect to localhost on port $ssh_port..."
        timeout 5 ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no -p $ssh_port localhost exit 2>/dev/null
        if [ $? -eq 0 ]; then
            print_success "SSH connection to localhost successful!"
        else
            print_error "Could not connect to SSH on localhost."
            print_error "This may indicate a configuration issue."
        fi
    fi
    
    echo
    read -p "Press Enter to return to the troubleshooting menu..."
    troubleshoot_issues
}

# Troubleshooting function for Polaris startup issues
troubleshoot_polaris_startup() {
    clear
    echo -e "${CYAN}${BOLD}Troubleshooting Polaris Startup Issues:${NC}"
    echo -e "${GREEN}─────────────────────────────────────────────────────${NC}"
    echo -e "${YELLOW}Common Polaris startup issues:${NC}"
    echo -e "1. Missing or incorrect configuration"
    echo -e "2. Docker not running"
    echo -e "3. Port conflicts"
    echo -e "4. Python environment issues"
    echo
    
    if ! check_polaris_installation; then
        print_error "Polaris is not installed or installation is incomplete."
        print_error "Please install Polaris first."
        read -p "Press Enter to return to the troubleshooting menu..."
        troubleshoot_issues
        return
    fi
    
    print_status "Checking Polaris configuration..."
    if [ -f "polaris-subnet/.env" ]; then
        print_success "Polaris .env configuration file found."
        
        # Check for essential configuration variables
        source polaris-subnet/.env
        local missing_vars=false
        
        for var in HOST API_PORT SSH_PORT_RANGE_START SSH_PORT_RANGE_END SSH_USER SSH_PASSWORD SSH_HOST SSH_PORT SERVER_URL; do
            if [ -z "${!var}" ]; then
                print_error "Missing configuration: $var"
                missing_vars=true
            fi
        done
        
        if [ "$missing_vars" = true ]; then
            print_warning "Some essential configuration variables are missing."
            read -p "Would you like to recreate the .env file? (y/n): " recreate_env
            if [[ $recreate_env =~ ^[Yy]$ ]]; then
                # Get user inputs to recreate .env
                cd polaris-subnet
                
                # Back up the existing .env file
                if [ -f ".env" ]; then
                    mv .env .env.backup.$(date +%Y%m%d_%H%M%S)
                fi
                
                # Create a new .env file
                print_status "Creating new .env file..."
                # Get public IP
                while true; do
                    read -p "Enter your public IP address: " public_ip
                    if [[ $public_ip =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
                        break
                    else
                        print_error "Invalid IP address format. Please try again."
                    fi
                done

                # Get SSH username
                read -p "Enter SSH username: " ssh_user

                # Get SSH password
                while true; do
                    read -s -p "Enter SSH password: " ssh_password
                    echo
                    read -s -p "Confirm SSH password: " ssh_password_confirm
                    echo
                    if [ "$ssh_password" = "$ssh_password_confirm" ]; then
                        break
                    else
                        print_error "Passwords do not match. Please try again."
                    fi
                done

                # Get port range
                get_port_range
                
                # Create .env file
                cat > .env << EOF
HOST=$public_ip
API_PORT=8000
SSH_PORT_RANGE_START=$port_start
SSH_PORT_RANGE_END=$port_end
SSH_PASSWORD=$ssh_password
SSH_USER=$ssh_user
SSH_HOST=$public_ip
SSH_PORT=$port_start
SERVER_URL=https://polaris-test-server.onrender.com/api/v1
EOF
                print_success "New .env file created."
                cd ..
            fi
        fi
    else
        print_error "Polaris configuration file (.env) not found."
        read -p "Would you like to recreate the .env file? (y/n): " recreate_env
        if [[ $recreate_env =~ ^[Yy]$ ]]; then
            # Similar logic to recreate .env as above
            print_status "Please recreate your Polaris configuration manually using option 2 from the main menu."
        fi
    fi
    
    # Check Docker status
    print_status "Checking Docker status..."
    if ! command_exists docker; then
        print_error "Docker is not installed. Installing Docker..."
        install_docker
    elif ! docker info &>/dev/null; then
        print_error "Docker daemon is not running."
        if pidof systemd >/dev/null; then
            print_status "Starting Docker service..."
            sudo systemctl start docker
        else
            print_status "Starting Docker service..."
            sudo service docker start
        fi
    else
        print_success "Docker is running."
    fi
    
    # Check for port conflicts
    print_status "Checking for port conflicts..."
    if [ -f "polaris-subnet/.env" ]; then
        source polaris-subnet/.env
        
        if [ ! -z "$API_PORT" ]; then
            if netstat -tuln 2>/dev/null | grep -q ":$API_PORT"; then
                process=$(sudo lsof -i :$API_PORT | tail -n 1)
                print_error "Port $API_PORT is already in use by another process:"
                echo "$process"
            else
                print_success "API port $API_PORT is available."
            fi
        fi
        
        if [ ! -z "$SSH_PORT" ]; then
            if netstat -tuln 2>/dev/null | grep -q ":$SSH_PORT"; then
                process=$(sudo lsof -i :$SSH_PORT | tail -n 1)
                print_error "SSH port $SSH_PORT is already in use by another process:"
                echo "$process"
            else
                print_success "SSH port $SSH_PORT is available."
            fi
        fi
    fi
    
    # Check Polaris status using its command
    print_status "Checking Polaris service status..."
    (
        cd polaris-subnet
        source venv/bin/activate
        polaris status
        deactivate
    )
    
    # Attempt to restart Polaris if needed
    read -p "Would you like to restart Polaris services? (y/n): " restart_polaris
    if [[ $restart_polaris =~ ^[Yy]$ ]]; then
        print_status "Restarting Polaris services..."
        (
            cd polaris-subnet
            source venv/bin/activate
            polaris stop
            sleep 2
            polaris start
            deactivate
        )
    fi
    
    echo
    read -p "Press Enter to return to the troubleshooting menu..."
    troubleshoot_issues
}

# Troubleshooting function for Python environment issues
troubleshoot_python_env() {
    clear
    echo -e "${CYAN}${BOLD}Troubleshooting Python Environment Issues:${NC}"
    echo -e "${GREEN}─────────────────────────────────────────────────────${NC}"
    echo -e "${YELLOW}Common Python environment issues:${NC}"
    echo -e "1. Missing or corrupted virtual environment"
    echo -e "2. Missing dependencies"
    echo -e "3. Python version incompatibility"
    echo -e "4. Permissions issues"
    echo
    
    # Check Python installation
    print_status "Checking Python installation..."
    if command_exists python3; then
        python_version=$(python3 --version)
        print_success "$python_version is installed."
    else
        print_error "Python 3 is not installed."
        print_status "Installing Python 3..."
        sudo apt-get update
        sudo apt-get install -y python3 python3-venv python3-pip
    fi
    
    # Check virtualenv
    print_status "Checking virtual environment..."
    local recreate_venv=false
    if [ -d "polaris-subnet/venv" ]; then
        if [ -f "polaris-subnet/venv/bin/activate" ]; then
            print_success "Virtual environment exists."
            
            # Check if the virtual environment is functional
            if (cd polaris-subnet && source venv/bin/activate && python3 -c "print('venv test')" &>/dev/null); then
                print_success "Virtual environment is working properly."
            else
                print_error "Virtual environment exists but appears to be corrupted."
                recreate_venv=true
            fi
        else
            print_error "Virtual environment exists but is missing activation script."
            recreate_venv=true
        fi
    else
        print_error "Virtual environment not found."
        recreate_venv=true
    fi
    
    # Recreate virtual environment if needed
    if [ "$recreate_venv" = true ]; then
        print_warning "Virtual environment needs to be recreated."
        read -p "Would you like to recreate the virtual environment? (y/n): " recreate_venv_confirm
        if [[ $recreate_venv_confirm =~ ^[Yy]$ ]]; then
            print_status "Recreating virtual environment..."
            
            if [ -d "polaris-subnet/venv" ]; then
                rm -rf polaris-subnet/venv
            fi
            
            (
                cd polaris-subnet
                python3 -m venv venv
                if [ $? -ne 0 ]; then
                    print_error "Failed to create virtual environment."
                    return 1
                fi
                
                source venv/bin/activate
                pip install --upgrade pip
                pip install -r requirements.txt
                pip install -e .
                pip install bittensor bittensor-cli communex==0.1.36.4
                deactivate
            )
            
            print_success "Virtual environment recreated successfully!"
        fi
    fi
    
    # Check dependencies
    print_status "Checking Python dependencies..."
    (
        cd polaris-subnet
        source venv/bin/activate
        
        # Check for missing packages in requirements.txt
        missing_packages=()
        if [ -f "requirements.txt" ]; then
            while read -r package; do
                # Skip empty lines and comments
                [[ -z "$package" || "$package" =~ ^# ]] && continue
                
                # Extract package name (remove version specifiers)
                package_name=$(echo "$package" | cut -d'=' -f1 | cut -d'>' -f1 | cut -d'<' -f1 | cut -d'~' -f1 | cut -d'!' -f1 | tr -d ' ')
                
                # Check if the package is installed
                pip show "$package_name" &>/dev/null || missing_packages+=("$package")
            done < requirements.txt
        fi
        
        # Report missing packages
        if [ ${#missing_packages[@]} -gt 0 ]; then
            print_error "Missing Python packages detected:"
            for pkg in "${missing_packages[@]}"; do
                echo -e "  - ${RED}$pkg${NC}"
            done
            
            read -p "Would you like to install the missing packages? (y/n): " install_missing
            if [[ $install_missing =~ ^[Yy]$ ]]; then
                print_status "Installing missing packages..."
                pip install -r requirements.txt
            fi
        else
            print_success "All required Python packages are installed."
        fi
        
        deactivate
    )
    
    echo
    read -p "Press Enter to return to the troubleshooting menu..."
    troubleshoot_issues
}

# Main loop
while true; do
    show_welcome_banner
    show_menu
done
