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

# Function to check and fix common issues that might prevent Polaris from running
check_and_fix_common_issues() {
    print_status "Performing pre-flight checks for Polaris..."
    
    # Get the script directory
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    print_status "Script is running from: $SCRIPT_DIR"
    
    # Check for polaris alias conflicts that might interfere with the command
    if grep -q "alias polaris=" ~/.bashrc 2>/dev/null; then
        print_warning "Found potentially conflicting 'polaris' alias in ~/.bashrc"
        echo -e "${YELLOW}This may redirect polaris commands to another program.${NC}"
        
        # Check what the alias is set to
        local alias_value=$(grep "alias polaris=" ~/.bashrc | sed 's/alias polaris=\(.*\)/\1/')
        print_warning "Current alias: polaris=$alias_value"
        
        read -p "Would you like to temporarily disable this alias for this session? (y/n): " disable_alias
        if [[ $disable_alias =~ ^[Yy]$ ]]; then
            print_status "Temporarily disabling polaris alias..."
            unalias polaris 2>/dev/null || true
            print_success "Alias disabled for this session only."
        fi
    fi
    
    # Check if polariscloud directory exists in expected location
    if [ -d "$SCRIPT_DIR/polariscloud" ]; then
        print_success "Found polariscloud directory at: $SCRIPT_DIR/polariscloud"
        
        # Check for virtual environment
        if [ -d "$SCRIPT_DIR/polariscloud/venv" ]; then
            print_success "Found virtual environment at: $SCRIPT_DIR/polariscloud/venv"
            
            if [ -f "$SCRIPT_DIR/polariscloud/venv/bin/activate" ]; then
                print_success "Found activation script: $SCRIPT_DIR/polariscloud/venv/bin/activate"
            else
                print_warning "Missing activation script in virtual environment!"
            fi
        else
            print_warning "No virtual environment found at: $SCRIPT_DIR/polariscloud/venv"
            print_warning "Installation may be incomplete or corrupted."
        fi
        
        # Check for .env file
        if [ -f "$SCRIPT_DIR/polariscloud/.env" ]; then
            print_success "Found .env configuration file"
            # Check if SSH_PASSWORD is set
            if grep -q "USE_SSH_KEY=true" "$SCRIPT_DIR/polariscloud/.env"; then
                print_success "Using SSH key authentication"
            else
                print_warning "USE_SSH_KEY not found in .env file!"
                print_warning "This may cause issues with authentication."
            fi
        else
            print_warning "No .env configuration file found!"
            print_warning "Polaris may not function correctly without proper configuration."
        fi
        
        # Check if polaris_run script exists and is executable
        if [ -f "$SCRIPT_DIR/polariscloud/polaris_run" ]; then
            print_success "Found polaris_run script at: $SCRIPT_DIR/polariscloud/polaris_run"
        else
            print_warning "polaris_run script not found!"
        fi
    else
        print_warning "polariscloud directory not found at: $SCRIPT_DIR/polariscloud"
        print_warning "Polaris may not function correctly without proper configuration."
    fi
}

# Function to detect public IP and ask for confirmation
get_public_ip() {
    print_status "Detecting your public IP address..."
    
    # Try multiple services in case one fails
    if curl -s https://api.ipify.org &>/dev/null; then
        detected_ip=$(curl -s https://api.ipify.org)
    elif curl -s https://ifconfig.me &>/dev/null; then
        detected_ip=$(curl -s https://ifconfig.me)
    elif curl -s https://icanhazip.com &>/dev/null; then
        detected_ip=$(curl -s https://icanhazip.com)
    else
        print_error "Could not automatically detect your public IP address."
        detected_ip=""
    fi
    
    # If we found an IP, ask for confirmation
    if [ ! -z "$detected_ip" ]; then
        # Make the detected IP highly visible
        echo
        echo -e "${YELLOW}================================${NC}"
        echo -e "${YELLOW}Detected public IP address: ${BOLD}$detected_ip${NC}"
        echo -e "${YELLOW}================================${NC}"
        echo
        read -p "Is this correct? (y/n): " confirm_ip
        
        if [[ $confirm_ip =~ ^[Yy]$ ]]; then
            public_ip=$detected_ip
            print_success "Using detected IP: $public_ip"
        else
            # If user says it's incorrect, ask for manual entry
            echo
            print_status "Please enter your correct IP address:"
            while true; do
                read -p "Enter your correct public IP address: " public_ip
                if [[ $public_ip =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
                    print_success "Using manual IP: $public_ip"
                    break
                else
                    print_error "Invalid IP address format. Please try again."
                fi
            done
        fi
    else
        # If detection failed, fall back to manual entry
        echo
        print_warning "Automatic IP detection failed. Please enter your IP manually:"
        while true; do
            read -p "Enter your public IP address: " public_ip
            if [[ $public_ip =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
                print_success "Using manual IP: $public_ip"
                break
            else
                print_error "Invalid IP address format. Please try again."
            fi
        done
    fi
    
    # Store result in a global variable instead of returning it
    # This allows the caller to get the value without capturing the output
    DETECTED_PUBLIC_IP="$public_ip"
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

    # Allow both Linux and macOS (Darwin)
    if [ "$os_name" != "Linux" ] && [ "$os_name" != "Darwin" ] && [ "$is_wsl" = false ]; then
        clear
        echo -e "${BG_RED}${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${BG_RED}${BOLD}║                 SYSTEM COMPATIBILITY ERROR                   ║${NC}"
        echo -e "${BG_RED}${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
        echo
        echo -e "${RED}${BOLD}⚠️  This script requires a Unix-based environment! ⚠️${NC}"
        echo -e "${YELLOW}Current system detected: ${BOLD}$os_name${NC}"
        echo
        echo -e "${BLUE}${BOLD}You have three options:${NC}"
        echo -e "${GREEN}${BOLD}1.${NC} ${BLUE}Run this script in a Linux environment (native Linux installation)${NC}"
        echo -e "${GREEN}${BOLD}2.${NC} ${BLUE}Run this script in macOS (native macOS installation)${NC}"
        echo -e "${GREEN}${BOLD}3.${NC} ${BLUE}Use Windows Subsystem for Linux (WSL) if you're on Windows${NC}"
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

    # Check if macOS
    if [ "$os_name" = "Darwin" ]; then
        print_success "Running in macOS environment - ${GREEN}Compatible ✓${NC}"
        echo -e "${YELLOW}Note: macOS uses launchd instead of systemd. Some features will use macOS alternatives.${NC}"
    # Check Linux distribution if needed
    elif [ "$is_wsl" = true ]; then
        print_success "Running in WSL environment - ${GREEN}Compatible ✓${NC}"
        echo -e "${YELLOW}Note: Docker in WSL may require special configuration.${NC}"
        echo -e "${YELLOW}Select option 5 from the menu for more information.${NC}"
    else
        # Check for systemd support
        if command_exists pidof && pidof systemd >/dev/null; then
            print_success "Running in Linux environment with systemd - ${GREEN}Compatible ✓${NC}"
        else
            print_warning "Running in Linux environment without systemd"
            echo -e "${YELLOW}Some features may require manual configuration${NC}"
        fi
    fi
}

# Function to check if we're running on macOS
is_macos() {
    [ "$(uname -s)" = "Darwin" ]
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
    
    # Check for python3-venv on Debian/Ubuntu systems
    if [ -f /etc/debian_version ]; then
        if ! dpkg -l | grep -q python3-venv; then
            print_warning "python3-venv is not installed (required for virtual environments)"
            missing_prereqs=true
            missing_packages+=" python3-venv"
        fi
    fi
    
    if [ "$missing_prereqs" = true ]; then
        echo
        print_warning "Missing prerequisites: ${missing_packages}"
        echo -e "${YELLOW}These packages are required for Polaris installation.${NC}"
        read -p "Would you like to install them now? (y/n): " install_prereqs
        if [[ $install_prereqs =~ ^[Yy]$ ]]; then
            print_status "Installing prerequisites..."
            if is_macos; then
                # Check if Homebrew is installed
                if ! command_exists brew; then
                    print_warning "Homebrew is not installed but needed to install packages on macOS"
                    print_status "Installing Homebrew..."
                    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
                    
                    # Add Homebrew to PATH based on Mac architecture
                    if [ -d "/opt/homebrew" ]; then
                        # Apple Silicon (M1/M2)
                        print_status "Setting up Homebrew environment..."
                        eval "$(/opt/homebrew/bin/brew shellenv)"
                        # Add to profile for future sessions
                        if ! grep -q "brew shellenv" ~/.zprofile; then
                            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
                        fi
                    elif [ -d "/usr/local/Homebrew" ]; then
                        # Intel Mac
                        print_status "Setting up Homebrew environment..."
                        eval "$(/usr/local/bin/brew shellenv)"
                        # Add to profile for future sessions
                        if ! grep -q "brew shellenv" ~/.zprofile; then
                            echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zprofile
                        fi
                    fi
                    
                    if ! command_exists brew; then
                        print_error "Failed to set up Homebrew. Please install prerequisites manually."
                        print_error "For wget: brew install wget"
                        print_error "For git: brew install git"
                        exit 1
                    fi
                fi
                
                # Install each missing package individually
                for pkg in ${missing_packages}; do
                    # Clean up leading/trailing spaces
                    pkg=$(echo "$pkg" | xargs)
                    if [ -n "$pkg" ]; then
                        print_status "Installing $pkg with Homebrew..."
                        brew install "$pkg"
                    fi
                done
            else
                sudo apt-get update
                sudo apt-get install -y $missing_packages
            fi
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
    local current_dir=$(pwd)
    
    # Add verbose debugging
    echo "=== POLARIS INSTALLATION CHECK DIAGNOSTICS ==="
    echo "Current directory: $current_dir"
    echo "Checking if polariscloud directory exists..."
    
    # First check if the repository exists
    if [ ! -d "polariscloud" ]; then
        echo "ERROR: polariscloud directory not found"
        return 1 # Not installed
    else
        echo "SUCCESS: polariscloud directory exists at: $(pwd)/polariscloud"
        ls -la polariscloud
        
        # Check if virtual environment exists inside polariscloud
        echo "Checking for virtual environment inside polariscloud..."
        if [ -d "polariscloud/venv" ]; then
            echo "SUCCESS: Virtual environment found at polariscloud/venv"
            
            # Check if the directory has the activation script
            if [ -f "polariscloud/venv/bin/activate" ]; then
                echo "SUCCESS: Virtual environment activation script found"
                
                # Check for polaris or pcli commands
                (
                    cd polariscloud
                    source venv/bin/activate
                    if command -v polaris &>/dev/null; then
                        echo "SUCCESS: polaris command found in PATH"
                    elif command -v pcli &>/dev/null; then
                        echo "SUCCESS: pcli command found in PATH"
                    else
                        echo "WARNING: Neither polaris nor pcli command found in PATH"
                    fi
                    deactivate
                )
            else
                echo "WARNING: Virtual environment exists but activation script not found"
            fi
        else
            echo "WARNING: Virtual environment not found inside polariscloud"
        fi
        
        echo "All checks passed (simplified), Polaris is considered installed"
        echo "=== END POLARIS INSTALLATION CHECK ==="
        return 0
    fi
}

# Function to backup Polaris configuration
backup_polaris_config() {
    if [ -d "polariscloud" ] && [ -f "polariscloud/.env" ]; then
        print_status "Creating backup of Polaris configuration..."
        
        local backup_dir="polaris_backup_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$backup_dir"
        
        # Backup .env file
        cp polariscloud/.env "$backup_dir/env_backup"
        
        # Backup other important configuration files if they exist
        if [ -d "polariscloud/config" ]; then
            cp -r polariscloud/config "$backup_dir/"
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
    if [ ! -d "polariscloud" ]; then
        print_error "Polaris repository not found!"
        return 1
    fi

    # Get the absolute path to polariscloud
    local polaris_dir=$(realpath "$(pwd)/polariscloud")
    print_status "Using Polaris directory: $polaris_dir"
    
    # Show simplified environment message
    clear
    echo -e "${CYAN}${BOLD}Welcome to Polaris Environment${NC}"
    echo -e "${GREEN}─────────────────────────────────────────────────────${NC}"
    echo -e "${YELLOW}Polaris directory: ${CYAN}$polaris_dir${NC}"
    echo
    echo -e "${YELLOW}Available Polaris commands:${NC}"
    echo -e "• ${CYAN}polaris start${NC}     - Start Polaris services"
    echo -e "• ${CYAN}polaris stop${NC}      - Stop Polaris services"
    echo -e "• ${CYAN}polaris status${NC}    - Check service status"
    echo -e "• ${CYAN}polaris logs${NC}      - View service logs"
    echo -e "• ${CYAN}polaris --help${NC}    - Show all available commands"
    echo
    
    # Change to polariscloud directory and activate virtual environment
    cd "$polaris_dir"
    
    # Check if virtual environment exists
    if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
        print_status "Activating virtual environment..."
        source "$polaris_dir/venv/bin/activate"
        print_success "Virtual environment activated."
        
        # Check if polaris command exists
        if command -v polaris &>/dev/null; then
            polaris_path=$(command -v polaris)
            print_success "Polaris command found at: $polaris_path"
        elif command -v pcli &>/dev/null; then
            pcli_path=$(command -v pcli)
            print_success "PCLI command found at: $pcli_path"
        elif [ -f "$polaris_dir/polaris_run" ]; then
            print_success "Using local Polaris runner script."
            print_status "Creating alias for polaris command..."
            alias polaris="$polaris_dir/polaris_run"
        else
            print_warning "Neither polaris nor pcli command found in PATH!"
            print_warning "Commands may not work correctly. Consider reinstalling Polaris."
        fi
    else
        print_warning "Virtual environment not found at $polaris_dir/venv!"
        print_warning "Commands may not work correctly. Consider reinstalling Polaris."
    fi
    
    echo -e "${GREEN}─────────────────────────────────────────────────────${NC}"
    echo -e "${YELLOW}You are now in the Polaris directory.${NC}"
    echo -e "${YELLOW}Type 'exit' to leave this environment.${NC}"
    echo

    # Start an interactive shell
    $SHELL

    # Deactivate virtual environment if activated
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate
    fi

    # Go back to the original directory if we changed to polariscloud
    cd - &>/dev/null
    echo -e "${GREEN}Exited Polaris environment.${NC}"
}

# Function to show menu and get user choice
show_menu() {
    # Check if Polaris was just installed
    if [ "${POLARIS_INSTALLED:-false}" = "true" ]; then
        local is_installed=true
        unset POLARIS_INSTALLED
    else
        local is_installed=$(check_polaris_installation && echo true || echo false)
    fi
    
    echo -e "${YELLOW}Available Options:${NC}"
    echo -e "${GREEN}─────────────────────────────────────────────────────${NC}"
    echo -e "1) ${CYAN}Install Polaris${NC}"
    echo -e "2) ${YELLOW}Reinstall Polaris${NC}"
    echo -e "3) ${GREEN}Enter Polaris Environment${NC}"
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
                # Re-check installation status after installation completes
                is_installed=$(check_polaris_installation && echo true || echo false)
            fi
            ;;
        2)
            if [ "$is_installed" = false ]; then
                print_warning "Polaris is not installed yet!"
                echo -e "Please choose ${CYAN}Install Polaris${NC} first."
                sleep 2
                return 0
            else
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
                    # Re-check installation status after reinstallation
                    is_installed=$(check_polaris_installation && echo true || echo false)
                fi
            fi
            ;;
        3)
            # Re-check installation status before entering the environment
            is_installed=$(check_polaris_installation && echo true || echo false)
            if [ "$is_installed" = false ]; then
                print_warning "Polaris is not installed yet!"
                echo -e "Please choose ${CYAN}Install Polaris${NC} first."
                sleep 2
                return 0
            else
                enter_polaris_environment
            fi
            ;;
        4)
            # Re-check installation status before uninstalling
            is_installed=$(check_polaris_installation && echo true || echo false)
            if [ "$is_installed" = false ]; then
                print_warning "Polaris is not installed yet!"
                echo -e "Nothing to uninstall."
                sleep 2
                return 0
            else
                # Ask for backup before uninstall
                read -p "Do you want to backup your configuration before uninstalling? (y/n): " backup_confirm
                if [[ $backup_confirm =~ ^[Yy]$ ]]; then
                    backup_polaris_config
                fi
                uninstall_polaris
            fi
            ;;
        5)
            # Check if we know Polaris was just installed
            if [ "${POLARIS_INSTALLED:-false}" = "true" ]; then
                is_installed=true
            else
                # Otherwise re-check installation status
                is_installed=$(check_polaris_installation && echo true || echo false)
            fi
            
            if [ "$is_installed" = true ]; then
                print_success "Polaris is installed (folder exists)."
                echo -e "${YELLOW}To use Polaris, select option 3 to enter Polaris environment.${NC}"
                echo -e "${YELLOW}NOTE: Only checking for polariscloud folder existence.${NC}"
                echo -e "${YELLOW}Other validation checks have been disabled for now.${NC}"
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
            # Re-check installation status before backup
            is_installed=$(check_polaris_installation && echo true || echo false)
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
        if is_macos; then
            # Check if Docker.app is running on macOS
            if ! pgrep -q "Docker"; then
                print_warning "Docker Desktop is not running on macOS"
                print_status "Starting Docker Desktop..."
                open -a Docker
                print_warning "Please wait for Docker Desktop to fully start"
                print_warning "This may take a minute or two..."
                # Give user time to let Docker start
                sleep 10
            fi
        elif pidof systemd >/dev/null && ! systemctl is-active --quiet docker; then
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
        
        if is_macos; then
            print_status "Installing Docker Desktop for Mac..."
            
            # Check if Homebrew is installed
            if ! command_exists brew; then
                print_status "Installing Homebrew first..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
                
                # Add Homebrew to PATH based on Mac architecture
                if [ -d "/opt/homebrew" ]; then
                    # Apple Silicon (M1/M2)
                    print_status "Setting up Homebrew environment..."
                    eval "$(/opt/homebrew/bin/brew shellenv)"
                    # Add to profile for future sessions
                    if ! grep -q "brew shellenv" ~/.zprofile; then
                        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
                    fi
                elif [ -d "/usr/local/Homebrew" ]; then
                    # Intel Mac
                    print_status "Setting up Homebrew environment..."
                    eval "$(/usr/local/bin/brew shellenv)"
                    # Add to profile for future sessions
                    if ! grep -q "brew shellenv" ~/.zprofile; then
                        echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zprofile
                    fi
                fi
                
                if ! command_exists brew; then
                    print_error "Failed to set up Homebrew. Cannot install Docker."
                    exit 1
                fi
            fi
            
            # Install Docker Desktop using Homebrew
            print_status "Installing Docker Desktop via Homebrew..."
            brew install --cask docker
            
            # Launch Docker Desktop
            print_status "Launching Docker Desktop. Please complete the setup if prompted..."
            open -a Docker
            
            print_warning "Docker Desktop is launching. You might need to:"
            echo -e "${YELLOW}  1. Complete the initial Docker setup if this is the first installation${NC}"
            echo -e "${YELLOW}  2. Accept the license agreement${NC}"
            echo -e "${YELLOW}  3. Provide your system password to allow Docker to install its components${NC}"
            echo
            print_warning "Please wait until Docker is fully started before continuing (may take a minute or two)."
            read -p "Press Enter once Docker is running..." 
            
            # Verify Docker is working
            if ! docker info &>/dev/null; then
                print_warning "Docker doesn't seem to be running correctly. Make sure Docker Desktop is running."
                print_warning "You might need to manually start Docker Desktop from your Applications folder."
                sleep 3
            fi
            
        else
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
        fi

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
    if is_macos; then
        print_status "Checking SSH server on macOS..."
        
        # Check if SSH service is enabled in macOS
        if sudo systemsetup -getremotelogin | grep -q "On"; then
            print_success "SSH server is already enabled on macOS"
        else
            print_status "Enabling SSH server on macOS..."
            print_warning "You may be prompted for your password to enable the SSH service"
            sudo systemsetup -setremotelogin on
            print_success "SSH server enabled on macOS"
        fi
    else
        # Linux SSH server installation
        if ! command_exists sshd; then
            print_status "Installing SSH server..."
            sudo apt-get update
            sudo apt-get install -y openssh-server
            
            if command_exists systemctl; then
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
            if command_exists systemctl && ! systemctl is-active --quiet ssh; then
                print_status "Starting SSH service..."
                sudo systemctl start ssh
            elif [ ! -z "$(ps -e | grep sshd)" ]; then
                print_success "SSH service is running"
            else
                print_warning "SSH service is not running."
                echo -e "${YELLOW}Try: sudo service ssh start${NC}"
            fi
        fi
    fi
}

# Function to install Python requirements
install_python_requirements() {
    print_status "Installing Python requirements..."
    
    if is_macos; then
        # For macOS, use Homebrew
        if ! command_exists brew; then
            print_status "Installing Homebrew first..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            
            # Add Homebrew to PATH based on Mac architecture
            if [ -d "/opt/homebrew" ]; then
                # Apple Silicon (M1/M2)
                print_status "Setting up Homebrew environment..."
                eval "$(/opt/homebrew/bin/brew shellenv)"
                # Add to profile for future sessions
                if ! grep -q "brew shellenv" ~/.zprofile; then
                    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
                fi
            elif [ -d "/usr/local/Homebrew" ]; then
                # Intel Mac
                print_status "Setting up Homebrew environment..."
                eval "$(/usr/local/bin/brew shellenv)"
                # Add to profile for future sessions
                if ! grep -q "brew shellenv" ~/.zprofile; then
                    echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zprofile
                fi
            fi
        fi

        # Check if Python 3.10 is already installed
        if brew list python@3.10 &>/dev/null; then
            print_success "Python 3.10 is already installed via Homebrew"
        else
            # Install Python 3.10
            print_status "Installing Python 3.10 via Homebrew..."
            brew install python@3.10
            
            # Make Python 3.10 the default
            print_status "Setting Python 3.10 as the default Python version..."
            brew link --force python@3.10
        fi

        # Verify Python installation and version
        if command_exists python3; then
            print_success "Python $(python3 --version) installed"
        else
            print_error "Python installation failed. Please install manually:"
            print_error "brew install python@3.10"
            exit 1
        fi
        
        # Install Rust
        if ! command_exists rustc; then
            print_status "Installing Rust..."
            brew install rust
        else
            print_success "Rust is already installed"
        fi
        
        # Install XCode Command Line Tools if needed
        if ! command_exists xcode-select || ! xcode-select -p &>/dev/null; then
            print_status "Installing XCode Command Line Tools..."
            xcode-select --install
            print_warning "If prompted, please complete the XCode Command Line Tools installation"
            print_warning "Press Enter once the installation is complete..."
            read -p ""
        fi
        
        # Install additional build dependencies
        print_status "Installing additional build dependencies..."
        brew install gcc cmake openssl
        
        # Install pip if not already available
        if ! command_exists pip3; then
            print_status "Installing pip..."
            curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
            python3 get-pip.py
            rm get-pip.py
        fi
    else
        # For Linux, use apt
        sudo apt-get update
        sudo apt-get install -y python3-venv python3-pip g++ rustc cargo build-essential python3-dev
    fi
    
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
    if [ -d "polariscloud" ]; then
        cd polariscloud
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
    if [ -d "polariscloud" ]; then
        cd polariscloud
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
        rm -rf polariscloud
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

    # Get absolute path to current directory to ensure consistency
    local current_dir=$(pwd)
    local polaris_dir="$current_dir/polariscloud"
    print_status "Installing Polaris in: $polaris_dir"
    
    # Check for and warn about any conflicting aliases
    print_status "Checking for conflicting aliases..."
    if grep -q "alias polaris=" ~/.bashrc 2>/dev/null; then
        print_warning "Found a 'polaris' alias in your ~/.bashrc file!"
        echo -e "${YELLOW}This may cause conflicts with the Polaris command.${NC}"
        read -p "Would you like to remove this alias? (y/n): " remove_alias
        if [[ $remove_alias =~ ^[Yy]$ ]]; then
            print_status "Removing 'polaris' alias from ~/.bashrc..."
            # Back up .bashrc first
            cp ~/.bashrc ~/.bashrc.bak.$(date +%Y%m%d%H%M%S)
            # Remove the alias line
            sed -i '/alias polaris=/d' ~/.bashrc
            print_success "Alias removed. You'll need to open a new terminal or run 'source ~/.bashrc' for changes to take effect."
        else
            print_warning "Alias not removed. This may cause the 'polaris' command to behave unexpectedly."
        fi
    fi

    # Check if polariscloud already exists
    if [ -d "polariscloud" ]; then
        print_status "Found existing polariscloud directory."
        
        # Check if virtual environment exists inside polariscloud
        if [ -d "polariscloud/venv" ]; then
            print_status "Found existing virtual environment inside polariscloud."
            # Activate the virtual environment and install polaris
            cd polariscloud
            source "$polaris_dir/venv/bin/activate"
            
            print_status "Installing Polaris in the existing environment..."
            
            # Install packages in the correct order to avoid dependency conflicts
            print_status "Installing Python dependencies in the correct order..."
            
            # First upgrade pip
            print_status "Upgrading pip..."
            # Check if we're dealing with externally managed Python
            if pip3 install --upgrade pip 2>&1 | grep -q "externally-managed-environment"; then
                print_warning "Detected externally-managed Python environment"
                print_warning "Using --break-system-packages flag to override (required for Python 3.12+ on some distros)"
                pip3 install --break-system-packages --upgrade pip
                
                # Define a helper function for pip installs
                pip_install() {
                    pip install --break-system-packages "$@"
                }
            else
                # Define a helper function for standard pip installs
                pip_install() {
                    pip install "$@"
                }
            fi
            
            # Install network-specific packages in the correct order
            print_status "Installing Bittensor and related packages..."
            pip_install bittensor
            check_command "Failed to install bittensor" 0
            
            pip_install bittensor-cli
            check_command "Failed to install bittensor-cli" 0
            
            pip_install communex==0.1.36.4
            check_command "Failed to install communex" 0

            # Now install the rest of the requirements
            print_status "Installing remaining dependencies..."
            if [ -f "requirements.txt" ]; then
                pip_install -r requirements.txt
                check_command "Failed to install requirements" 0
            else
                print_warning "No requirements.txt found. Installing common dependencies..."
                pip_install click tabulate GitPython click-spinner rich loguru inquirer requests xlsxwriter pyyaml psutil python-dotenv pid
            fi

            # Install Polaris in development mode
            print_status "Installing Polaris in development mode..."
            pip_install -e .
            check_command "Failed to install Polaris" 0
            
            # Verify polaris command is available
            if command -v polaris &>/dev/null || command -v pcli &>/dev/null; then
                polaris_path=$(which polaris 2>/dev/null || which pcli 2>/dev/null)
                print_success "Polaris command successfully installed at: $polaris_path"
            else
                print_warning "Could not find polaris or pcli command in PATH."
                print_status "Creating runner script for Polaris..."
            fi
        else
            # ADDED THIS ELSE BRANCH - Handle case where polariscloud exists but no venv
            print_status "Directory exists but no virtual environment found."
            print_status "Creating new virtual environment..."
            
            # Change to polariscloud directory
            cd polariscloud
            
            # Ensure python3-venv is installed on Ubuntu/Debian systems
            if [ -f /etc/debian_version ]; then
                # Check if python3-venv is installed
                if ! dpkg -l | grep -q python3-venv; then
                    print_status "Installing python3-venv package required for virtual environments..."
                    sudo apt-get update
                    # Determine Python version and install the appropriate venv package
                    python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
                    if apt-cache search python${python_version}-venv | grep -q python${python_version}-venv; then
                        sudo apt-get install -y python${python_version}-venv
                    else
                        # Fallback to generic python3-venv if specific version not available
                        sudo apt-get install -y python3-venv
                    fi
                    print_success "Python virtual environment package installed"
                else
                    print_success "Python virtual environment package already installed"
                fi
            fi
            
            # Create virtual environment
            print_status "Creating Python virtual environment..."
            python3 -m venv venv
            check_command "Failed to create virtual environment" 1
            
            # Activate the virtual environment
            print_status "Activating virtual environment..."
            source "$polaris_dir/venv/bin/activate"
            check_command "Failed to activate virtual environment" 1
            
            # Upgrade pip
            print_status "Upgrading pip..."
            # Check if we're dealing with externally managed Python
            if pip3 install --upgrade pip 2>&1 | grep -q "externally-managed-environment"; then
                print_warning "Detected externally-managed Python environment"
                print_warning "Using --break-system-packages flag to override (required for Python 3.12+ on some distros)"
                pip3 install --break-system-packages --upgrade pip
                
                # Define a helper function for pip installs
                pip_install() {
                    pip install --break-system-packages "$@"
                }
            else
                # Define a helper function for standard pip installs
                pip_install() {
                    pip install "$@"
                }
            fi
            
            # Check if requirements.txt exists
            if [ ! -f "requirements.txt" ]; then
                print_warning "requirements.txt not found!"
                print_status "Creating a basic requirements.txt file..."
                cat > requirements.txt << EOF
click
tabulate
GitPython
click-spinner
rich
loguru
inquirer
requests
xlsxwriter
pyyaml
psutil
python-dotenv
pid
EOF
            fi
            
            # Install bittensor and related packages first
            print_status "Installing Bittensor and related packages..."
            pip_install bittensor
            check_command "Failed to install bittensor" 0
            
            pip_install bittensor-cli
            check_command "Failed to install bittensor-cli" 0
            
            pip_install communex==0.1.36.4
            check_command "Failed to install communex" 0
            
            # Install requirements
            print_status "Installing Python requirements..."
            pip_install -r requirements.txt
            check_command "Failed to install requirements" 0
            
            # Install Polaris in development mode
            print_status "Installing Polaris in development mode..."
            pip_install -e .
            check_command "Failed to install Polaris" 0
            
            # Verify polaris command is available
            if command -v polaris &>/dev/null || command -v pcli &>/dev/null; then
                polaris_path=$(which polaris 2>/dev/null || which pcli 2>/dev/null)
                print_success "Polaris command successfully installed at: $polaris_path"
            else
                print_warning "Could not find polaris or pcli command in PATH."
                print_status "Creating runner script for Polaris..."
            fi
        fi
        
        # Setup configuration regardless of whether we just installed or already had a venv
        # Stay in the polariscloud directory if we're already there, otherwise go there
        if [ ! "$PWD" = "$polaris_dir" ]; then
            cd "$polaris_dir"
        fi
        
        # Create a more robust direct runner script for convenience
        print_status "Creating robust polaris runner script..."
        cat > polaris_run << EOF
#!/bin/bash

# polaris_run - Direct runner for Polaris commands
# This script activates the virtual environment and runs polaris with any arguments

# Get the directory of this script (works even with symlinks)
SCRIPT_DIR="\$(cd "\$(dirname "\$(readlink -f "\${BASH_SOURCE[0]}")")" && pwd)"
echo "Running Polaris from directory: \$SCRIPT_DIR"

# Ensure we're in the polariscloud directory
cd "\$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ] || [ ! -f "venv/bin/activate" ]; then
    echo "Error: Virtual environment not found at \$SCRIPT_DIR/venv"
    echo "Please make sure Polaris is properly installed."
    exit 1
fi

# Activate the virtual environment
echo "Activating virtual environment at: \$SCRIPT_DIR/venv"
source "\$SCRIPT_DIR/venv/bin/activate"

# Check if activation was successful
if [ -z "\$VIRTUAL_ENV" ]; then
    echo "Error: Failed to activate virtual environment."
    exit 1
fi

echo "Successfully activated the virtual environment"
echo "Python interpreter: \$(command -v python3)"

# Check if polaris command exists in PATH or try alternatives
if command -v polaris &>/dev/null; then
    echo "Found polaris command: \$(command -v polaris)"
    polaris "\$@"
elif command -v pcli &>/dev/null; then
    echo "Found pcli command: \$(command -v pcli)"
    pcli "\$@"
elif [ -f "\$SCRIPT_DIR/polaris" ] && [ ! -L "\$SCRIPT_DIR/polaris" ]; then
    # If polaris exists and is not a symlink (to avoid infinite recursion)
    echo "Using local polaris script at: \$SCRIPT_DIR/polaris"
    "\$SCRIPT_DIR/polaris" "\$@"
elif [ -f "\$SCRIPT_DIR/polaris/__main__.py" ]; then
    # Try running it as a module if it exists
    echo "Running polaris module directly with Python"
    python3 -m polaris "\$@"
elif [ -f "\$SCRIPT_DIR/main.py" ]; then
    # Try running main.py directly
    echo "Running main.py directly with Python"
    python3 "\$SCRIPT_DIR/main.py" "\$@"
else
    echo "Error: Could not find polaris or pcli command."
    echo "Available commands in PATH: \$(which python3)"
    echo "Contents of \$SCRIPT_DIR/venv/bin:"
    ls -la "\$SCRIPT_DIR/venv/bin"
    echo "Please ensure Polaris is properly installed."
    exit 1
fi

RESULT=\$?

# Deactivate the virtual environment when done
echo "Deactivating virtual environment"
deactivate 2>/dev/null || true

exit \$RESULT
EOF

        chmod +x polaris_run
        
        # Create a symlink called polaris if it doesn't exist
        if [ ! -f "polaris" ] || [ -L "polaris" ]; then
            ln -sf polaris_run polaris
            chmod +x polaris
        fi
        
        # Setup bash profile to include polariscloud in PATH
        print_status "Setting up PATH to include the Polaris directory..."
        
        # Determine which shell profile file to use
        local profile_file=""
        if [ -f ~/.bash_profile ]; then
            profile_file=~/.bash_profile
        elif [ -f ~/.profile ]; then
            profile_file=~/.profile
        else
            profile_file=~/.bashrc
        fi
        
        # Add polariscloud to PATH if not already there
        if ! grep -q "export PATH=.*$polaris_dir" "$profile_file" 2>/dev/null; then
            echo "# Added by Polaris installer" >> "$profile_file"
            echo "export PATH=\"$polaris_dir:\$PATH\"" >> "$profile_file"
            print_success "Added Polaris directory to your PATH in $profile_file"
            print_warning "You'll need to open a new terminal or run 'source $profile_file' for this to take effect."
        else
            print_success "Polaris directory is already in your PATH."
        fi
        
        # Continue with the rest of the installation...
        
        # Create a shell wrapper in /usr/local/bin if possible
        if [ -d "/usr/local/bin" ] && [ -w "/usr/local/bin" ]; then
            print_status "Creating system-wide polaris command..."
            
            cat > /usr/local/bin/polaris << EOF
#!/bin/bash
# System-wide wrapper for Polaris commands

# Run the polaris command from the installed directory
"$polaris_dir/polaris_run" "\$@"
EOF
            chmod +x /usr/local/bin/polaris
            print_success "Created system-wide 'polaris' command in /usr/local/bin"
        elif [ -d "/usr/local/bin" ]; then
            print_warning "Could not create system-wide command (permission denied)."
            print_warning "To create it manually, run: sudo ln -sf \"$polaris_dir/polaris_run\" /usr/local/bin/polaris"
        fi
        
        # Always create the environment file
        print_status "Setting up environment configuration..."
        
        # Get public IP
        get_public_ip
        public_ip="$DETECTED_PUBLIC_IP"
        
        # Always get SSH configuration
        print_status "Setting up SSH configuration..."
        setup_ssh_configuration
        
        # Create .env file
        print_status "Creating .env file with configuration..."
        create_env_file
        
        # Validate .env file
        if [ -f ".env" ]; then
            print_success "Environment file created successfully."
            
            # Display configuration
            echo -e "${YELLOW}Environment Configuration:${NC}"
            echo -e "${CYAN}HOST:${NC} $public_ip"
            echo -e "${CYAN}SSH User:${NC} $ssh_user"
            echo -e "${CYAN}SSH Ports:${NC} $port_start-$port_end"
            echo -e "${CYAN}Primary SSH Port:${NC} $port_start"
            
            # Check if critical variables are set
            source .env
            if [ -z "$USE_SSH_KEY" ]; then
                print_warning "USE_SSH_KEY is not set in .env file."
                print_status "Setting up SSH key authentication..."
                echo "USE_SSH_KEY=true" >> .env
                print_success "SSH key authentication configured."
            fi
        else
            print_error "Failed to create .env file."
            print_status "Creating a minimal .env file with defaults..."
            
            # Create a minimal .env file with placeholders
            cat > .env << EOF
# Polaris Environment Configuration - Default Minimal Setup
# Values in [brackets] need your attention

# Server Configuration
HOST=[server_ip]
API_PORT=8000

# SSH Configuration
SSH_PORT_RANGE_START=11000
SSH_PORT_RANGE_END=11010
SSH_USER=[ssh_user]
SSH_HOST=[ssh_host]
SSH_PORT=11000
USE_SSH_KEY=true

# API Configuration
SERVER_URL=https://polaris-test-server.onrender.com/api/v1
EOF
            # Fill in what we know
            sed -i "s/\[server_ip\]/$public_ip/g" .env
            sed -i "s/\[ssh_host\]/$public_ip/g" .env
            sed -i "s/\[ssh_user\]/$(whoami)/g" .env
            
            print_warning "Created minimal .env file with default values."
            print_warning "Default SSH port range: 11000-11010"
            print_warning "Using public key authentication by default"
        fi
        
        # Return to parent directory if we started there
        if [ "$PWD" != "$current_dir" ]; then
            cd "$current_dir"
        fi
        
        print_success "Polaris installation and configuration completed successfully!"
        
        # Set the flag to indicate installation is complete
        POLARIS_INSTALLED=true
        export POLARIS_INSTALLED
        
        # Ask user if they want to start Polaris immediately
        ask_to_start_polaris
        
        return 0
    fi
    
    # Setup for fresh installation when polariscloud doesn't exist
    if [ ! -d "polariscloud" ]; then
        print_status "Creating polariscloud directory..."
        mkdir -p polariscloud
    fi

    # Change directory to polariscloud
                cd polariscloud
    print_status "Setting up Polaris in $(pwd)..."
    
    # Ensure python3-venv is installed on Ubuntu/Debian systems
    if [ -f /etc/debian_version ]; then
        # Check if python3-venv is installed
        if ! dpkg -l | grep -q python3-venv; then
            print_status "Installing python3-venv package required for virtual environments..."
            sudo apt-get update
            # Determine Python version and install the appropriate venv package
            python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
            if apt-cache search python${python_version}-venv | grep -q python${python_version}-venv; then
                sudo apt-get install -y python${python_version}-venv
            else
                # Fallback to generic python3-venv if specific version not available
                sudo apt-get install -y python3-venv
            fi
            print_success "Python virtual environment package installed"
        else
            print_success "Python virtual environment package already installed"
        fi
    fi
    
    # Create virtual environment
    print_status "Creating virtual environment..."
    python3 -m venv venv
    check_command "Failed to create virtual environment" 1
    
    # Activate the virtual environment
    source "$polaris_dir/venv/bin/activate"
    check_command "Failed to activate virtual environment" 1
    
    # Upgrade pip
    print_status "Upgrading pip..."
    # Check if we're dealing with externally managed Python
    if pip3 install --upgrade pip 2>&1 | grep -q "externally-managed-environment"; then
        print_warning "Detected externally-managed Python environment"
        print_warning "Using --break-system-packages flag to override (required for Python 3.12+ on some distros)"
        pip3 install --break-system-packages --upgrade pip
        
        # Define a helper function for pip installs
        pip_install() {
            pip install --break-system-packages "$@"
        }
    else
        # Define a helper function for standard pip installs
        pip_install() {
            pip install "$@"
        }
    fi
    check_command "Failed to upgrade pip" 0
    
    # Check if requirements.txt exists
    if [ ! -f "requirements.txt" ]; then
        print_warning "requirements.txt not found!"
        print_status "Creating a basic requirements.txt file..."
        cat > requirements.txt << EOF
click
tabulate
GitPython
click-spinner
rich
loguru
inquirer
requests
xlsxwriter
pyyaml
psutil
python-dotenv
pid
EOF
    fi
    
    # Install bittensor and related packages first
    print_status "Installing Bittensor and related packages..."
    pip_install bittensor
    check_command "Failed to install bittensor" 0
    
    pip_install bittensor-cli
    check_command "Failed to install bittensor-cli" 0
    
    pip_install communex==0.1.36.4
    check_command "Failed to install communex" 0
    
    # Install requirements
    print_status "Installing Python requirements..."
    pip_install -r requirements.txt
    check_command "Failed to install requirements" 0
    
    # Install Polaris in development mode
    print_status "Installing Polaris in development mode..."
    pip_install -e .
    check_command "Failed to install Polaris" 0
    
    # Create a more robust direct runner script for convenience
    print_status "Creating robust polaris runner script..."
    cat > polaris_run << EOF
#!/bin/bash

# polaris_run - Direct runner for Polaris commands
# This script activates the virtual environment and runs polaris with any arguments

# Get the directory of this script (works even with symlinks)
SCRIPT_DIR="\$(cd "\$(dirname "\$(readlink -f "\${BASH_SOURCE[0]}")")" && pwd)"
echo "Running Polaris from directory: \$SCRIPT_DIR"

# Ensure we're in the polariscloud directory
cd "\$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ] || [ ! -f "venv/bin/activate" ]; then
    echo "Error: Virtual environment not found at \$SCRIPT_DIR/venv"
    echo "Please make sure Polaris is properly installed."
    exit 1
fi

# Activate the virtual environment
echo "Activating virtual environment at: \$SCRIPT_DIR/venv"
source "\$SCRIPT_DIR/venv/bin/activate"

# Check if activation was successful
if [ -z "\$VIRTUAL_ENV" ]; then
    echo "Error: Failed to activate virtual environment."
    exit 1
fi

echo "Successfully activated the virtual environment"
echo "Python interpreter: \$(command -v python3)"

# Check if polaris command exists in PATH or try alternatives
if command -v polaris &>/dev/null; then
    echo "Found polaris command: \$(command -v polaris)"
    polaris "\$@"
elif command -v pcli &>/dev/null; then
    echo "Found pcli command: \$(command -v pcli)"
    pcli "\$@"
elif [ -f "\$SCRIPT_DIR/polaris" ] && [ ! -L "\$SCRIPT_DIR/polaris" ]; then
    # If polaris exists and is not a symlink (to avoid infinite recursion)
    echo "Using local polaris script at: \$SCRIPT_DIR/polaris"
    "\$SCRIPT_DIR/polaris" "\$@"
elif [ -f "\$SCRIPT_DIR/polaris/__main__.py" ]; then
    # Try running it as a module if it exists
    echo "Running polaris module directly with Python"
    python3 -m polaris "\$@"
elif [ -f "\$SCRIPT_DIR/main.py" ]; then
    # Try running main.py directly
    echo "Running main.py directly with Python"
    python3 "\$SCRIPT_DIR/main.py" "\$@"
else
    echo "Error: Could not find polaris or pcli command."
    echo "Available commands in PATH: \$(which python3)"
    echo "Contents of \$SCRIPT_DIR/venv/bin:"
    ls -la "\$SCRIPT_DIR/venv/bin"
    echo "Please ensure Polaris is properly installed."
    exit 1
fi

RESULT=\$?

# Deactivate the virtual environment when done
echo "Deactivating virtual environment"
deactivate 2>/dev/null || true

exit \$RESULT
EOF
    chmod +x polaris_run
    
    # Create a symlink called polaris if it doesn't exist
    if [ ! -f "polaris" ] || [ -L "polaris" ]; then
        ln -sf polaris_run polaris
        chmod +x polaris
    fi
    
    # Setup bash profile to include polariscloud in PATH
    print_status "Setting up PATH to include the Polaris directory..."
    
    # Determine which shell profile file to use
    local profile_file=""
    if [ -f ~/.bash_profile ]; then
        profile_file=~/.bash_profile
    elif [ -f ~/.profile ]; then
        profile_file=~/.profile
    else
        profile_file=~/.bashrc
    fi
    
    # Add polariscloud to PATH if not already there
    if ! grep -q "export PATH=.*$polaris_dir" "$profile_file" 2>/dev/null; then
        echo "# Added by Polaris installer" >> "$profile_file"
        echo "export PATH=\"$polaris_dir:\$PATH\"" >> "$profile_file"
        print_success "Added Polaris directory to your PATH in $profile_file"
        print_warning "You'll need to open a new terminal or run 'source $profile_file' for this to take effect."
    else
        print_success "Polaris directory is already in your PATH."
    fi
    
    # Continue with the rest of the installation...
    
    # Create a shell wrapper in /usr/local/bin if possible
    if [ -d "/usr/local/bin" ] && [ -w "/usr/local/bin" ]; then
        print_status "Creating system-wide polaris command..."
        
        cat > /usr/local/bin/polaris << EOF
#!/bin/bash
# System-wide wrapper for Polaris commands

# Run the polaris command from the installed directory
"$polaris_dir/polaris_run" "\$@"
EOF
        chmod +x /usr/local/bin/polaris
        print_success "Created system-wide 'polaris' command in /usr/local/bin"
    elif [ -d "/usr/local/bin" ]; then
        print_warning "Could not create system-wide command (permission denied)."
        print_warning "To create it manually, run: sudo ln -sf \"$polaris_dir/polaris_run\" /usr/local/bin/polaris"
    fi
    
    # Always create the environment file
    print_status "Setting up environment configuration..."
    
    # Get public IP
    get_public_ip
    public_ip="$DETECTED_PUBLIC_IP"
    
    # Always get SSH configuration
    print_status "Setting up SSH configuration..."
    setup_ssh_configuration
    
    # Create .env file
    print_status "Creating .env file with configuration..."
    create_env_file
    
    # Validate .env file
    if [ -f ".env" ]; then
        print_success "Environment file created successfully."
        
        # Display configuration
        echo -e "${YELLOW}Environment Configuration:${NC}"
        echo -e "${CYAN}HOST:${NC} $public_ip"
        echo -e "${CYAN}SSH User:${NC} $ssh_user"
        echo -e "${CYAN}SSH Ports:${NC} $port_start-$port_end"
        echo -e "${CYAN}Primary SSH Port:${NC} $port_start"
        
        # Check if critical variables are set
        source .env
        if [ -z "$USE_SSH_KEY" ]; then
            print_warning "USE_SSH_KEY is not set in .env file."
            print_status "Setting up SSH key authentication..."
            echo "USE_SSH_KEY=true" >> .env
            print_success "SSH key authentication configured."
        fi
    else
        print_error "Failed to create .env file."
        print_status "Creating a minimal .env file with defaults..."
        
        # Create a minimal .env file with placeholders
        cat > .env << EOF
# Polaris Environment Configuration - Default Minimal Setup
# Values in [brackets] need your attention

# Server Configuration
HOST=[server_ip]
API_PORT=8000

# SSH Configuration
SSH_PORT_RANGE_START=11000
SSH_PORT_RANGE_END=11010
SSH_USER=[ssh_user]
SSH_HOST=[ssh_host]
SSH_PORT=11000
USE_SSH_KEY=true

# API Configuration
SERVER_URL=https://polaris-test-server.onrender.com/api/v1
EOF
        # Fill in what we know
        sed -i "s/\[server_ip\]/$public_ip/g" .env
        sed -i "s/\[ssh_host\]/$public_ip/g" .env
        sed -i "s/\[ssh_user\]/$(whoami)/g" .env
        
        print_warning "Created minimal .env file with default values."
        print_warning "Default SSH port range: 11000-11010"
        print_warning "Using public key authentication by default"
    fi
    
    # Return to parent directory if we started there
    if [ "$PWD" != "$current_dir" ]; then
        cd "$current_dir"
    fi
    
    print_success "Polaris installation and configuration completed successfully!"
    
    # Set the flag to indicate installation is complete
    POLARIS_INSTALLED=true
    export POLARIS_INSTALLED
    
    # Ask user if they want to start Polaris immediately
    ask_to_start_polaris
}

# Function to set up SSH configuration
setup_ssh_configuration() {
    # Get SSH username
    read -p "Enter SSH username: " ssh_user

    # Default to SSH key authentication
    print_status "Setting up SSH key authentication (recommended)..."
    setup_ssh_key_auth
    use_ssh_key=true

    # Get port range
    get_port_range
}

# Function to set up SSH key authentication
setup_ssh_key_auth() {
    print_status "Setting up SSH key authentication..."
    
    # Check if ssh-keygen is installed
    if ! command_exists ssh-keygen; then
        print_warning "ssh-keygen command not found. Installing OpenSSH..."
        
        # Check the operating system and install ssh
        if is_macos; then
            print_status "Installing SSH on macOS..."
            # On macOS, SSH should be pre-installed, just need to enable it
            sudo launchctl load -w /System/Library/LaunchDaemons/ssh.plist
            sudo mkdir -p /etc/ssh
        else
            # Linux installation
            print_status "Installing SSH on Linux..."
            sudo apt-get update
            sudo apt-get install -y openssh-server
            
            if command_exists systemctl; then
                sudo systemctl enable ssh
                sudo systemctl start ssh
            else
                print_warning "Could not start SSH service automatically."
                print_warning "You may need to start it manually using appropriate commands for your system."
                echo -e "${YELLOW}Try: sudo service ssh start${NC}"
            fi
        fi
        
        # Check again if ssh-keygen is available
        if ! command_exists ssh-keygen; then
            print_error "Failed to install ssh-keygen. Please install OpenSSH manually."
            return 1
        fi
        
        print_success "OpenSSH installed successfully"
    else
        print_success "OpenSSH is already installed"
    fi
    
    # Make sure .ssh directory exists
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    
    print_success "SSH setup complete!"
    echo
}

# Function to create .env file
create_env_file() {
    print_status "Creating .env file template..."
    
    # First create the template with placeholder brackets
    cat > .env << EOF
# Polaris Environment Configuration
# Values in [brackets] are placeholders that will be replaced

# Server Configuration
HOST=[server_ip]
API_PORT=8000

# SSH Configuration
SSH_PORT_RANGE_START=[port_start]
SSH_PORT_RANGE_END=[port_end]
SSH_USER=[ssh_user]
SSH_HOST=[ssh_host]
SSH_PORT=[ssh_port]
USE_SSH_KEY=true

# API Configuration
SERVER_URL=https://polaris-test-server.onrender.com/api/v1
EOF

    # Now replace the placeholders with actual values
    print_status "Filling in configuration values..."
    
    # Replace server IP
    sed -i "s/\[server_ip\]/$public_ip/g" .env
    
    # Replace SSH config
    sed -i "s/\[port_start\]/$port_start/g" .env
    sed -i "s/\[port_end\]/$port_end/g" .env
    sed -i "s/\[ssh_user\]/$ssh_user/g" .env
    sed -i "s/\[ssh_host\]/$public_ip/g" .env
    sed -i "s/\[ssh_port\]/$port_start/g" .env
    
    # Display the final configuration
    print_success "Environment file created with your configuration"
    echo
    echo -e "${YELLOW}Your Polaris configuration:${NC}"
    echo -e "${CYAN}HOST:${NC} $public_ip"
    echo -e "${CYAN}SSH User:${NC} $ssh_user"
    echo -e "${CYAN}SSH Ports:${NC} $port_start-$port_end"
    echo -e "${CYAN}Primary SSH Port:${NC} $port_start"
    echo -e "${CYAN}Authentication:${NC} SSH Key (public key authentication)"
    
    # Final check - ensure no placeholders remain
    if grep -q "\[.*\]" .env; then
        print_warning "Some configuration values could not be filled:"
        grep "\[.*\]" .env
        
        print_status "Setting default values for any remaining placeholders..."
        
        # Check again after fixes
        if grep -q "\[.*\]" .env; then
            print_warning "Please review your .env file and manually update any remaining placeholders."
        else
            print_success "All placeholders have been filled with default values."
        fi
    else
        print_success "All configuration values successfully applied!"
    fi
}

# Function to ask user if they want to start Polaris
ask_to_start_polaris() {
    # Get absolute path to polariscloud directory
    local polaris_dir=$(realpath "$(pwd)/polariscloud")
    
    # Show clean success message and available commands
    clear
    echo -e "${GREEN}─────────────────────────────────────────────────────${NC}"
    echo -e "${CYAN}${BOLD}Success! Polaris Installation Complete${NC}"
    echo -e "${GREEN}─────────────────────────────────────────────────────${NC}"
    echo -e "${YELLOW}Available Polaris Commands:${NC}"
    echo -e "• ${CYAN}polaris start${NC}     - Start Polaris services"
    echo -e "• ${CYAN}polaris stop${NC}      - Stop Polaris services"
    echo -e "• ${CYAN}polaris status${NC}    - Check service status"
    echo -e "• ${CYAN}polaris logs${NC}      - View service logs"
    echo -e "• ${CYAN}polaris --help${NC}    - Show all available commands"
    echo
    print_warning "Important: Please log out and log back in for Docker group changes to take effect."
    echo
    
    echo -e "${CYAN}Would you like to start Polaris now? (y/n)${NC}"
    read -p "${CYAN}> ${NC}" start_polaris_now
    if [[ $start_polaris_now =~ ^[Yy]$ ]]; then
        # First check that polariscloud directory exists
        if [ ! -d "$polaris_dir" ]; then
            print_error "Cannot find polariscloud directory at: $polaris_dir"
            print_warning "Please make sure Polaris is properly installed."
            read -p "Press Enter to continue to main menu..."
            return 1
        fi
        
        # Navigate to polariscloud directory
        cd "$polaris_dir"
        print_status "Starting Polaris services from: $(pwd)"
        
        # Check if virtual environment exists
        if [ ! -d "venv" ] || [ ! -f "venv/bin/activate" ]; then
            print_error "Virtual environment not found at $(pwd)/venv"
            print_warning "Creating a new virtual environment..."
            
            # Try to create a new virtual environment
            python3 -m venv venv
            if [ ! -f "venv/bin/activate" ]; then
                print_error "Failed to create virtual environment. Please reinstall Polaris."
                cd - &>/dev/null
                read -p "Press Enter to continue to main menu..."
                return 1
            fi
        fi
        
        # Use polaris_run script if available (most reliable method)
        if [ -f "polaris_run" ] && [ -x "polaris_run" ]; then
            print_status "Using polaris_run script to start services..."
            ./polaris_run start
            start_success=$?
            
            if [ $start_success -eq 0 ]; then
                print_success "Polaris services started successfully with polaris_run!"
                echo
                print_status "Showing current status:"
                ./polaris_run status
                echo
            else
                print_error "Failed to start Polaris services with polaris_run. Trying alternative methods..."
            fi
        else
            # Activate the virtual environment with proper error handling
            print_status "Activating virtual environment..."
            if [ -f "venv/bin/activate" ]; then
                source "./venv/bin/activate"
                
                # Check if activation was successful 
                if [ -z "$VIRTUAL_ENV" ]; then
                    print_error "Failed to activate virtual environment."
                    cd - &>/dev/null
                    read -p "Press Enter to continue to main menu..."
                    return 1
                fi
                
                print_success "Virtual environment activated."
                
                # Try both command names
                start_success=1
                if command -v polaris &>/dev/null; then
                    print_status "Found polaris command, starting services..."
                    polaris start
                    start_success=$?
                elif command -v pcli &>/dev/null; then
                    print_status "Found pcli command, starting services..."
                    pcli start
                    start_success=$?
                else
                    print_error "Neither polaris nor pcli command found in PATH"
                    print_error "Available commands in PATH: $(which python3)"
                    print_status "Contents of venv/bin:"
                    ls -la venv/bin
                    start_success=1
                fi
                
                if [ $start_success -eq 0 ]; then
                    print_success "Polaris services started successfully!"
                    echo
                    print_status "Showing current status:"
                    if command -v polaris &>/dev/null; then
                        polaris status
                    elif command -v pcli &>/dev/null; then
                        pcli status
                    fi
                    echo
                    read -p "Press Enter to enter the Polaris environment..."
                    # Start interactive shell in the environment
                    $SHELL
                else
                    print_error "Failed to start Polaris services. Please check the logs."
                    
                    # Attempt to run directly with python as a last resort
                    print_status "Attempting to start directly with Python as a fallback..."
                    
                    if [ -f "main.py" ]; then
                        print_status "Found main.py, attempting to run..."
                        python3 main.py start
                    elif [ -f "polaris/__main__.py" ]; then
                        print_status "Found polaris module, attempting to run..."
                        python3 -m polaris start
                    else
                        print_error "Could not find Python entry points to run Polaris."
                    fi
                fi
                
                # When shell exits, deactivate the environment properly
                if [ -n "$VIRTUAL_ENV" ]; then
                    print_status "Deactivating virtual environment..."
                    deactivate || true  # Don't fail if deactivate isn't defined
                fi
            else
                print_error "Activation script not found at: $(pwd)/venv/bin/activate"
                print_warning "Polaris installation appears to be incomplete."
            fi
        fi
        
        # Return to the original directory
        cd - &>/dev/null
    fi
    
    echo
    read -p "Press Enter to continue to main menu..."
    
    # Force immediate check to update status
    check_polaris_installation
}

# Main loop
while true; do
    show_welcome_banner
    show_menu
done