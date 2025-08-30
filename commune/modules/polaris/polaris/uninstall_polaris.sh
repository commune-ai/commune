#!/bin/bash

# uninstall_polaris.sh - Script to remove Polaris Compute Subnet components for a fresh install

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}===============================================${NC}"
echo -e "${BLUE}    Polaris Compute Subnet - Uninstaller       ${NC}"
echo -e "${BLUE}===============================================${NC}"

# Check if we're in the right directory
if [ ! -f "setup.py" ] || [ ! -d "polaris_cli" ]; then
    echo -e "${RED}Error: This script must be run from the root of the polaris-subnet repository.${NC}"
    exit 1
fi

echo -e "${YELLOW}This script will remove all Polaris components from your system to allow a fresh install.${NC}"
echo -e "${RED}Warning: This will stop any running Polaris processes and delete configuration files.${NC}"
read -p "Are you sure you want to continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}Uninstallation cancelled.${NC}"
    exit 0
fi

# Step 1: Stop any running Polaris processes
echo -e "${BLUE}Stopping any running Polaris processes...${NC}"
if command -v polaris &> /dev/null; then
    if source venv/bin/activate 2>/dev/null && polaris status 2>/dev/null | grep -q "is running"; then
        echo -e "${YELLOW}Polaris processes are running. Attempting to stop them...${NC}"
        polaris stop || echo -e "${YELLOW}Could not stop Polaris processes automatically. You may need to stop them manually.${NC}"
    else
        echo -e "${GREEN}No Polaris processes appear to be running.${NC}"
    fi
else
    echo -e "${GREEN}Polaris command not found, likely no running processes.${NC}"
fi

# Step 2: Remove the virtual environment
echo -e "${BLUE}Removing virtual environment...${NC}"
if [ -d "venv" ]; then
    rm -rf venv
    echo -e "${GREEN}Virtual environment removed.${NC}"
else
    echo -e "${GREEN}No virtual environment found.${NC}"
fi

# Step 3: Remove environment configuration
echo -e "${BLUE}Removing environment configuration files...${NC}"
if [ -f ".env" ]; then
    rm .env
    echo -e "${GREEN}.env file removed.${NC}"
else
    echo -e "${GREEN}No .env file found.${NC}"
fi

# Step 4: Remove activation script
echo -e "${BLUE}Removing activation script...${NC}"
if [ -f "activate_polaris.sh" ]; then
    rm activate_polaris.sh
    echo -e "${GREEN}Activation script removed.${NC}"
else
    echo -e "${GREEN}No activation script found.${NC}"
fi

# Step 5: Clean Python cache files
echo -e "${BLUE}Cleaning Python cache files...${NC}"
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type f -name "*.pyd" -delete
find . -type d -name "*.egg-info" -exec rm -rf {} +
find . -type d -name "*.dist-info" -exec rm -rf {} +
find . -type d -name "*.egg" -exec rm -rf {} +
find . -type d -name ".pytest_cache" -exec rm -rf {} +
echo -e "${GREEN}Python cache files cleaned.${NC}"

# Step 6: Remove any other temporary files
echo -e "${BLUE}Removing any temporary files...${NC}"
rm -rf .coverage htmlcov .pytest_cache build dist logs/* 2>/dev/null
echo -e "${GREEN}Temporary files removed.${NC}"

echo -e "${GREEN}===============================================${NC}"
echo -e "${GREEN}    Uninstallation Complete!                   ${NC}"
echo -e "${GREEN}===============================================${NC}"
echo -e "${BLUE}You can now run the installation script again with:${NC}"
echo -e "   ${YELLOW}chmod +x install_polaris.sh${NC}"
echo -e "   ${YELLOW}./install_polaris.sh${NC}"
echo -e "${BLUE}Happy testing!${NC}" 