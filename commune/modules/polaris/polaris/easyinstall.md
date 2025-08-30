# Polaris Compute Subnet - Easy Installation Guide

This guide provides simple instructions for installing the Polaris Compute Subnet using our automated installation script.

## What is Polaris Compute Subnet?

Polaris Compute Subnet is a modern development workspace manager for distributed compute resources. It simplifies managing compute resources, monitoring their status, and automating key tasks in a distributed environment.

Key features:
- Register and manage compute resources
- Monitor system status
- Manage SSH connections
- Direct connection support
- Cross-platform support (Windows, Linux, and macOS)

## Prerequisites

Before using the installation script, please ensure you have:

1. **Git**: To clone the repository
2. **Python 3.6+**: Python 3.6 or higher is required
3. **Admin/sudo access**: For installing system dependencies (on Linux)
4. **Internet connection**: To download dependencies
5. **Docker**: Installed and running (see README.md for detailed setup instructions)
6. **SSH Service**: Running on your machine (see README.md for setup details)
7. **Public IP Address**: Your compute node should have a public IP or be accessible from the internet

## Network Configuration Requirements

Polaris requires specific network settings to operate properly. To simplify the installation process:

- The script will **automatically configure** most settings for you
- You will only need to provide your **SSH username and password**
- All other values are set to recommended defaults

Here's what happens during configuration:
- Your public IP is automatically detected and set for HOST and SSH_HOST
- Default ports are configured (API_PORT=8000, SSH_PORT=11000, etc.)
- The script will interactively ask you only for SSH credentials

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/bigideainc/polaris-subnet.git
cd polaris-subnet
```

### 2. Make the Installation Script Executable

```bash
chmod +x install_polaris.sh
```

### 3. Run the Installation Script

```bash
./install_polaris.sh
```

The script will automatically:
- Switch to the correct branch (`cli_changes`)
- Install system dependencies
- Set up a Python virtual environment
- Install all required Python packages
- Configure environment settings (.env file)
- Verify the installation
- Create an activation script

### 4. Provide SSH Credentials

During installation, you'll be prompted to enter only two pieces of information:

1. **SSH Username**: Your system username for SSH connections
2. **SSH Password**: A secure password for SSH authentication

Example prompt:
```
===== REQUIRED CONFIGURATION =====
Please provide your SSH credentials:
SSH Username (your system username to connect via SSH):
> your_username

SSH Password (used for secure connections):
> 
```

All other configuration values will be automatically set to recommended defaults. If you need to customize other settings, you'll have the option to edit the full .env file.

### 5. Optional: Customize Other Settings

After providing your SSH credentials, you'll be asked if you want to customize any other settings:

```
Would you like to customize any other settings in the .env file? (y/n)
```

If you answer "y", a text editor will open where you can modify any of these pre-configured values:

```dotenv
# Network configuration
HOST=your_detected_ip_address
API_PORT=8000

# SSH configuration
SSH_PORT_RANGE_START=11000
SSH_PORT_RANGE_END=11002
SSH_PORT=11000

# Server configuration
SERVER_URL=https://polaris-test-server.onrender.com/api/v1
```

For most users, the default values will work perfectly, and no additional customization is needed.

### 6. Follow the On-Screen Instructions

At the end of the installation, you'll be asked if you want to start Polaris immediately. You can choose 'y' to start it right away or 'n' to start it later.

## What the Installation Script Does

Our automated script performs the following tasks:

- **Environment Detection**: Detects your operating system and installs appropriate dependencies
- **Dependency Installation**: 
  - Installs Rust and other required system packages
  - Sets up Python virtual environment
  - Installs Python dependencies including bittensor and communex
- **Configuration**: 
  - Detects and sets your public IP address
  - Prompts for SSH credentials (username and password)
  - Sets all other values to recommended defaults
- **Verification**: Ensures Polaris is installed correctly
- **Convenience**: Creates an activation script for future use

## Using Polaris After Installation

After installation is complete, you can use Polaris in two ways:

### Option 1: Using the Activation Script

```bash
source ./activate_polaris.sh
polaris start
```

### Option 2: Manual Activation

```bash
source venv/bin/activate
polaris start
```

## Key Commands

Once Polaris is installed and activated, here are some common commands:

- `polaris start` - Start Polaris and choose between Miner or Validator mode
- `polaris status` - Check if Polaris and Compute Subnet are running
- `polaris stop` - Stop Polaris and Compute Subnet background processes
- `polaris --help` - View all available commands

## Verifying Public IP Access

To verify that your compute node is accessible from the internet:

1. **Check your public IP**:
   ```bash
   curl ifconfig.me
   ```

2. **Test SSH connection**:
   ```bash
   ssh your_username@your_public_ip -p 11000
   ```

3. **Check port status**:
   You can use an online port checking service like [CanYouSeeMe.org](https://canyouseeme.org/) to confirm your ports are open and reachable.

## Troubleshooting

If you encounter issues during installation:

1. **Script fails at system dependencies**: 
   - Ensure you have admin/sudo privileges
   - Try installing the listed dependencies manually

2. **Python virtual environment issues**:
   - Ensure you have Python 3.6+ installed
   - Try creating and activating the virtual environment manually:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Package installation failures**:
   - Check your internet connection
   - Try installing packages individually:
     ```bash
     pip install -r requirements.txt
     pip install bittensor bittensor-cli
     pip install communex==0.1.36.4
     pip install -e .
     ```

4. **Branch issues**:
   - If the script can't find the `cli_changes` branch, try:
     ```bash
     git fetch --all
     git checkout -b cli_changes origin/cli_changes
     ```

5. **Network configuration issues**:
   - If your public IP wasn't detected automatically, you can manually edit the .env file:
     ```bash
     nano .env  # Or any text editor you prefer
     ```
   - Verify that ports 8000 and 11000-11002 are open in your firewall/router
   - If behind a NAT, ensure proper port forwarding is configured in your router
   - Check that your SSH service is running properly

## System Requirements

- **RAM**: 4GB minimum, 8GB+ recommended
- **Storage**: 1GB+ free space
- **CPU**: Modern multi-core processor
- **Network**: Active internet connection with public IP or port forwarding
- **OS**: Linux (Ubuntu/Debian preferred), macOS, or Windows with WSL

## Support

If you encounter any issues with the installation script or Polaris Compute Subnet, please:

1. Check if there is a solution in this guide
2. Visit our GitHub repository issues section
3. Contact our support team on Discord: [compute-33](https://discord.com/channels/941362322000203776/1324582017513422870)

## License

Polaris Compute Subnet is licensed under the MIT License. See the LICENSE file for details. 