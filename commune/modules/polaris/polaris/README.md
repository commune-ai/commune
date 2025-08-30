# Polaris Compute Subnet

A modern development workspace manager for distributed compute resources. Polaris simplifies managing compute resources, monitoring their status, and automating key tasks in a distributed environment.

---

## Features

- **Register and manage compute resources:** Add and monitor distributed compute nodes.
- **Monitor system status:** View system health and active processes.
- **Manage SSH connections:** Automate and configure secure SSH connections.
- **Direct connection support:** Establish secure connections using your public IP.
- **Cross-platform support:** Works on Windows, Linux, and macOS.

---

## Prerequisites

### 1. Docker Installed and Running

**Installation:**
- Visit [Docker's official site](https://docs.docker.com/get-docker/) for installation instructions tailored to your operating system.

**Running Docker:**
- **Verify Installation:**
  ```bash
  docker --version
  ```
  **Expected Output:**
  ```
  Docker version 20.10.7, build f0df350
  ```
- **Check Docker Status:**
  ```bash
  docker info
  ```
  **Expected Output:**
  ```
  Client:
   Debug Mode: false

  Server:
   Containers: 5
   Running: 2
   Paused: 0
   Stopped: 3
   ...
  ```
- **Start Docker Manually (if not running):**
  - **Windows/Mac:** Launch Docker Desktop from your applications menu.
  - **Linux:** Start Docker service using your system's service manager, e.g.,
    ```bash
    sudo systemctl start docker
    ```

### 2. SSH Service Running

**Check SSH Status:**
- **Linux:**
  ```bash
  sudo systemctl status ssh
  ```
  **Expected Output:**
  ```
  ● ssh.service - OpenBSD Secure Shell server
     Loaded: loaded (/lib/systemd/system/ssh.service; enabled; vendor preset: enabled)
     Active: active (running) since Tue 2025-01-07 05:03:14 UTC; 1h 30min ago
     ...
  ```
- **Windows:**
  - **Enable OpenSSH Server:**
    1. Go to **Settings** > **Apps** > **Optional Features**.
    2. Click **Add a feature**.
    3. Find and install **OpenSSH Server**.
  - **Start SSH Service:**
    ```powershell
    Start-Service sshd
    ```
  - **Set SSH to Start Automatically:**
    ```powershell
    Set-Service -Name sshd -StartupType 'Automatic'
    ```

**Start SSH if Needed:**
- **Linux:**
  ```bash
  sudo systemctl start ssh
  ```
- **Windows:**
  ```powershell
  Start-Service sshd
  ```

**Verify SSH Configuration:**
- **Linux:** Check `/etc/ssh/sshd_config` for correct settings.
- **Windows:** Ensure OpenSSH Server is properly configured via **Settings** or configuration files.

### 3. Public IP Address for Your Compute Node

**Objective:**  
Ensure your compute node has a public IP address or is directly accessible from the internet. This allows other nodes and services to connect to your machine without intermediary solutions.

**Steps to Make Your Compute IP Public:**

#### A. Verify Your Current Public IP

1. **Check Public IP:**
   - Open a terminal on your compute node and run:
     ```bash
     curl ifconfig.me
     ```
     or
     ```bash
     curl ipinfo.io/ip
     ```
   - The returned IP should be a public IP. If it shows a private IP (e.g., 192.168.x.x, 10.x.x.x, or 172.16.x.x to 172.31.x.x), then your machine is behind a NAT and does not have a direct public IP.

#### B. Configuring Public IP Access

If your machine does not have a public IP, follow one of these approaches to expose it:

1. **Direct Public IP from ISP:**
   - **Request a Public IP:** Contact your Internet Service Provider (ISP) to request a static or dynamic public IP assignment for your machine.
   - **Configure Network Interface:** 
     - If given a static IP, configure your network interface with the provided IP, subnet mask, gateway, and DNS settings.
     - On Linux, for example, you might update your network configuration file or use `nmcli`/`ifconfig` depending on your distribution.

2. **Port Forwarding Through a Router:**
   - **Access Router Settings:**
     - Log into your router's administration panel (usually accessed via a web browser at an address like `192.168.1.1`).
   - **Configure Port Forwarding:**
     1. Locate the port forwarding section.
     2. Add new port forwarding rules to forward external traffic to your compute node's internal IP:
        - Forward the API port (e.g., 8000)
        - Forward the SSH port range (e.g., 11000-11002)
   - **Save and Apply:** Save changes and restart the router if necessary.
   - **Determine Your Public IP:** Find your router's public IP by checking the router status page or using a service like [WhatIsMyIP.com](https://www.whatismyip.com/).

3. **Dynamic DNS (if you have a dynamic public IP):**
   - **Set Up Dynamic DNS:** If your ISP assigns a dynamic public IP, use a Dynamic DNS service (like No-IP, DynDNS, etc.) to associate a domain name with your changing IP.
   - **Configure Your Router or Client:**
     - Many routers support Dynamic DNS configuration directly. Input your Dynamic DNS credentials into the router's DDNS settings.
     - Alternatively, run a Dynamic DNS client on your compute node to update the DNS record whenever your IP changes.

#### C. Verify Public Accessibility

Once configured, verify that your compute node is accessible from the internet:

- **Test SSH Connection:**
  ```bash
  ssh user@your_public_ip -p <ssh_port>
  ```
  Replace `your_public_ip` with your public IP and `<ssh_port>` with your configured SSH port (e.g., 11000).

- **Check Port Status:**
  Use an online port checking service like [CanYouSeeMe.org](https://canyouseeme.org/) to confirm the relevant ports are open and reachable.

**Security Considerations:**

- **Firewall Rules:**  
  Ensure your firewall (both on the compute node and network level) allows incoming connections on the necessary ports, but also restricts access to trusted IPs when possible.

- **Strong Authentication:**  
  Use strong passwords, SSH keys, or other authentication methods to secure direct access to your compute node.

---

## Setting Up a Linux Environment for Windows and macOS Users

While Polaris is designed to be cross-platform, certain features work best in a Linux environment. Windows and macOS users should set up a Linux virtual environment for optimal performance.

### For Windows Users: Windows Subsystem for Linux (WSL)

WSL allows you to run a Linux environment directly on Windows without the need for a traditional virtual machine.

1. **Install WSL**:
   Open PowerShell as Administrator and run:
   ```powershell
   wsl --install
   ```
   This will install WSL 2 with Ubuntu by default.

2. **Restart your computer** when prompted.

3. **Set up your Ubuntu user account**:
   After restart, a Ubuntu terminal will open automatically.
   Create a username and password when prompted.

4. **Update and upgrade packages**:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

5. **Install Docker in WSL**:
   ```bash
   # Install Docker prerequisites
   sudo apt install -y apt-transport-https ca-certificates curl software-properties-common

   # Add Docker's official GPG key
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

   # Add Docker repository
   sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

   # Install Docker
   sudo apt update
   sudo apt install -y docker-ce

   # Start Docker service
   sudo service docker start

   # Add your user to the Docker group
   sudo usermod -aG docker $USER
   ```

6. **Install SSH server**:
   ```bash
   sudo apt install -y openssh-server
   sudo service ssh start
   sudo systemctl enable ssh
   ```

7. **Now follow the Polaris installation steps** in your WSL terminal.

### For macOS Users: Docker Desktop with Ubuntu Container

1. **Install Docker Desktop**:
   Download and install Docker Desktop from the [official website](https://www.docker.com/products/docker-desktop).

2. **Run an Ubuntu container**:
   Open Terminal and run:
   ```bash
   docker run -it --name polaris-env ubuntu:20.04
   ```

3. **Set up the Ubuntu environment**:
   ```bash
   apt update && apt upgrade -y
   apt install -y python3 python3-pip git curl openssh-server sudo
   
   # Create a non-root user (optional but recommended)
   adduser polaris
   usermod -aG sudo polaris
   su - polaris
   ```

4. **Start SSH server**:
   ```bash
   sudo service ssh start
   ```

5. **Now follow the Polaris installation steps** within your Ubuntu container.

### Alternative: Using a Full Virtual Machine

If you prefer a full virtual machine:

1. **Download and install VirtualBox** from [virtualbox.org](https://www.virtualbox.org/).

2. **Download Ubuntu 20.04 LTS** from [ubuntu.com](https://ubuntu.com/download/desktop).

3. **Create a new virtual machine**:
   - Open VirtualBox and click "New"
   - Name: Polaris
   - Type: Linux
   - Version: Ubuntu (64-bit)
   - Allocate at least 4GB RAM
   - Create a virtual hard disk (at least 30GB)

4. **Install Ubuntu**:
   - Start the VM
   - Select the Ubuntu ISO
   - Follow the installation wizard

5. **Update the system**:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

6. **Install necessary packages**:
   ```bash
   sudo apt install -y git python3-pip docker.io openssh-server
   sudo systemctl enable --now docker
   sudo usermod -aG docker $USER
   ```

7. **Now follow the Polaris installation steps** within your Ubuntu VM.

## Installation and Setup

### 1. Clone the Repository

```bash
# Clone the Polaris repository
git clone https://github.com/bigideainc/polaris-subnet.git
cd polaris-subnet
```

### 2. Create and Activate a Virtual Environment

It's best practice to use a Python virtual environment to isolate dependencies.

#### Create the Virtual Environment

```bash
# Create a virtual environment named 'venv'
python3 -m venv venv
```

*Note:* If your system uses `python` instead of `python3`, adjust the command accordingly:
```bash
python -m venv venv
```

#### Activate the Virtual Environment

- **On macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```
- **On Windows (Command Prompt):**
  ```batch
  venv\Scripts\activate.bat
  ```
- **On Windows (PowerShell):**
  ```powershell
  venv\Scripts\Activate.ps1
  ```

Once activated, your command prompt should indicate the virtual environment is active (e.g., it may start with `(venv)`).

### 3. Install Required System Packages

Before installing Python dependencies, you need to install several system packages and network-specific requirements:

```bash
# Install system dependencies
sudo apt install g++ rustc cargo build-essential python3-dev

# Install network-specific packages
pip install bittensor
pip install bittensor-cli
pip install communex==0.1.36.4
```

### 4. Clone and Install Polaris

After installing the prerequisites, clone the Polaris repository and install it:

```bash
# Clone the repository
git clone https://github.com/bigideainc/polaris-subnet.git
cd polaris-subnet

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Install Polaris in development mode
pip install -e .
```

### 5. Configure SSH and Network Settings

Polaris uses SSH and network port configuration for secure connections. Add your SSH and network settings to a `.env` file at the root of the project:

```dotenv
# .env file
HOST=xx.xx.xx.xx
API_PORT=8000
SSH_PORT_RANGE_START=11xxx
SSH_PORT_RANGE_END=11xx2
SSH_PASSWORD=...
SSH_USER=pol_user1
SSH_HOST=xx.xx.xx.xx
SSH_PORT=1xxxx
SERVER_URL=https://polaris-test-server.onrender.com/api/v1
```

### 6. Verify Installation

Check that Polaris is installed correctly by running:

```bash
polaris --help
```

## Polaris Commands

### polaris register

This command initiates the registration process for your node on the Polaris network.

```bash
polaris register
```

When you run this command, you'll see a registration type selection prompt with arrow key navigation:

```
2025-03-04 19:15:09,023 [INFO] Enabling default logging (Warning level)
2025-03-04 19:15:09,023 |     INFO     | Enabling default logging (Warning level)
🔑 Select registration type: (Use arrow keys)
» Commune Miner Node
  Bittensor Miner Node
  Polaris Miner Node (Coming Soon)
  Independent Miner
```

#### Commune Miner Node Registration

If you select "Commune Miner Node", you'll be asked if you already have a Commune key:

```
2025-03-04 19:15:09,023 [INFO] Enabling default logging (Warning level)
2025-03-04 19:15:09,023 |     INFO     | Enabling default logging (Warning level)
🔑 Select registration type: Commune Miner Node
? Do you have an existing Commune key? (Y/n) 
```

If you answer "No", you'll be prompted to create a new key:

```
2025-03-04 19:15:09,023 [INFO] Enabling default logging (Warning level)
2025-03-04 19:15:09,023 |     INFO     | Enabling default logging (Warning level)
🔑 Select registration type: Commune Miner Node
? Do you have an existing Commune key? No
Enter a name for your new Commune key: poly
```

After creating the key, the system checks your COMAI balance and may display a warning if it's insufficient:

```
Key poly created successfully.
⚠️Low Balance Warning━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WARNING: Your current balance is 0.0 COMAI.
A minimum of 10 COMAI is recommended for registration.
You may proceed, but some network features might be limited.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Network Selection ━━━━━━━━━━━━━━━━━━━━━━━━━━━
Available networks:
1. Mainnet (netuid=33)
2. Testnet (netuid=12)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Select network (enter 1 for Mainnet or 2 for Testnet) [1/2] (1): 
```

After selecting your network (Mainnet or Testnet), the system will confirm your selection:

```
Select network (enter 1 for Mainnet or 2 for Testnet) [1/2] (1): 1
Selected network: Mainnet (netuid=33)
```

#### Bittensor Miner Node Registration

If you select "Bittensor Miner Node", you'll see this process:

```
2025-03-04 19:32:57,920 [INFO] Enabling default logging (Warning level)
2025-03-04 19:32:57,920 |     INFO     | Enabling default logging (Warning level)
🔑 Select registration type: Bittensor Miner Node
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Bittensor Setup ━━━━━━━━━━━━━━━━━━━━━━━━━
Bittensor Wallet Configuration
You'll need a wallet to participate in the Bittensor subnet
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

? Do you already have a Bittensor wallet? (Y/n)
```

If you answer "No", you'll be prompted to create a new wallet:

```
? Do you already have a Bittensor wallet? No
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Wallet Creation ━━━━━━━━━━━━━━━━━━━━━━━━━
Creating a New Bittensor Wallet
You will need to provide a name for your new wallet.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

? Enter a name for your new wallet: poly
```

The wallet creation process includes several steps:

1. **Create Coldkey:**
   ```
   Creating new coldkey...
   Enter the path to the wallets directory (~/bittensor/wallets/): 
   Choose the number of words [12/15/18/21/24]: 12
   
   IMPORTANT: Store this mnemonic in a secure (preferable offline place), as anyone who has possession of this mnemonic can use it to regenerate the key and access your tokens.
   
   The mnemonic to the new coldkey is: awkward giant record strong sugar ensure lens inch dinner kite fatigue orbit
   You can use the mnemonic to recreate the key with "btcli" in case it gets lost.
   ```

2. **Set Password:**
   ```
   Enter your password: 
   Password not strong enough. Try increasing the length of the password or the password complexity.
   Enter your password again: 
   Retype your password: 
   Encrypting...
   Coldkey created
   ```

3. **Create Hotkey:**
   ```
   Creating new hotkey...
   Choose the number of words [12/15/18/21/24]: 12
   
   IMPORTANT: Store this mnemonic in a secure (preferable offline place), as anyone who has possession of this mnemonic can use it to regenerate the key and access your tokens.
   
   The mnemonic to the new hotkey is: spoon month first wild travel oppose skin birth roast vague what patient
   You can use the mnemonic to recreate the key with "btcli" in case it gets lost.
   Hotkey created
   Wallet created successfully!
   ```

4. **Network Selection:**
   ```
   ? Select the network to register on: (Use arrow keys)
   » Mainnet (netuid 100)
     Testnet (netuid 12)
   ```

#### Polaris Miner Node Registration

If you select "Polaris Miner Node (Coming Soon)", you'll see:

```
2025-03-04 19:35:27,624 [INFO] Enabling default logging (Warning level)
2025-03-04 19:35:27,624 |     INFO     | Enabling default logging (Warning level)
🔑 Select registration type: Polaris Miner Node (Coming Soon)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 🏗️ Coming Soon ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Polaris Miner Node is coming soon!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

This feature is not yet implemented and will return you to the command prompt.

### polaris start

This command starts the Polaris service and selected compute processes.

```bash
polaris start
```

When you run this command, you'll be prompted to select which mode you want to run:

```
🚀 Select mode:
> Miner
  Validator
```

After selecting and confirming, upon successful startup, you'll see the Polaris dashboard:

```
                                                POLARIS SUBNET

                              ____        __            _     
                             / __ \____  / /___ _______(_)____
                            / /_/ / __ \/ / __ `/ ___/ / ___/
                           / ____/ /_/ / / /_/ / /  / (__  ) 
                          /_/    \____/_/\__,_/_/  /_/____/  
    
                           ♦ The Best Place to List Your GPUs ♦

                        Welcome to the Polaris Compute Subnet!

         ♦ Our Mission is to be the Best Place on This Planet to List Your GPUs - We're just getting
                                          started! ♦

                                 Powering GPU Computation

┌───────────────────────────────────────────────┬─────────────────────────────────────────────────┐
│ Setup Commands                                │ Service Management                              │
│ • register - Register as a new miner (required│ • start - Start Polaris and selected compute pro│
│ • update subnet - Update the Polaris repositor│ • stop - Stop running processes                 │
│                                               │ • status - Check if services are running        │
└───────────────────────────────────────────────┴─────────────────────────────────────────────────┘
┌───────────────────────────────────────────────┬─────────────────────────────────────────────────┐
│ Monitoring & Logs                             │ Bittensor Integration                           │
│ • logs - View logs without process monitoring │ Polaris integrates with Bittensor to provide a d│
│ • monitor - Monitor miner heartbeat signals in│ • Wallet Management - Create or use existing Bit│
│ • check-main - Check if main process is runnin│ • Validator Mode - Run as a Bittensor subnet val│
│ • view-compute - View pod compute resources   │ • Network Registration - Register with Bittensor│
│                                               │ • Heartbeat Service - Maintain connection with t│
└───────────────────────────────────────────────┴─────────────────────────────────────────────────┘

                                    Quick Start Guide
1. First register as a miner
2. Then start your preferred service type
3. Check status to verify everything is running
4. Use logs to monitor operation
5. Use stop when you want to shut down services

Examples:
$ polaris register - Register as a new miner
$ polaris start - Start the Polaris services
$ polaris status - Check which services are running
$ polaris stop - Stop running services
$ polaris logs - View service logs
```

### polaris status

This command checks if Polaris services are running.

```bash
polaris status
```

Example output when services are running:

```
[green]polaris is running with PID 22845.[/green]
[green]heartbeat service is running with PID 22846.[/green]
[green]system process is running with PID 22847.[/green]
```

Example output when services are not running:

```
[yellow]polaris is not running.[/yellow]
[yellow]heartbeat service is not running.[/yellow]
[yellow]system process is not running.[/yellow]
```

### polaris logs

This command displays log files without process monitoring.

```bash
polaris logs
```

Example output showing active logs:

```
API logs: /home/polaris/pol/polaris-subnet/logs/api_server.log
2025-03-04 19:32:57,920 [INFO] API server started
2025-03-04 19:33:12,345 [INFO] Handling heartbeat request
2025-03-04 19:33:42,671 [INFO] Compute resource status: ACTIVE
...
```

### polaris monitor

This command monitors miner heartbeat signals in real-time.

```bash
polaris monitor
```

Example display showing a real-time heartbeat monitor:

```
╔════════════════════════════ Polaris Heartbeat Monitor ════════════════════════════╗
║                                                                                   ║
║  Miner ID: rNzk6vnQd1j1qWt8Ajue                                                   ║
║  Status: ONLINE                                                                   ║
║  Last Heartbeat: 2025-03-04 19:42:15.123 UTC (2 seconds ago)                      ║
║                                                                                   ║
║  System Metrics:                                                                  ║
║  ├─ CPU: 12%                                                                      ║
║  ├─ Memory: 2.3GB/7.6GB (30%)                                                     ║
║  ├─ Disk: 324GB/1006GB (32%)                                                      ║
║  └─ Network: 2.1 Mbps ↓ / 0.4 Mbps ↑                                              ║
║                                                                                   ║
║  Press Ctrl+C to exit                                                             ║
║                                                                                   ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
```

### polaris check-main

This command checks if the main process is running and displays its logs.

```bash
polaris check-main
```

Example output:

```
[green]Main process is running with PID 22847.[/green]
[cyan]Recent logs:[/cyan]
2025-03-04 19:36:30,543 [INFO] System process started
2025-03-04 19:36:45,217 [INFO] Resource monitoring initialized
2025-03-04 19:37:01,342 [INFO] Heartbeat registered
```

### polaris view-compute

This command displays information about your compute resources.

```bash
polaris view-compute
```

Example output showing your registered compute resources:

```
Pod Details
-----------
ID: rNzk6vnQd1j1qWt8Ajue
Name: poly's Compute Pod
Location: Kampala, Central Region, UG
Description: Independent miner compute resources

Compute Resources
-----------------
ID                              | Type | Location                  | RAM   | Storage     | CPU                   
--------------------------------| ---- | ------------------------- | ----- | ----------- | ----------------------
aa54c5de-8ee8-41b8-aa89-f623d6b | CPU  | Kampala, Central Region   | 7.6GB | HDD 1006GB  | Intel i7-1255U, 12 cores
```

For complete technical documentation, including API endpoints, data models, error handling, and more, please refer to the [Technical Documentation](./docs/technical.md).

---

## Requirements

- **Python:** Version 3.6 or higher.
- **Operating Systems:** Compatible with Windows, Linux, and macOS.
- **Docker:** Installed and running.
- **SSH Service:** Active and properly configured on your machine.
- **Public IP:** Your compute node should be publicly accessible, either via a direct public IP or properly configured port forwarding.

---

## Author

**Polaris Team**  
Hit us up on Discord: [compute-33](https://discord.com/channels/941362322000203776/1324582017513422870)

---

*For further assistance or inquiries, please reach out to the Polaris Team.*
