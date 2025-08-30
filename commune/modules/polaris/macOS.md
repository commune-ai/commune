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
  -Mac:** Launch Docker Desktop from your applications menu.


## SSH Setup and Configuration on macOS
### **Checking SSH Status**
On macOS, `systemctl` is not available. Instead, use `launchctl` to check the SSH service status:
```sh
sudo launchctl list | grep ssh
```
If SSH is running, you will see an output similar to:
```
-    0    com.openssh.sshd
```
If there is no output, SSH is not running.

---

### **Starting SSH Service**
#### **Enable SSH at Boot**
To ensure SSH starts automatically when your Mac boots up, run:
```sh
sudo systemsetup -setremotelogin on
```

---

### **Restarting SSH Service**
If you make changes to the SSH configuration, restart the SSH service with:
```sh
sudo launchctl stop com.openssh.sshd
sudo launchctl start com.openssh.sshd
```

---

### **Changing SSH Port**
1. Open the SSH configuration file:
   ```sh
   sudo nano /etc/ssh/sshd_config
   ```
2. Find the line:
   ```sh
   #Port 22
   ```
3. Uncomment it and change the port number, e.g.:
   ```sh
   Port 2222
   ```
4. Save the file (`CTRL + X`, then `Y`, then `Enter`).
5. Restart SSH:
   ```sh
   sudo launchctl stop com.openssh.sshd
   sudo launchctl start com.openssh.sshd
   ```

---

### **Opening a Range of Ports in the macOS Firewall**
To allow a range of ports (e.g., 2200-4300), modify the firewall rules:
```sh
sudo nano /etc/pf.conf
```
Add this line:
```sh
pass in proto tcp from any to any port 2200:4300
```
Save the file, then reload the firewall:
```sh
sudo pfctl -f /etc/pf.conf
```

---

### **Verifying SSH Listening Ports**
To check which ports SSH is listening on:
```sh
sudo lsof -i -P | grep ssh
```

---

### **Testing SSH Connection**
To connect to your Mac via SSH from another device, use:
```sh
ssh -p <your-port> username@your-mac-ip
```
For example, if using port 2222:
```sh
ssh -p 2222 user@192.168.1.100
```

---

### **check for public IP address**
```
curl ifconfig.me

```

### **Now your macOS SSH server is set up and configured! 🚀**


# Installing Homebrew, Rust, and Cargo on macOS

This guide provides step-by-step instructions to install Homebrew, Rust, and Cargo on macOS efficiently.

## 1. Install Homebrew
Homebrew is a package manager for macOS that simplifies software installation.

Open your terminal and run the following command to install Homebrew and add it to your shell profile:

```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile

eval "$(/opt/homebrew/bin/brew shellenv)"
```

### Verify Homebrew Installation
Run the following command to check if Homebrew is installed correctly:
```sh
brew --version
```

## 2. Install Rust and Cargo using Homebrew
Rust is a systems programming language, and Cargo is its package manager and build system.

Run the following command to install Rust and Cargo using Homebrew:
```sh
brew install rust && rustc --version && cargo --version
```

## Alternative Installation Method (Using rustup)
For the latest version and better integration, you can install Rust using rustup:
```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh && source $HOME/.cargo/env
```

## Conclusion
You have now installed Homebrew, Rust, and Cargo on your macOS system. You are ready to start developing with Rust!

For more information, check out the official documentation:
- Homebrew: [https://brew.sh](https://brew.sh)
- Rust: [https://www.rust-lang.org](https://www.rust-lang.org)
# Ensure Python 3.10 on macOS is installed
## Installation Steps
1. Open your terminal.
2. Run the following command to install Python 3.10 and link it as the default version:
   ```sh
   brew install python@3.10 && brew link python@3.10 --force --overwrite
   ```

## Verify Installation
After installation, verify that Python 3.10 and pip are installed correctly by running:

```sh
python3 --version
pip3 --version
```

Expected output should show Python 3.10.x and the corresponding pip version.

## Troubleshooting
- If `python3` does not point to Python 3.10, you can explicitly use:
  ```sh
  /usr/local/bin/python3.10 --version
  ```
- If `pip3` is not found, try:
  ```sh
  python3 -m ensurepip --default-pip
  ```
### 2. Create and Activate a Virtual Environment

It's best practice to use a Python virtual environment to isolate dependencies.

#### Create the Virtual Environment

```bash
# Create a virtual environment named 'venv'
python3.10 -m venv venv
```

#### Activate the Virtual Environment

- **On macOS:**
  ```bash
  source venv/bin/activate
  ```
  ### 3. Install Required System Packages

Before installing Python dependencies, you need to install several system packages and network-specific requirements:

```
# Install network-specific packages
pip install bittensor
pip install bittensor-cli
pip install communex==0.1.36.4
```

### 4. Clone and Install Polaris

After installing the prerequisites, clone the Polaris repository and install it:

```bash
# Clone the repository
git clone https://github.com/bigideaafrica/polariscloud.git
cd polariscloud

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
### Install dependencies

#### Install dependencies from requirements.txt
```
pip install -r requirements.txt

```

#### Install Polaris in development mode
```
pip install -e .

```

### 6. Verify Installation

Check that Polaris is installed correctly by running:

```bash
polaris --help
```

## Polaris Commands

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

- **Python:** Version 3.10 
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

