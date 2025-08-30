# Polaris Subnet

## Subnet Purpose and Objectives

**Polaris** is a Bittensor subnet designed as a decentralized GPU compute marketplace and management layer. Its primary goal is to **connect miners who have spare GPU resources with the Bittensor network**, enabling those GPUs to be used for AI tasks and other compute-intensive workloads in a trustless environment.

Polaris's **mission** is to be *"the best place on this planet to list your GPUs,"* simplifying how operators join the Bittensor compute network. To achieve this, Polaris offers user-friendly tools to register and manage compute nodes across distributed environments.

Key objectives of the Polaris subnet include:

- **Decentralized Compute Market:** Create a trustless marketplace for GPU compute resources
- **Robust Validation:** Ensure reliable and accurate performance measurement
- **Ease of Use and Management:** Provide intuitive tools for node operators
- **Seamless Bittensor Integration:** Maintain compatibility with the Bittensor ecosystem

## Technical Requirements

Running a Polaris node (either as a miner or validator) requires meeting certain hardware, software, and network prerequisites. Below are the minimum and recommended requirements to ensure stable operation:

### Hardware Requirements

- **Operating System:** A Linux environment is required to run Polaris. Native installation on Ubuntu 20.04/22.04 (or other Debian-based distributions) is recommended. Windows 10/11 users can participate via WSL2, and macOS users can run a Linux VM or use Docker.
- **CPU & RAM:** 
  - 64-bit multi-core processor (Intel/AMD) with 2-4+ cores
  - Miners: Minimum 8GB RAM (16GB recommended)
  - Validators: Similar CPU/RAM requirements
- **GPU (Mining Nodes Only):**
  - Minimum: NVIDIA GPU with ~4GB VRAM (e.g., GTX 1660/RTX 2060)
  - Recommended: High-performance GPUs (RTX 3090/4090, A100/H100)
  - Multiple GPUs supported
- **Storage:** 
  - Minimum: 2GB free space
  - Recommended: 50+ GB for workloads/datasets
  - SSD recommended for better performance
- **Network:**
  - Reliable internet connection
  - 5+ Mbps upload/download (10+ Mbps recommended)
  - Public IP address or proper port forwarding
  - Ports: 8000 (API) and 11000-11002 (SSH)

### Software Requirements

- **Python:** Version 3.8+ (3.10 recommended)
- **Docker:** For containerized environments
- **Git:** For repository management
- **OpenSSH Server:** For secure connections
- **NVIDIA Drivers:** Latest version for your GPU
- **NVIDIA Container Toolkit:** For Docker GPU support

**Note on GPUs and Drivers:** Polaris itself does not impose specific GPU model requirements, but the **better the GPU, the higher the potential rewards**. Modern NVIDIA GPUs (Turing architecture or newer) have proven effective on Bittensor's compute tasks. Ensure you have the latest NVIDIA drivers installed and disable power-saving modes for optimal performance.

## Competitiveness Guidance

### Performance Measurement

In the Polaris subnet, **miners' performance is evaluated by validators** using various metrics:
- GPU capability (model and CUDA cores)
- Number of GPUs
- Available VRAM
- Network bandwidth
- Actual computational throughput

**Higher-performance miners will earn higher scores and greater rewards**, as the Bittensor consensus rewards contributions proportionally to their value.

### Staying Competitive

To remain competitive as a **miner** on Polaris:

- **Use High-Performance Hardware:** Invest in quality GPUs and maintain optimal performance
- **Maintain Excellent Uptime:** Keep your node running consistently
- **Optimize Network Settings:** Ensure stable connectivity and proper port forwarding
- **Stay Updated:** Keep your Polaris installation current with `polaris update subnet`
- **Monitor Performance:** Use built-in tools like `polaris monitor` and `polaris logs`
- **No Custom Model Training Needed:** Polaris focuses on raw compute power, not AI model quality

For **validators**, competitiveness focuses on:
- Continuous availability
- Accurate performance measurement
- Reliable hardware for handling multiple connections
- Professional operation mindset
- Regular software updates

## Installation Instructions

#### For  **[MacOs](https://github.com/bigideaafrica/polaris/blob/release/macOS.md)** Installation follow the link.

### Miner Installation

1. **Prepare the System:**
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io git openssh-server python3.10 python3-pip
```

2. **Clone the Repository:**
```bash
git clone https://github.com/bigideaafrica/polaris.git
cd polaris
```

3. **Run the Installation Script:**
```bash
chmod +x polaris_manager.sh
./polaris_manager.sh
```

4. **Follow the Interactive Setup:**
   - Choose "Install Polaris"
   - System will check requirements
   - Set up Python virtual environment
   - Configure network settings
   - Initialize Polaris environment

5. **Register Your Node:**
```bash
polaris register
```
   - Choose "Bittensor Miner Node"
   - Create or use existing wallet
   - Select network (Mainnet netuid 49 or Testnet netuid 100)

6. **Start Mining:**
```bash
polaris start
```

### Validator Installation

1. **Create Wallets:**
```bash
pip install bittensor-cli==9.1.0
btcli wallet new_coldkey --wallet.name <your_wallet_name>
btcli wallet new_hotkey --wallet.name <your_wallet_name> --wallet.hotkey default
```

2. **Register Keys:**
```bash
btcli subnet register --netuid 49 --subtensor.network finney --wallet.name <your_wallet_name> --wallet.hotkey default
```

3. **Verify Registration:**
```bash
btcli wallet overview --wallet.name <your_wallet_name> --subtensor.network finney
```

4. **Run Validator:**
```bash
docker pull bateesa/polaris-validator
docker run --rm -it -v ~/.bittensor:/root/.bittensor -e WALLET_NAME=<your_wallet_name> -e WALLET_HOTKEY=default bateesa/polaris-validator
```

## Usage

### Common Commands

- `polaris start` - Start Polaris services
- `polaris stop` - Stop Polaris services
- `polaris status` - Check service status
- `polaris logs` - View service logs
- `polaris monitor` - View real-time metrics
- `polaris update subnet` - Update to latest version
- `polaris --help` - Show all available commands

### Monitoring and Management

- Use `polaris monitor` for real-time performance tracking
- Check logs with `polaris logs` for troubleshooting
- Monitor system resources and GPU usage
- Keep software updated with `polaris update subnet`

## FAQ

### General Questions

**Q: Do I need an AI model or to train anything to mine on Polaris?**  
A: No – Polaris is a **compute-focused subnet**, meaning miners provide raw computing power (GPU cycles) rather than serving a machine learning model. You don't have to train or fine-tune any neural network to compete on Polaris. Your node's performance is determined by your hardware and how well it runs the subnet's tasks, not by accuracy of AI responses. This is different from some other subnets where model quality matters. In summary, **hardware > algorithms** on Polaris – keep your GPUs running optimally and you're set.

**Q: What kind of rewards will I earn, and how are they given?**  
A: Polaris is integrated into the Bittensor network, so miners earn **TAO**, Bittensor's native token, as a reward for their contributions. The network's consensus (Yuma) allocates block rewards to Polaris validators, who then distribute rewards to miners based on the performance scores of each mine. Practically, this means if your miner is consistently providing strong compute service, your Bittensor wallet's TAO balance will increase over time. You do not earn a separate token – it's all TAO, maintaining consistency with the rest of the Bittensor ecosystem. (If you're on testnet, the rewards are test-TAO with no real value, just for experimentation.)

**Q: What is the difference between a miner and a validator on Polaris? Can I run both?**  
A: In the Bittensor subnet design, **miners** provide the service (in Polaris's case, GPU compute) and **validators** provide oversight (they verify miners' work and uphold the blockchain consensus). On Polaris, a miner node will dedicate its GPU to running tasks and responding to validator queries (heartbeats, benchmarks), whereas a validator node will connect to many miners to measure their performance and produce blocks accordingly. **Most participants will be miners.** Validators are typically a limited set of nodes (determined by the subnet governance or genesis). Unless you have been specifically included or elected as a validator, you should run a miner. It's technically possible to run both on separate machines – one as a validator, one as a miner – but running a validator without authorization won't be effective. The Polaris software does allow you to start in "Validator" mode, but if you are not recognized as a validator by the network, that instance won't produce blocks or earn rewards. Therefore, **most users should choose Miner mode**.

### Technical Questions

**Q: I already have a Bittensor wallet (from another subnet or an earlier version). Can I use that on Polaris?**  
A: Yes, you can use an existing Bittensor wallet to register on Polaris – in fact, the installer will ask if you have one during `polaris register`. If you say yes, you'll provide the name (and ensure the corresponding hotkey/coldkey files are in place). However, be mindful that some users have reported needing to **regenerate or re-import wallets** for Polaris specifically. If the registration process indicates an issue with your wallet or if the wallet was created long ago (with old formats), you can create a fresh wallet through the Polaris CLI. This doesn't affect your other subnets; you can have multiple wallets. Also note, each wallet on Bittensor is tied to a network (via netuid). If your wallet is already registered on another subnet, it can still register on Polaris independently (as subnets have separate UID spaces). The key point is to **have your wallet's secret keys available** on the machine (usually in `~/.bittensor/wallets/<walletname>`). If Polaris can't find or use them, consider regenerating the wallet through the prompts.

**Q: The installer or `polaris register` is asking about a *Commune* key and COMAI balance. What is this?**  
A: Polaris includes an optional concept called **Commune** – this appears to be a community orchestration layer or alternate network mode. If you choose "Commune Miner Node" during registration, it deals with a **Commune key** and mentions a COMAI balance (which seems to be a separate token or credit system). This path is likely for connecting your node to a community-run system or sidechain that might require a certain balance of a token (COMAI) to participate. For **most users, this is not needed**, and you can ignore the Commune option unless you are specifically joining that community experiment. It's safe to proceed with **Bittensor Miner Node** registration (which doesn't involve COMAI) and skip Commune-related questions. The **Independent Miner** option is another advanced mode (possibly to run without any network – for development or local testing). In short, unless you know you need the Commune mode, stick to the default Bittensor registration for Polaris.

**Q: Can I run multiple Polaris miners to utilize multiple GPUs or machines?**  
A: Yes. Each Polaris miner is essentially a separate process tied to a Bittensor wallet identity. If you have a multi-GPU rig, you have a couple of options:  
(a) **Run one Polaris instance that uses all GPUs.** Polaris will detect all available GPUs on the system (for example, via CUDA) and the validators' benchmarking will recognize the total power. This means a single registered miner could be credited for having multiple GPUs. This approach is simpler (one software instance), but note that Bittensor's reward mechanism may not perfectly linearize rewards for multiple GPUs under one UID – if one process holds many GPUs, it will perform very well, likely keeping that UID safe in top ranks, but you won't get "multiple shares" of rewards beyond what one UID can earn.  
(b) **Run multiple instances (one per GPU)**, each with its own wallet (or sub-wallet). This would register multiple UIDs on the subnet, one per GPU. This can potentially maximize your reward share if the subnet's design rewards separate identities more, but it comes with overhead: you'll need to allocate specific GPUs to each instance (using environment variables like `CUDA_VISIBLE_DEVICES` or similar so they don't overlap) and ensure each has unique ports. This is an advanced setup – you'd basically run each instance in a separate folder or container, each on a different port range (e.g., second miner uses API port 8001 and SSH ports 11003-11005, etc.).

### Troubleshooting

**Q: I completed installation, but my miner doesn't seem to be getting any rewards or tasks. How do I know it's working?**  
A: First, check that your miner is **registered and running**: use `polaris status` to ensure the process is active and heartbeats are being sent. The Polaris logs (`polaris logs`) will show periodic heartbeat messages or any incoming requests from validators. In the early stages, you might not "see" tasks running because the system could be in a bootstrap phase or tasks are intermittent. Keep in mind that rewards are not instantaneous – they are distributed over blocks. It might take some time (minutes to hours) for your miner to accumulate noticeable TAO, especially if the network is large and your hardware is average. To verify, you can use the Bittensor CLI or taostats to check your wallet's stake and rank on subnet 49. For example, `btcli overview --subnet 49 --wallet <yourwallet>` (if using bittensor-cli) can show your stake and other stats. If after some time you see no activity: ensure your **ports are open** (validators might be unable to reach you if, say, port 8000 is blocked – check firewall settings and port forwarding) and that your `.env` has the correct public IP. One common oversight is that the machine's IP changed (for instance, if on DHCP) – you may need to update the HOST in `.env` and restart Polaris. Another tip: try the **testnet** first (netuid 12) where it's easier to see if you can get in, as it's less competitive, then switch to mainnet.

**Q: My Polaris node fails to start or crashes after a while. What can I do?**  
A: Here are a few troubleshooting steps:  
- **Check the logs:** Run `polaris logs` to get details on why it stopped. The logs might show Python exceptions, network errors, etc. If you see messages about failing to connect or bind to a port, it could be a networking issue (port in use or blocked). If you see errors about missing modules or similar, the installation might have had an issue – consider re-running the installer's **Reinstall** option to fix any dependency problem.  
- **Ensure correct Python version:** Polaris is tested with Python 3.10. If you have multiple Python versions, the virtual environment might have been set up with an older version that has issues. The installer usually handles this, but if in doubt, manually create a Python 3.10 venv and install Polaris there.  
- **Wallet issues:** If the service fails around the time of registration (for example, complaining about keys), you might need to generate a new wallet as mentioned above.  
- **Update and Reinstall:** Polaris Manager offers a **Reinstall** option which performs a clean setup while preserving your config. This can resolve issues by resetting the environment to a known good state. Make sure to stop any running Polaris processes before reinstalling.  
- **WSL specifics:** If running on Windows WSL, make sure you launch Polaris from within the WSL Ubuntu environment (not from Windows directly). Also, WSL networking can sometimes reset after a Windows reboot – verify that your WSL instance still has Docker running and the IP has not changed. Running `polaris status` in WSL will confirm if it's active or not.  
- **Hardware stability:** For crashes that occur after running for some time, consider hardware factors – an overheating GPU can cause the miner process (or even the system) to crash. Check GPU temperatures and consider underclocking slightly if it's running too hot. Also ensure your system isn't running out of memory (monitor RAM and swap usage).

**Q: How do I update or uninstall Polaris if needed?**  
A: Updating is straightforward – use the CLI command `polaris update subnet` to fetch and apply the latest code update. This will likely stop your node, update the repository (via git pull or pip update), and then you can restart the node. It's good practice to update periodically, especially if the project announces new releases. For uninstalling, if you used the installer script, there is an **Uninstall Polaris** option in the Polaris Manager menu. This will stop services and attempt to remove all Polaris components (it won't remove Docker or system packages like Python, but will remove the Polaris-specific files and environments. If you installed manually, you can stop the services (`polaris stop`), then simply delete the repository/venv folder. Also, you might want to remove the entry from your PATH (if any) and any systemd service that was created. The uninstall script should handle most of that. Remember to also **deregister your miner** if you no longer want it on the network – currently, Bittensor will automatically drop miners that stop responding after a certain period, but to be graceful you might use a `btcli unregister` command if one exists (this might be advanced and not necessary; simply stopping the node is fine in practice).

**Q: Where can I learn more or get support for Polaris?**  
A: Beyond this README, you can refer to the official Polaris documentation (if provided by Big Idea Africa or the maintainers) and the broader Bittensor documentation for subnets. Bittensor's community forums, Discord, and GitHub discussions are valuable resources. If you encounter issues specific to Polaris, check if the repository has a **Discussions** or **Issues** section and post there. For general questions about how subnets work, the Bittensor docs site has an *Understanding Subnets* section. Since Polaris is a community-driven subnet (with Big Idea Africa leading it), they might have announcements or Medium articles out – keep an eye on any links or references in the repo for external resources. Lastly, engaging with other subnet miners can be very insightful; many share tips on hardware tuning, best providers to use for cloud GPUs, etc. The field is evolving quickly, and staying connected will help you remain competitive and informed.

## Support and Resources

- Official Polaris documentation
- Bittensor community forums and Discord
- GitHub Discussions and Issues
- Community channels for hardware and provider recommendations

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute Polaris's code in accordance with the MIT terms.

---

*Polaris Subnet is an open-source initiative by Big Idea Africa, aiming to democratize access to AI compute. By participating in Polaris, you're contributing to a more decentralized and accessible future for AI development. Happy mining!* 🚀

## Recent Updates

### Public Key Authentication is Now Default

Polaris now uses SSH key-based authentication by default:

- No more password prompts during installation
- Automatic exchange of SSH public keys during registration
- Enhanced security through public key cryptography
- Simplified onboarding process for new miners

This change eliminates the need to transmit or store passwords, providing a more secure and streamlined experience for all users.

## Getting Started

To get started with Polaris, follow these steps:

1. Register your compute resources using the `polaris register` command
2. Start the Polaris services with `polaris start`
3. Monitor your resources with `polaris status`

For detailed instructions, refer to our documentation.

## License

Copyright © 2024 Polaris
