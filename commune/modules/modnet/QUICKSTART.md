# Quick Start Guide - How to Run Mod-Net

## Prerequisites
1. Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
2. Install Python 3.10+
3. Install IPFS: https://docs.ipfs.tech/install/
4. Install Substrate dependencies: `curl https://getsubstrate.io -sSf | bash -s -- --fast`

## Step 1: Clone and Build
```bash
# Clone the repository
git clone --recursive https://github.com/Bakobiibizo/mod-net-modules.git
cd mod-net-modules/modules

# Build the node (this takes a while)
cargo build --release
```

## Step 2: Set up Python Environment
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt
```

## Step 3: Start IPFS
```bash
# In a new terminal, start IPFS daemon
ipfs daemon
```

## Step 4: Run the Blockchain Node
```bash
# Start the development chain
./target/release/solochain-template-node --dev
```

## Optional: Run Development Tools
```bash
# Interactive development tools menu
./run_checks.sh

# Or run specific tools:
./run_checks.sh all          # Run everything
./run_checks.sh formatters   # Run all formatters
./run_checks.sh tests        # Run all tests
./run_checks.sh rust         # Run all Rust tools
./run_checks.sh python       # Run all Python tools
```

## Access the Node
- RPC endpoint: ws://localhost:9944
- Polkadot.js UI: https://polkadot.js.org/apps/?rpc=ws://localhost:9944

## Common Commands
- Purge chain data: `./target/release/solochain-template-node purge-chain --dev`
- Run with debug logs: `RUST_BACKTRACE=1 ./target/release/solochain-template-node -ldebug --dev`
- Persist chain state: `./target/release/solochain-template-node --dev --base-path ./my-chain-state/`

## Docker Alternative
```bash
# Build Docker image
docker build -t modnet-node .

# Run container
docker run -p 30333:30333 -p 9933:9933 -p 9944:9944 -p 9615:9615 modnet-node --dev
```