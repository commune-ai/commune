# Development Guide

This guide will help you set up a development environment for the Mod-Net Module Registry.

## Prerequisites

- Rust 1.70+ (install via [rustup](https://rustup.rs/))
- Python 3.10+
- IPFS 0.15.0+
- Node.js 18+ (for frontend development)
- Git
- UV package manager (`pip install uv`)

## Environment Setup

### 1. Clone the Repository

```bash
git clone --recursive https://github.com/Bakobiibizo/mod-net-modules.git
cd mod-net-modules/modules
```

### 2. Set Up Rust

```bash
# Install Rust toolchain
rustup toolchain install stable
rustup target add wasm32-unknown-unknown --toolchain stable

# Install wasm-opt for optimized builds
cargo install wasm-opt
```

### 3. Set Up Python Environment

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
uv pip install -r requirements-dev.txt
```

### 4. Install IPFS

```bash
# Install IPFS
ipfs init
ipfs config --json API.HTTPHeaders.Access-Control-Allow-Origin '["*"]'
ipfs config --json API.HTTPHeaders.Access-Control-Allow-Methods '["PUT", "GET", "POST"]'
```

## Building the Project

### Build Substrate Node

```bash
# Debug build
cargo build

# Release build (recommended for development)
cargo build --release
```

### Build Python Package

```bash
# Install in development mode
uv pip install -e .

# Or build a wheel
python -m build
```

## Running Tests

### Rust Tests

```bash
# Run all tests
cargo test --all

# Run specific test
cargo test -p pallet-module-registry
```

### Python Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_module_registry.py -v
```

### Integration Tests

```bash
# Start local development node
./target/release/node-template --dev

# In a separate terminal, run integration tests
pytest tests/integration/
```

## Development Workflow

### 1. Start Local Development Node

```bash
# Start a development node with detailed logging
RUST_LOG=debug,txpool=debug \
  ./target/release/node-template \
  --dev \
  --tmp \
  --rpc-external \
  --rpc-methods=unsafe \
  --rpc-cors=all
```

### 2. Start IPFS Daemon

```bash
# Run in a separate terminal
ipfs daemon
```

### 3. Interact with the Node

#### Using Python Client

```python
from modnet import ModuleRegistryClient

# Connect to local node
client = ModuleRegistryClient("ws://127.0.0.1:9944")

# Register a module
await client.register_module("0x...", "Qm...")

# Get module info
module = await client.get_module("0x...")
print(module)
```

#### Using Polkadot.js Apps

1. Open https://polkadot.js.org/apps/
2. Go to Settings > Developer
3. Add custom types (if needed)
4. Connect to your local node (ws://127.0.0.1:9944)

## Code Style

### Rust

```bash
# Format code
cargo fmt --all

# Lint code
cargo clippy --all-targets -- -D warnings
```

### Python

```bash
# Format code
black .
isort . --profile black

# Lint code
ruff check .
mypy .
```

## Debugging

### Rust Debugging

1. Use `println!` for quick debugging
2. For more complex debugging, use the `log` crate
3. Enable debug logs:
   ```bash
   RUST_LOG=debug ./target/release/node-template --dev
   ```

### Python Debugging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your code here
```

## Performance Profiling

### Rust Profiling

```bash
# Build with debug symbols
cargo build --release --debuginfo=2

# Profile with perf
perf record --call-graph dwarf ./target/release/node-template --dev
perf report
```

### Python Profiling

```bash
python -m cProfile -o profile.prof your_script.py
snakeviz profile.prof
```

## Documentation

### Building Documentation

```bash
# Rust docs
cargo doc --open

# Python docs
cd commune-ipfs
make docs
```

### Writing Documentation

- Use Markdown for all documentation
- Follow Google-style docstrings for Python
- Use Rust's documentation comments (`///` and `//!`)
- Keep documentation up-to-date with code changes

## Git Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "feat(module): add new feature"
   ```

3. Push your branch:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a Pull Request on GitHub

## Troubleshooting

### Common Issues

1. **Build Failures**
   - Run `cargo clean` and rebuild
   - Ensure all submodules are updated: `git submodule update --init --recursive`

2. **Node Not Starting**
   - Delete chain data: `rm -rf /tmp/substrate*`
   - Check for port conflicts

3. **Python Import Errors**
   - Ensure virtual environment is activated
   - Run `uv pip install -e .` in the project root

4. **IPFS Connection Issues**
   - Ensure IPFS daemon is running
   - Check API port (default: 5001)

## Getting Help

- Check the [FAQ](./FAQ.md)
- Open an [issue](https://github.com/Bakobiibizo/mod-net-modules/issues)
- Join our [Discord/Slack] channel (link TBD)
