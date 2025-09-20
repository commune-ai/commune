#!/bin/bash

# Build script for the Substrate node with module registry pallet

set -e

echo "Building Substrate node with module registry pallet..."

# Check if cargo is installed
if ! command -v cargo &> /dev/null; then
    echo "Cargo not found! Please install Rust and Cargo first."
    echo "Visit: https://rustup.rs/"
    exit 1
fi

# Install required components
echo "Installing required Rust components..."
rustup default stable
rustup update
rustup target add wasm32-unknown-unknown

# Build the node
echo "Building the node..."
cargo build --release

# Build runtime
echo "Building runtime..."
cargo build --release -p node-runtime

echo "Build complete! Binary located at: target/release/substrate-node"
echo "Run with: ./target/release/substrate-node --dev"