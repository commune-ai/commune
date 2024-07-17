#!/usr/bin/env bash
# This script is meant to be run on Unix/Linux based systems
set -e

# Install cargo and Rust
curl https://sh.rustup.rs -sSf | sh -s -- -y

export PATH="$HOME/.cargo/bin:$PATH"
. "$HOME/.cargo/env"
echo "*** Initialized WASM build environment with Rust 1.68.1"

# Install cargo and Rust nightly

rustup install nightly-2023-01-01
rustup override set nightly-2023-01-01
rustup target add wasm32-unknown-unknown

# Install dependencies
apt-get update
# Use the "yes" command to automatically provide 'Y' as the answer
yes | apt-get install libclang-dev
yes | apt-get install protobuf-compiler

