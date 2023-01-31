#/bin/bash

# Install cargo and Rust
curl https://sh.rustup.rs -sSf | sh -s -- -y
PATH="/root/.cargo/bin:${PATH}"

apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
    apt-get install --no-install-recommends --assume-yes \
      protobuf-compiler

rustup update nightly
rustup target add wasm32-unknown-unknown --toolchain nightly
apt-get install make
apt-get install -y pkg-config

# CONTRACTS STUFF
apt install binaryen
apt-get install libssl-dev
cargo install cargo-dylint dylint-link
cargo install cargo-contract --force

rustup component add rust-src --toolchain nightly-x86_64-unknown-linux-gnu