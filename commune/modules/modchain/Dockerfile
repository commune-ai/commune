FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    clang \
    curl \
    libssl-dev \
    llvm \
    libudev-dev \
    make \
    protobuf-compiler \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Add wasm target
RUN rustup target add wasm32-unknown-unknown

# Clone and build Substrate node template
WORKDIR /app
cargo build --release

# Expose ports
EXPOSE 9944 9933 30333

# Set the startup command
CMD ["./target/release/node-template", "--dev", "--ws-external", "--rpc-external", "--rpc-cors=all"]