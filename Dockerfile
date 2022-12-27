# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04

RUN rm -f /etc/apt/sources.list.d/*.list
WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt upgrade -y
RUN apt install -y curl sudo nano git htop netcat wget unzip python3-dev python3-pip tmux apt-utils cmake build-essential

## Upgrade pip
RUN pip3 install --upgrade pip
RUN apt install -y protobuf-compiler

# PYTHON LAND

# RUN python3 pip install https://github.com/opentensor/cubit/releases/download/v1.1.2/cubit-1.1.2-cp37-cp37m-linux_x86_64.whl
RUN python3 -m pip install -U "ray[default]"
RUN python3 -m pip install -U streamlit
RUN python3 -m pip install -U gradio
RUN python3 -m pip install -U plotly
RUN python3 -m pip install -U sentence-transformers
RUN python3 -m pip install -U datasets
RUN python3 -m pip install hub
RUN python3 -m pip install -U accelerate
RUN python3 -m pip install jupyterlab
RUN python3 -m pip install aiofiles
RUN python3 -m pip install web3


ADD ./bittensor /app/bittensor
RUN python3 -m pip install -e /app/bittensor

# BITTENSOR USES AN OLDER PROTOBUF, SO LETS OVERRIDE IT
RUN python3 -m pip install --upgrade protobuf

# BITTENSOR USES AN OLDER PROTOBUF, SO LETS OVERRIDE IT
RUN python3 -m pip install --upgrade torch

RUN pip install --upgrade substrate-interface
# This makes it compatible with streamlit
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python 

RUN alias python=python3
# ADD /usr/local/cuda/bin/nvcc /usr/local/cuda/bin/nvcc
# RUN python3 -m pip install deepspeed

# JAVASCRIPT LAND

# we add sprinkle of npm for hardhat smart contract tings
ENV NODE_VERSION=16.17.1
RUN apt install -y curl
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
ENV NVM_DIR=/root/.nvm
RUN . "$NVM_DIR/nvm.sh" && nvm install ${NODE_VERSION}
RUN . "$NVM_DIR/nvm.sh" && nvm use v${NODE_VERSION}
RUN . "$NVM_DIR/nvm.sh" && nvm alias default v${NODE_VERSION}
ENV PATH="/root/.nvm/versions/node/v${NODE_VERSION}/bin/:${PATH}"
RUN node --version
RUN npm --version
RUN npm install --save-dev hardhat
RUN npm install --save-dev @nomicfoundation/hardhat-toolbox
COPY hardhat.config.js .
RUN npx hardhat
RUN npm install @openzeppelin/contracts
RUN npm install @uniswap/v3-periphery
RUN npm install @uniswap/v2-periphery
# RUN npm install --global @ceramicnetwork/cli @glazed/cli

# RUST LAND


# Install cargo and Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
    apt-get install --no-install-recommends --assume-yes \
      protobuf-compiler

RUN rustup update nightly
RUN rustup target add wasm32-unknown-unknown --toolchain nightly
RUN apt-get install make
RUN apt-get install -y pkg-config

# CONTRACTS STUFF
RUN apt install binaryen
RUN apt-get install libssl-dev
RUN cargo install cargo-dylint dylint-link
RUN cargo install cargo-contract --force

RUN rustup component add rust-src --toolchain nightly-x86_64-unknown-linux-gnu


