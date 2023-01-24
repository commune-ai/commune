
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




# NPM LAND

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

RUN npm i -g pm2

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


# COPY SRC AND INSTALL AS A PACKAGE

COPY ./commune /app/commune
COPY ./scripts /app/scripts

COPY ./subtensor /app/subtensor
COPY ./requirements.txt /app/requirements.txt
COPY ./setup.py /app/setup.py
COPY ./Dockerfile /app/Dockerfile
COPY ./start.sh /app/start.sh
COPY ./README.md /app/README.md


# PYTHON LAND

# COPY ./bittensor /app/bittensor
# RUN pip install -e /app/bittensor
# RUN pip install https://github.com/opentensor/cubit/releases/download/v1.1.2/cubit-1.1.2-cp310-cp310-linux_x86_64.whl

# RUN pip install -U "ray[default]"
# RUN pip install -U streamlit
# RUN pip install -U plotly
# RUN pip install -U datasets
# RUN pip install hub
# RUN pip install -U accelerate
# RUN pip install jupyterlab
# RUN pip install aiofiles
# RUN pip install web3
# RUN pip install --upgrade protobuf
# RUN export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python 
# RUN pip install accelerate
# RUN pip install --upgrade torch
# RUN pip install nvidia-ml-py3

# COPY ./diffusers /app/diffusers
# RUN pip install -e /app/diffusers

RUN pip install -e .