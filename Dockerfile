
# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04
WORKDIR /commune
RUN rm -f /etc/apt/sources.list.d/*.list
ARG DEBIAN_FRONTEND=noninteractive
# INSTALL APT PACKAGES
RUN apt update && apt upgrade -y
RUN apt install -y curl sudo nano git htop netcat wget unzip python3-dev python3-pip tmux apt-utils cmake build-essential protobuf-compiler


# VOLUMES FOR MODULES
COPY ./commune /commune/commune
COPY ./requirements.txt /commune/requirements.txt
COPY ./setup.py /commune/setup.py
COPY ./README.md /commune/README.md
COPY ./bin /commune/bin

# MAKE A SCRIPTS DIRECTORY
RUN mkdir /commune/scripts
# INSTALL PYTHON ENV
COPY ./scripts/install_python_env.sh /commune/scripts/install_python_env.sh
RUN ./scripts/install_python_env.sh
# INSTALL NPM ENV
COPY ./scripts/install_npm_env.sh /commune/scripts/install_npm_env.sh
RUN ./scripts/install_npm_env.sh
# INSTALL RUST ENV
COPY ./scripts/install_rust_env.sh /commune/scripts/install_rust_env.sh
RUN ./scripts/install_rust_env.sh

# INSTALL COMMUNE
RUN pip3 install -e .

# BUILD SUBDSPACE
COPY ./subspace /commune/subspace
# RUN cd ./subspace && cargo build --release

