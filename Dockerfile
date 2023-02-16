
# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04
WORKDIR /app
ARG DEBIAN_FRONTEND=noninteractive
# INSTALL APT PACKAGES
RUN apt update && apt upgrade -y
RUN apt install -y curl sudo nano git htop netcat wget unzip python3-dev python3-pip tmux apt-utils cmake build-essential protobuf-compiler

COPY ./commune /app/commune
COPY ./requirements.txt /app/requirements.txt
COPY ./setup.py /app/setup.py
COPY ./README.md /app/README.md

COPY ./scripts /app/scripts
# INSTALL PYTHON ENV
RUN ./scripts/install_python_env.sh
# INSTALL NPM ENV
RUN ./scripts/install_npm_env.sh
