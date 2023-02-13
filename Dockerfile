
# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04
RUN rm -f /etc/apt/sources.list.d/*.list
WORKDIR /app
ARG DEBIAN_FRONTEND=noninteractive

# INSTALL APT PACKAGES
RUN apt update && apt upgrade -y
RUN apt install -y curl sudo nano git htop netcat wget unzip python3-dev python3-pip tmux apt-utils cmake build-essential

## Upgrade pip
RUN pip3 install --upgrade pip
RUN apt install -y protobuf-compiler
RUN apt install -y curl
COPY ./scripts /app/scripts

# PYTHON LAND
# INSTALL PACKAGES BEFORE COMMUNE (This is a hack to get around the bittensor install)
RUN pip install bittensor

# INSTALL CUBIT
RUN git clone https://github.com/opentensor/cubit.git /cubit
RUN  pip install -e /cubit

# BITTENSOR FIXES FOR NOW
# RUN pip install --upgrade substrate-interface
RUN pip install --upgrade torch


# INSTALL COMMUNE
COPY ./commune /app/commune
COPY ./requirements.txt /app/requirements.txt
COPY ./setup.py /app/setup.py
COPY ./README.md /app/README.md

# INSTALL PACKAGES AFTER COMMUNE
RUN pip install -e .
RUN pip install openai
RUN pip install google-search-results
RUN pip install wikipedia
RUN pip install pytest
RUN pip install jupyterlab
RUN pip install accelerate
RUN pip install nvidia-ml-py3

# INSTALL NPM ENV
RUN ./scripts/install_npm_env.sh
