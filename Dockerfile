# THE GENERAL CONTAINER FOR CONNECTING ALL THE ENVIRONMENTS 😈
FROM ubuntu:22.04

#SYSTEM
ARG DEBIAN_FRONTEND=noninteractive
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN apt-get update 

#RUST
RUN apt-get install curl nano build-essential cargo libstd-rust-dev -y

#JS 
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
RUN apt-get install -y nodejs
RUN npm install -g pm2

#PYTHON
RUN apt-get install python3 python3-pip python3-venv -y
WORKDIR /app

# make /commune equal to the current directory
COPY . /commune
RUN pip install -e /commune

# ENTRYPOINT 
ENTRYPOINT [ "tail", "-f", "/dev/null"]