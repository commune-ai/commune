# THE GENERAL CONTAINER FOR CONNECTING ALL THE ENVIRONMENTS 😈
FROM ubuntu:22.04
#SYSTEM
ARG DEBIAN_FRONTEND=noninteractive
RUN usermod -s /bin/bash root
RUN apt-get update 
#RUST
RUN apt-get install curl nano build-essential cargo libstd-rust-dev -y
#NPM/JS 
RUN apt-get install -y nodejs npm
RUN npm install -g pm2 
#PYTHON
RUN apt-get install python3 python3-pip python3-venv -y
COPY . /commune
# RUN git clone -b main https://github.com/commune-ai/commune.git /commune 
RUN pip install -e /commune
WORKDIR /app
# TODO DOCKERIZE THE ENTRYPOINT
ENTRYPOINT [ "tail", "-f", "/dev/null"]