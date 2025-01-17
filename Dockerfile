# GENERAL CONTAINER, REPORTING FOR OPENNESS SIR
FROM ubuntu:22.04

#SYSTEM
ARG DEBIAN_FRONTEND=noninteractive
RUN usermod -s /bin/bash root
RUN apt-get update 

#RUST ENVIRONMENT
RUN apt-get install curl nano build-essential cargo libstd-rust-dev -y

#NPM ENVIRONMENT
RUN apt-get install -y nodejs npm
RUN npm install -g pm2 

#PYTHON ENVIRONMENT
RUN apt-get install python3 python3-pip python3-venv -y
WORKDIR /app

# make /commune equal to the current directory
COPY . .
RUN pip install -e ./commune

# ENTRYPOINT 
ENTRYPOINT [ "tail", "-f", "/dev/null"]