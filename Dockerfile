# GENERAL CONTAINER, REPORTING FOR OPENNESS SIR
FROM ubuntu:22.04

#SYSTEM ENVIRONMENT
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

# MODULE ENVIRONMENT (ANYTHING YOU WANT TO INSTALL, DO IT HERE)
COPY . /commune
RUN pip install -e /commune
WORKDIR /app
# ENTRYPOINT 
ENTRYPOINT [ "tail", "-f", "/dev/null"]