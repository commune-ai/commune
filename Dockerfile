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

# INSTALL DOCKER
RUN apt-get install -y apt-transport-https ca-certificates curl gnupg2 software-properties-common
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
RUN add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
RUN apt-get install -y docker.io
# install make file 
RUN apt-get install -y make

# Create docker group and add user to it
RUN groupadd docker || true
RUN usermod -aG docker root

# EXPOSE DOCKER SOCKET
EXPOSE 2375

# MODULE ENVIRONMENT
WORKDIR /app
COPY . .
RUN pip install -e ./

ENTRYPOINT [ "tail", "-f", "/dev/null"]