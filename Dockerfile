# THIS CONTAINER IS A NPM, RUST, PYTHON, DOCKER INTO ONE CONTAINER
# THIS GENERAL CONTAINER IS THE CORE OF COMMUNE, USE IT AS YOU WISH AT YOUR OWN RISK

FROM ubuntu:22.04

#SYSTEM ENVIRONMENT
ARG DEBIAN_FRONTEND=noninteractive
RUN usermod -s /bin/bash root
RUN apt-get update 

#RUST ENVIRONMENT
RUN apt-get install curl nano build-essential cargo libstd-rust-dev make -y

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
RUN groupadd docker || true
RUN usermod -aG docker root
EXPOSE 2375

# MODULE ENVIRONMENT
WORKDIR /app
COPY . .
RUN pip install -e ./

# ENTRYPOINT (default to container running in the background in case of no command)
ENTRYPOINT [ "tail", "-f", "/dev/null"]