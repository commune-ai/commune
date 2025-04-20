# THIS CONTAINER IS A NPM, RUST, PYTHON, DOCKER INTO ONE CONTAINER
# THIS GENERAL CONTAINER IS THE CORE OF COMMUNE, USE IT AS YOU WISH AT YOUR OWN RISK

FROM ubuntu:22.04

#SYSTEM ENVIRONMENT
ARG DEBIAN_FRONTEND=noninteractive
RUN usermod -s /bin/bash root
RUN apt-get update 

# RUST 
RUN apt-get install -y nano build-essential cargo libstd-rust-dev git

# PYTHON
RUN apt-get install -y python3 python3-pip python3-venv
RUN python3 --version

# NPM
RUN apt-get install -y npm 
RUN npm install -g pm2

# DOCKER
RUN apt-get install -y apt-transport-https ca-certificates curl gnupg2 software-properties-common
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
RUN add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
RUN apt-get install -y docker.io
RUN groupadd docker || true
RUN usermod -aG docker root
EXPOSE 2375 

# APP # COPY YOUR APP HERE
WORKDIR /app
COPY . .
RUN pip install -e ./ 

# ENTRYPOINT (default to container running in the background in case of no command)
ENTRYPOINT [ "tail", "-f", "/dev/null"]