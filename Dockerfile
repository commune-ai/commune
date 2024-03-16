# Use the specified base image
FROM linuxserver/code-server:latest

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Install nano, python3, and pip
RUN apt-get update && \
    apt-get install -y nano python3 python-is-python3 python3-pip iputils-ping git git-lfs openssh-server

USER root

RUN ssh-keygen -A

RUN (crontab -l 2>/dev/null; echo "@reboot /usr/sbin/sshd -D") | crontab -
RUN (crontab -l 2>/dev/null; echo "@reboot /etc/init.d/ssh start") | crontab -

RUN usermod -s /bin/bash abc
RUN usermod -s /bin/bash root

### For Docker in Docker ###
# Install prerequisites
RUN apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common

RUN pip install -q nvidia-smi

# Set the working directory to /workspace
WORKDIR /commune

# install npm and pm2 (for process management)
RUN apt-get install build-essential software-properties-common
RUN apt-get install nodejs npm -y
RUN npm install pm2 -g

# Copy the contents of the local directory "../" to the /workspace directory in the container
# install Commune

# install python libraries for commune
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY ./ /commune
RUN pip install -e .