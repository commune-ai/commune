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


# Copy the contents of the local directory "../" to the /workspace directory in the container
COPY ./ /workspace

### For Docker in Docker ###
# Install prerequisites
RUN apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common

RUN pip install -q nvidia-smi

# # Add the Docker repository to APT sources
# RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add - \
#     && add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

# # Install Docker
# RUN apt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io



# 50/50 on removing this.
#RUN apt-get update

# Set the working directory to /workspace
WORKDIR /workspace

# RUN git clone https://github.com/commune-ai/commune.git
# # now go to commune folder/repo
# WORKDIR /workspace/commune

# install python libraries for commune
RUN pip install -r requirements.txt

# Work around for bug that added a "x" at the end of the setup.py file
#RUN sed -i '$ d' setup.py

################# missing items (PM2) ################
#ENV NODE_VERSION 21.7.0



# RUN wget -q -O setup.deb.sh https://raw.githubusercontent.com/Unitech/pm2/master/packager/setup.deb.sh
# RUN bash setup.deb.sh
#RUN curl -sL https://raw.githubusercontent.com/Unitech/pm2/master/packager/setup.deb.sh | sudo -E bash -
## Alternate approach

RUN apt-get install build-essential software-properties-common
RUN apt-get install nodejs npm -y
RUN npm install pm2 -g

################################################################

# install Commune
RUN pip install -e .


# Clean up
#BREAKS WAY TOO MUCH IN CODE SERVER## RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*