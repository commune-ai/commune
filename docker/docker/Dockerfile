FROM ubuntu:22.04

WORKDIR /commune
RUN rm -f /etc/apt/sources.list.d/*.list
ARG DEBIAN_FRONTEND=noninteractive
# INSTALL APT PACKAGES (NOT PYTHON)
RUN apt update
RUN apt upgrade -y
RUN apt install -y curl sudo git htop wget unzip tmux apt-utils cmake build-essential protobuf-compiler
#  INSTALL PYTHON PACKAGES
RUN apt install -y python3-dev python3-pip
COPY ./commune /commune/commune
COPY ./requirements.txt /commune/requirements.txt
COPY ./setup.py /commune/setup.py
COPY ./README.md /commune/README.md
COPY ./bin /commune/bin
RUN pip3 install -e .


# INTSALL NPM PACKAGES
RUN apt-get install -y nodejs npm
RUN npm install -g pm2
