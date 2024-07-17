# THE GENERAL CONTAINER FOR CONNECTING ALL THE ENVIRONMENTS 😈

FROM ubuntu:22.04

#SYSTEM
WORKDIR /app
ARG DEBIAN_FRONTEND=noninteractive
RUN usermod -s /bin/bash root
RUN apt-get update 

#RUST
RUN apt-get install curl nano build-essential cargo libstd-rust-dev -y

#JS 
RUN apt-get install -y nodejs npm
RUN npm install -g pm2 

#PYTHON
ENV PYTHONUNBUFFERED True 
RUN apt-get install python3-pip python3 python3-dev python-is-python3 -y
RUN python -m pip install --upgrade pip

# INSTALL COMMUNE FROM SOURCE
COPY ./commune /app/commune
COPY ./setup.py /app/setup.py
COPY ./requirements.txt /app/requirements.txt
COPY ./README.md /app/README.md
RUN pip install -e ./

# IMPORT EVERYTHING ELSE
COPY ./ /app

ENTRYPOINT [ "tail", "-f", "/dev/null"]