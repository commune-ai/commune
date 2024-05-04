#FROM ubuntu:22.04

#probably better:
FROM python:3.12-slim-bookworm


ENV PYTHONUNBUFFERED True
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /commune

RUN usermod -s /bin/bash root

#RUN apt-get update && apt-get upgrade -y
RUN apt-get update
RUN apt-get install curl nano python3 python3-dev python-is-python3 build-essential cargo libstd-rust-dev -y
RUN python -m pip install --upgrade pip
RUN pip install setuptools wheel 

#RUN apt-get update && \
#    apt-get install -y curl nano python3 python3-dev python3-pip build-essential cmake apt-utils protobuf-compiler

#RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

COPY ./requirements.txt /commune/requirements.txt
RUN pip install -r requirements.txt

RUN apt-get install -y nodejs npm
RUN npm install -g pm2
COPY . /commune
RUN pip install -e ./
ENTRYPOINT [ "tail", "-f", "/dev/null"]