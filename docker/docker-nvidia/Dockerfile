FROM ubuntu:22.04
FROM python:3.11


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
# RUN pip3 install -e .


# Configure Poetry
ENV POETRY_VERSION=1.8.2
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache

# Install poetry separated from system interpreter
RUN python3 -m venv $POETRY_VENV \
	&& $POETRY_VENV/bin/pip install -U pip setuptools \
	&& $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

# Add `poetry` to PATH
ENV PATH="${PATH}:${POETRY_VENV}/bin"

# Install dependencies
COPY ./pyproject.toml /commune/pyproject.toml
COPY ./poetry.lock /commune/poetry.lock 
RUN poetry install


# INTSALL NPM PACKAGES
RUN apt-get install -y nodejs npm
RUN npm install -g pm2
