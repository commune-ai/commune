FROM ubuntu:22.04

# Set the working directory in the container
WORKDIR /app
# Set environment variables

ENV PYTHONUNBUFFERED True  
ARG DEBIAN_FRONTEND=noninteractive
RUN usermod -s /bin/bash root
RUN apt-get update 

# INSTALL RUST ENV
RUN apt-get install curl nano build-essential cargo libstd-rust-dev -y

# INSTALL NODEJS ENV
RUN apt-get install -y nodejs npm
# install pm2 for process management (currently required for commune)
RUN npm install -g pm2 

# INSTALL PYTHON ENV
RUN apt-get install python3-pip python3 python3-dev python-is-python3 -y
RUN python -m pip install --upgrade pip

# --- INSTALL POETRY ---
# RUN pip install poetry
# COPY ./pyproject.toml /app/pyproject.toml
# COPY ./poetry.lock /app/poetry.lock
# COPY ./ /app
# RUN poetry install
# INSTALL THE COMMUNE REPO FROM SOURCE SO IT WORKS OUT OF THE BOX WHEN YOU ENTER
COPY ./ /app/commune
RUN pip install -e ./

ENTRYPOINT [ "tail", "-f", "/dev/null"]