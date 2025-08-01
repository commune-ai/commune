FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# ESSENTIAL DEPENDENCIES
RUN apt-get update && apt-get install -y \
    curl git build-essential nano jq vim software-properties-common \
    python3 python3-pip python3-venv \
    npm gnupg2 ca-certificates apt-transport-https \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# PM2
RUN npm install -g pm2

# Rust via rustup (more modern than apt)
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# DOCKER
RUN apt-get update && apt-get install -y \
    docker.io \
    && systemctl enable docker \
    && usermod -aG docker root

# Install Docker Compose
RUN curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" -o /usr/local/bin/docker-compose && \
    chmod +x /usr/local/bin/docker-compose

# Workdir + Install App
WORKDIR /root/commune
COPY . .
RUN pip install -e .

# Default CMD (replace with ENTRYPOINT if needed)
CMD ["tail", "-f", "/dev/null"]
