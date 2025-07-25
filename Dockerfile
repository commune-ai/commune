FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# Essentials
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

# Docker + Docker Compose
RUN install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    chmod a+r /etc/apt/keyrings/docker.gpg && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
    | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin && \
    groupadd docker || true && usermod -aG docker root

# Workdir + Install App
WORKDIR /root/commune
COPY . .
RUN pip install -e .

# Default CMD (replace with ENTRYPOINT if needed)
CMD ["tail", "-f", "/dev/null"]
