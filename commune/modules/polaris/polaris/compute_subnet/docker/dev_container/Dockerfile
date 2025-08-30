FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openssh-server \
    git \
    curl \
    vim \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# SSH server configuration
RUN mkdir /var/run/sshd && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config && \
    echo 'AuthorizedKeysFile %h/.ssh/authorized_keys' >> /etc/ssh/sshd_config && \
    echo 'PermitRootLogin no' >> /etc/ssh/sshd_config

# Create dev user with sudo access
RUN useradd -m -d /home/devuser -s /bin/bash devuser && \
    adduser devuser sudo && \
    echo "devuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Setup SSH directory and workspace
RUN mkdir -p /home/devuser/.ssh /workspace && \
    chmod 700 /home/devuser/.ssh && \
    chown -R devuser:devuser /home/devuser/.ssh /workspace

# Add development tools and configurations
COPY configs/.bashrc /home/devuser/.bashrc
COPY configs/.vimrc /home/devuser/.vimrc
RUN chown devuser:devuser /home/devuser/.bashrc /home/devuser/.vimrc

# Install Python development tools
RUN pip install --no-cache-dir \
    ipython \
    pylint \
    black \
    pytest

# Set working directory
WORKDIR /workspace

# Expose SSH port
EXPOSE 22

# Copy and setup entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Switch to non-root user for better security
USER devuser

# Start SSH server
ENTRYPOINT ["/entrypoint.sh"]