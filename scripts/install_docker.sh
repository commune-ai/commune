#!/bin/bash
# Install Docker
apt-get install -y docker-ce docker-ce-cli containerd.io

# Verify that Docker has been installed correctly
groupadd docker
usermod -aG docker $USER
chmod 666 /var/run/docker.sock


