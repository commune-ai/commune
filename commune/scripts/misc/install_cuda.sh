#!/bin/bash

# Add NVIDIA package repository key
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2204-12-1-local/7fa2af80.pub
sudo apt-get update

# Install CUDA and its dependencies
sudo apt-get -y install cuda

# Install the latest NVIDIA driver
sudo ubuntu-drivers autoinstall

# Install the latest NVIDIA toolkit
sudo apt-get -y install nvidia-cuda-toolkit

# Install the latest CUDA toolkit
sudo apt-get -y install cuda-toolkit
# rm cuda
rm cuda-*