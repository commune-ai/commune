groupadd docker
usermod -aG docker $USER
chmod 666 /var/run/docker.sock