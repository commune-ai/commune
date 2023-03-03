sudo apt-get purge docker-engine
sudo apt-get autoremove --purge docker-engine
sudo rm -rf /var/lib/docker
sudo apt-get install --reinstall docker-ce