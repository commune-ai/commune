sudo apt update
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
git clone https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed
cd docker
sudo docker build -t deepspeed .
sudo systemctl stop docker
sudo systemctl disable docker