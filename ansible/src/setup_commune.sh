#/bin/bash
if [ -d "commune" ] 
then
    
    echo "Commune installed"

else
    git clone https://github.com/commune-ai/commune.git
fi

original_path=${PWD}
cd commune
sudo git stash
sudo git clean -d -f
sudo git pull

sudo docker compose up -d

cd ${original_path}
