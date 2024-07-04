#/bin/bash
if [ -d "/commune" ] 
then
    git clone https://github.com/commune-ai/commune.git
else
    echo "Commune installed"
fi

cd commune

git pull

sudo docker-compose up -d