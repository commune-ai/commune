#/bin/bash
git clone https://github.com/commune-ai/commune.git
cd commune

sudo docker exec -it commune bash -c "btcli list"