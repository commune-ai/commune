PWD=$(pwd)
NAME=$(basename $PWD)
CONTAINER_NAME=$NAME
SHM_SIZE=4g
docker exec -it $CONTAINER_NAME /bin/bash