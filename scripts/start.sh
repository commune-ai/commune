
REPO_PATH=$(pwd) ;  
NAME=$(basename $REPO_PATH)

if [ -z $1 ]; then
  CONTAINER_NAME=$NAME
else
  CONTAINER_NAME=$1
fi
echo "STARING($CONTAINER_NAME)"
if [ $(docker ps -q -f name=$CONTAINER_NAME) ]; then
  ./scripts/stop.sh
fi
  eval "docker run -d \
    --name $CONTAINER_NAME \
    -v $REPO_PATH:/app \
    -v ~/.$NAME:/root/.$NAME \
    --network=host \
    --restart unless-stopped \
    --privileged \
    --shm-size 4g \
    $NAME"

