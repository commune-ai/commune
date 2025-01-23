
REPO_PATH=$(pwd) ;  
NAME=$(basename $REPO_PATH)
if [ -z $1 ]; then
  CONTAINER_NAME=$NAME
else
  CONTAINER_NAME=$1
fi
if [ $(docker ps -q -f name=$CONTAINER_NAME) ]; then
  ./scripts/stop.sh
fi

echo "STARTING($CONTAINER_NAME)"

eval "docker run -d \
  --name $CONTAINER_NAME \
  --network=host --restart unless-stopped --privileged --shm-size 4g \
  -v $REPO_PATH:/$NAME -v ~/.$NAME:/root/.$NAME \
  $NAME"

