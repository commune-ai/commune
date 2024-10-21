
PWD=$(pwd)
NAME=$(basename $PWD)
CONTAINER_NAME=$NAME
SHM_SIZE=4g
REPO_PATH=$(pwd)
DOCKER_REPO_PATH=/$NAME
CACHE_PATH=~/.$NAME
DOCKER_CACHE_PATH=/root/.$NAME

CONTAINER_EXISTS=$(docker ps -q -f name=$NAME)  
if [ $CONTAINER_EXISTS ]; then
  echo "STOPPING CONTAINER $NAME"
  docker kill $NAME
  CONTAINER_ID=$(docker ps -aq -f name=$NAME)
  docker rm $NAME
fi

CMD_STR="docker run -d \
  --name $CONTAINER_NAME \
  --shm-size $SHM_SIZE \
  -v $CACHE_PATH:$DOCKER_CACHE_PATH \
  -v $REPO_PATH:$DOCKER_REPO_PATH \
  --network=host \
  --restart unless-stopped \
  --privileged
  $CONTAINER_NAME
"

eval $CMD_STR