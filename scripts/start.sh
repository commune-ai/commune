#

# if commune does not exist build it

# if docker is not running, start it

NAME=commune
PWD=~/$NAME
IMAGE_NAME=$NAME
CONTAINER_NAME=$NAME
ENABLE_DOCKER=true
COMMUNE_CACHE_PATH=.commune

IMAGE_PATH=./
SHM_SIZE=4g
# RESOLVE PORT RANGE
START_PORT=50050
END_PORT=50100

CONTAINER_EXISTS=$(docker ps -q -f name=$CONTAINER_NAME)  
if [ $CONTAINER_EXISTS ]; then
  echo "STOPPING CONTAINER $CONTAINER_NAME"
  docker kill $CONTAINER_NAME
  CONTAINER_ID=$(docker ps -aq -f name=$CONTAINER_NAME)
  docker rm $CONTAINER_NAME
fi

CMD_STR="docker run -d \
  --name $CONTAINER_NAME \
  --shm-size $SHM_SIZE \
  -v ~/.$NAME:/root/.$NAME \
  -v $PWD:/$NAME \
  -p $START_PORT-$END_PORT:$START_PORT-$END_PORT \
  --restart unless-stopped \
  --privileged
  $CONTAINER_NAME
"


eval $CMD_STR