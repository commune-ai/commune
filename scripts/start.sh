
PWD=$(pwd)
NAME=$(basename $PWD); 
REPO_PATH=$(pwd) ; DOCKER_REPO_PATH=/$IMAGE_NAME
CACHE_PATH=~/.$NAME ; DOCKER_CACHE_PATH=/root/.$IMAGE_NAME
SHM_SIZE=4g

CONTAINER_PARAMS=" -d \
  --name $NAME \
  --shm-size $SHM_SIZE \
  -v $REPO_PATH:$DOCKER_REPO_PATH \
  -v $CACHE_PATH:$DOCKER_CACHE_PATH \
  --network=host \
  --restart unless-stopped \
  --privileged \
  $NAME"

CONTAINER_EXISTS=$(docker ps -q -f name=$NAME)  
if [ $CONTAINER_EXISTS ]; then
  echo "STOPPING CONTAINER $NAME"
  docker kill $NAME
  CONTAINER_ID=$(docker ps -aq -f name=$NAME)
  docker rm $NAME
fi
eval docker run $CONTAINER_PARAMS