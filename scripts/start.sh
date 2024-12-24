
NAME=$(basename $(pwd)); 
if [ $(docker ps -q -f name=$NAME) ]; then
  ./scripts/stop.sh
fi
  REPO_PATH=$(pwd) ; DOCKER_REPO_PATH=/$NAME
  CACHE_PATH=~/.$NAME ; DOCKER_CACHE_PATH=/root/.$NAME
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
  echo "Starting $NAME"
  eval docker run $CONTAINER_PARAMS