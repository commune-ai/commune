
PWD=$(pwd)
NAME=$(basename $PWD)
PWD=~/$NAME
CONTAINER_NAME=$NAME
SHM_SIZE=4g
# RESOLVE PORT RANGE

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
  -v ~/.$NAME:/root/.$NAME \
  -v ~/$NAME:/$NAME \
  --network=host \
  --restart unless-stopped \
  --privileged
  $CONTAINER_NAME
"


eval $CMD_STR