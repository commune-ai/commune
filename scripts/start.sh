#

# if commune does not exist build it

# if docker is not running, start it

IMAGE_NAME=comtensor
IMAGE_PATH=./
CONTAINER_NAME=comtensor
SHM_SIZE=4g
# RESOLVE PORT RANGE
START_PORT=50050
END_PORT=50100
# if the image doesnt exist, build it
if [ "$1" == "--build" ] || [ "$1" == "-b" ]; then
  BUILD=true
else
  BUILD=false
fi
if [ "$BUILD" == true ]; then
  docker build -t $IMAGE_NAME $IMAGE_PATH
fi




CONTAINER_EXISTS=$(docker ps -q -f name=$CONTAINER_NAME)  
if [ $CONTAINER_EXISTS ]; then

  echo "STOPPING CONTAINER $CONTAINER_NAME"
  docker stop $CONTAINER_NAME

fi

CONTAINER_ID=$(docker ps -aqf "name=$CONTAINER_NAME")
if [ $CONTAINER_ID ]; then
  echo "REMOVING CONTIANER NAME=$CONTAINER_NAME ID=$CONTAINER_ID"
  docker rm $CONTAINER_ID
fi

CMD_STR="docker run -d \
  --name $CONTAINER_NAME \
  --shm-size $SHM_SIZE \
  -v ~/.commune:/root/.commune \
  -v ~/.bittensor:/root/.bittensor \
  -v $PWD:/app \
  -p $START_PORT-$END_PORT:$START_PORT-$END_PORT \
  --restart unless-stopped \
  $CONTAINER_NAME"

eval $CMD_STR

if [ "$1" == "--port-range" ] || [ "$1" == "-pr" ]; then
  # add the port range flag by taking the next two arguments
  # if '-' in the second argument, split it and set the start and end ports
  if [[ $2 == *-* ]]; then
    IFS='-' read -r -a PORTS <<< "$2"
    START_PORT=${PORTS[0]}
    END_PORT=${PORTS[1]}
  else
    START_PORT=$2
    END_PORT=$3
    echo "Setting port range to $PORT_RANGE"
  fi
fi
PORT_RANGE=$START_PORT-$END_PORT
echo "Setting port range to $PORT_RANGE"
docker exec $CONTAINER_NAME bash -c "c set_port_range $PORT_RANGE"
docker exec $CONTAINER_NAME bash -c "c serve"
# docker exec $CONTAINER_NAME bash -c "c app arena.app"
# get the timestamp as a random seed

SEED=$(date +%s)
echo "SEED=$SEED"
# scan for open port ranges of 100

# if [ "$1" == "--c" ] || [ "$1" == "-s" ]; then
#   echo "Scanning for open port ranges of 100"
#   for i in {50000..50250..100}; do
