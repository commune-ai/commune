#

# if commune does not exist build it

# if docker is not running, start it

CONTAINER_NAME=commune
START_PORT=50050
END_PORT=50250

if [ "$1" == "--port-range" ]; then
  # add the port range flag by taking the next two arguments
  # if '-' in the second argument, split it and set the start and end ports
  if [[ $2 == *-* ]]; then
    IFS='-' read -r -a PORTS <<< "$2"
    START_PORT=${PORTS[0]}
    END_PORT=${PORTS[1]}
  else
    START_PORT=$2
    END_PORT=$3
    PORT_RANGE=$START_PORT-$END_PORT
    echo "Setting port range to $PORT_RANGE"
  fi
fi

PORT_RANGE=$START_PORT-$END_PORT

if [ "$BUILD" == true ]; then
  docker build -t commune .
fi

CONTAINER_EXISTS=$(docker ps -q -f name=$CONTAINER_NAME)  


# build if it does not exist
if [ ! $CONTAINER_EXISTS ]; then
    docker run -d --name commune --shm-size 4gb \
    -v ~/.commune:/root/.commune \
    -v $PWD:/app \
    --network host \
    # -p $PORT_RANGE:$PORT_RANGE \
    --restart unless-stopped \
    commune
fi



