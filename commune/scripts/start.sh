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
if [ $CONTAINER_EXISTS ]; then
  echo  "HEY"
  echo "Stopping and removing existing container"
  docker stop $CONTAINER_NAME
  # get the container id
  CONTAINER_ID=$(docker ps -aqf "name=$CONTAINER_NAME")
  echo "Container ID: $CONTAINER_ID"
  docker rm $CONTAINER_ID
  echo "Container removed $CONTAINER_IDS"
fi



docker run -d --name commune --shm-size 4gb \
  -v ~/.commune:/root/.commune \
  -v $PWD:/app \
  -p $PORT_RANGE:$PORT_RANGE \
  --restart unless-stopped \
  commune

# run c set_port_range to set the port range
# run c set_port_range 50050-50250

echo "Setting port range to $PORT_RANGE"
docker exec commune bash -c "c set_port_range $PORT_RANGE"
