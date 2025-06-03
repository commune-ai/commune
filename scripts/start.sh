# starts the container with the name of the current directory


TEST=false
NAME=$(basename $(pwd)) # default name is the current directory name
IMAGE=$(basename $(pwd)) # get the name of the current directory
BUILD=false # default to not build the image
SHM_SIZE=4g # default shared memory size
# get ~/.commune directory
CACHE_DIR=~/.commune
# get the absolute path of the cache directory
CACHE_DIR=$(realpath $CACHE_DIR)


VOLUME_PART=" -v $(pwd):/app -v /var/run/docker.sock:/var/run/docker.sock -v $CACHE_DIR:/root/.commune" # mount the current directory to /app in the container
NETWORK_PART="--network host" # use the host network to allow access to local services
# ARGS


if [ $# -gt 0 ]; then
  for arg in "$@"; do
    # include the name of the container
    if [ "$arg" == "--build" ]; then
      BUILD=true
      shift
    elif [ "$arg" == "--test" ]; then
      TEST=true
      shift

    elif [[ "$arg" == "--image="* ]]; then
      # remove the --image= prefix
      IMAGE="${arg#--image=}"
      shift

    elif [[ "$arg" == "--name="* ]]; then
      # remove the --name= prefix
      NAME="${arg#--name=}"
      echo "NAME=$NAME"
      shift
    elif [[ "$arg" == "--shm-size="* ]]; then
      # remove the --shm-size= prefix
      SHM_SIZE="${arg#--shm-size=}"
      # check if the shm size is valid
      if ! [[ "$SHM_SIZE" =~ ^[0-9]+[gGmM]$ ]]; then
        echo "Invalid shm size: $SHM_SIZE"
        exit 1
      fi
      shift

    elif [[ "$arg" == "--port="* ]]; then
      # remove the --port= prefix
      PORT="${arg#--port=}"
      # check if the port is valid
      if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
        echo "Invalid port number: $PORT"
        exit 1
      fi
      # add the port mapping to the docker run command
      NETWORK_PART=" -p $PORT:$PORT" # map the port to the container
      shift
    elif [[ "$arg" == "--"* ]]; then
      echo "Unknown argument: $arg"
      exit 1
    else
      shift
    fi

  done
fi

echo $BUILD

if [ "$BUILD" == true ]; then
  echo "BUILDING(name=$NAME repo=$IMAGE)"
  docker build -t $IMAGE .
fi

# check if the container with the name already exists
CONTAINER_EXISTS=$(docker ps -q -f name=$NAME)
if [ $CONTAINER_EXISTS ]; then
  echo "Container with name $NAME already exists. Stopping it first..."
  # stop the existing container
  docker kill $NAME
  docker rm $NAME
fi

# # run the container with the name of the current directory
# docker run -d --name $NAME \
#   $NETWORK_PART \
#   $VOLUME_PART \
#   --restart unless-stopped  --privileged --shm-size $SHM_SIZE \
#   $IMAGE

# CONTAINER_ID=$(docker ps -q -f name=$NAME)
# echo_msg="name=$NAME id=$CONTAINER_ID image=$IMAGE"
# if PORT="${PORT:-}" ; then
#   echo_msg="$echo_msg port=$PORT"
# fi
# echo "CONTAINER($echo_msg)"
# # check if the container started successfully

# # 
# if [ "$TEST" == true ]; then
#   echo "Running tests..."
#   # run the tests in the container
#   docker exec -it $CONTAINER_ID /bin/bash -c "c test"
# else 
#   echo "Container started successfully."
#   if [ -n "$PORT" ]; then
#     echo "You can access the service at http://localhost:$PORT"
#   else
#     echo "No port mapping specified. Access the service using the container's internal network."
#   fi
#   echo "To enter the container, run: docker exec -it $NAME /bin/bash"
# fi

docker compose up -d