# starts the container with the name of the current directory


PWD=$(pwd)
REPO=$(basename $(pwd)) # get the name of the current directory
# if no argument is passed, start the container with the name of the current directory

case $i in
    --port=*)
    PORT="${i#*=}"
    shift
    ;;
    --pwd=*)
    PWD="${i#*=}"
    shift
    ;;
    --name=*)
    NAME="${i#*=}"
    shift
    ;;


    *)

esac


docker run -d \
  --name $NAME \
  --network=host \
  --restart unless-stopped \
  --privileged --shm-size 4g \
  -v $PWD:/app \
  -v /var/run/docker.sock:/var/run/docker.sock \
  $REPO

CONTAINER_ID=$(docker ps -q -f name=$NAME)
echo "STARTING(name=$NAME repo=$REPO container=$CONTAINER_ID)"



# Path: run/stop.sh