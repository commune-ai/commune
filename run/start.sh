# starts the container with the name of the current directory


PWD=$(pwd)
REPO=$(basename $(pwd)) # get the name of the current directory
# if no argument is passed, start the container with the name of the current directory
if [ -z $1 ]; then
  NAME=$REPO
else
  NAME=$1
fi
if [ $(docker ps -q -f name=$NAME) ]; then
  ./run/stop.sh $NAME
fi

docker run -d \
  --name $NAME \
  --network=host \
  --restart unless-stopped \
  --privileged --shm-size 4g \
  -v $PWD:/app -v ~/.$REPO:/root/.$REPO \
  -v /var/run/docker.sock:/var/run/docker.sock \
  $REPO

CONTAINER_ID=$(docker ps -q -f name=$NAME)
echo "STARTING(name=$NAME repo=$REPO container=$CONTAINER_ID)"



# Path: run/stop.sh