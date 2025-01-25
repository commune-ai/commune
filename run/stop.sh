
# if no argument is passed, stop the container with the name of the current directory

# if an argument is passed, stop the container with the name of the argument
REPO=$(basename $(pwd))

if [ $# -eq 0 ]; then
  NAME=$(basename $(pwd))
else
  NAME=$1
fi
CONTAINER_EXISTS=$(docker ps -q -f name=$NAME)  
CONTAINER_ID=$(docker ps -aq -f name=$NAME)

echo "STOPING(name=$NAME repo=$REPO container=$CONTAINER_ID)"
if [ $CONTAINER_EXISTS ]; then
  docker kill $NAME
  docker rm $NAME
fi
