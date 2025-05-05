
# if no argument is passed, stop the container with the name of the current directory

# if an argument is passed, stop the container with the name of the argument
if [ $# -eq 0 ]; then
  NAME=$(basename $(pwd))
else
  NAME=$1
fi
CONTAINER_EXISTS=$(docker ps -q -f name=$NAME)  
if [ $CONTAINER_EXISTS ]; then
  CONTAINER_ID=$(docker ps -aq -f name=$NAME)
  echo "STOPING(name=$NAME container=$CONTAINER_ID)"
  docker kill $NAME
  docker rm $NAME
fi
