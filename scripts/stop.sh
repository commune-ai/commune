
# if no argument is passed, stop the container with the name of the current directory

# Check if first argument starts with --name=
if [[ "$1" == "--name="* ]]; then
  # Extract name from --name=value format
  NAME="${1#--name=}"
# Check if first argument is --name
elif [[ "$1" == "--name" ]]; then
  # Use the second argument as the name
  if [ -n "$2" ]; then
    NAME="$2"
  else
    echo "Error: --name requires a value"
    exit 1
  fi
# If no --name argument, use the first argument directly or default to repo name
elif [ -z "$1" ]; then
  NAME=$(basename $(pwd))
else
  NAME=$1
fi

CONTAINER_EXISTS=$(docker ps -q -f name=$NAME)  
if [ $CONTAINER_EXISTS ]; then
  CONTAINER_ID=$(docker ps -aq -f name=$NAME)
  echo "STOPPING(name=$NAME container=$CONTAINER_ID)"
  docker kill $NAME
  docker rm $NAME
else
  echo "No container named $NAME is currently running."
fi
