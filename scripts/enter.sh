
REPO_PATH=$(pwd) ;  
REPO_NAME=$(basename $REPO_PATH)

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
  NAME=$REPO_NAME
else
  NAME=$1
fi

echo "Entering container: $NAME"
docker exec -it $NAME /bin/bash
