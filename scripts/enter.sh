

# If no --name argument, use the first argument directly or default to repo name
if [ -z "$1" ]; then
  NAME=$(basename $(pwd))
else 
  NAME=$1
fi

echo "Entering container: $NAME"
docker exec -it $NAME /bin/bash
