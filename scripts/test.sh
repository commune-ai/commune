
PWD=$(pwd); NAME=$(basename $PWD)
CONTAINER_EXISTS=$(docker ps -a | grep $NAME)
if [ -z "$CONTAINER_EXISTS" ]; then
    make run
fi
docker exec -it $NAME c pytest
