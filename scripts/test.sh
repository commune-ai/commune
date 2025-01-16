
NAME=$(basename $(pwd));
CONTAINER_EXISTS=$(docker ps -a | grep $NAME)
if [ -z "$CONTAINER_EXISTS" ]; then
    ./scripts/start.sh
fi
docker exec -it $NAME pytest /$NAME/tests/test_server.py
