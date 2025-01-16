
REPO_PATH=$(pwd) ;  
NAME=$(basename $REPO_PATH)
CONTAINER_NAME=$NAME-test
echo "STARING($CONTAINER_NAME)"
if [ $(docker ps -q -f name=$CONTAINER_NAME) ]; then
  ./scripts/stop.sh $CONTAINER_NAME
fi
  eval "docker run -d --name $CONTAINER_NAME -v $REPO_PATH:/app $NAME"
docker exec -it $CONTAINER_NAME pytest /app/tests
./scripts/stop.sh $CONTAINER_NAME
