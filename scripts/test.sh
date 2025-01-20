
PWD=$(pwd) ;  
NAME=$(basename $PWD)
CONTAINER_NAME=$NAME-test
echo "STARING($CONTAINER_NAME)"
./scripts/stop.sh $CONTAINER_NAME
eval "docker run -d --name $CONTAINER_NAME -v $PWD:/app $NAME"
docker exec -it $CONTAINER_NAME pytest /app/tests
./scripts/stop.sh $CONTAINER_NAME
