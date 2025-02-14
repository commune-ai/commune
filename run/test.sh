
PWD=$(pwd) ;  
REPO=$(basename $PWD)
NAME=$REPO-test
./run/stop.sh $NAME
docker run -d --name $NAME -v $PWD:/app $REPO
docker exec -it $NAME pytest /app/tests
./run/stop.sh $NAME
