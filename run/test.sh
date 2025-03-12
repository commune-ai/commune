
PWD=$(pwd) ;  
REPO=$(basename $PWD)
NAME=$REPO-test
TEST_PATH=/app/$REPO/test.py
TEST_CMD="pytest $TEST_PATH"
./run/stop.sh $NAME
docker run -d --name $NAME -v $PWD:/app $REPO
docker exec -it $NAME bash -c "$TEST_CMD"
./run/stop.sh $NAME
