
PWD=$(pwd)
NAME=$(basename $PWD)
docker build -t $NAME $PWD