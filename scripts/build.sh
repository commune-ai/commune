NAME=$(basename $(pwd))
docker build -t $NAME $(pwd)