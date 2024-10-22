PWD=$(pwd)
NAME=$(basename $PWD)
docker exec -it $NAME /bin/bash