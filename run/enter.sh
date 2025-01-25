
REPO_PATH=$(pwd) ;  
REPO_NAME=$(basename $REPO_PATH)
if [ -z $1 ]; then
  NAME=$REPO_NAME
else
  NAME=$1
fi
docker exec -it $NAME /bin/bash