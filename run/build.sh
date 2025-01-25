REPO_NAME=$(basename $(pwd))
if [ -z $1 ]; then
  NAME=$REPO_NAME
else
  NAME=$1
fi
docker build -t $NAME $(pwd)