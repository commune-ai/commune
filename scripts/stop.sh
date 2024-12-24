NAME=$(basename $(pwd))
CONTAINER_EXISTS=$(docker ps -q -f name=$NAME)  
if [ $CONTAINER_EXISTS ]; then
  echo "STOPPING --> $NAME"
  docker kill $NAME
  CONTAINER_ID=$(docker ps -aq -f name=$NAME)
  docker rm $NAME
fi
