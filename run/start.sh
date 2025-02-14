
REPO=$(basename $(pwd))
if [ -z $1 ]; then
  NAME=$REPO
else
  NAME=$1
fi
if [ $(docker ps -q -f name=$NAME) ]; then
  ./run/stop.sh $NAME
fi

docker run -d \
  --name $NAME \
  --network=host \
  --restart unless-stopped \
  --privileged \
  --shm-size 4g \
  -v $REPO:/app -v ~/.$REPO:/root/.$REPO \
  -v /var/run/docker.sock:/var/run/docker.sock \ 
  $REPO
echo "STARTING(name=$NAME repo=$REPO)"


