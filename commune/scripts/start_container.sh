#

docker run -d --name commune --shm-size 4gb \
  -v ~/.commune:/root/.commune \
  -v $(pwd):/app \
  -p 50050-50250:50050-50250 \
  --restart unless-stopped \
  commune


  # -v /var/run/docker.sock:/var/run/docker.sock \
