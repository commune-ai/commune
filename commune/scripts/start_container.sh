docker run \
  --name commune \
  --shm-size 4gb \
  -v ~/.commune:/root/.commune \
  -v $(pwd):/app \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -p 50050-50250:50050-50250 \
  --restart unless-stopped \