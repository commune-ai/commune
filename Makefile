



COMMUNE=commune
SUBSPACE=subspace
SUBTENSOR=0.0.0.0:9944

PYTHON=python3

start:
	chmod +x ./start.sh ; ./start.sh

build:
	docker-compose build

down:
	docker-compose down
stop:
	make down
up:
	docker-compose up -d
restart:
	make down && make up
	
enter:
	docker exec -it commune bash

