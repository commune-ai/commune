



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

restart:
	make down && make up
logs:
	./$(COMMUNE).sh --commune

subspace:
	make bash arg=$(SUBSPACE)

enter: 
	make bash arg=$(COMMUNE)

prune_volumes:	
	docker system prune --all --volumes

bash:
	docker exec -it ${arg} bash


kill_all:
	docker kill $(docker ps -q)

logs:
	docker logs ${arg} --tail=100 --follow


enter_backend:
	docker exec -it $(COMMUNE) bash

pull:
	git submodule update --force --recursive --init --remote
	