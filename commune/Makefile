down:
	./script/start.sh --backend --ganache --ipfs --down
stop:
	make down
up:
	./script/start.sh --backend --ganache --ipfs
start:
	./script/start.sh --${arg}


start_latest:
	make up_latest

bash_backend: 
	docker exec -it backend bash

backend: 
	docker exec -it backend bash

restart:
	make down && make up;


prune_volumes:	
	docker system prune --all --volumes

bash:
	docker exec -it ${arg} bash

app:
	make streamlit
kill_all:
	docker kill $(docker ps -q)

logs:
	docker logs ${arg} --tail=100 --follow

streamlit:
	docker exec -it backend bash -c "streamlit run commune/${arg}/module.py "
enter_backend:
	docker exec -it backend bash
pull:
	git submodule update --init --recursive
	
kill_all:
	docker kill $(docker ps -q) 

jupyter:
	docker exec -it backend bash -c "jupyter lab --allow-root --ip=0.0.0.0 --port=8888"

python:
	docker exec -it backend bash -c "python commune/${arg}/module.py"

exec:
	docker exec -it backend bash -c "${arg}"

api:
	docker exec -it backend bash -c "python commune/gradio/api/module.py --api"


gradio_api:
	docker exec -it backend bash -c "python commune/gradio/api/module.py --api"

gradio_client:
	make app arg=gradio/client

backend:
	make bash arg=backend
