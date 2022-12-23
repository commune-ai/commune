



down:
	./start.sh --all --down

stop:
	make down
up:
	./start.sh --light
start:
	./start.sh --${arg} 
logs:
	./start.sh --${arg}

build:
	./start.sh --build --${arg}


enter: 
	make bash arg=commune

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
	docker exec -it commune bash -c "streamlit run commune/${arg}.py "
	
enter_backend:
	docker exec -it backend bash

pull:
	git submodule update --init --recursive
	
kill_all:
	docker kill $(docker ps -q) 

python:
	docker exec -it backend bash -c "python ${arg}.py"

exec:

	docker exec -it backend bash -c "${arg}"

register:
	cd backend ; source env/bin/activate ; python commune/bittensor/bittensor_module.py --index=${arg}


env_up:
	python3 -m venv env; source env/bin/activate;

build_commune_protos:
	python -m grpc_tools.protoc --proto_path=${PWD}/commune/proto ${PWD}/commune/proto/commune.proto --python_out=${PWD}/commune/proto --grpc_python_out=${PWD}/commune/proto

build_bittensor_protos:
	python -m grpc_tools.protoc --proto_path=${PWD}/bittensor/bittensor/_proto ${PWD}/bittensor/bittensor/_proto/bittensor.proto --python_out=${PWD}/bittensor/bittensor/_proto --grpc_python_out=${PWD}/bittensor/bittensor/_proto

server:
	docker exec -it commune bash -c "streamlit run commune/model/remote/remote_model_server.py "

client:
	docker exec -it commune bash -c "streamlit run commune/model/remote/remote_model_client.py"