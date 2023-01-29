



LIB=commune


down:
	./start.sh --all --down

stop:
	make down
up:
	./start.sh --all
start:
	./start.sh --${arg} 
logs:
	./start.sh --${arg}

build:
	./start.sh --build --${arg}

subspace:
	make bash arg=subspace

enter: 
	make bash arg=$(LIB)
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
	
kill_all_containers:
	docker kill $(docker ps -q) 

python:
	docker exec -it backend bash -c "python ${arg}.py"

exec:

	docker exec -it backend bash -c "${arg}"

build_protos:
	python -m grpc_tools.protoc --proto_path=${PWD}/commune/proto ${PWD}/commune/proto/commune.proto --python_out=${PWD}/commune/proto --grpc_python_out=${PWD}/commune/proto


api:  
	uvicorn commune.api.api:app --reload

ray_start:
	ray start --head --port=6379 --redis-port=6379 --object-manager-port=8076 --node-manager-port=8077 --num-cpus=4 --num-gpus=0 --memory=1000000000 --object-store-memory=1000000000

ray_stop:
	ray stop

ray_status:
	ray status