



COMMUNE=commune
SUBSPACE=subspace
SUBTENSOR=0.0.0.0:9944

PYTHON=python3

down:
	docker-compose down
stop:
	make down
up:
	docker-compose up -d
start:
	make start
restart:
	make down && make up
logs:
	./$(COMMUNE).sh --commune

build:
	docker-compose build

subspace:
	make bash arg=$(SUBSPACE)

enter: 
	make bash arg=$(COMMUNE)
restart:
	make down && make up

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
	git submodule update --init --recursive
	
kill_all_containers:
	docker kill $(docker ps -q) 

python:
	docker exec -it $(COMMUNE) bash -c "python ${arg}.py"

exec:
	docker exec -it com bash -c "${arg}"

build_protos:
	python -m grpc_tools.protoc --proto_path=${PWD}/$(COMMUNE)/proto ${PWD}/$(COMMUNE)/proto/commune.proto --python_out=${PWD}/$(COMMUNE)/proto --grpc_python_out=${PWD}/$(COMMUNE)/proto


api:  
	uvicorn $(COMMUNE).api.api:app --reload

ray_start:
	ray start --head --port=6379 --redis-port=6379 --object-manager-port=8076 --node-manager-port=8077 --num-cpus=4 --num-gpus=0 --memory=1000000000 --object-store-memory=1000000000

ray_stop:
	ray stop

ray_status:
	ray status

miner:
	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python pm2 start commune/model/${mode}/model.py --name miner_${coldkey}_${hotkey}_${mode} --time --interpreter $(PYTHON) --  --logging.debug  --subtensor.chain_endpoint 194.163.191.101:9944 --wallet.name ${coldkey} --wallet.hotkey ${hotkey} --axon.port ${port}


vali:
	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python pm2 start commune/block/bittensor/neuron/validator/neuron.py --name vali_${coldkey}_${hotkey} --time --interpreter python3 -- --logging.debug --subtensor.network local  --neuron.device cuda:1 --wallet.name ${coldkey} --wallet.hotkey ${hotkey} --logging.trace True --logging.record_log True  --neuron.print_neuron_stats True

dashboard:
	streamlit run commune/dashboard.py 

sand:
	streamlit run sandbox.py

sandbox_docker:
	make exec arg="make sandbox"

python_pm2:
	pm2 start {file} --name {name} --interpreter python3

streamlit_pm2:
	pm2 start {file} --name {name} --interpreter streamlit -- --server.port {port}
register_loop:
	pm2 start commune/block/bittensor/bittensor_module.py --name register_loop --interpreter python3 -- --fn register_loop


st_ensemble:
	streamlit run commune/model/ensemble/model.py
st:
	pm2 start commune/${arg}.py --name st_${arg} --interpreter python3 -- -m streamlit run  

enver_env:
	source env/bin/activate

venv_build:
	./scripts/install_python_env.sh

venv_up:
	source venv/bin/activate

register:
	python3 commune/bittensor/bittensor_module.py -fn register_wallet -kwargs "{'dev_id': 1, 'wallet': 'ensemble_0.1'}"

start_env:
	chmod +x ./scripts && ./scripts/start_python_env.sh

# ANSIBLE CHADGPT PLAYS

play:
	ansible-playbook -i ./ansible/inventory.yaml ./ansible/playbooks/${arg}.yaml

gen_keys:
	make play arg=gen_keys

update:
	make play arg=update

sync:
	make play arg=sync
register:
	make ConfigMixinplay arg=register


ping:
	ansible all -m ping -i ./ansible/inventory.yaml

docker_build:
	make play arg=docker_build

chad:
	make play arg=chad

docker_ps:
	make play arg=docker_ps

list_keys:
	make play arg=list_keys

cmd:
	ansible all -i ansible/inventory.yaml -m shell -a "${arg}"

ssh:
	ssh -i ~/.ssh/id_rsa paperspace@${arg}

miner_vali:
	pm2 start commune/validator/validator.py --interpreter python3 --name miner -- --logging.debug --subtensor.network finney
miner:
	pm2 delete miner; pm2 start ./bittensor/_neuron/text/core_server/main.py --interpreter python3 --name miner -- --logging.debug --subtensor.network finney --neuron.autocast --neuron.local_train --neuron.model_name EleutherAI/gpt-j-6b --wallet.name collective --netuid 3
install: 
	make pull; chmod +x ./scripts/* ; ./scripts/install_commune.sh

miner_van:
	pm2 delete miner-van; pm2 start ./bittensor/bittensor/_neuron/text/core_server/main.py --name miner-van --time --interpreter python3 -- --logging.debug --neuron.model_name ~/models/gpt-j-6B-vR --neuron.device cuda:6 --neuron.autocast --no_prompt true --subtensor.network finney  --wallet.reregister false --neuron.no_set_weights --wallet.name collective --wallet.hotkey 0 --axon.port 9270 --netuid 3