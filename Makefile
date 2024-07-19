



CONTAINER=commune
SCRIPTS_PATH=./scripts

# include the arguments following this
build:
	${SCRIPTS_PATH}/build.sh 
start:
	${SCRIPTS_PATH}/start.sh 
up: 
	make start
	
restart: 
	docker restart ${CONTAINER}
down:
	docker kill ${CONTAINER} ; docker rm ${CONTAINER}
kill:
	docker kill ${CONTAINER} ; docker rm ${CONTAINER}
enter:
	docker exec -it ${CONTAINER} bash
tests: 
	docker exec ${CONTAINER} bash -c "pytest commune/tests"
	
install_venv:
	./commune/scripts/install_python_venv.sh
enter_env: 
	bash -c "source ./env/bin/activate"
create_env:
	python3 -m venv env

chmod_scripts:
	chmod +x ${SCRIPTS_PATH}/*.sh

app:
	docker exec ${CONTAINER} c app arena.app

apps:
	docker exec ${CONTAINER} c app/apps