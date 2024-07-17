



CONTAINER=commune
SCRIPTS_PATH=./${CONTAINER}/scripts
build:
	${SCRIPTS_PATH}/build.sh
start:
	${SCRIPTS_PATH}/start.sh
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