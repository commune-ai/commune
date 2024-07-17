



CONTAINER=commune
SCRIPTS_PATH=./${CONTAINER}/scripts
install_docker:
	${SCRIPTS_PATH}/install_docker.sh
build:
	${SCRIPTS_PATH}/build_container.sh
up:
	${SCRIPTS_PATH}/start_container.sh
down:
	docker kill ${CONTAINER} ; docker rm ${CONTAINER}
enter:
	docker exec -it ${CONTAINER} bash
run_test: 
	docker exec ${CONTAINER} bash -c "pytest commune/tests"
install_venv:
	./commune/scripts/install_python_venv.sh
enter_env: 
	bash -c "source ./env/bin/activate"
create_env:
	python3 -m venv env