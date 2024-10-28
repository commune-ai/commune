SCRIPTS_PATH=./scripts
build:
	${SCRIPTS_PATH}/build.sh 
run:
	${SCRIPTS_PATH}/run.sh 
stop:
	${SCRIPTS_PATH}/stop.sh 
enter:
	${SCRIPTS_PATH}/enter.sh
test:
	${SCRIPTS_PATH}/test.sh
chmod:
	chmod +x ${SCRIPTS_PATH}/*
up: 
	make run
down:
	make stop
start:
	make run

