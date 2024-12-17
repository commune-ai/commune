SCRIPTS_PATH=./scripts
build:
	${SCRIPTS_PATH}/build.sh 
start:
	${SCRIPTS_PATH}/start.sh 
stop:
	${SCRIPTS_PATH}/stop.sh 
enter:
	${SCRIPTS_PATH}/enter.sh
test:
	${SCRIPTS_PATH}/test.sh
chmod:
	chmod +x ${SCRIPTS_PATH}/*
up: 
	make start
down:
	make stop

