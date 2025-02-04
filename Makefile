SCRIPTS_PATH=./run
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
freshtest:
	make build && make test
install:
	${SCRIPTS_PATH}/install.sh	
restart:
	make stop
	make start
chmod:
	chmod +x ${SCRIPTS_PATH}/*
up: 
	make start
down:
	make stop

