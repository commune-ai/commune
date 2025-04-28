
SCRIPTS_PATH=./scripts
build:
	${SCRIPTS_PATH}/start.sh --build
start:
	${SCRIPTS_PATH}/start.sh 
stop:
	${SCRIPTS_PATH}/stop.sh 
enter:
	${SCRIPTS_PATH}/enter.sh 
test:
	${SCRIPTS_PATH}/start.sh --test
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
# New commands with argument passing
start-with-name:
	${SCRIPTS_PATH}/start.sh --name=$(name)
enter-with-name:
	${SCRIPTS_PATH}/enter.sh $(name)
stop-with-name:
	${SCRIPTS_PATH}/stop.sh $(name)
