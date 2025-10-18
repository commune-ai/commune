

SCRIPTS_PATH=./scripts
# CHMOD: change the mode of the scripts
chmod:
	chmod +x ${SCRIPTS_PATH}/*
# BUILD: build the app
# START: start the app
start:
	${SCRIPTS_PATH}/start.sh
up: 
	make start
# STOP: stop the app
stop:
	${SCRIPTS_PATH}/stop.sh
down:
	make stop

build:
	${SCRIPTS_PATH}/build.sh
enter: 
	${SCRIPTS_PATH}/enter.sh
