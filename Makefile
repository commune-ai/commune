



COMMUNE=commune
SUBSPACE=subspace
SUBTENSOR=0.0.0.0:9944

PYTHON=python3

start:
	chmod +x ./start.sh ; ./start.sh

build:
	docker-compose build
build_image:
	docker build -t commune .
start_container:
	docker run -d -p 5000:5000 commune

down:
	docker-compose down
stop:
	make down
up:
	docker-compose up -d
restart:
	make down && make up
	
enter:
	docker exec -it commune bash
test: 
	docker exec -it commune bash -c "pytest -v"
install_venv:
	./commune/scripts/install_python_venv.sh
enter_env: 
	bash -c "source ./env/bin/activate"
create_env:
	python3 -m venv env