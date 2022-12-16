env_up:
	python3 -m venv env; source env/bin/activate;

build_protos:
	python -m grpc_tools.protoc --proto_path=${PWD}/commune/proto ${PWD}/commune/proto/commune.proto --python_out=${PWD}/commune/proto --grpc_python_out=${PWD}/commune/proto