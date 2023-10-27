start_env:
	chmod +x ./scripts && ./scripts/start_python_env.sh

# ANSIBLE CHADGPT PLAYS

play:
	ansible-playbook -i inventory.yaml playbooks/${arg}.yaml

gen_keys:
	make play arg=gen_keys

update:
	make play arg=update

sync:
	make play arg=sync
register:
	make play arg=register

ping:
	ansible all -m ping -i inventory.yaml

docker_build:
	make play arg=docker_build

chad:
	make play arg=chad

docker_ps:
	make play arg=docker_ps

list_keys:
	make play arg=list_keys

cmd:
	ansible all -i inventory.yaml -m shell -a "${arg}"
sandbox:
	make play arg=sandbox

ssh:
	ssh -i ~/.ssh/id_rsa paperspace@${arg}