import commune as c
import os
num_gpus = c.num_gpus()
subspace = c.module('subspace')()
docker = c.module('docker')()
my_ip = c.ip()
gpu_idx = 0
netuid= 14
num_gpus = c.num_gpus()
stake = 300
refresh = False
key = 'module'
timeout= 120
n = 24
address2key = c.address2key()
subnet_name = subspace.netuid2subnet(netuid)

def get_keys(n=n):
    keys = []
    for i in range(n):
        key = f'miner_{i}'
        if not c.key_exists(key):
            c.add_key(key)
        keys += [key]
    return keys

keys = subspace.my_keys(netuid=netuid)

def register_miner(key):
    port = c.free_port()
    while port in used_ports:
        port = c.free_port()
    key_address = c.get_key(key).ss58_address
    is_registered = bool(key_address in registered_keys)
    if is_registered:
        return {'msg': 'already registered'}

    subnet_prefix = subnet_name.lower()
    name = f"{subnet_prefix}_{key.replace('::', '_')}"
    address = f'{c.ip()}:{port}'

def run_miner(key, device=0, refresh=False):
    device = device % num_gpus
    print('FAMMMM')
    module_info = subspace.get_module(key, netuid=netuid)
    global gpu_idx
    home_dir = '/home/fam'
    commune_mount = f'{home_dir}/.commune:/root/.commune'
    huggingface_mount = f'{home_dir}/.cache/huggingface:/root/.cache/huggingface'
    name = module_info['name']
    if docker.exists(name):
        if refresh:
            docker.kill(name)
        else:
            return {'msg': 'already running', 'name': name, 'key': key}

    address = module_info['address']
    port = int(address.split(':')[-1])
    ip = address.split(':')[0]
    key_name = address2key.get(key, key)
    cmd = f'docker run --gpus=\'"device={device}"\' -d --network host --restart always \
        -v {huggingface_mount}\
        -v {commune_mount} \
        --name {name} \
        mos4ic/mosaic-subnet:latest \
        python mosaic_subnet/cli.py miner {key_name} 0.0.0.0 {port}'

    return c.cmd(cmd)
futures = []
for i, key in enumerate(keys):
    print(f'Running miner')
    future = c.submit(run_miner, kwargs={'key': key, 'device': i , 'refresh': refresh })
    futures += [future]

for f in c.as_completed(futures, timeout=timeout):
    print(f.result())
