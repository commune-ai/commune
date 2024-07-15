
import commune as c
import os


class Miner(c.Module): 
    
    def __init__(self, netuid = 2, max_miners = 10):
        self.miner_key_prefix = 'miner_'
        self.subspace = c.module('subspace')()
        self.docker = c.module('docker')()
        self.netuid = netuid
        self.subnet_name = self.subspace.netuid2subnet(netuid)
        self.max_miners = max_miners


    def keys(self):
        keys =  c.keys(self.miner_key_prefix)
        keys = list(filter(lambda k: int(k.split('_')[-1]) < self.max_miners, keys))
        return keys

    def transfer_to_miners(self, amount=200):
        return self.subspace.transfer_multiple(self.key_addresses(), amount)


    def miner2balance(self, timeout=30, max_age=30, **kwargs):
        key_addresses = self.key_addresses()
        future2key = {c.submit(self.subspace.get_balance, kwargs={'key': k, 'max_age': max_age, **kwargs}): k for k in key_addresses}
        miner2balance = {}
        for f in c.as_completed(future2key, timeout=timeout):
            key = future2key.pop(f)
            print(key)
            miner2balance[key] = f.result()
        return miner2balance



    def key_addresses(self):
        key2address = c.key2address()
        return [key2address[miner] for miner in self.keys()]




    def add_keys(self, n=24):
        for i in range(n):
            c.add_key( f"miner_{i}")
        
    used_ports = []
    def register_miner(self, key):
        port = c.free_port()
        keys = self.keys()
       
        while port in self.used_ports:
            port = c.free_port()
        key_address = c.get_key(key).ss58_address
        if self.subspace.is_registered(key_address, netuid=self.netuid):
            return {'msg': 'already registered'}
        subnet_prefix = self.subnet_name.lower()
        name = f"{subnet_prefix}_{key.replace('::', '_')}"
        address = f'{c.ip()}:{port}'
        return self.subspace.register(name=name, address=address, module_key=key, netuid=self.netuid, key=key)

    def run_miner(key, refresh=False):
        device = device % num_gpus
        module_info = self.subspace.get_module(key, netuid=self.netuid)
        home_dir = '/home/fam'
        commune_mount = f'{home_dir}/.commune:/root/.commune'
        huggingface_mount = f'{home_dir}/.cache/huggingface:/root/.cache/huggingface'
        name = module_info['name']
        if self.docker.exists(name):
            if refresh:
                self.docker.kill(name)
            else:
                return {'msg': 'already running', 'name': name, 'key': key}
        address = module_info['address']
        port = int(address.split(':')[-1])
        ip = address.split(':')[0]
        key_name = address2key.get(key, key)
        name = name or f"comchat/{key}"
        cmd = f"comx module serve comchat.miner.model.Miner {key} --subnets-whitelist {self.netuid} --ip 0.0.0.0 --port {port}"
        cmd = f"pm2 start {cmd} --name {name}"
        return c.cmd(cmd)
        futures = []

    def register_miners(self, timeout=60):
        keys = self.keys()
        futures = []
        for i, key in enumerate(keys):
            future = c.submit(self.register_miner, kwargs={'key': key})
            futures += [future]
        for f in c.as_completed(futures, timeout=timeout):
            print(f.result())
        
