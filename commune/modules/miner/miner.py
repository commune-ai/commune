
import commune as c
import os


class Miner(c.Module): 
    
    def __init__(self, 
                netuid = 15, 
                n = 42, 
                key : str =None, 
                treasury_key_address:str = None,
                stake=1, 
                miner_key_prefix = 'miner_', 
                max_age=600, 
                update=False,
                key_path='miner_mems'
                ):
        
        self.miner_key_prefix = miner_key_prefix
        self.subspace = c.module('subspace')()
        self.docker = c.module('docker')()
        self.pm2 = c.module('pm2')()
        self.netuid = netuid
        self.max_age = max_age
        self.update = update
        self.subnet = self.subnet_params['name']
        self.subnet_prefix = self.subnet.lower() + '_'
        self.n = int(n)
        self.key = c.get_key(key)
        self.treasury_key_address = treasury_key_address or self.key.ss58_address
        self.stake = stake
        self.key_path = self.resolve_path(key_path, extension='json')

    @property
    def subnet_params(self):
        return self.subspace.subnet_params(netuid=self.netuid, max_age=self.max_age)


    def keys(self, names = False):
        keys =  c.keys(self.miner_key_prefix)
        keys = list(filter(lambda k: int(k.split('_')[-1]) < self.n, keys))
        return keys

    def add_keys(self):
        for i in range(self.n):
            name = f"miner_{i}"
            c.add_key(name)


    def transfer_to_miners(self, amount=None):
        amount = amount or self.stake
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




    used_ports = []

    def is_running(self, name):
        return name in c.pm2ls(name)
    def register_miner(self, key, controller_key=None):
        if controller_key == None:
            controller_key = self.key.ss58_address 
        port = c.free_port()
        while port in self.used_ports:
            port = c.free_port()
        key_address = c.get_key(key).ss58_address
        if self.subspace.is_registered(key_address, netuid=self.netuid):
            return {'msg': 'already registered'}
        subnet_prefix = self.subnet.lower()
        name = f"{subnet_prefix}_{key.replace('::', '_')}"
        address = f'{c.ip()}:{port}'
        c.print(f"Registering {name} at {address}")
        return self.subspace.register(name=name, 
                                     address=address, 
                                     module_key=key,
                                     netuid=self.netuid, 
                                     key=controller_key, 
                                     stake=self.stake)

    def kill_miner(self, name):
        return self.pm2.kill(name)
    
    
    def run_miner(self, key, refresh=False):
        address2key = c.address2key()
        module_info = self.subspace.get_module(key, netuid=self.netuid)
        name = module_info['name']
        address = module_info['address']
        port = int(address.split(':')[-1])
        ip = address.split(':')[0]
        key_name = address2key.get(key, key)
        key = address2key.get(key, key)
        if self.is_running(name):
            if refresh:
                self.kill_miner(name)
            else:
                return {'msg': 'already running', 'name': name, 'key': key}
        
        cmd = f"comx module serve comchat.miner.model.Miner {key} --subnets-whitelist {self.netuid} --ip 0.0.0.0 --port {port}"
        cmd = f'pm2 start "{cmd}" --name {name}'
        
        return c.cmd(cmd)

    def run_miners(self, refresh=False, **kwargs):
        keys = self.registered_keys(**kwargs)
        futures = []
        for i, key in enumerate(keys):
            future = c.submit(self.run_miner, kwargs={'key': key, 'refresh': refresh})
            futures += [future]
        for f in c.as_completed(futures):
            print(f.result())


    def registered_keys(self, names=False, prefix='miner_', **kwargs):
        keys = self.subspace.keys(netuid=self.netuid, **kwargs)
        address2key = c.address2key()
        address2key = {k: v for k,v in address2key.items() if v.startswith(prefix)}
        miner_key_addresses = list(address2key.keys())
        keys =  list(filter(lambda k: k in miner_key_addresses, keys))
        if names:
            return [address2key[k] for k in keys]
        return keys

            
    
    def uids(self):
        key2uid = self.subspace.key2uid(netuid=self.netuid)
        return [key2uid[key] for key in self.registered_keys()]
    




    def register_miners(self, timeout=60, parallel=False, controller_key=None):
        keys = self.keys()
        futures = []
        c.print('Registering miners ...')
        c.print(keys)
        results = []
        if parallel:
            for i, key in enumerate(keys):
                futures += [c.submit(self.register_miner, kwargs={'key': key, 'controller_key': controller_key or key})]
            for f in c.as_completed(futures, timeout=timeout):
                results.append(f.result())
                c.print(results[-1])
        else:
            for i, key in enumerate(keys):
                results.append(self.register_miner(key))
                c.print(results[-1])

    def servers(self):
        return c.pm2ls()

    def modules(self, max_age=600, update=False, **kwargs):
        modules = self.get('modules', None,  max_age=max_age, update=update,  **kwargs)
        if modules == None:
            keys = self.registered_keys()
            modules = self.subspace.get_modules(keys, netuid=self.netuid)
            self.put('modules', modules)
        return modules

    def sync(self, timeout=60, **kwargs):
        modules = self.modules(**kwargs)
        futures = []
        ip = c.ip(update=1)
        free_ports = c.free_ports(n=len(modules))
        port_range = c.port_range() # if its in the port range

        for i, module in enumerate(modules):
            key = module['key']
            module_ip = module['address'].split(':')[0]
            module_port = int(module['address'].split(':')[-1])
            within_port_range = bool(module_port >= port_range[0] and module_port <= port_range[1])
            if within_port_range:
                if module_ip == ip:
                    emoji = '👍'
                    c.print(f"{emoji} {key} {emoji}", color='yellow')
                    continue
            address =  f'{ip}:{free_ports[i]}'
            c.print(f"Updating {key} ({module['address']} --> {address})", color='yellow')
            future = c.submit(self.subspace.update_module, kwargs={'module': key, 'address': address, 'netuid': self.netuid})
            futures += [future]

        for f in c.as_completed(futures, timeout=timeout):
            print(f.result())

    def unstake_and_transfer_back(self, key, amount=20):
        assert self.subspace.is_registered(key, netuid=self.netuid), f'{key} is not registered'
        self.subspace.unstake(key, netuid=self.netuid, amount=amount, key=key)
        self.subspace.transfer(key=key, dest=self.key.ss58_address, amount=amount)

    def unstake_many(self, amount=50, transfer_back=True):
        keys = self.registered_keys()
        futures = []
        for key in keys:
            future = c.submit(self.unstake_and_transfer_back, kwargs={'key': key, 'amount': amount})
            futures += [future]
        for f in c.as_completed(futures):
            print(f.result())
            
    def sand(self):
        keys = c.keys()
        rm_keys = []
        for k in keys:
            if '.' in k and k.startswith(self.miner_key_prefix):
                rm_keys.append(k)
        return c.rm_keys(rm_keys)

    def leaderboard(self, 
                avoid_keys=['stake_from', 'key', 
                            'vote_staleness', 
                            'last_update', 
                            'dividends',
                            'delegation_fee'], 
                sort_by='emission',
                reverse=False,
    ):
        modules = self.modules()
        for key in avoid_keys:
            for module in modules:
                module.pop(key, None)
        
        df =  c.df(modules)
        df = df.sort_values(by=sort_by, ascending=reverse)
        return df

    def save_keys(self, path='miner_mems'):
        keys = self.keys()
        key2mnemonic = {}
        for key in keys:
            mnemonic = c.get_key(key).mnemonic
            key2mnemonic[key] = mnemonic
        c.put_json(self.key_path, key2mnemonic)
        return {"msg": "keys saved", "path": self.key_path}

    def load_keys(self):
        key2mnemonic = c.get_json(self.key_path)
        return key2mnemonic
        


print(Miner.run(__name__))
                