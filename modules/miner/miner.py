
import commune as c
import os


class Miner(c.Module): 
    
    def __init__(self, 
                netuid = 15, 
                n = 10, 
                key : str =None, 
                name_prefix = None,
                key_prefix = 'miner', 
                max_age=100600, 
                update=False,
                use_subnet_prefix=True,
                key_path='miner_mems',
                prefix_seperator = "::"
                ):
        self.key_prefix = key_prefix 
        self.docker = c.module('docker')()
        self.set_subnet(netuid=netuid, max_age=max_age, update=update)
        self.n = int(n)
        self.key = c.get_key(key)
        self.key_path = self.resolve_path(key_path, extension='json')
        # miner name prefix
        self.use_subnet_prefix = use_subnet_prefix
        self.prefix_seperator = prefix_seperator
        self.name_prefix = (self.subnet["name"].lower() if self.use_subnet_prefix else name_prefix) 
        self.name_prefix = self.name_prefix + self.prefix_seperator
        self.resolve_keys()

    def set_subnet(self, netuid=None,  max_age=10000,  update=False,  **kwargs):
        if not hasattr(self, 'subspace'):
            self.subspace = c.module('subspace')()
        if netuid == None:
            netuid = self.netuid
        params = self.subspace.subnet_params(netuid=netuid)
        keys = self.subspace.keys(netuid=netuid)
        uids = list(range(len(keys)))
        self.subnet = {
            'name': params['name'],
            'keys': keys,
            'uids': uids,
            'netuid': netuid,
            'params': params

        }
        self.netuid = netuid

    def resolve_keys(self):
        key2exist = self.key2exist()
        keys = self.key_names()
        for key in keys:
            if not key2exist[key]:
                c.print(c.add_key(key))
        return self.key2exist()

    def keys(self, names = False):
        keys =  c.keys()

        def filter_key(k):
            try:
                return k.startswith(self.key_prefix) and int(k[len(self.key_prefix):]) < self.n
            except:
                return False
        keys = list(filter(filter_key, keys))
        if names:
            address2key = c.address2key()
            return [address2key[k] for k in keys]
        return keys
    
    def key2exist(self):
        keys = self.key_names()
        key2address = c.key2address()
        return {k: k in key2address for k in keys}
    
    def key_names(self):
        return [self.key_prefix + str(i) for i in range(self.n)]
    
    def key_addresses(self):
        return [c.get_key(key).ss58_address for key in self.key_names()]
    
    def names(self):
        return [self.get_miner_name(key) for key in self.keys()]

    
    def get_miner_name(self, key):

        return self.name_prefix + key
    


    def add_keys(self):
        for i in range(self.n):
            name = f"miner_{i}"
            c.add_key(name)

    def transfer_to_miners(self, amount):
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
    

    def resolve_controller(self, controller):
        if controller == None:
            return self.key.ss58_address
        return controller

    def register_miner(self, key, controller=None, stake=1,  nonce=None):
        controller = self.resolve_controller(controller)
        key_address = c.get_key(key).ss58_address
        name = self.get_miner_name(key)
        if self.subspace.is_registered(key_address, netuid=self.netuid):
            return {'msg': 'already registered'}
        port = c.free_port()
        key_address = c.get_key(key).ss58_address
        address = f'{c.ip()}:{port}'
        return self.subspace.register(name=name, 
                                     address=address, 
                                     module_key=key,
                                     netuid=self.netuid, 
                                     key=controller, 
                                     stake=stake, 
                                     nonce=nonce)

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

    def unregisered_keys(self):
        key_addresses = self.key_addresses()
        return [k for k in key_addresses if k not in self.subnet['keys']]

    def registered_keys(self,  **kwargs):
        return [k for k,v in self.key2reg().items()if v ]

    def uids(self):
        key2uid = dict(zip(self.subnet['keys'] , self.subnet['uids']))
        return [key2uid[key] for key in self.registered_keys()]

    def resolve_key_address(self, key):
        if c.key_exists(key):
            return c.get_key(key).ss58_address
        else:
            return key


    def is_registered(self, key):
        key = self.resolve_key_address(key)
        return key in self.subnet["keys"]

    def register_miners(self, timeout=60, parallel=False, controller=None):
        keys = self.keys()
        c.print('Registering miners ...')
        futures = []
        results = []
        if parallel:
            nonce = self.subspace.get_nonce(controller)
        
        if parallel:
            for i, key in enumerate(keys):
                if self.is_registered(key):
                    print(f'{key} is already registered')
                    continue 
                
                futures += [c.submit(self.register_miner, kwargs={'key': key, 'controller': controller, 'nonce': nonce}, timeout=timeout)]
                nonce += 1
            for f in c.as_completed(futures, timeout=timeout):
                results.append(f.result())
                c.print(results[-1])
        else:
            for i, key in enumerate(keys):
                if self.is_registered(key):
                    print(f'{key} is already registered')
                    continue
                results.append(self.register_miner(key))
                c.print(results[-1])
    register_keys = register_miners

    def servers(self):
        return c.pm2ls()

    def modules(self, max_age=600, update=False, **kwargs):
        modules = self.get('modules', None,  max_age=max_age, update=update,  **kwargs)
        if modules == None:
            keys = self.registered_keys()
            modules = self.subspace.get_modules(keys, netuid=self.netuid)
            self.put('modules', modules)
        return modules
    
    def leaderboard(self, *args, **kwargs):
        modules = self.modules(*args, **kwargs)
        df = c.df(modules)
        return df

    def unstake_and_transfer_back(self, key, amount=20):
        assert self.subspace.is_registered(key, netuid=self.netuid), f'{key} is not registered'
        self.subspace.unstake(key=key, netuid=self.netuid, amount=amount, key=key)
        self.subspace.transfer(key=key, dest=self.key.ss58_address, amount=amount)

    def unstake_many(self, amount=50, transfer_back=True):
        keys = self.registered_keys()
        futures = []
        for key in keys:
            future = c.submit(self.unstake_and_transfer_back, kwargs={'key': key, 'amount': amount})
            futures += [future]
        for f in c.as_completed(futures):
            print(f.result())

    def leaderboard(self, 
                avoid_keys=['stake_from', 
                            'key', 
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
        c.put_json(self.key_path, self.key_state())
        return {"msg": "keys saved", "path": self.key_path}
    
    def key_state(self):
        keys = self.keys()
        key2mnemonic = {}
        for key in keys:
            mnemonic = c.get_key(key).mnemonic
            key2mnemonic[key] = mnemonic
        return key2mnemonic

    def load_keys(self):
        key2mnemonic = c.get_json(self.key_path)
        return key2mnemonic
    
    def rename_keys(self, new_prefix):
        keys = self.keys()
        old_prefix = self.key_prefix
        for key in keys:
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):] 
                c.print(c.rename_key(key, new_key))
                print(f"{key} --> {new_key}")
        self.key_prefix = new_prefix
        return {"msg": "keys renamed", "new_prefix": new_prefix, "old_prefix": old_prefix}
        


                