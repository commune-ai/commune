import commune as c
from typing import List

class Miner(c.Module):
    description: str
    whitelist: List[str]
    def __init__(self):
        super().__init__()
        self.description = 'Eden Miner v1'
        self.whitelist = ['forward'] 
        
    def forward(self, a=1, b=1):
        return a + b

    def launcher_keys(self):
        keys = c.keys()
        return [k for k in keys if k.startswith('module::')]
    
    def transfer2launchers(self, amount=10, **kwargs):
        destinations = self.launcher_keys()
        amounts = [amount] * len(destinations)
        return c.transfer_many(amounts=amounts, destinations=destinations, **kwargs)

    @classmethod
    def register_many(cls, key2address ,
        timeout=60,
        netuid = 0):
        futures = []
        launcher_keys = c.launcher_keys()
        future2launcher = {}
        future2module = {}
        registered_keys = c.m('subspace')().keys(netuid=netuid)
        progress = c.tqdm(total=len(key2address))
        while len(key2address) > 0:
            modules = list(key2address.keys())
            for i, module in enumerate(modules):
                module_key = key2address[module]
                if module_key in registered_keys:
                    c.print(f"Skipping {module} with key {module}")
                    key2address.pop(module)
                    progress.update(1)
                    continue
                c.print(f"Registering {module} with key {module}")
                launcher_key = launcher_keys[i % len(launcher_keys)]
                kwargs=dict(name=module, module_key=module_key, serve=True, key=launcher_key)
                future = c.submit(c.register, kwargs=kwargs, timeout=timeout)
                future2launcher[future] = launcher_key
                future2module[future] = module

            futures = list(future2launcher.keys())

            for f in c.as_completed(futures, timeout=timeout):
                module = future2module.pop(f)
                launcher_key = future2launcher.pop(f)
                module_key = key2address.pop(module)
                c.print(f"Registered {module} module_key:{module_key} launcher_key:{launcher_key}")
                r = f.result()
                if c.is_error(r):
                    progress.update(1)