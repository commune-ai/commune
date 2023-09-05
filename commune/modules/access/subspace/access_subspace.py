import commune as c

class AccessSubspace(c.Module):
    sync_time = 0
    def __init__(self, module, **kwargs):
        config = self.set_config(kwargs)
        self.module = module
        self.sync()


    def sync(self):
        sync_time  = c.time() - self.sync_time
        if sync_time >  self.config.sync_interval :
            self.sync_time = c.time()
        else:
            return
        if not hasattr(self, 'subspace'):
            self.subspace = c.module('subspace')(network=self.config.network, netuid=self.config.netuid)
        self.stake_to = self.subspace.stake_to()
        self.key_stake = {k:sum([_[1] for _ in v ]) for k,v in self.stake_to.items() if len(v) > 0}

    def verify(self, input:dict) -> dict:
        self.sync()
        address = input['address']
        if c.is_admin(address):
            return input
        stake = self.key_stake.get(address, 0)
        stake_to = self.stake_to.get(address, {})

        # allow keys that stake to the module to access the module
        stake_to_module = stake_to.get(self.module.key.ss58_address, 0)
        if stake_to_module == 0:
            assert stake > self.config.min_stake, f"Min stake of {address} should be {self.config.min_stake}"

        return input