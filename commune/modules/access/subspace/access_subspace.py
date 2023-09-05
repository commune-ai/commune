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
            self.subspace = c.module('subspace')(network=config.network, netuid=config.netuid)
        self.stake_to = self.subspace.stake_to()
        self.key_stake = {k:sum(v.values()) for k,v in self.stake_to.items()}

    def verify(self, input:dict) -> dict:
        self.sync()
        stake = self.key_stake.get(input['address'], 0)
        stake_to = self.stake_to.get(input['address'], {})

        # allow keys that stake to the module to access the module
        stake_to_module = stake_to.get(self.module.key.ss58_address, 0)
        if stake_to_module == 0:
            assert stake > self.config.min_stake, f"Address {input['address']} does not have enough stake to access this function"

        return input