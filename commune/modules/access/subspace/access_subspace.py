import commune as c

class AccessSubspace(c.Module):
    def __init__(self, **kwargs):
        config = self.set_config(config=kwargs)

        self.subspace = c.module('subspace')(network=config.network, netuid=config.netuid)

    def run(self):
        print('Base run')
    


