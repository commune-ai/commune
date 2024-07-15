import commune as c


class Miner(c.Module): 

    def __init__(self, ):
        self.miner_key_prefix = 'miner_'
        self.subspace = c.module('subspace')()

    def keys(self):
        return c.keys(self.miner_key_prefix)


    def key_addresses(self):
        key2address = c.key2address()
        return [key2address[miner] for miner in self.keys()]


    def registered(self, netuid=None):
        return c.registered(self.miner_key_prefix, netuid)

    def stats(self, netuid=0):
        return c.s
        