import commune as c

class Subnet(c.Module):

    def __init__(self, network='main', **kwargs):
        self.subspace = c.module('subspace')(network=network, **kwargs)

    def emissions(self, netuid = 0, network = "main", block=None, update=False, **kwargs):
        return self.subspace.query_vector('Emission', network=network, netuid=netuid, block=block, update=update, **kwargs)
    
    