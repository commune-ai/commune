import commune as c

class Subnet(c.Module):

    def emissions(self, netuid = 0, network = "main", block=None, update=False, **kwargs):

        return self.query_vector('Emission', network=network, netuid=netuid, block=block, update=update, **kwargs)
    