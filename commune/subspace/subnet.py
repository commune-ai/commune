import commune as c

class Subnet(c.Module):

    def __init__(self, network='main', **kwargs):
        self.subspace = c.module('subspace')(network=network, **kwargs)

    def emissions(self, netuid = 0, network = "main", block=None, update=False, **kwargs):
        return self.subspace.query_vector('Emission', network=network, netuid=netuid, block=block, update=update, **kwargs)

    def subnet2stake(self, network=None, update=False) -> dict:
        subnet2stake = {}
        for subnet_name in self.subnet_names(network=network):
            c.print(f'Getting stake for subnet {subnet_name}')
            subnet2stake[subnet_name] = self.my_total_stake(network=network, netuid=subnet_name , update=update)
        return subnet2stake
