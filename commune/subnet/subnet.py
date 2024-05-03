import commune as c
from commune.subspace.subspace import Subspace

subspace = Subspace()

class Subnet(c.Module):
    def __init__(self, network='subspace', netuid=1):
        self.network = c.module(network)(netuid=netuid)
        self.set_config(kwargs=locals())
    def call(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y
    
    def subnets(self, search=None, **kwargs):
        modules = c.modules(search=search, **kwargs)
        subnets = []
        for module in modules:
            if 'vali' in module and module.startswith('subnet'):
                subnet = module.split('.')[1]
                if (module[:len(module)-len('.vali')] + '.miner') in modules:
                    subnets.append(subnet)
        return subnets
    
    def my_trees(self, search=None, **kwargs):
        for tree in c.trees(search=search, **kwargs):
            c.print(tree)
            c.print(tree, 'This is the tree, it is a Munch object')
            return tree

    @classmethod
    def testnet(cls, miners=10, valis=1, **kwargs):
        futures = []
        for miner in range(miners):
            f = c.submit(c.serve, [f"subnet.miner::{miner}"])
            futures += [f]
        for vali in range(valis):
            f = c.submit(c.serve,[f"subnet.vali::{vali}"])
            futures += [f]
        
        for future in c.as_completed(futures):
            c.print(future.result())
        return {'miners': miners, 'valis': valis}

    @classmethod
    def repos(cls, search=None, **kwargs):
        repos = c.repos(search=search, **kwargs)
        return repos

    def new_repo(self, search=None, **kwargs):
        repos = c.repos(search=search, **kwargs)
        return repos
        

    def emissions(self, netuid = 0, network = "main", block=None, update=True, **kwargs):

        return subspace.query_vector('Emission', network=network, netuid=netuid, block=block, update=update, **kwargs)
    

    def my_subnets(self, *args, **kwargs):
        return self.network.my_subnets(*args, **kwargs)
    

    def subnets(self, *args, **kwargs):
        return self.network.subnets(*args, **kwargs)
    