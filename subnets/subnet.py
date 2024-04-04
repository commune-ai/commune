import commune as c

class Subnet(c.Module):
    def __init__(self, a=1, b=2):
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


    def new_subnet(self, search=None, **kwargs):
        modules = c.modules(search=search, **kwargs)
        subnets = []
        for module in modules:
            if 'vali' in module and module.startswith('subnet'):
                subnet = module.split('.')[1]
                if (module[:len(module)-len('.vali')] + '.miner') in modules:
                    subnets.append(subnet)
        return subnets


    @classmethod
    def repos(cls, search=None, **kwargs):
        repos = c.repos(search=search, **kwargs)
        return repos

    def new_repo(self, search=None, **kwargs):
        repos = c.repos(search=search, **kwargs)
        return repos
        