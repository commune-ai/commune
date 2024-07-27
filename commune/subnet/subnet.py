import commune as c
class Subnet(c.Module):
    def __init__(self, netuid=0, api='subspace.api'):
        self.api = c.connect(api)
        self.netuid = netuid
        self.sync()
        
    def sync(self):
        self.__dict__.update(self.api.subnet_state(netuid=self.netuid))
        return self.__dict__
    
    def set_weights(self, uids, weights):
        return self.api.set_weights(uids=uids, weights=weights, netuid=self.netuid, key=self.key, nonce=self.nonce)
    
    def get_nonce(self):
        return self.api.get_nonce(self.key)
    
    def incentives(self, update=False):
        return self.api.incentives(netuid=self.netuid, update=update)
    
    def get_weights(self):
        return self.api.get_weights(netuid=self.netuid, key=self.key)
    
    def get_module(self):
        return self.api.get_module(netuid=self.netuid, key=self.key)
    
    def set_module(self, module):
        return self.api.set_module(netuid=self.netuid, key=self.key, module=module)
    
    def get_key(self):
        return self.api.get_key(netuid=self.netuid)
    
    def set_key(self, key):
        return self.api.set_key(netuid=self.netuid, key=key)
    
    def get_params(self):
        return self.api.get_params(netuid=self.netuid)
    
    def set_params(self, params):
        return self.api.set_params(netuid=self.netuid, params=params)
    
    def get_modules(self):
        return self.api.get_modules(netuid=self.netuid)
    
    def get_subnets(self):
        return self.api.get_subnets(netuid=self.netuid)
    
    def set_subnets(self, subnets):
        return self.api.set_subnets(netuid=self.netuid, subnets=subnets)
    
    def get_subnet(self):
        return self.api.get_subnet(netuid=self.netuid)
    
    def set_subnet(self, subnet):
        return self.api.set_subnet(netuid=self.netuid, subnet=subnet)
    
    def get_subnet2netuid(self):
        return self.api.get_subnet2netuid(netuid=self.netuid)
    
    def set_subnet2netuid(self, subnet2netuid):
        return self.api.set
    
