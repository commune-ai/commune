import commune as c


class Vali(c.m('vali')):

    whitelist = ['get_module', 'eval_module', 'leaderboard']

    def __init__(self, 
                 search='subspace', 
                 reference='subspace', 
                 netuid = 0,
                 network = 'subspace',
                 **kwargs):

        self.init_vali(locals())
        c.print(self.config)
        self.reference = c.m(reference)()
        self.sync_time = 10

    def get_module_key(self):
        keys = self.reference.keys(netuid = self.config.netuid)
        return c.shuffle(keys)[0]

    def score(self, module):
        if isinstance(module, str):
            module = c.connect(module, network = self.config.network)
        key = self.get_module_key()
        local_output = self.reference.get_module(key)
        remote_output = module.get_module(key)
        remote_hash = c.hash(remote_output)
        local_hash = c.hash(local_output)

        if local_hash == remote_hash:
            return 1
        else:   
            return 0
        
    def get_module(self, module, **kwargs):
        return self.reference.get_module(module, **kwargs)
        

