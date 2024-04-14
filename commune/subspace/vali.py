import commune as c


class Vali(c.m('vali')):
    def __init__(self, 
                 search='subspace', 
                 reference='subspace', 
                 netuid = 0,
                 network = 'local',
                 **kwargs):
        self.init_vali(kwargs)
        self.reference = c.m(reference)()
        self.sync_time = 10

    def get_module_key(self):
        keys = self.reference.keys(netuid = self.config.netuid)
        return c.shuffle(keys)[0]


    def score_module(self, module):
        key = self.get_module_key()
        local_output = self.reference.get_module(key)
        remote_output = module.get_module(key)

        c.print('remote_output', remote_output)
        c.print('local_output', local_output)
        remote_hash = c.hash(remote_output)
        local_hash = c.hash(local_output)

        if local_hash == remote_hash:
            return 1
        else:   
            return 0
        

