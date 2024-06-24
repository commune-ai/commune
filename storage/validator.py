import commune as c
import random


class Vali(c.m('vali')):

    whitelist = ['get_module', 'eval_module', 'leaderboard']

    def __init__(self, 
                 search='storage', 
                 netuid = 0,
                 alpha = 1.0,
                 network = 'subspace:test',
                 **kwargs):

        self.init_vali(locals())

    @classmethod
    def get_sample(cls):
        return random.choice(range(100))

    def score(self, module):
        sample = self.get_sample()
        sample_hash = c.hash(sample)
        
        module.put_item(sample_hash, sample)
        remote_data = module.get_item(sample_hash)

        remote_data_hash = c.hash(remote_data)

        if sample_hash == remote_data_hash:
            return 1
        else:
            return 0

