import commune as c
import random
import torch

class Vali(c.Module):

    whitelist = ['get_module', 'eval_module', 'leaderboard']

    def __init__(self, 
                 search='matmul.miner', 
                 netuid = 0,
                 alpha = 1.0,
                 network = 'local',
                 **kwargs):
        self.init_vali(locals(), module=self)
        self.local_miner = c.module('matmul.miner')()

    def get_sample(self, n=10):
        return dict( x =  torch.rand(n, n) ,
                y = torch.rand(n, n))

    def score(self, module):
        if isinstance(module, str):
            module = c.connect(module)
        sample = self.get_sample()
        local_response = self.local_miner.forward(sample['x'], sample['y'])
        t0 = c.time()
        remote_response = module.forward(sample['x'], sample['y'])
        t1 = c.time()
        latency = t1 - t0
        score = max(min(1 - torch.norm(local_response - remote_response), 0), 1)
        return {'w': score, 'latency': latency}

