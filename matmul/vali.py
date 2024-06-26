import commune as c
import random
import torch

class Vali(c.Vali):

    whitelist = ['get_module', 'eval_module', 'leaderboard']

    def __init__(self, 
                 search='matmul.miner', 
                 netuid = 0,
                 alpha = 1.0,
                 n = 10,
                 network = 'local',
                 **kwargs):
        self.init_vali(locals())
        self.local_miner = c.module('matmul.miner')()

    def get_sample(self, n=10):
        return dict( x =  torch.rand(n, n) ,
                y = torch.rand(n, n))

    def clip_distance(self, x, y):
        return min(max(torch.norm(x - y).item(), 0), 1)
    
    def score(self, module):
        if isinstance(module, str):
            module = c.connect(module)
        sample = self.get_sample()
        local_response = self.local_miner.forward(sample['x'], sample['y'], n=self.config.n)
        t0 = c.time()
        remote_response = module.forward(sample['x'], sample['y'] + 1, n=self.config.n)
        t1 = c.time()
        latency = t1 - t0
        score = self.clip_distance(x=local_response, y=remote_response)
        return {'w': score, 'latency': latency}

