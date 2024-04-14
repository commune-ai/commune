import commune as c

class Wombo(c.Module):
    def __init__(self, network = 'local'):
        self.set_config(locals()) # send locals() to init
    
    def testnet(self, miners=3, valis=1):
        miners = [f'subnet.miner::{i}' for i in range(miners)] 
        valis = [f'subnet.vali::{i}' for i in range(valis)]
        results = []
        for s in miners + valis:
            results += [c.submit(c.serve, params=[s])]
        results = c.wait(results)
        return results