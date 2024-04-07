import commune as c

class Subnet(c.Module):
    def __init__(self, network = 'local'):
        self.init(locals()) # send locals() to init
    
    def test(self, n=3):
        for i in range(n):
            c.print(c.serve(f'subnet.miner::{i}'))
        c.print(c.serve('subnet.vali', kwargs={'network': 'local'}))
    

    def testnet(self, miners=6, valis=2):
        servers = [f'subnet.miner::{i}' for i in range(miners)]
        servers += [f'subnet.vali::{i}' for i in range(valis)]
        c.server_many(servers)