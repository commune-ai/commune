import commune as c
import bt as bt

class Bittensor(c.Module):
    def __init__(self, **kwargs):
        pass

    def metagraph(self, netuid: int=0):
        return dir(bt.metagraph(netuid=netuid))
    

    def subnets_paths(self):
        return bt.subnets.paths