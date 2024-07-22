import commune as c
import bittensor as bt

class Bittensor(c.Module):
    def __init__(self, **kwargs):
        pass

    def metagraph(self, netuid: int=0):
        return bt.metagraph(netuid=netuid)
    meta = metagraph