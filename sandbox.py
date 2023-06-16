import commune as c

bt = c.module('bittensor')()
meta = bt.metagraph
c.print(meta.neurons)