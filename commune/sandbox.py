import commune as c
import torch
app = c.module('subspace.app')()
c.print(app.global_state())



# c.print(subspace.set_weights(weights=weights, uids=uids, key=key, netuid=netuid))

