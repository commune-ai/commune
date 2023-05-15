import commune as c

# c.print(c.call('dataset', 'sample')['input_ids'].shape)

import bittensor

subtensor = bittensor.subtensor()

print(subtensor.query_subtensor('Burn', None, [3]).value)