import commune as c

# c.print(c.call('dataset', 'sample')['input_ids'].shape)

import bittensor
print(isinstance(bittensor.wallet(), bittensor.Wallet))