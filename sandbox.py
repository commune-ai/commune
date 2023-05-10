import commune as c

# c.print(c.call('dataset', 'sample')['input_ids'].shape)

import bittensor
print(c.import_object('bittensor.neurons.core_server.server'))
print(bittensor.neurons.core_server.server)