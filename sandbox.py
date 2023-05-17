import commune as c

# bt = c.module('bittensor')
# print(bt.get_metagraph())


server = c.import_object('commune.bittensor.neuron.text.server')
server.serve()
print(server)
