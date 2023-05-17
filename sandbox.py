import commune as c


server = c.import_object('commune.bittensor.neuron.text.server')
server.serve()
print(server)