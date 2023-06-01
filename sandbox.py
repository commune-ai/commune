import commune as c

module = c.connect('bittensor')
c.print(module.info())

# model.forward(sample['input_ids'])