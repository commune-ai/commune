import commune as c

sample = c.connect('dataset')
model = c.module('model.dendrite')

model.forward(sample['input_ids'])