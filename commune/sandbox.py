import commune

batch_size = 10
sequence_length = 256
dataset = commune.connect('dataset')
sample = dataset.sample(batch_size=batch_size,sequence_length=sequence_length)
model = commune.connect('model')
sample.pop('attention_mask')
# output = model.forward(**sample)
print(sample['input_ids'].shape)
# print(output)