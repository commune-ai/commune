import commune

batch_size = 10
sequence_length = 10
dataset = commune.connect('dataset')
sample = dataset.sample(batch_size=batch_size,sequence_length=sequence_length)

print(sample)