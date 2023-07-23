import commune as c

# data = c.connect('dataset.hf')
data = c.connect('dataset.hf')
c.print(data.info())

t = c.time()
num_steps = 10000
for i in range(num_steps):
    data.sample()

    c.print(i/(c.time() - t))