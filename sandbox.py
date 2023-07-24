import commune as c
import torch

data = c.connect('dataset.hf')

c.print(data.info())

t = c.time()
num_steps = 10
for i in range(num_steps):
    x = data.sample(idx=[0,1,2,3])
    c.print(x)

    c.print(i/(c.time() - t))