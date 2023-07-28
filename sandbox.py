import commune as c
import torch

module = c.connect('module')


t = c.time()
num_steps = 100
jobs = []
for i in range(num_steps):
    x = module.info(return_future=True)
    jobs.append(x)
    seconds = c.time() - t

jobs = c.gather(jobs)

jobs_per_second = len(jobs)/(c.time() - t)
c.print(jobs_per_second, color='green')