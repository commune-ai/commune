import argparse
import commune as c
futures = []
for i in range(10):
    futures += [c.submit(c.serve, dict(module='model.openai', tag=str(i)))]

for f in c.as_completed(futures):
    print(f.result(), 'This is the result of the future')
