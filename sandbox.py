import argparse
import commune as c
futures = []
n = 30
for i in range(n):
    c.print(i)
    futures += [c.submit(c.register, dict(module=f'agi::{i}'))]

c.wait(futures)
