import commune as c

servers = c.servers('subspace::')

for s in servers:
    c.print(c.register(s))
