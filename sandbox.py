import commune as c

# addresses = c.addresses('module', network='remote')
# c.print(c.call(addresses[0], fn='submit', kwargs={'fn': 'subspace.start_node', 'kwargs': {'node': 'alice'}}))
c.print(c.rcmd('cd commune/subspace; git checkout main', verbose=True))
