import commune as c

addresses = c.addresses(network='remote')
c.print(c.call('module', fn='submit', kwargs={'fn': 'subspace.start_node', 'kwargs': {'node': 'alice'}}))