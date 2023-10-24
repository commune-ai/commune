import commune as c

# x = 'fam wahtdup'
# servers = c.servers(network='remote')
# c.print(servers)
# c.print(c.submit('module', fn = 'submit',  kwargs=dict(fn='print', args=[x], network='remote')))

# root_key = c.root_key()
# root_key_address = root_key.ss58_address
# c.rcmd(f'c add_admin {root_key_address}')
# output = c.rcmd('c get_address subspace')
# network = 'subspace'
# c.rm_namespace(network=network)
# for v in output.values():
#     c.add_server(v[0], network=network)


c.print(c.rcmd('c ip update=True'))
