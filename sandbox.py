import commune as c

# x = 'fam wahtdup'
# servers = c.servers(network='remote')
# c.print(servers)
# c.print(c.submit('module', fn = 'submit',  kwargs=dict(fn='print', args=[x], network='remote')))


output = c.rcmd('c addy')
network = 'remote'
c.rm_namespace(network=network)
for v in output.values():
    c.add_server(v[0], network=network)
