import commune as c
modules = ['key', 'namespace', 'server', 'subspace', 'module']
for module in modules:
    c.print('Testing module: ' + module)
    c.module(module).test()
