import commune as c

# s = c.module('subspace')()


module = c.connect('module')
c.print(module.server_info)
# auth = c.module()().auth()
# c.print(c.verify(auth))