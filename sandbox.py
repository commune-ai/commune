import commune as c

# s = c.module('subspace')()

auth = c.module()().auth()
c.print(c.verify(auth))