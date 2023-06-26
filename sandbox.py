import commune as c

s = c.module('subspace')()

# s.state_dict(save=True, cache=True)
s.state_dict(load=True, cache=True)
c.print('bro')