import commune as c
import torch


# s = c.module('subspace')()
x = {'bro': c.tensor([0,1]*100000)}
c.print(c.sizeof(x))
# auth = c.module()().auth()
# c.print(c.verify(auth))