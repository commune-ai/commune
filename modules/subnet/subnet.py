import commune as c
import pandas as pd

class Subnet(c.Module):
    def score_module(self, module):
        a = c.random_int()
        b = c.random_int()
        output = module.forward(a, b)

        if output == a + b:
            return 1
        else:
            return 0
        
